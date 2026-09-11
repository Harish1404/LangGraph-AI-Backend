"""
LangSmith LLM-as-a-Judge Evaluation Script
───────────────────────────────────────────
Runs the LangGraph agent against a LangSmith dataset and scores every
response with four LLM judges powered by `openevals`:

    1. Correctness      — reference-based  (CORRECTNESS_PROMPT)
    2. Relevance        — reference-free   (ANSWER_RELEVANCE_PROMPT)
    3. Faithfulness     — context-based    (RAG_GROUNDEDNESS_PROMPT)
    4. Conciseness      — reference-free   (CONCISENESS_PROMPT)

Usage:
    python -m app.eval.evaluate_agent --dataset "YOUR_DATASET_NAME"

The script bootstraps MongoDB (for the RAG vector store) and compiles the
LangGraph graph with an in-memory checkpointer — no running FastAPI server
needed.  Results are posted to LangSmith and visible in the dashboard under
Datasets & Testing.

Prerequisites:
    pip install openevals
"""

import argparse
import asyncio
import logging
import uuid

from langchain_core.messages import HumanMessage
from langsmith.evaluation import aevaluate

from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    CONCISENESS_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
)

from app.core.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.rag.rag_pipeline import rag_pipeline
from app.ai.agents.graph import get_chat_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Judge model ──────────────────────────────────────────────────────────────
# Uses your MISTRAL_API_KEY from .env — lightweight, fast, and highly reliable
# for structured evaluation reasoning.
DEFAULT_JUDGE_MODEL = "mistralai:mistral-small-latest"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Extractors
# ═══════════════════════════════════════════════════════════════════════════════
# The dataset was created from LangGraph traces, so the structure is:
#   inputs  = {"messages": [{"type": "human", "content": "..."}]}
#   outputs = {"messages": [..., {"type": "ai", "content": "..."}],
#              "context": "...", "route": "RAG"|"DIRECT"|...}


def extract_question(inputs: dict) -> str:
    """Pull the user question out of the dataset's nested message format."""
    messages = inputs.get("messages", [])
    if messages and isinstance(messages[-1], dict):
        return messages[-1].get("content", "")
    # Fallback: maybe it was stored flat
    return str(inputs.get("user_prompt", inputs.get("question", "")))


def extract_reference(outputs: dict) -> str:
    """Pull the expected AI answer from the dataset outputs."""
    if not outputs:
        return ""
    messages = outputs.get("messages", [])
    # The last message in the trace is the AI response
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("type") == "ai":
            return msg.get("content", "")
    return ""


def extract_reference_context(outputs: dict) -> str:
    """Pull the RAG context from the dataset outputs (for faithfulness baseline)."""
    if not outputs:
        return ""
    return outputs.get("context", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Target Agent Function
# ═══════════════════════════════════════════════════════════════════════════════


async def predict_agent(inputs: dict) -> dict:
    """
    Invokes the LangGraph agent for a single evaluation example.

    Each call gets a fresh thread_id so conversation state from one test case
    never leaks into another.
    """
    question = extract_question(inputs)
    thread_id = f"eval_{uuid.uuid4()}"
    graph = get_chat_graph()

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_prompt": question,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract the final AI message
    last_message = result["messages"][-1]
    output_text = (
        last_message.content
        if hasattr(last_message, "content")
        else str(last_message)
    )

    return {
        "output": output_text,
        "route": result.get("route", "DIRECT"),
        "context": result.get("context", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Judge Evaluators  (openevals)
# ═══════════════════════════════════════════════════════════════════════════════


def build_evaluators(judge_model: str = DEFAULT_JUDGE_MODEL):
    """Builds the 4 openevals LLM-as-a-judge functions."""

    def _correctness_evaluator(run, example) -> dict:
        """Correctness: reference-based comparison."""
        judge = create_llm_as_judge(
            prompt=CORRECTNESS_PROMPT,
            feedback_key="correctness",
            model=judge_model,
        )
        return judge(
            inputs=extract_question(example.inputs),
            outputs=run.outputs["output"],
            reference_outputs=extract_reference(example.outputs),
        )

    def _relevance_evaluator(run, example) -> dict:
        """Relevance: reference-free check against user question."""
        judge = create_llm_as_judge(
            prompt=ANSWER_RELEVANCE_PROMPT,
            feedback_key="relevance",
            model=judge_model,
        )
        return judge(
            inputs=extract_question(example.inputs),
            outputs=run.outputs["output"],
        )

    def _faithfulness_evaluator(run, example) -> dict:
        """Faithfulness / Groundedness: context-based check against hallucinations."""
        context = run.outputs.get("context", "")
        if not context:
            context = extract_reference_context(example.outputs)
        if not context:
            context = "No context was provided — this was a direct answer without retrieval."

        judge = create_llm_as_judge(
            prompt=RAG_GROUNDEDNESS_PROMPT,
            feedback_key="faithfulness",
            model=judge_model,
        )
        return judge(
            inputs=extract_question(example.inputs),
            outputs=run.outputs["output"],
            context=context,
        )

    def _conciseness_evaluator(run, example) -> dict:
        """Conciseness: checks for succinctness without unnecessary fluff."""
        judge = create_llm_as_judge(
            prompt=CONCISENESS_PROMPT,
            feedback_key="conciseness",
            model=judge_model,
        )
        return judge(
            inputs=extract_question(example.inputs),
            outputs=run.outputs["output"],
        )

    return [
        _correctness_evaluator,
        _relevance_evaluator,
        _faithfulness_evaluator,
        _conciseness_evaluator,
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap & Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main(
    dataset_name: str,
    experiment_prefix: str,
    max_concurrency: int,
    judge_model: str = DEFAULT_JUDGE_MODEL,
):
    """
    1. Connect to MongoDB (needed for RAG vector store)
    2. Ingest documents into the vector store
    3. Run aevaluate against the LangSmith dataset
    4. Clean up
    """
    logger.info("=" * 60)
    logger.info("LangSmith LLM-as-a-Judge Evaluation")
    logger.info("=" * 60)

    # ── Step 1: Bootstrap MongoDB ────────────────────────────────────────────
    logger.info("Connecting to MongoDB (for RAG vector store)...")
    await connect_to_mongo()

    # ── Step 2: Ingest RAG documents ─────────────────────────────────────────
    logger.info("Ingesting documents into vector store...")
    chunk_count = await rag_pipeline.ingest("uploads")
    logger.info(f"RAG ingestion complete: {chunk_count} new chunk(s) indexed.")

    # ── Step 3: Verify graph is ready ────────────────────────────────────────
    # get_chat_graph() auto-falls-back to MemorySaver when called outside
    # the FastAPI lifespan — no MongoDB checkpointer needed for eval.
    graph = get_chat_graph()
    logger.info(f"Graph compiled: {type(graph).__name__}")

    # ── Step 4: Build evaluators ─────────────────────────────────────────────
    evaluators = build_evaluators(judge_model=judge_model)
    logger.info(f"Built {len(evaluators)} evaluators: Correctness, Relevance, Faithfulness, Conciseness")
    logger.info(f"Judge model: {judge_model}")

    # ── Step 5: Run evaluation ───────────────────────────────────────────────
    logger.info(f"Starting evaluation against dataset: '{dataset_name}'")
    logger.info(f"Experiment prefix: '{experiment_prefix}'")
    logger.info(f"Max concurrency: {max_concurrency}")
    logger.info("-" * 60)

    try:
        results = await aevaluate(
            predict_agent,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            max_concurrency=max_concurrency,
        )

        logger.info("-" * 60)
        logger.info("Evaluation completed successfully!")
        logger.info(
            "View results in LangSmith -> Datasets & Testing -> "
            f"'{dataset_name}' -> Experiments"
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
    finally:
        # ── Step 6: Cleanup ──────────────────────────────────────────────────
        await close_mongo_connection()
        logger.info("MongoDB connection closed. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge evaluation against a LangSmith dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the LangSmith dataset to evaluate against.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="langgraph-agent-eval",
        help="Experiment prefix shown in the LangSmith dashboard (default: langgraph-agent-eval).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max number of concurrent evaluation runs (default: 2).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use for scoring (default: {DEFAULT_JUDGE_MODEL}).",
    )

    args = parser.parse_args()

    asyncio.run(main(
        dataset_name=args.dataset,
        experiment_prefix=args.prefix,
        max_concurrency=args.concurrency,
        judge_model=args.judge_model,
    ))
