"""Search the vector store and put the result in the state."""

import logging

from app.ai.agents.state import RagState
from app.prompts.rag_prompt import build_context_text
from app.rag.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)


async def retrieve(state: RagState) -> dict:
    """
    Retrieve resume chunks for whichever route brought us here.

    Searches with the router's *rewritten* question, not the raw one: a
    follow-up like "and where did he study?" has no searchable terms of its own,
    so embedding it directly returns nothing useful no matter how good the
    vector store is.

    This one node serves both the RAG and the BOTH route. They used to be two
    identical nodes (`retrieve` and `retrieve_for_both`) that differed only in
    where the graph went afterwards — which is now a conditional edge instead.
    """
    retrieved_chunks = await rag_pipeline.retrieve(state["search_query"])
    context_text = build_context_text(retrieved_chunks)

    return {"context": context_text}
