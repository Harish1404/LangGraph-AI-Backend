"""Answer strictly from the retrieved resume chunks."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.ai.agents.state import RagState
from app.ai.chat import _build_models
from app.core.config import settings
from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT
from app.prompts.router_prompt import DIRECT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def generate(state: RagState) -> dict:
    """
    Produce the final answer for the RAG route.

    The node name matters: app/ai/chat.py streams tokens only from nodes listed
    in ANSWER_NODES, and it matches on the *inner* node name of a subgraph.
    Renaming this to something else silently stops the answer reaching the
    browser.

    No tools are bound here — the RAG route is retrieval only.
    """
    llm_with_fallbacks, _ = _build_models(
        settings.light_max_tokens, settings.reasoning_max_tokens
    )

    # Not messages[-1]: see the note on `user_prompt` in agents/state.py.
    user_prompt = state.get("user_prompt") or state["messages"][-1].content
    context = state.get("context", "")

    if context:
        system_prompt = RAG_SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuestion: {user_prompt}"
    else:
        # Retrieval came back empty. Falling back to the model's own knowledge
        # beats answering from a prompt that promises context and has none.
        logger.info("RAG retrieval returned no context; answering directly.")
        system_prompt = DIRECT_SYSTEM_PROMPT
        user_content = user_prompt

    history = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]

    # Drop the raw question off the tail and rebuild it with the context folded
    # in — that rebuilt message is the whole point of the RAG route.
    messages = [
        SystemMessage(content=system_prompt),
        *history[:-1],
        HumanMessage(content=user_content),
    ]

    response = await llm_with_fallbacks.ainvoke(messages)

    return {"messages": [response]}
