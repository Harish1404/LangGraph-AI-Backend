"""The supervisor's first act: work out which agent should handle this turn."""

import logging

from app.ai.agents.router import query_router
from app.ai.agents.state import ChatState

logger = logging.getLogger(__name__)


async def route_query(state: ChatState) -> dict:
    """
    Classify the user's question and rewrite it.

    The router decides which path (RAG / TOOL / BOTH / DIRECT) the question
    should take, and also rewrites follow-up questions like "and where did he
    study?" into standalone ones the vector store can actually search for.

    This node is also the per-turn reset. Everything except `messages` belongs
    to a single turn, and with a durable checkpointer the whole state comes back
    on the next one — so a stale `context` or `denied_tools` would leak into an
    unrelated question if it were not cleared here.
    """
    # The last message is always the user's latest question
    last_message = state["messages"][-1]
    user_prompt = last_message.content

    # History for the router = everything except the latest message
    history = state["messages"][:-1]

    decision = await query_router.route(user_prompt, history)

    logger.info(
        f"Router selected: {decision.route} for query: {user_prompt!r} "
        f"(searching for: {decision.standalone_question!r})"
    )

    return {
        "route": decision.route,
        "search_query": decision.standalone_question,
        "user_prompt": user_prompt,
        "context": "",
        "denied_tools": [],
    }
