"""The human-in-the-loop gate. Runs before any tool does."""

import logging
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt

from app.ai.agents.state import ToolState
from app.core.config import settings

logger = logging.getLogger(__name__)


async def approve_tools(state: ToolState) -> Command[Literal["tools", "__end__"]]:
    """
    Ask the user before running a gated tool.

    Only tools listed in HITL_TOOLS are gated; everything else falls straight
    through, so the common case costs one set lookup.

    `interrupt()` does not block. It raises, LangGraph checkpoints the graph
    exactly where it stands, and the whole run unwinds — through this subgraph
    and out of the supervisor above it. When the caller resumes with
    Command(resume=...), **this node re-executes from its first line** and the
    interrupt() call returns the resume value instead of raising. That is why
    nothing above it may have a side effect: it all runs twice.

    Returns a Command rather than a dict so it can pick its own next node.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    gated = [tc for tc in tool_calls if tc["name"] in settings.HITL_TOOLS]

    if not gated:
        return Command(goto="tools")

    logger.info(f"Tool approval required for: {[tc['name'] for tc in gated]}")

    decision = interrupt({
        "type": "tool_approval",
        "tool_calls": [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in gated
        ],
    })

    # Anything that is not an explicit "accept" is a refusal. Fail closed: a
    # malformed resume payload must not be able to run a gated tool.
    action = decision.get("action") if isinstance(decision, dict) else None

    if action == "accept":
        logger.info(f"Tool approval GRANTED for: {[tc['name'] for tc in gated]}")
        return Command(goto="tools")

    # ── Refused ──────────────────────────────────────────────────────────────
    reason = decision.get("reason") if isinstance(decision, dict) else None
    denial = "The user declined to run this tool, so it produced no result."
    if reason:
        denial += f" Their reason: {reason}"

    logger.info(f"Tool approval DENIED for: {[tc['name'] for tc in gated]}")

    # A subgraph cannot goto a node in its parent, so the refusal path ends this
    # agent and lets the supervisor take it from here: it sees a non-empty
    # `denied_tools` on the way out and routes to its own `generate`, which
    # answers from the model's own knowledge. The tool is not retried — the user
    # already said no once.
    #
    # One ToolMessage per *pending* call, not just the gated ones. An AIMessage
    # whose tool_calls are not all answered is an invalid history: the next
    # provider call fails with a 400 rather than degrading. So every id gets
    # closed out, whether it was the one the user objected to or not.
    return Command(
        goto=END,
        update={
            "messages": [
                ToolMessage(
                    content=denial,
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ],
            "route": "DIRECT",
            "denied_tools": [tc["name"] for tc in gated],
        },
    )
