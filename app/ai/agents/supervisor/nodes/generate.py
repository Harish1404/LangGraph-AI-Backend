"""The supervisor's own answering node: no retrieval, no tools."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.ai.agents.state import ChatState
from app.ai.chat import _build_models
from app.core.config import settings
from app.prompts.router_prompt import DENIED_TOOL_NOTE, DIRECT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def generate(state: ChatState) -> dict:
    """
    Answer from the model's own knowledge.

    Two ways to get here, and they share a prompt:

      DIRECT   the router said this needs neither the resume nor a tool.

      refused  the tool agent asked to run a gated tool and the user said no,
               so it ended with `denied_tools` set and the supervisor sent the
               turn here instead of to END.

    The node name matters — see the note in rag_agent/nodes/generate.py.
    """
    llm_with_fallbacks, _ = _build_models(
        settings.light_max_tokens, settings.reasoning_max_tokens
    )

    # Not messages[-1]: on the refusal path that is a denial ToolMessage, not
    # the question. See the note on `user_prompt` in agents/state.py.
    user_prompt = state.get("user_prompt") or state["messages"][-1].content
    denied_tools = state.get("denied_tools") or []

    system_prompt = DIRECT_SYSTEM_PROMPT

    # Arriving here after a refused tool call. Without this the model sees a
    # ToolMessage saying "declined" and a system prompt that knows nothing about
    # tools, and tends to either invent the answer or ask to try again.
    if denied_tools:
        system_prompt += DENIED_TOOL_NOTE.format(tools=", ".join(denied_tools))

    history = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]

    if isinstance(state["messages"][-1], ToolMessage):
        # Rejection path. The AIMessage/ToolMessage pairing has to stay intact —
        # an assistant tool_call with no matching tool result is a 400 from both
        # Groq and Gemini — so append the question rather than replacing the tail.
        messages = [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=user_prompt),
        ]
    else:
        messages = [
            SystemMessage(content=system_prompt),
            *history[:-1],
            HumanMessage(content=user_prompt),
        ]

    response = await llm_with_fallbacks.ainvoke(messages)

    return {"messages": [response]}
