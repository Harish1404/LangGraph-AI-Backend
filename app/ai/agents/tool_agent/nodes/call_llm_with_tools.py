"""Ask the tool-aware model what to do, and then what it made of the results."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.ai.agents.state import ToolState
from app.ai.chat import _build_models
from app.core.config import settings
from app.prompts.router_prompt import BOTH_SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def call_llm_with_tools(state: ToolState) -> dict:
    """
    Call the LLM with the toolbelt bound.

    Runs twice per tool round-trip: once to decide (it either answers outright
    or asks for tools), and once more after the ToolNode to turn the raw tool
    output into an answer.

    On the BOTH route the retrieved resume context is folded into the prompt, so
    the model can read a city out of the resume before calling the weather tool.

    The node name matters — see the note in rag_agent/nodes/generate.py.
    """
    _, llm_with_tools = _build_models(
        settings.light_max_tokens, settings.reasoning_max_tokens
    )

    # NOT messages[-1]: on the second pass that is a ToolMessage, and using it
    # here would ask the model to answer its own tool output instead of the
    # question. See the note on `user_prompt` in agents/state.py.
    user_prompt = state.get("user_prompt") or state["messages"][-1].content
    context = state.get("context", "")
    route = state.get("route", "TOOL")

    if route == "BOTH" and context:
        system_prompt = BOTH_SYSTEM_PROMPT
        user_content = f"Resume context:\n{context}\n\nQuestion: {user_prompt}"
    else:
        system_prompt = TOOL_SYSTEM_PROMPT
        user_content = user_prompt

    history = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]

    if isinstance(state["messages"][-1], ToolMessage):
        # Second pass, coming back from the tool node. History already ends with
        # the AIMessage that requested the tools followed by the ToolMessages
        # answering them, and that pairing has to survive intact — Gemini and
        # Groq both reject a request where an assistant tool_call has no
        # matching tool result. So replay it verbatim; do not rebuild anything.
        messages = [SystemMessage(content=system_prompt), *history]
    else:
        # First pass. Drop the raw question and rebuild it with the RAG context
        # folded in, which is the whole point of the BOTH route.
        messages = [
            SystemMessage(content=system_prompt),
            *history[:-1],
            HumanMessage(content=user_content),
        ]

    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}
