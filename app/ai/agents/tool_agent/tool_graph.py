"""
The tool agent — a subgraph that owns everything tool-calling.

    START → call_llm_with_tools ──┬── no tool_calls ──→ END
                                  └── tool_calls ────→ approve_tools
                                                          ├─ accept → tools
                                                          └─ deny   → END

    tools → call_llm_with_tools   (loop back so the model can use the results)

It serves two of the supervisor's routes: TOOL (tools alone) and the tail of
BOTH (tools with the RAG agent's context already in the state).

Two things to know about the exits:

  - Tool calls never reach the ToolNode directly. They go through
    approve_tools, which is what forwards them on — or refuses.

  - The refusal exit leaves `denied_tools` set on the way out. The supervisor
    reads that and sends the turn to its own `generate` for a tool-free answer.

Compiled with a bare .compile(): a subgraph must NOT be given its own
checkpointer. The supervisor's propagates down, and that is precisely what lets
the interrupt() in approve_tools survive the round-trip to the browser.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.ai.agents.state import ToolState
from app.ai.agents.tool_agent.nodes.approve_tools import approve_tools
from app.ai.agents.tool_agent.nodes.call_llm_with_tools import call_llm_with_tools
from app.ai.agents.tool_agent.toolbelt import TOOLS


def should_continue(state: ToolState) -> Literal["approve", "end"]:
    """After the model call: did it ask for a tool, or is the answer ready?"""
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "approve"

    return "end"


# LangGraph's built-in ToolNode reads tool_calls off the last AIMessage, runs
# each tool, and appends the ToolMessages — no hand-written loop needed.
tools_node = ToolNode(TOOLS)


# Which of this agent's nodes stream text meant for the user. Declared next to
# add_node so a rename has to touch both lines at once — app/ai/chat.py filters
# the token stream on these names, and a mismatch drops the answer silently.
ANSWER_NODES = {"tool_llm"}


builder = StateGraph(ToolState)

builder.add_node("tool_llm", call_llm_with_tools)
builder.add_node("approve_tools", approve_tools)
builder.add_node("tools", tools_node)

builder.add_edge(START, "tool_llm")
builder.add_conditional_edges(
    "tool_llm",
    should_continue,
    {
        "approve": "approve_tools",
        "end": END,
    },
)

# approve_tools has no static edges — it returns Command(goto=...), and LangGraph
# reads its possible destinations off the Command[Literal["tools", "__end__"]]
# return annotation.

builder.add_edge("tools", "tool_llm")

tool_graph = builder.compile()
tool_graph.name = "tool_agent"
