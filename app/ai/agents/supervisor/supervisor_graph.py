"""
The supervisor — the parent graph that orchestrates every agent.

Graph shape
───────────

    START → route_query
              ├── RAG    → [rag_agent] ────────────────────→ END
              ├── BOTH   → [rag_agent] → [tool_agent] ─────→ (denied? generate : END)
              ├── TOOL   → [tool_agent] ───────────────────→ (denied? generate : END)
              └── DIRECT → generate ───────────────────────→ END

`rag_agent` and `tool_agent` are not nodes in the ordinary sense — they are
whole compiled subgraphs dropped in as nodes. That works because their state
schemas (RagState, ToolState in agents/state.py) are subsets of ChatState:
LangGraph filters the state down on the way in and merges the returned keys
back on the way out. No adapter code in between.

What the supervisor keeps for itself
────────────────────────────────────

  routing   route_query classifies the turn and resets the per-turn state.

  DIRECT    `generate` — the tool-free, retrieval-free answer. Kept here rather
            than in an agent of its own because it is also the landing spot for
            a refused tool call, and a subgraph cannot goto a parent node.

  BOTH      the only route that spans two agents. The supervisor sequences it:
            the RAG agent retrieves and stops, then the tool agent answers with
            that context in its prompt.

Adding an agent later is: a new folder with its own compiled subgraph, one
add_node, and one more branch in pick_route.

Why the compile is deferred
───────────────────────────

The checkpointer needs a live MongoDB connection, and app/db/mongodb.py has not
connected yet when this module is first imported. So the lifespan in
app/main.py calls init_chat_graph() once, and everything else goes through
get_chat_graph(). The subgraphs, by contrast, compile at import time with no
checkpointer at all — the parent's propagates down to them.
"""

from functools import lru_cache
import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.ai.agents.rag_agent.rag_graph import rag_graph
from app.ai.agents.state import ChatState
from app.ai.agents.supervisor.nodes.generate import generate
from app.ai.agents.supervisor.nodes.route_query import route_query
from app.ai.agents.tool_agent.tool_graph import tool_graph

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional edges — the supervisor's dispatch logic
# ═══════════════════════════════════════════════════════════════════════════════

def pick_route(state: ChatState) -> Literal["rag_agent", "tool_agent", "generate"]:
    """
    After route_query: which agent handles this turn?

    RAG and BOTH both start at the RAG agent. The difference is what the agent
    does once it has the context — it reads `route` itself and stops early on
    BOTH — and where the supervisor sends things afterwards (see after_rag).
    """
    route = state.get("route", "DIRECT")

    if route in ("RAG", "BOTH"):
        return "rag_agent"
    elif route == "TOOL":
        return "tool_agent"
    else:
        return "generate"


def after_rag(state: ChatState) -> Literal["tool_agent", "end"]:
    """
    After the RAG agent: is this a BOTH turn that still needs the tool agent?

    On RAG the agent has already produced the answer, so the turn is over.
    """
    if state.get("route") == "BOTH":
        return "tool_agent"

    return "end"


def after_tools(state: ChatState) -> Literal["generate", "end"]:
    """
    After the tool agent: did the user refuse a gated tool?

    The tool agent cannot jump to a node in this graph, so it signals the
    refusal by leaving `denied_tools` populated on its way out. That is what
    sends the turn to `generate` for a tool-free answer instead of to END.
    """
    if state.get("denied_tools"):
        return "generate"

    return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# Assemble
# ═══════════════════════════════════════════════════════════════════════════════

# Which of the supervisor's OWN nodes stream text meant for the user. `router`
# is deliberately absent: it runs a structured-output classifier, and when Groq
# rate-limits and the Gemini fallback takes over, that call streams its raw
# {"route": ..., "standalone_question": ...} JSON as ordinary content. See
# ANSWER_NODES in agents/graph.py for how these are combined.
ANSWER_NODES = {"generate"}


builder = StateGraph(ChatState)

builder.add_node("router", route_query)
builder.add_node("generate", generate)
# The two subgraphs, added exactly like any other node.
builder.add_node("rag_agent", rag_graph)
builder.add_node("tool_agent", tool_graph)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    pick_route,
    {
        "rag_agent": "rag_agent",
        "tool_agent": "tool_agent",
        "generate": "generate",
    },
)

builder.add_conditional_edges(
    "rag_agent",
    after_rag,
    {
        "tool_agent": "tool_agent",
        "end": END,
    },
)

builder.add_conditional_edges(
    "tool_agent",
    after_tools,
    {
        "generate": "generate",
        "end": END,
    },
)

builder.add_edge("generate", END)


# ═══════════════════════════════════════════════════════════════════════════════
# Compile with a checkpointer (this is the memory)
# ═══════════════════════════════════════════════════════════════════════════════
# The checkpointer stores the full graph state in MongoDB, keyed by thread_id.
# At runtime you pass config={"configurable": {"thread_id": conversation_id}}
# and LangGraph loads and saves that thread's history for you — including the
# subgraphs' state, which is why they must not carry checkpointers of their own.

_chat_graph = None


@lru_cache(maxsize=4)
def _compile_graph(checkpointer_instance):
    """
    Cached compilation of the StateGraph against a checkpointer instance.
    Prevents redundant compilation latency on graph retrieval.
    """
    return builder.compile(checkpointer=checkpointer_instance)


def init_chat_graph(checkpointer) -> None:
    """Compile the graph against a checkpointer. Called once, from the lifespan."""
    global _chat_graph
    _chat_graph = _compile_graph(checkpointer)
    logger.info(
        "LangGraph supervisor compiled and cached with %s "
        "(agents: rag_agent, tool_agent).",
        type(checkpointer).__name__,
    )


def get_chat_graph():
    """
    The compiled supervisor graph.

    The MemorySaver fallback exists so that importing this module outside the
    app (a test, a script, a notebook) still works. It is loud because in a
    running server it means state is silently not being persisted, and — worse
    — that tool approvals cannot survive the round-trip to the browser.
    """
    global _chat_graph

    if _chat_graph is None:
        logger.warning(
            "get_chat_graph() called before init_chat_graph(); falling back to "
            "MemorySaver. Conversation state will NOT persist and tool approval "
            "will not survive a restart."
        )
        _chat_graph = _compile_graph(MemorySaver())

    return _chat_graph
