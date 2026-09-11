"""
The RAG agent — a subgraph that owns everything retrieval.

    START → retrieve ──┬── route == BOTH ──→ END      (context only)
                       └── otherwise ──────→ generate → END

It serves two of the supervisor's routes:

  RAG   the agent retrieves *and* answers. The supervisor wires it straight
        to END.

  BOTH  the agent only retrieves, then hands back. The supervisor passes the
        state on to the tool agent, which does the answering — it needs the
        resume context in the prompt to read a city out of it before calling
        the weather tool. Answering here would waste a model call and produce
        a reply the user never sees.

Compiled with a bare .compile(): a subgraph must NOT be given its own
checkpointer. The supervisor's propagates down, and that shared checkpoint is
what keeps this agent's work on the same thread as everything else.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.ai.agents.rag_agent.nodes.generate import generate
from app.ai.agents.rag_agent.nodes.retrieve import retrieve
from app.ai.agents.state import RagState


def should_answer(state: RagState) -> Literal["generate", "end"]:
    """After retrieve: answer here, or hand the context back to the supervisor?"""
    if state.get("route") == "BOTH":
        return "end"

    return "generate"


# Which of this agent's nodes stream text meant for the user. Declared next to
# add_node so a rename has to touch both lines at once — app/ai/chat.py filters
# the token stream on these names, and a mismatch drops the answer silently.
ANSWER_NODES = {"generate"}


builder = StateGraph(RagState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve",
    should_answer,
    {
        "generate": "generate",
        "end": END,
    },
)
builder.add_edge("generate", END)

rag_graph = builder.compile()
rag_graph.name = "rag_agent"
