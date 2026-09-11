"""
The stable entry point into the multi-agent system.

Everything outside this package (app/main.py, app/ai/chat.py,
app/eval/evaluate_agent.py) imports the graph from here rather than reaching
into app.ai.agents.supervisor directly. That keeps the agent layout free to
move around without touching the rest of the app.

The graph itself lives in supervisor/supervisor_graph.py; the individual agents
live in rag_agent/ and tool_agent/, each with its own compiled subgraph and its
own nodes/ folder.
"""

from app.ai.agents.rag_agent.rag_graph import ANSWER_NODES as _RAG_ANSWER_NODES
from app.ai.agents.supervisor.supervisor_graph import (
    ANSWER_NODES as _SUPERVISOR_ANSWER_NODES,
    get_chat_graph,
    init_chat_graph,
)
from app.ai.agents.tool_agent.tool_graph import ANSWER_NODES as _TOOL_ANSWER_NODES


# Every node in the system that streams text meant for the user, gathered from
# the agents themselves rather than restated here.
#
# app/ai/chat.py filters the token stream on `langgraph_node`, which for a
# subgraph is the *inner* node name — so this flat union is the right shape.
# Not every LLM call in the system is an answer: the router runs a classifier
# whose raw JSON would otherwise be prepended to the reply and saved into the
# transcript as part of it.
#
# Collecting these from each agent module is deliberate. The previous hardcoded
# literal in chat.py silently stopped matching when `call_llm_with_tools` was
# renamed to `tool_llm`, which dropped every TOOL and BOTH answer on the floor
# with no error anywhere. Now a rename that misses its ANSWER_NODES entry is a
# one-line diff away from the add_node it belongs to.
ANSWER_NODES = (
    _SUPERVISOR_ANSWER_NODES | _RAG_ANSWER_NODES | _TOOL_ANSWER_NODES
)

__all__ = ["ANSWER_NODES", "get_chat_graph", "init_chat_graph"]
