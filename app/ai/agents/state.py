"""
The state schemas every graph in this package reads and writes.

There is one parent state (ChatState, owned by the supervisor) and one smaller
state per agent. The agent states are deliberate *subsets* of the parent: that
is what lets a compiled subgraph be dropped straight into the supervisor as a
node. LangGraph filters the parent state down to the subgraph's schema on the
way in, and merges whatever keys it returns back on the way out.

Two rules make that work, and breaking either one fails quietly:

  1. A key that appears in more than one schema must have the *same name* in
     all of them. There is no renaming layer.

  2. `messages` must carry the `add_messages` reducer in every schema. Without
     it the subgraph's return value *replaces* the conversation history instead
     of appending to it, and the turn silently loses everything said before.
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """The supervisor's state — the union of everything any agent needs."""

    messages: Annotated[Sequence[BaseMessage], add_messages]  # history (auto-appended)
    route: str                               # "RAG" | "TOOL" | "BOTH" | "DIRECT"
    search_query: str                        # rewritten question from the router
    context: str                             # retrieved RAG chunks (empty if unused)
    user_prompt: str                         # this turn's question, verbatim
    denied_tools: list[str]                  # tools the user refused this turn


# A note on the last two fields.
#
# `user_prompt` exists because messages[-1] is NOT reliably the user's question.
# After a trip through the tool node the last message is a ToolMessage, and on
# the rejection path it is a *denial* ToolMessage.  Nodes that need the question
# read this instead of guessing from the tail of the list.
#
# Every field except `messages` is per-turn and must be cleared by route_query.
# With a durable checkpointer the whole state is reloaded on the next turn, so a
# stale `context` or `denied_tools` would otherwise leak into an unrelated
# question. The old MemorySaver hid this by forgetting everything on restart.


class RagState(TypedDict):
    """What the RAG agent sees. No `denied_tools` — it never touches tools."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str                               # read to tell RAG apart from BOTH
    search_query: str                        # what retrieve() actually embeds
    context: str
    user_prompt: str


class ToolState(TypedDict):
    """What the tool agent sees. No `search_query` — it does not retrieve."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str                               # picks the TOOL vs BOTH system prompt
    context: str                             # populated by the RAG agent on BOTH
    user_prompt: str
    denied_tools: list[str]                  # written on the refusal path
