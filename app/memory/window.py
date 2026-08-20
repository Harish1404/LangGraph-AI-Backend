"""
The window buffer: how much of a conversation the model gets to see.

This is the async-native equivalent of the old
`ConversationBufferWindowMemory(k=...)` — keep the last k exchanges, drop
everything older. That class was deleted in LangChain 1.0 along with the rest
of `langchain.memory`, and its MongoDB backend was blocking pymongo, so the
same behaviour is reproduced here over the existing Motor client.

A *turn* is a user message plus its answer. The window is measured in turns
rather than raw messages deliberately: a window that cuts between a question
and its answer hands the model a dangling question and produces visibly
confused replies.
"""

import logging
from typing import Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import settings
from app.memory.store import conversation_store

logger = logging.getLogger(__name__)


def _window_size(turn_count: int) -> int:
    """Long threads earn a slightly wider window."""
    if turn_count > settings.LARGE_HISTORY_THRESHOLD:
        return settings.WINDOW_K_LARGE
    return settings.WINDOW_K


def to_messages(docs: Sequence[dict]) -> list[BaseMessage]:
    """
    Maps stored documents to LangChain messages.

    Only plain user/assistant text is reconstructed. Tool activity is stored as
    metadata but never rebuilt into AIMessage.tool_calls / ToolMessage pairs —
    replaying a tool call without its result makes the Groq API reject the
    whole request, and the finished answer already contains everything the
    model needs from that tool.
    """
    messages: list[BaseMessage] = []
    for doc in docs:
        content = doc.get("content") or ""
        if not content.strip():
            continue
        if doc.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


async def load(conversation_id: str, exclude_latest: bool = True) -> list[BaseMessage]:
    """
    Returns the last k turns of `conversation_id` as LangChain messages.

    `exclude_latest` drops the newest stored message, because the caller writes
    the incoming question to the database *before* loading history (so it
    survives a crash mid-answer) and then appends that same question to the
    prompt itself. Without this the model would see the question twice.
    """
    try:
        turn_count = await conversation_store.count_turns(conversation_id)
        k = _window_size(turn_count)

        limit = k * 2 + (1 if exclude_latest else 0)
        docs = await conversation_store.get_recent_messages(conversation_id, limit)

        if exclude_latest and docs:
            docs = docs[:-1]

        messages = to_messages(docs)
        logger.info(
            f"Memory window for {conversation_id}: {len(messages)} message(s) "
            f"(k={k} turns, thread has ~{turn_count} turn(s))"
        )
        return messages

    except Exception as e:
        # Losing memory degrades the answer; failing the request loses it
        # entirely. Degrade.
        logger.error(f"Could not load memory for {conversation_id}: {e}")
        return []


def as_text(messages: Optional[Sequence[BaseMessage]], empty: str = "(no previous messages)") -> str:
    """
    Flattens messages into a plain transcript for the router prompt.

    The router is a small, fast classifier fed through a single prompt
    template, so it takes history as text rather than as a message list.
    """
    if not messages:
        return empty

    lines = []
    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)
