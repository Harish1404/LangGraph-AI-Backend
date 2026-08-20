"""
Request and response models for the chat and conversation endpoints.

These also translate between the MongoDB document shape (`_id`) and the API
shape (`conversation_id` / `message_id`), so the database layout is never
exposed directly to a client.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    A single user turn.

    `conversation_id` is optional: omitting it starts a new thread, which is
    what makes the very first message of a chat work without a separate
    "create conversation" round-trip.

    There is deliberately no `user_id` field. It used to be one, defaulting to
    "default_user", which meant the caller declared who they were — anyone
    could act as anyone by editing one line of JSON. Identity now comes from
    the session cookie via app/api/deps.py, and removing the field is what
    makes spoofing impossible rather than merely discouraged.
    """
    user_prompt: str = Field(..., min_length=1, max_length=8000)
    conversation_id: Optional[str] = None


class ConversationCreate(BaseModel):
    """Same rule as ChatRequest: the owner comes from the session, not the body."""
    title: Optional[str] = Field(default=None, max_length=200)


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ── Responses ─────────────────────────────────────────────────────────────────

class ConversationOut(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    message_count: int
    last_message_preview: str = ""
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    message_id: str
    conversation_id: str
    seq: int
    role: Literal["user", "assistant"]
    content: str
    route: Optional[str] = None
    # Which tools ran for this answer. Display only — never replayed to the model.
    tool_calls: list[dict] = []
    partial: bool = False
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []


# ── Document -> model ─────────────────────────────────────────────────────────

def conversation_out(doc: dict) -> ConversationOut:
    return ConversationOut(
        conversation_id=doc["_id"],
        user_id=doc.get("user_id", ""),
        title=doc.get("title", ""),
        message_count=doc.get("message_count", 0),
        last_message_preview=doc.get("last_message_preview", ""),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


def message_out(doc: dict) -> MessageOut:
    return MessageOut(
        message_id=doc["_id"],
        conversation_id=doc["conversation_id"],
        seq=doc["seq"],
        role=doc["role"],
        content=doc.get("content", ""),
        route=doc.get("route"),
        tool_calls=doc.get("tool_calls") or [],
        partial=doc.get("partial", False),
        created_at=doc["created_at"],
    )


def conversation_detail(doc: dict, messages: list[dict]) -> ConversationDetail:
    return ConversationDetail(
        **conversation_out(doc).model_dump(),
        messages=[message_out(m) for m in messages],
    )
