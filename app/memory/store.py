"""
Persistence for conversations and messages.

This layer knows about MongoDB and nothing else — no LangChain types, no LLM
calls. The window-buffer logic that sits on top of it lives in
app/memory/window.py, and the two are kept apart so the storage schema can
change without touching how memory is assembled for the model.

Why this is hand-written rather than langchain's ConversationBufferWindowMemory:
that class was removed in LangChain 1.0 (`langchain.memory` no longer exists),
and its MongoDB backend was built on blocking pymongo, which would stall the
event loop inside these async endpoints. The window semantics it provided are
reproduced natively in window.py.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from pymongo import ReturnDocument

from app.core.ids import new_conversation_id, new_message_id
from app.db.mongodb import get_conversations_collection, get_messages_collection

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New chat"
TITLE_MAX_CHARS = 60
PREVIEW_MAX_CHARS = 120


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _derive_title(text: str) -> str:
    """Turns the first user message into a sidebar title, Claude-style."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return DEFAULT_TITLE
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned
    return cleaned[:TITLE_MAX_CHARS].rstrip() + "…"


class ConversationNotFound(Exception):
    """Raised when a write targets a conversation that does not exist."""


class ConversationStore:
    """CRUD over the `conversations` and `messages` collections."""

    # Resolved per call rather than in __init__, because the Motor client does
    # not exist until connect_to_mongo() runs in the app lifespan, and this
    # module is imported long before that.
    @property
    def conversations(self):
        return get_conversations_collection()

    @property
    def messages(self):
        return get_messages_collection()

    # ── Conversations ────────────────────────────────────────────────────────

    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
    ) -> dict:
        now = _now()
        doc = {
            "_id": new_conversation_id(),
            "user_id": user_id,
            "title": title or DEFAULT_TITLE,
            "message_count": 0,
            "last_message_preview": "",
            "created_at": now,
            "updated_at": now,
            "deleted": False,
        }
        await self.conversations.insert_one(doc)
        logger.info(f"Created conversation {doc['_id']} for user {user_id!r}")
        return doc

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> Optional[dict]:
        """
        Returns None for a missing, soft-deleted, or someone else's conversation.

        `user_id` is required, and that is the fix for a real hole: it used to
        be optional, and the query skipped the owner clause whenever it was
        falsy. Every caller that omitted it — which was every /conversations/{id}
        route — could therefore read any thread by id. Making the argument
        mandatory means the filter cannot be forgotten again.

        Not-yours is reported as None, so the route answers 404 rather than 403
        and the API does not confirm that an id exists to someone who cannot
        see it.
        """
        return await self.conversations.find_one(
            {"_id": conversation_id, "user_id": user_id, "deleted": {"$ne": True}}
        )

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0,
    ) -> list[dict]:
        """Newest-activity-first, the order a chat sidebar wants."""
        cursor = (
            self.conversations
            .find({"user_id": user_id, "deleted": {"$ne": True}})
            .sort("updated_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def rename_conversation(
        self, conversation_id: str, title: str, user_id: str
    ) -> Optional[dict]:
        """
        The owner clause is repeated here rather than left to the route's
        earlier existence check. Re-checking inside the write closes the gap
        between "we looked it up" and "we changed it", and means a future
        caller cannot reach this method without an owner.
        """
        return await self.conversations.find_one_and_update(
            {"_id": conversation_id, "user_id": user_id, "deleted": {"$ne": True}},
            {"$set": {"title": title, "updated_at": _now()}},
            return_document=ReturnDocument.AFTER,
        )

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """
        Soft delete. The messages stay on disk, so a thread can be restored and
        so an accidental click is not a permanent loss.
        """
        result = await self.conversations.update_one(
            {"_id": conversation_id, "user_id": user_id, "deleted": {"$ne": True}},
            {"$set": {"deleted": True, "updated_at": _now()}},
        )
        return result.modified_count > 0

    # ── Messages ─────────────────────────────────────────────────────────────

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        route: Optional[str] = None,
        tool_calls: Optional[list] = None,
        partial: bool = False,
    ) -> dict:
        """
        Appends a message and returns the stored document.

        The sequence number comes from a single atomic $inc on the parent
        conversation, so two concurrent requests on the same thread can never
        be handed the same `seq` — which is what keeps the replayed history in
        a coherent order under load.
        """
        conversation = await self.conversations.find_one_and_update(
            {"_id": conversation_id, "deleted": {"$ne": True}},
            {"$inc": {"message_count": 1}, "$set": {"updated_at": _now()}},
            return_document=ReturnDocument.AFTER,
        )
        if conversation is None:
            raise ConversationNotFound(conversation_id)

        seq = conversation["message_count"] - 1

        doc = {
            "_id": new_message_id(),
            "conversation_id": conversation_id,
            "seq": seq,
            "role": role,
            "content": content,
            "route": route,
            # Recorded for observability only. These are NEVER replayed to the
            # model: an assistant message carrying tool_calls without its
            # matching ToolMessage is rejected outright by the Groq API.
            "tool_calls": tool_calls or [],
            "partial": partial,
            "created_at": _now(),
        }
        await self.messages.insert_one(doc)

        updates: dict = {"last_message_preview": " ".join(content.split())[:PREVIEW_MAX_CHARS]}

        # The first user message names the thread, unless it was named at creation.
        if seq == 0 and role == "user" and conversation.get("title") == DEFAULT_TITLE:
            updates["title"] = _derive_title(content)

        await self.conversations.update_one({"_id": conversation_id}, {"$set": updates})

        return doc

    async def get_recent_messages(self, conversation_id: str, limit: int) -> list[dict]:
        """
        The last `limit` messages, oldest-first.

        Sorted descending in the query so the index does the work and only the
        tail is read, then reversed in memory — the alternative (ascending sort
        plus skip) has to walk the whole thread on every turn.
        """
        if limit <= 0:
            return []

        cursor = (
            self.messages
            .find({"conversation_id": conversation_id})
            .sort("seq", -1)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        docs.reverse()
        return docs

    async def get_transcript(
        self,
        conversation_id: str,
        limit: int = 200,
        before_seq: Optional[int] = None,
    ) -> list[dict]:
        """The full thread for display, paginated backwards from `before_seq`."""
        query: dict = {"conversation_id": conversation_id}
        if before_seq is not None:
            query["seq"] = {"$lt": before_seq}

        cursor = self.messages.find(query).sort("seq", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        docs.reverse()
        return docs

    async def count_turns(self, conversation_id: str) -> int:
        """
        Roughly how many exchanges have happened.

        Read off the conversation's own counter instead of a count_documents()
        on messages — it is one indexed lookup by _id rather than a scan, on a
        value that is already maintained.
        """
        conversation = await self.conversations.find_one(
            {"_id": conversation_id}, {"message_count": 1}
        )
        if not conversation:
            return 0
        return conversation.get("message_count", 0) // 2


# One shared instance, same pattern as rag_pipeline and query_router.
conversation_store = ConversationStore()
