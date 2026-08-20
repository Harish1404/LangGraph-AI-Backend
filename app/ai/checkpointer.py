"""
The LangGraph checkpointer — durable graph state in MongoDB.

Why this is a separate client from app/db/mongodb.py
────────────────────────────────────────────────────
`langgraph-checkpoint-mongodb` 0.4.0 ships a single `MongoDBSaver` that takes a
**synchronous** `pymongo.MongoClient`. Its async surface (`aput`, `aget_tuple`,
`alist`, `aput_writes`) is real — each one wraps the blocking call in a thread
via run_in_executor — so it is safe inside FastAPI's event loop. But it cannot
be handed the `AsyncIOMotorClient` that `connect_to_mongo()` opens, because that
client's methods return coroutines the saver never awaits.

(Older releases exposed an `AsyncMongoDBSaver` built on motor. It no longer
exists in 0.4.0 — `langgraph.checkpoint.mongodb.aio` is gone.)

So: one extra connection pool to the same MongoDB, owned by this module and
closed on shutdown. That is the whole cost.

What lands in Mongo
───────────────────
Two collections, both created and indexed by the saver's constructor:

  checkpoints        one document per superstep, keyed by (thread_id,
                     checkpoint_ns, checkpoint_id) — the full graph state,
                     including the real message objects and their tool_calls
  checkpoint_writes  the pending writes of a step that has not committed,
                     which is what a paused tool-approval interrupt lives in

`thread_id` is the conversation_id (see app/ai/chat.py), so a conversation and
its graph state share one key.
"""

import asyncio
import logging

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level so shutdown can close it. Only build_checkpointer() writes it.
_client: MongoClient | None = None


def _build_sync() -> MongoDBSaver:
    """
    Blocking. The constructor calls create_index on both collections, so this
    does real I/O and must not run on the event loop — see build_checkpointer.
    """
    global _client

    if not settings.MONGO_URL:
        raise RuntimeError(
            "MONGO_URL is not set — the LangGraph checkpointer cannot start. "
            "Conversation state would be lost on every restart."
        )

    _client = MongoClient(settings.MONGO_URL)

    # Seconds, or None to keep checkpoints forever. `or None` also catches the
    # 0 default, which would otherwise become an expireAfterSeconds=0 index and
    # reap every checkpoint the moment it was written.
    ttl = settings.CHECKPOINT_TTL_DAYS * 86400 or None

    return MongoDBSaver(
        client=_client,
        db_name=settings.CHECKPOINT_DB_NAME or "rag_db",
        checkpoint_collection_name=settings.CHECKPOINT_COLLECTION,
        writes_collection_name=settings.CHECKPOINT_WRITES_COLLECTION,
        ttl=ttl,
    )


async def build_checkpointer() -> MongoDBSaver:
    """Called once from the FastAPI lifespan, after connect_to_mongo()."""
    saver = await asyncio.to_thread(_build_sync)

    logger.info(
        "LangGraph checkpointer ready — db=%s collections=%s/%s ttl=%s",
        settings.CHECKPOINT_DB_NAME or "rag_db",
        settings.CHECKPOINT_COLLECTION,
        settings.CHECKPOINT_WRITES_COLLECTION,
        f"{settings.CHECKPOINT_TTL_DAYS}d" if settings.CHECKPOINT_TTL_DAYS else "none",
    )
    return saver


async def close_checkpointer() -> None:
    """Closes this module's pool. Safe to call when nothing was ever opened."""
    global _client

    if _client is not None:
        await asyncio.to_thread(_client.close)
        _client = None
        logger.info("🔒 LangGraph checkpointer connection closed.")
