"""
Shared helpers for LangSmith tracing.

Anything built on LangChain (the chat models, the router's LCEL chain, the
@tool, the embedding model) reports to LangSmith on its own as soon as
LANGSMITH_TRACING=true. What it does *not* do is connect those reports to each
other, or show the plain-Python parts of the app at all.

@traceable fills both gaps. These helpers exist so the same four fixes aren't
rewritten in every file:

  drop_self                -> keep `self` out of traced method inputs
  drop_self_and_embedding  -> also keep a 768-float vector out of the payload
  join_tokens              -> turn a stream of tokens back into one answer
  as_documents             -> let LangSmith render retrieved chunks properly
  set_run_metadata         -> tag the run with something known only at runtime
"""

import asyncio
import logging
from typing import Any, Sequence

from langsmith.run_helpers import get_current_run_tree

from app.core.config import settings

logger = logging.getLogger(__name__)

# Same logger name as app/db/mongodb.py, so the LangSmith and MongoDB checks
# line up in the startup output.
startup_logger = logging.getLogger("uvicorn")


# ── process_inputs hooks ──────────────────────────────────────────────────────

def drop_self(inputs: dict) -> dict:
    """
    Removes `self` from the traced inputs of an instance method.

    Defensive only: langsmith already binds the signature and omits `self`
    before process_inputs ever runs, so in practice this is a no-op. It is kept
    as the base for the hooks below, and so the behaviour stays correct if that
    ever changes.

    Note this is also why a hook CANNOT reach `self` to pull extra fields off
    the instance — for that, see set_run_inputs().
    """
    return {key: value for key, value in inputs.items() if key != "self"}


def drop_self_and_embedding(inputs: dict) -> dict:
    """
    Same as drop_self, but also replaces `query_embedding` with a short label.

    A query embedding is a list of 768 floats. Logged verbatim it would be
    attached to every retrieval span, bloating the upload and burying the
    inputs that are actually worth reading.
    """
    cleaned = drop_self(inputs)

    embedding = cleaned.get("query_embedding")
    if isinstance(embedding, (list, tuple)):
        cleaned["query_embedding"] = f"<{len(embedding)}-dim vector>"

    return cleaned


def summarize_messages(inputs: dict) -> dict:
    """
    Condenses a LangChain message list down to role + a short text preview.

    Used for the tool loop, where dumping full message objects (including the
    retrieved context and raw tool output) makes the span unreadable.
    """
    cleaned = drop_self(inputs)

    messages = cleaned.get("messages")
    if isinstance(messages, (list, tuple)):
        cleaned["messages"] = [
            {
                "type": getattr(message, "type", type(message).__name__),
                "preview": str(getattr(message, "content", ""))[:300],
            }
            for message in messages
        ]

    return cleaned


# ── process_outputs hooks ─────────────────────────────────────────────────────

def as_documents(chunks: Any) -> dict:
    """
    Maps our retrieved chunks into the shape LangSmith's document viewer wants.

    A run_type="retriever" span gets a dedicated document renderer in the
    dashboard, but only when each item carries a `page_content` key. Our chunks
    use `text` (see VectorStore.vector_search), so they need translating.
    """
    if not isinstance(chunks, (list, tuple)):
        return {"documents": chunks}

    return {
        "documents": [
            {
                "page_content": chunk.get("text", ""),
                "metadata": {
                    "source": chunk.get("source", "unknown"),
                    "score": chunk.get("score"),
                    "id": str(chunk.get("_id", "")),
                },
            }
            if isinstance(chunk, dict) else {"page_content": str(chunk), "metadata": {}}
            for chunk in chunks
        ]
    }


# ── reduce_fn hook ────────────────────────────────────────────────────────────

def join_tokens(chunks: Sequence) -> dict:
    """
    Joins the tokens yielded by an async generator back into one string.

    @traceable records a generator's output as the list of everything it
    yielded. For a streaming answer that is hundreds of 2-3 character
    fragments — technically correct, completely unreadable. This gives the
    dashboard the finished answer instead.
    """
    return {"output": "".join(str(chunk) for chunk in chunks)}


# ── runtime metadata ──────────────────────────────────────────────────────────

def set_run_inputs(**values: Any) -> None:
    """
    Sets the traced inputs of the currently active run.

    Needed for methods whose only argument is `self` — ChatService.chat() and
    its branch generators. langsmith strips `self`, so their spans would
    otherwise show empty inputs and the dashboard would display the answer
    with no sign of the question that produced it.

    Safe to call after the run has started: RunTree.patch() re-sends inputs
    when the run ends.
    """
    try:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return
        if run_tree.inputs is None:
            run_tree.inputs = {}
        run_tree.inputs.update(values)
    except Exception as e:
        logger.debug(f"Could not attach run inputs {values}: {e}")


def set_run_metadata(**values: Any) -> None:
    """
    Attaches metadata to the currently active run, for values that are only
    known once the function is already running (such as the chosen route).

    Makes traces filterable in the dashboard, e.g. metadata.route = "TOOL".
    Silently does nothing when tracing is off, so the app never depends on it.
    """
    try:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return
        if run_tree.metadata is None:
            run_tree.metadata = {}
        run_tree.metadata.update(values)
    except Exception as e:
        # Tracing must never break the request it is observing.
        logger.debug(f"Could not attach run metadata {values}: {e}")


# ── connection check ──────────────────────────────────────────────────────────

def _explain(error: Exception) -> str:
    """Turns LangSmith's terser failures into something actionable."""
    text = str(error)

    if "Forbidden" in text or "403" in text or "org-scoped" in text:
        return (
            f"{text[:200]} "
            "-- HINT: the API key looks org-scoped. Set LANGSMITH_WORKSPACE_ID "
            "in .env to the workspace that owns this project."
        )
    if "401" in text or "Invalid token" in text or "Unauthorized" in text:
        return f"{text[:200]} -- HINT: LANGSMITH_API_KEY is invalid or revoked."
    return text[:250]


async def verify_langsmith_connection() -> dict:
    """
    Pings LangSmith the same way connect_to_mongo() pings Atlas.

    read_project() is a good ping because it exercises all three things that
    can be misconfigured at once: the API key, the workspace scoping, and
    whether the target project actually exists.

    Unlike the MongoDB check this NEVER raises. Tracing is observability — a
    LangSmith outage or an expired key must not stop the chatbot from serving.
    """
    if not settings.tracing_enabled:
        startup_logger.info(
            "LangSmith tracing: DISABLED "
            "(set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)"
        )
        return {"status": "disabled"}

    try:
        startup_logger.info("⏳ Verifying LangSmith connection...")

        # The LangSmith SDK is synchronous and does blocking HTTP, so it is
        # pushed off the event loop rather than called directly.
        def _ping():
            from langsmith import Client
            return Client().read_project(project_name=settings.LANGSMITH_PROJECT)

        project = await asyncio.to_thread(_ping)

        startup_logger.info(
            f"✅ LangSmith Connected -> project '{settings.LANGSMITH_PROJECT}'"
        )
        return {
            "status": "ok",
            "project": settings.LANGSMITH_PROJECT,
            "project_id": str(project.id),
        }

    except Exception as e:
        detail = _explain(e)
        startup_logger.error(f"❌ LangSmith check failed: {detail}")
        startup_logger.error("   Traces will NOT reach the dashboard for this run.")
        return {"status": "error", "project": settings.LANGSMITH_PROJECT, "detail": detail}
