"""
The /chatbot endpoints — one turn of a text conversation, and the approval
round-trip a turn takes when it wants to run a gated tool.

Memory is handled by the graph's MongoDB checkpointer, keyed by conversation_id,
so there is no manual history loading here.

    POST /chatbot                  ->  SSE: token* then (done | interrupt | error)
    POST /chatbot/{id}/resume      ->  same, continuing from an interrupt
    GET  /chatbot/{id}/pending     ->  what an interrupted thread is waiting on
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.ai.chat import get_pending_approval, resume_chat, stream_chat
from app.api.deps import get_current_user_id
from app.core.ids import is_conversation_id
from app.memory.store import conversation_store
from app.schemas.chat import ChatRequest, ToolDecision

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chatbot"])

# Proxies love to buffer text/event-stream into oblivion. Without no-cache and
# X-Accel-Buffering the tokens arrive in one lump at the end, which looks
# identical to the server having hung.
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


async def _owned_conversation(conversation_id: str, user_id: str) -> dict:
    """
    The thread, or 404.

    Load-bearing on the resume endpoint in a way it is not on /chatbot: the
    graph keys its state by conversation_id alone, so without this check any
    signed-in caller who guessed a conv_<uuid> could approve someone else's
    pending tool call. Ownership is the only thing standing in the way.
    """
    if not is_conversation_id(conversation_id):
        raise HTTPException(status_code=422, detail="Malformed conversation_id")

    conversation = await conversation_store.get_conversation(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


@router.post("/chatbot")
async def chatbot(body: ChatRequest, user_id: str = Depends(get_current_user_id)):
    """
    One turn of a conversation, powered by LangGraph.

    Omit `conversation_id` to start a new thread; the id of the thread the
    message landed in comes back in the `X-Conversation-Id` response header,
    because the body itself is a raw token stream with nowhere to put it.

    The owner comes from the session cookie, never the body — posting into
    someone else's thread now fails the ownership check below rather than
    succeeding because the caller claimed their id.
    """
    if body.conversation_id:
        conversation = await _owned_conversation(body.conversation_id, user_id)
    else:
        # An implicit new chat, so the first message of a thread does not need
        # a separate round-trip to POST /conversations.
        conversation = await conversation_store.create_conversation(user_id)

    conversation_id = conversation["_id"]

    # Written before the stream opens, for two reasons: the question survives a
    # crash mid-answer, and any failure above surfaces as a proper JSON 4xx
    # rather than an error string buried in the middle of the token stream.
    await conversation_store.append_message(conversation_id, "user", body.user_prompt)

    # ── LangGraph handles memory automatically via the checkpointer ──
    # No more: history = await window.load(conversation_id)
    # No more: service = ChatService(user_prompt=..., history=...)
    # Just stream the graph:

    return StreamingResponse(
        stream_chat(body.user_prompt, conversation_id),
        media_type="text/event-stream",
        status_code=200,
        headers={"X-Conversation-Id": conversation_id, **STREAM_HEADERS},
    )


@router.post("/chatbot/{conversation_id}/resume")
async def resume(
    conversation_id: str,
    body: ToolDecision,
    user_id: str = Depends(get_current_user_id),
):
    """
    Answer a pending tool-approval prompt and let the turn finish.

    The question is not resent — it is still in the checkpoint. This picks the
    graph up from exactly where `interrupt()` left it, so the response is the
    rest of the same answer, streamed the same way.

    Accepting runs the tool. Refusing does not: the model is told the tool was
    declined and answers from its own knowledge instead.

    409 rather than 404 when nothing is pending, because the thread does exist —
    it is just not waiting on anything, usually because another tab already
    answered the prompt.
    """
    await _owned_conversation(conversation_id, user_id)

    pending = await get_pending_approval(conversation_id)
    if pending is None:
        raise HTTPException(
            status_code=409, detail="This conversation is not awaiting approval"
        )

    logger.info(
        f"Tool decision for {conversation_id}: {body.action}"
        + (f" ({body.reason})" if body.reason else "")
    )

    return StreamingResponse(
        resume_chat(conversation_id, body.model_dump()),
        media_type="text/event-stream",
        status_code=200,
        headers={"X-Conversation-Id": conversation_id, **STREAM_HEADERS},
    )


@router.get("/chatbot/{conversation_id}/pending")
async def pending(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    What this thread is waiting on, if anything.

    For reconnecting: a refreshed tab — or a client whose stream died while the
    server restarted — would otherwise see a conversation that just stops
    mid-turn, with no way to discover there is a prompt still open on it.
    """
    await _owned_conversation(conversation_id, user_id)

    return {"pending": await get_pending_approval(conversation_id)}
