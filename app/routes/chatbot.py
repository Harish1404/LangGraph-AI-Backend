"""
The /chatbot endpoint — one turn of a text conversation.

This is the simplified version that uses the LangGraph workflow.
Memory is handled by the graph's MemorySaver checkpointer, so there
is no manual history loading here.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.ai.chat import stream_chat
from app.api.deps import get_current_user_id
from app.core.ids import is_conversation_id
from app.memory.store import conversation_store
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chatbot"])


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
        if not is_conversation_id(body.conversation_id):
            raise HTTPException(status_code=422, detail="Malformed conversation_id")

        conversation = await conversation_store.get_conversation(
            body.conversation_id, user_id
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # An implicit new chat, so the first message of a thread does not need
        # a separate round-trip to POST /conversations.
        conversation = await conversation_store.create_conversation(user_id)

    conversation_id = conversation["_id"]

    # Written before the stream opens, for two reasons: the question survives a
    # crash mid-answer, and any failure above surfaces as a proper JSON 4xx
    # rather than an error string buried in the middle of the token stream.
    await conversation_store.append_message(conversation_id, "user", body.user_prompt)

    # ── LangGraph handles memory automatically via MemorySaver ──
    # No more: history = await window.load(conversation_id)
    # No more: service = ChatService(user_prompt=..., history=...)
    # Just stream the graph:

    return StreamingResponse(
        stream_chat(body.user_prompt, conversation_id),
        media_type="text/event-stream",
        status_code=200,
        headers={"X-Conversation-Id": conversation_id},
    )
