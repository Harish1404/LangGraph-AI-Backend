"""
The voice turn, end to end.

    mic PCM --ws--> [buffer] --> Groq Whisper --> ChatService --> sentence
    chunks --> ElevenLabs ws --> PCM --ws--> browser

Push-to-talk, so a turn is bounded by explicit start/end messages from the
client rather than by voice activity detection. That is what removes VAD,
echo cancellation and barge-in handling from this file entirely.

Frame types are self-describing: binary frames are always audio, text frames
are always JSON control messages, so no envelope or length prefix is needed.
"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai.chat import ChatService
from app.ai.voice import ElevenLabsStream, sentence_chunks, stream_tts, transcribe_pcm
from app.api.deps import get_current_user_ws
from app.core.config import settings
from app.core.ids import is_conversation_id
from app.memory import window
from app.memory.store import conversation_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice"])

# CORSMiddleware does not apply to WebSocket handshakes — Starlette runs it on
# HTTP requests only — so the check has to happen here or not at all.
ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}
if settings.FRONTEND_URL:
    ALLOWED_ORIGINS.add(settings.FRONTEND_URL.rstrip("/"))

# A 16 kHz mono PCM16 second is 32 KB. This caps one utterance at ~60s, which
# stops a stuck client from growing the buffer without bound.
MAX_UTTERANCE_BYTES = 32_000 * 60

# Below this there is no speech worth sending — a click on the button, or the
# mic opening and closing. ~0.25s.
MIN_UTTERANCE_BYTES = 8_000


class Turn:
    """Timing for one turn, so the latency budget can be measured not guessed."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        self.marks.append((label, (time.perf_counter() - self.t0) * 1000))

    def report(self) -> str:
        out, prev = [], 0.0
        for label, ms in self.marks:
            out.append(f"{label}=+{ms - prev:.0f}ms")
            prev = ms
        total = self.marks[-1][1] if self.marks else 0
        return "  ".join(out) + f"  TOTAL={total:.0f}ms"


@router.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin is not None and origin.rstrip("/") not in ALLOWED_ORIGINS:
        logger.warning(f"Rejected voice socket from origin {origin!r}")
        await websocket.close(code=1008)
        return

    # The origin check above proves where the page came from, not who is using
    # it. Authentication rides the session cookie, which the browser attaches
    # to the handshake automatically — the WebSocket API cannot set an
    # Authorization header, and a token in the query string would end up in
    # every access log.
    user = await get_current_user_ws(websocket)
    if user is None:
        logger.warning("Rejected unauthenticated voice socket")
        await websocket.close(code=1008, reason="Not authenticated")
        return

    user_id = user["_id"]

    await websocket.accept()
    logger.info(f"Voice socket connected for {user_id}")

    audio = bytearray()
    conversation_id: str | None = None

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Binary frame -> mic audio for the utterance in progress.
            if (chunk := message.get("bytes")) is not None:
                if len(audio) < MAX_UTTERANCE_BYTES:
                    audio.extend(chunk)
                continue

            if (text := message.get("text")) is None:
                continue

            msg = json.loads(text)
            kind = msg.get("type")

            if kind == "start":
                audio.clear()
                requested = msg.get("conversation_id")
                # Validated here rather than trusted, same as POST /chatbot.
                if requested and is_conversation_id(requested):
                    conversation_id = requested

            elif kind == "cancel":
                audio.clear()

            elif kind == "end":
                if len(audio) < MIN_UTTERANCE_BYTES:
                    await websocket.send_json(
                        {"type": "error", "message": "Didn't catch that — hold the button and speak."}
                    )
                    audio.clear()
                    continue

                pcm, audio = bytes(audio), bytearray()
                conversation_id = await run_turn(
                    websocket, pcm, conversation_id, user_id
                )

    except WebSocketDisconnect:
        logger.info("Voice socket disconnected")
    except Exception:
        logger.exception("Voice socket failed")
        try:
            await websocket.send_json({"type": "error", "message": "Voice pipeline error."})
        except Exception:
            pass


async def run_turn(
    websocket: WebSocket, pcm: bytes, conversation_id: str | None, user_id: str
) -> str | None:
    """One utterance in, one spoken answer out. Returns the conversation id."""
    turn = Turn()
    seconds = len(pcm) / 2 / settings.mic_sample_rate

    # Finding the conversation does not depend on what was said, so it runs
    # alongside transcription instead of queueing behind it.
    conv_task = asyncio.create_task(_resolve_conversation(conversation_id, user_id))

    transcript = await transcribe_pcm(pcm)
    turn.mark("stt")

    if not transcript:
        conv_task.cancel()
        await websocket.send_json(
            {"type": "error", "message": "Didn't catch that — try again."}
        )
        return conversation_id

    await websocket.send_json({"type": "transcript", "text": transcript})

    # The ElevenLabs handshake takes a few hundred ms and needs no text, so it
    # is started now and overlapped with the router and the LLM. By the time
    # the first sentence exists the socket is already open and can speak it.
    tts = ElevenLabsStream()
    opening = asyncio.create_task(tts.open())

    try:
        conversation = await conv_task
        conversation_id = conversation["_id"]

        # Same bookkeeping as POST /chatbot: the question is written before the
        # answer starts so it survives a crash mid-turn, and history is loaded
        # after that write so the model is not shown its own question twice.
        await conversation_store.append_message(conversation_id, "user", transcript)
        history = await window.load(conversation_id)
        turn.mark("memory")

        service = ChatService(
            user_prompt=transcript,
            conversation_id=conversation_id,
            history=history,
            voice_mode=True,
        )

        first_audio = False

        async def captioned_tokens():
            """Forward tokens to the browser as captions on the way to the TTS.

            The client gets text well before the audio for the same words,
            which makes the wait legible instead of silent.
            """
            async for token in service.chat():
                await websocket.send_json({"type": "token", "text": token})
                yield token

        async def on_audio(buf: bytes):
            nonlocal first_audio
            if not first_audio:
                first_audio = True
                turn.mark("first_audio")
            await websocket.send_bytes(buf)

        await stream_tts(
            sentence_chunks(captioned_tokens()), on_audio, stream=tts, opening=opening
        )
        turn.mark("tts_done")
    except BaseException:
        # The socket was opened speculatively; if the turn dies before
        # stream_tts takes ownership it still has to be cleaned up.
        opening.cancel()
        await tts.close()
        raise

    await websocket.send_json({"type": "done", "conversation_id": conversation_id})

    logger.info(f"voice turn ({seconds:.1f}s audio)  {turn.report()}")
    return conversation_id


async def _resolve_conversation(conversation_id: str | None, user_id: str) -> dict:
    """The existing thread if it is valid and ours, otherwise a fresh one."""
    if conversation_id:
        conversation = await conversation_store.get_conversation(
            conversation_id, user_id
        )
        if conversation is not None:
            return conversation
    return await conversation_store.create_conversation(user_id)
