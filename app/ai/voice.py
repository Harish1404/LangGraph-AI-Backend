"""
The audio ends of the voice pipeline: speech in, speech out.

Everything here works on raw PCM16 little-endian mono, because that is what the
browser can produce without an encoder and consume without a decoder. Keeping
the wire format decoder-free on both ends is what removes ffmpeg, pydub and
MediaSource from the design — the bytes are inspectable at every hop.

Nothing in this module knows about FastAPI or about the LLM; app/routes/voice.py
is what wires the two halves together.
"""

import asyncio
import base64
import hashlib
import json
import logging
import re
import wave
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import httpx
import websockets
from groq import AsyncGroq

from app.core.config import settings

logger = logging.getLogger(__name__)

STT_MODEL = "whisper-large-v3-turbo"

# Written next to the app rather than in a temp dir so it survives a reload and
# is easy to inspect (and delete) while teaching. Gitignored.
_CACHE_DIR = Path(__file__).resolve().parents[2] / ".tts_cache"

_groq_client: AsyncGroq | None = None


def _groq() -> AsyncGroq:
    """One client for the process, created on first use.

    Deliberately lazy: building it at import time binds it to whichever event
    loop happens to be current, which under uvicorn --reload is not the loop
    that ends up serving requests.
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


# ── Speech to text ───────────────────────────────────────────────────────────


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono in a 44-byte WAV header.

    Groq needs a container it can sniff; it will not take headerless PCM. The
    stdlib `wave` module writes one with no third-party dependency at all, which
    is the whole reason this pipeline needs neither ffmpeg nor pydub.
    """
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def transcribe_pcm(pcm: bytes, sample_rate: int | None = None) -> str:
    """Transcribe one whole utterance of raw PCM16 mono.

    Returns the transcript VERBATIM. The previous version of this file ran the
    transcript through Gemini with a "summarize short and crisp" prompt and
    returned the summary, which silently destroyed the user's actual words.
    """
    sample_rate = sample_rate or settings.mic_sample_rate
    wav = pcm_to_wav(pcm, sample_rate)

    resp = await _groq().audio.transcriptions.create(
        # The filename is not cosmetic — Groq picks the decoder off the
        # extension, so a wrong one is rejected even when the bytes are valid.
        file=("utterance.wav", wav, "audio/wav"),
        model=STT_MODEL,
        response_format="text",
    )
    # response_format="text" yields a bare string; the SDK still wraps it in an
    # object for some versions, so both shapes are accepted.
    text = resp if isinstance(resp, str) else getattr(resp, "text", "")
    return text.strip()


# ── Chunking the token stream for TTS ────────────────────────────────────────

# Sentence end followed by whitespace. The lookbehind keeps the punctuation
# attached to the sentence it belongs to, which matters because ElevenLabs uses
# it for prosody — stripping it makes the speech sound flat and run-on.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# Long enough that a clause carries some prosody, short enough that first audio
# is not held hostage by a model that forgot to punctuate.
_MAX_CHUNK_CHARS = 90

# Words whose trailing period is not a full stop. Without this, "Dr. Reddy"
# is flushed as two utterances and spoken with a pause in the middle of the
# name. A short explicit list beats a length heuristic, which cannot tell
# "Dr." from a genuinely short sentence like "Hello there."
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
    "inc", "ltd", "co", "no", "fig", "eg", "ie", "approx", "dept",
    "est", "vol", "ref", "pvt",
}

_LAST_WORD = re.compile(r"([A-Za-z]+)\.\s*$")


def _is_abbreviation(head: str) -> bool:
    """True if `head` ends on an abbreviation rather than a real sentence."""
    m = _LAST_WORD.search(head)
    return bool(m) and m.group(1).lower() in _ABBREVIATIONS


async def sentence_chunks(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup an LLM token stream into speakable chunks.

    This is the single highest-leverage piece of the latency budget. Feeding
    ElevenLabs one token at a time produces choppy prosody; feeding it the whole
    answer means the first word is not spoken until the last one is generated.
    Flushing on sentence boundaries starts audio after the first sentence.
    """
    buf = ""
    async for token in tokens:
        buf += token
        # Flush every complete sentence sitting in the buffer, keeping the
        # trailing partial one behind for the next token.
        while True:
            match = _BOUNDARY.search(buf)
            if not match:
                break
            # "Dr. " is not the end of a sentence — keep it in the buffer and
            # look for the next boundary instead.
            while _is_abbreviation(buf[: match.end()]):
                later = _BOUNDARY.search(buf, match.end())
                if later is None:
                    match = None
                    break
                match = later
            if match is None:
                break
            head, buf = buf[: match.end()], buf[match.end() :]
            if head.strip():
                yield head

        # No punctuation in sight and the buffer is getting long — cut at the
        # last word break so a rambling answer still starts speaking.
        if len(buf) >= _MAX_CHUNK_CHARS:
            cut = buf.rfind(" ", 0, _MAX_CHUNK_CHARS)
            if cut <= 0:
                cut = _MAX_CHUNK_CHARS
            head, buf = buf[:cut], buf[cut:]
            if head.strip():
                yield head

    if buf.strip():
        yield buf


# ── Text to speech ───────────────────────────────────────────────────────────


def _cache_path(text: str, voice_id: str, model_id: str) -> Path:
    key = hashlib.sha256(f"{voice_id}|{model_id}|{text}".encode()).hexdigest()
    return _CACHE_DIR / f"{key}.pcm"


class ElevenLabsStream:
    """A single turn's worth of streaming TTS over the ElevenLabs WebSocket.

    Text goes in as the LLM produces it and audio comes back while the model is
    still writing — that overlap is what the REST endpoints cannot do, since
    they need the complete text up front.

    One instance per turn. Opening per turn rather than holding a long-lived
    socket also sidesteps the 20s inactivity timeout, which a user pausing to
    think would otherwise trip.
    """

    def __init__(self, voice_id: str | None = None, model_id: str | None = None):
        self.voice_id = voice_id or settings.eleven_voice_id
        self.model_id = model_id or settings.eleven_model_id
        self._ws = None
        self._spoken: list[str] = []

    @property
    def url(self) -> str:
        return (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
            f"?model_id={self.model_id}"
            f"&output_format={settings.eleven_output_format}"
        )

    async def open(self) -> None:
        self._ws = await websockets.connect(
            self.url,
            additional_headers={"xi-api-key": settings.elevenlabs_api_key},
            max_size=None,
        )
        # ElevenLabs requires an initialising message before any text. A bare
        # space is the documented way to send settings without speaking a word.
        await self._ws.send(
            json.dumps(
                {
                    "text": " ",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                }
            )
        )

    async def send_text(self, text: str) -> None:
        self._spoken.append(text)
        await self._ws.send(json.dumps({"text": text, "try_trigger_generation": True}))

    async def done_sending(self) -> None:
        """Signal end of input. An empty string is EOS in this protocol."""
        await self._ws.send(json.dumps({"text": ""}))

    async def audio(self) -> AsyncIterator[bytes]:
        """Yield decoded PCM as it arrives, until the server says isFinal."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("audio"):
                    # Audio arrives base64-encoded inside JSON, not as binary
                    # frames — so it has to be decoded before being forwarded.
                    yield base64.b64decode(msg["audio"])
                if msg.get("isFinal"):
                    break
        except websockets.exceptions.ConnectionClosed as e:
            # 1008 here is nearly always the free-tier library-voice rejection
            # or an invalid voice id; both are worth surfacing loudly.
            if e.code != 1000:
                logger.error(f"ElevenLabs closed the socket: {e.code} {e.reason}")
                raise

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    @property
    def spoken_text(self) -> str:
        return "".join(self._spoken)


async def stream_tts(
    chunks: AsyncIterator[str],
    on_audio,
    voice_id: str | None = None,
    stream: "ElevenLabsStream | None" = None,
    opening: asyncio.Task | None = None,
) -> None:
    """Push text chunks through ElevenLabs, handing each PCM buffer to `on_audio`.

    Sending and receiving run concurrently: waiting for the audio of sentence
    one before sending sentence two would serialise the very overlap that makes
    this fast.

    `stream`/`opening` let the caller start the ElevenLabs handshake early, in
    parallel with the router and the LLM. That connect costs a few hundred ms
    and is pure dead time if it only begins once the first sentence exists.
    """
    stream = stream or ElevenLabsStream(voice_id=voice_id)
    if opening is not None:
        await opening
    else:
        await stream.open()

    async def pump_text():
        try:
            async for chunk in chunks:
                await stream.send_text(chunk)
        finally:
            await stream.done_sending()

    sender = asyncio.create_task(pump_text())
    try:
        async for pcm in stream.audio():
            await on_audio(pcm)
        await sender
    finally:
        sender.cancel()
        await stream.close()


async def synth_cached(text: str, voice_id: str | None = None) -> bytes | None:
    """Whole-utterance TTS with an on-disk cache. Returns None on a miss.

    Only used for fixed phrases (greetings, error lines) and for replaying test
    utterances during development. Real answers go through `stream_tts`, since
    caching them would defeat the point of streaming.
    """
    if not settings.tts_cache_enabled:
        return None
    path = _cache_path(text, voice_id or settings.eleven_voice_id, settings.eleven_model_id)
    if path.exists():
        logger.info(f"TTS cache hit ({len(text)} chars, 0 credits): {text[:40]!r}")
        return path.read_bytes()
    return None


def cache_put(text: str, pcm: bytes, voice_id: str | None = None) -> None:
    if not settings.tts_cache_enabled or not pcm:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(text, voice_id or settings.eleven_voice_id, settings.eleven_model_id).write_bytes(pcm)


# ── Quota ────────────────────────────────────────────────────────────────────


async def warm_up() -> None:
    """Pay the TLS/connection cost at startup instead of on the first turn.

    Measured on this machine: the first Groq transcription takes ~1000ms and
    every one after it ~230ms. That second number is the real cost of the
    model; the first is connection setup. Without this the first student to
    press the mic button gets a visibly worse experience than everyone after.

    Sends 100ms of silence — Whisper returns empty, which is fine and free.
    """
    try:
        t = asyncio.get_event_loop().time()
        await transcribe_pcm(b"\x00\x00" * (settings.mic_sample_rate // 10))
        logger.info(
            f"Groq STT warmed up in {(asyncio.get_event_loop().time() - t) * 1000:.0f}ms"
        )
    except Exception as e:
        # Never block startup on this — it is an optimisation, not a dependency.
        logger.warning(f"STT warm-up skipped: {e}")


async def remaining_credits() -> int | None:
    """Credits left this month, or None if the key cannot read it.

    Needs the `user_read` scope. A key scoped to text-to-speech only still works
    fine for the pipeline, so a failure here is logged and shrugged off rather
    than raised — it is telemetry, not a dependency.
    """
    if not settings.elevenlabs_api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
        if r.status_code != 200:
            logger.info(
                "ElevenLabs quota unavailable (%s) — key is missing the "
                "`user_read` scope. Regenerate it with that scope to enable "
                "credit logging.",
                r.status_code,
            )
            return None
        d = r.json()
        return d["character_limit"] - d["character_count"]
    except Exception as e:
        logger.warning(f"Could not read ElevenLabs quota: {e}")
        return None
