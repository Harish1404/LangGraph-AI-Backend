from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    FRONTEND_URL = os.getenv("FRONTEND_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    WEATHER_WEBHOOK_URL = os.getenv("WEATHER_WEBHOOK_URL")

    # ── DeepSeek, via OpenRouter ─────────────────────────────────────────────
    # The primary answering model. Reached through OpenRouter rather than
    # DeepSeek's own API because the direct account has no balance: every
    # completion on that key comes back 402 "Insufficient Balance", which would
    # make the primary fail on every request and fall through to Mistral —
    # adding a round-trip instead of removing one.
    #
    # DEEPSEEK_API_KEY is recorded for the day that account is funded. Switching
    # back is then two lines: this key, and base_url "https://api.deepseek.com"
    # with the bare model id "deepseek-v4-flash" (no "deepseek/" prefix).
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    # Note: "deepseek/deepseek-v4-flash-latest" is NOT a valid id on OpenRouter
    # and returns 400.
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash")
    # LangSmith. Listed here for visibility only — the SDK reads these straight
    # out of os.environ, so it is the load_dotenv() call above that enables it.
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
    # Required because the API key is org-scoped rather than workspace-scoped.
    # The SDK turns this into the X-Tenant-Id header; without it every
    # workspace-scoped call (projects, runs, run ingestion) returns 403.
    LANGSMITH_WORKSPACE_ID = os.getenv("LANGSMITH_WORKSPACE_ID")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")

    MONGO_URL = os.getenv("MONGO_URL")
    DB_NAME = os.getenv("DB_NAME")

    # ── LangGraph checkpointer ───────────────────────────────────────────────
    # Where the graph's own state lives, keyed by thread_id (= conversation_id).
    # Separate from the `messages` collection: that one holds flattened text for
    # the UI, this one holds the real message objects, tool_calls, and the
    # paused-mid-approval state that a tool interrupt leaves behind.
    CHECKPOINT_DB_NAME = os.getenv("CHECKPOINT_DB_NAME") or os.getenv("DB_NAME")
    CHECKPOINT_COLLECTION = os.getenv("CHECKPOINT_COLLECTION", "checkpoints")
    CHECKPOINT_WRITES_COLLECTION = os.getenv(
        "CHECKPOINT_WRITES_COLLECTION", "checkpoint_writes"
    )
    # 0 keeps checkpoints forever. Every superstep writes a document, so a busy
    # thread accumulates them quickly; a TTL is the only thing that bounds it.
    CHECKPOINT_TTL_DAYS = int(os.getenv("CHECKPOINT_TTL_DAYS", "0"))

    # ── Human-in-the-loop tool approval ──────────────────────────────────────
    # Tools named here pause the graph and wait for the user to approve before
    # they run. An allowlist rather than a code-level rule, so gating a new tool
    # (the GitHub MCP write endpoints, when they land) is an env change.
    HITL_TOOLS = {
        name.strip() for name in os.getenv("HITL_TOOLS", "").split(",") if name.strip()
    }

    # ── Answer length (text chat) ────────────────────────────────────────────
    # Two budgets, because the chain mixes two kinds of model.
    #
    # A budget is a hard ceiling: the model stops dead when it is reached,
    # finish_reason comes back "length", and the reply ends mid-word. When that
    # happens the stream emits a `truncated` event and the UI says so rather
    # than leaving it a mystery — see _stream_graph in app/ai/chat.py.
    #
    # LIGHT applies to the models that emit only visible text: DeepSeek (with
    # reasoning explicitly disabled — see app/ai/models.py), Mistral and Gemini.
    # Every token of it reaches the reader.
    #
    # REASONING applies to openai/gpt-oss-20b alone, which spends hidden
    # reasoning tokens out of the same budget. A measured deep-dive answer used
    # 1375 reasoning tokens of 3617, so it needs materially more than the light
    # tier to produce a comparable answer.
    #
    # On the numbers: a budget is a ceiling, not a target — measured on the
    # light tier, a two-sentence answer spent 61 tokens and a 200-word one 255,
    # and those cost exactly the same whatever this is set to. Only answers that
    # genuinely need the room draw on it, so a generous ceiling is close to free.
    #
    # 2500 is sized from the answers that were being cut: a deep dive with tables
    # measured ~2250 visible tokens. At the earlier 600 it stopped mid-row.
    LIGHT_MAX_TOKENS = int(os.getenv("LIGHT_MAX_TOKENS", "2500"))
    # 4000 = the light tier's 2500 visible, plus ~1400 of headroom for the
    # hidden reasoning. Sized so both tiers yield a comparable answer: leaving
    # this at 3000 while LIGHT rose to 2500 would make the reasoning model the
    # *weaker* one (~1600 visible), so an answer would quietly get shorter
    # whenever the chain fell through to it.
    REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "4000"))

    ELEVEN_API = os.getenv("ELEVEN_API")

    # ── Voice mode ───────────────────────────────────────────────────────────
    # Flash v2.5 is the cheap, fast model: ~75ms to first byte and 0.5 credits
    # per character instead of 1. On the free tier's 10k credits/month that is
    # the difference between ~10 and ~20 thousand characters of speech.
    # Free accounts cannot use *library* voices over the API — the socket closes
    # with 1008 "Free users cannot use library voices". Verified working on this
    # key: Adam pNInz6obpgDQGcFmaJgB, Antoni ErXwobaYiN019PkySvjV,
    # Sarah EXAVITQu4vr4xnSDxMaL, Arnold VR6AewLTigWG4xSOukaG,
    # George JBFqnCBsd6RMkjVDRZzb, Jessica cgSgspJ2msm6clMCkdW9,
    # Daniel onwK4e9ZLuTAKqWW03F9. Rachel/Josh/Domi/Sam are library-gated.
    ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2_5")
    # pcm_24000 is NOT tier-gated (only pcm_44100/wav_44100 need Pro), which is
    # what lets the browser play raw frames through Web Audio with no decoder.
    ELEVEN_OUTPUT_FORMAT = os.getenv("ELEVEN_OUTPUT_FORMAT", "pcm_24000")
    TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    # What the browser's AudioWorklet is asked to produce. Whisper downsamples
    # to 16k internally anyway, so sending more than this is wasted bandwidth.
    MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
    # Spoken answers are capped far below the text path's budget. Long answers are
    # bad voice UX, slow to first audio, and on the free tier a single 2000-char
    # reply costs 1000 of the month's 10000 credits.
    VOICE_MAX_TOKENS = int(os.getenv("VOICE_MAX_TOKENS", "120"))
    # Replays identical phrases from disk during development instead of paying
    # for them again. Turn off to measure real end-to-end latency.
    TTS_CACHE_ENABLED = os.getenv("TTS_CACHE_ENABLED", "true").lower() == "true"

    # ── Conversation memory (window buffer) ──────────────────────────────────
    # How many past TURNS (a user message + its answer) are replayed to the
    # model. Tunable without a code edit, because the right number depends on
    # the model's context window and how chatty the answers are.
    #
    # Only the VOICE path still reads these — app/memory/window.py, via
    # app/routes/voice.py. Text chat gets its history from the checkpointer
    # instead. They stay because window.py:29-31 reads all three and swallows
    # the AttributeError, so removing them silently costs voice mode its memory
    # rather than failing loudly.
    WINDOW_K = int(os.getenv("WINDOW_K", "4"))
    WINDOW_K_LARGE = int(os.getenv("WINDOW_K_LARGE", "5"))
    LARGE_HISTORY_THRESHOLD = int(os.getenv("LARGE_HISTORY_THRESHOLD", "100"))

    GITHUB_PAT = os.getenv("GITHUB_PAT")

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Clerk is the identity provider (who someone is); this backend is the
    # session authority (whether they may act right now). Clerk verifies the
    # sign-in, we mint our own short access token plus a rotating refresh token
    # and enforce is_banned/is_verified locally on every request.
    CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
    CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY")
    # Signing secret for the Clerk webhook endpoint. Without webhook sync, a
    # revocation on Clerk's side would never reach our refresh-token store.
    CLERK_WEBHOOK_SECRET = os.getenv("CLERK_WEBHOOK_SECRET")
    # `azp` allowlist: which origins may present a Clerk token to us.
    CLERK_AUTHORIZED_PARTIES = [
        origin
        for origin in (
            os.getenv("FRONTEND_URL"),
            "http://localhost:3000",
        )
        if origin
    ]

    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    # Short on purpose. The access token is verified without a database read,
    # so its lifetime is also the worst-case delay before a ban takes effect.
    ACCESS_TOKEN_TTL_MIN = int(os.getenv("ACCESS_TOKEN_TTL_MIN", "15"))
    REFRESH_TOKEN_TTL_DAYS = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "30"))

    ACCESS_COOKIE_NAME = os.getenv("ACCESS_COOKIE_NAME", "access_token")
    REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "refresh_token")
    # The refresh cookie is scoped to its own endpoint, so it is never sent on
    # ordinary API calls — far less exposure than a site-wide cookie.
    REFRESH_COOKIE_PATH = os.getenv("REFRESH_COOKIE_PATH", "/auth/refresh")
    COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    # localhost:3000 -> localhost:8000 is same-site (a port is not part of the
    # site), so "lax" works in development. Cross-domain production needs
    # "none", which browsers only honour together with Secure.
    COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
    COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN") or None

    @property
    def auth_configured(self) -> bool:
        """False when the auth env vars are missing, so startup can say so."""
        return bool(self.CLERK_SECRET_KEY and self.JWT_SECRET)

    @property
    def tracing_enabled(self) -> bool:
        return str(self.LANGSMITH_TRACING).lower() == "true" and bool(self.LANGSMITH_API_KEY)

    # lowercase aliases — this is what ChatService / ChainService actually read
    gemini_api_key = GEMINI_API_KEY
    groq_api_key = GROQ_API_KEY
    mistral_api_key = MISTRAL_API_KEY
    weather_webhook_url = WEATHER_WEBHOOK_URL
    langsmith_api_key = LANGSMITH_API_KEY
    langsmith_tracing = LANGSMITH_TRACING
    langsmith_project = LANGSMITH_PROJECT
    elevenlabs_api_key = ELEVEN_API
    eleven_voice_id = ELEVEN_VOICE_ID
    eleven_model_id = ELEVEN_MODEL_ID
    eleven_output_format = ELEVEN_OUTPUT_FORMAT
    tts_sample_rate = TTS_SAMPLE_RATE
    mic_sample_rate = MIC_SAMPLE_RATE
    voice_max_tokens = VOICE_MAX_TOKENS
    light_max_tokens = LIGHT_MAX_TOKENS
    reasoning_max_tokens = REASONING_MAX_TOKENS
    deepseek_api_key = DEEPSEEK_API_KEY
    openrouter_api_key = OPENROUTER_API_KEY
    openrouter_base_url = OPENROUTER_BASE_URL
    deepseek_model = DEEPSEEK_MODEL
    tts_cache_enabled = TTS_CACHE_ENABLED
    clerk_secret_key = CLERK_SECRET_KEY
    clerk_webhook_secret = CLERK_WEBHOOK_SECRET
    jwt_secret = JWT_SECRET
    jwt_algorithm = JWT_ALGORITHM

settings = Settings()

