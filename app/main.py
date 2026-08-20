import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.auth import router as auth_router
from app.routes.chatbot import router as chatbot_router
from app.routes.conversations import router as conversations_router
from app.routes.voice import router as voice_router
from app.routes.webhooks import router as webhooks_router
from app.rag.rag_pipeline import rag_pipeline
from app.core.tracing import verify_langsmith_connection
from app.db.mongodb import (
    connect_to_mongo,
    close_mongo_connection,
    get_database_client,
    ensure_auth_indexes,
    ensure_chat_indexes,
)
from app.tools.weather import close_client as close_weather_client
from app.ai.voice import warm_up as warm_up_stt, remaining_credits
from app.ai.chat import warm_up_models, warm_up_llm


# Configure logging to output INFO level logs to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App startup")

    # Ping LangSmith before Mongo, so the boot log reads top-down:
    # tracing -> database -> ingestion.
    app.state.langsmith = await verify_langsmith_connection()

    await connect_to_mongo()
    await ensure_chat_indexes()
    await ensure_auth_indexes()

    # Loud rather than silent: without these the auth routes cannot mint a
    # session, and the failure would otherwise only show up as a 401 at
    # sign-in with no hint as to why.
    if not settings.auth_configured:
        logger.warning(
            "Auth is NOT configured — set CLERK_SECRET_KEY and JWT_SECRET in .env. "
            "Every protected endpoint will return 401 until then."
        )

    chunk_count = await rag_pipeline.ingest("uploads")
    logger.info(f"RAG startup ingestion complete: {chunk_count} new chunk(s) indexed.")

    # The first Groq transcription of a process pays ~1s of connection setup
    # that every later one does not. Spending it here means the first person to
    # press the mic button gets the same latency as everyone after them.
    warm_up_models()
    await asyncio.gather(warm_up_stt(), warm_up_llm())

    credits = await remaining_credits()
    if credits is not None:
        logger.info(f"ElevenLabs credits remaining this month: {credits}")

    yield
    logger.info("App shutdown")

    # Runs upload from a background thread; this makes sure the last few traces
    # are sent before the process goes away (e.g. Ctrl+C under --reload).
    if settings.tracing_enabled:
        try:
            from langsmith import Client
            Client().flush()
            logger.info("LangSmith traces flushed.")
        except Exception as e:
            logger.warning(f"Could not flush LangSmith traces: {e}")

    await close_weather_client()
    await close_mongo_connection()


app = FastAPI(lifespan=lifespan)

# Dev defaults. `expose_headers` is the part that matters: without it a browser
# cannot read X-Conversation-Id off the streaming response at all, so a new
# chat would have no way to learn its own id. Tighten allow_origins for prod.
origins = [
    origin
    for origin in (
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    if origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id"],
)

app.include_router(auth_router)
# Machine-to-machine, authenticated by Svix signature rather than a session —
# see app/routes/webhooks.py.
app.include_router(webhooks_router)
app.include_router(chatbot_router)
app.include_router(conversations_router)
app.include_router(voice_router)



@app.get("/")
def landing_page():
    return {
        "message": " Hi this is your backend server"
    }

@app.get("/health")
async def health():
    """
    Reports the live state of both external dependencies.

    Checked on every call rather than echoing the startup result, so you can
    confirm a config fix without restarting the server. Never raises — a broken
    dependency shows up in the body, not as a 500.
    """
    try:
        await get_database_client().admin.command("ping")
        mongodb = {"status": "ok"}
    except Exception as e:
        mongodb = {"status": "error", "detail": str(e)[:250]}

    langsmith = await verify_langsmith_connection()

    healthy = mongodb["status"] == "ok" and langsmith["status"] in ("ok", "disabled")

    return {
        "status": "ok" if healthy else "degraded",
        "mongodb": mongodb,
        "langsmith": langsmith,
    }
