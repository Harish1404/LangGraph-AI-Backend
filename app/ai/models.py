"""
Where every chat model in the app is constructed.

One module rather than one per caller, because `app/ai/chat.py` and
`app/ai/agents/router.py` both need the same providers in the same order, and
`chat.py` already imports `router.py` — so the shared code cannot live in
either of them without closing an import cycle. This module imports only
`config`, so it is always safe to import from anywhere.

The chain:

    mistral-small  →  gpt-oss-20b  →  gemini-3.5-flash-lite

Measured time-to-first-token, 3 runs each, same prompt:

    mistral-small-latest      0.44s min / 0.56s median
    gpt-oss-20b (Groq)        rate-limited (429) during the benchmark
    gemini-3.5-flash-lite     3.21s min / 3.30s median

Mistral leads on reliability and speed.
Gemini is last because it is roughly six times slower to first token.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-provider factories
# ═══════════════════════════════════════════════════════════════════════════════

def mistral(max_tokens: int, temperature: float = 0.7) -> ChatMistralAI:
    """
    The primary. Fastest to first token of anything in the chain (0.44s min).
    """
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=settings.mistral_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def groq(max_tokens: int, temperature: float = 0.7) -> ChatGroq:
    """
    First fallback — the reasoning model in the chain.

    Give it `REASONING_MAX_TOKENS` rather than the light budget: its hidden
    reasoning comes out of the same allowance, so at the light tier it would
    produce far less visible text than the others.
    """
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=settings.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def gemini(max_tokens: int, temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Last resort. Slowest to first token, but a genuinely independent provider."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# The chain
# ═══════════════════════════════════════════════════════════════════════════════

def ordered_models(
    light_tokens: int,
    reasoning_tokens: int,
    temperature: float = 0.7,
) -> list:
    """
    Every model, primary first, each with the budget its tier calls for.

    Returned as a plain list rather than an assembled `RunnableWithFallbacks`
    because callers need the individual models: `bind_tools()` and
    `with_structured_output()` have to be applied to each *model* before
    `with_fallbacks()` is, since `with_fallbacks()` returns a
    `RunnableWithFallbacks` that has neither method.
    """
    return [
        mistral(light_tokens, temperature),
        groq(reasoning_tokens, temperature),
        gemini(light_tokens, temperature),
    ]


def describe_chain() -> str:
    """The chain as a one-line string, for the startup log."""
    return " -> ".join([
        "mistral-small-latest",
        "openai/gpt-oss-20b",
        "gemini-3.5-flash-lite",
    ])


def log_chain() -> None:
    """
    Say which chain was built, once, at startup.
    """
    logger.info("Model chain: %s", describe_chain())

