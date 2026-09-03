"""
Where every chat model in the app is constructed.

One module rather than one per caller, because `app/ai/chat.py` and
`app/ai/agents/router.py` both need the same providers in the same order, and
`chat.py` already imports `router.py` — so the shared code cannot live in
either of them without closing an import cycle. This module imports only
`config`, so it is always safe to import from anywhere.

The chain:

    mistral-small  →  gpt-oss-20b  →  deepseek-v4-flash  →  gemini-2.5-flash

Measured time-to-first-token, 3 runs each, same prompt:

    mistral-small-latest      0.44s min / 0.56s median
    deepseek (sort=latency)   0.55s min / 0.56s median
    gpt-oss-20b (Groq)        rate-limited (429) during the benchmark
    gemini-2.5-flash          3.21s min / 3.30s median

Mistral leads on reliability rather than raw speed — it and DeepSeek are
effectively tied on first-token latency, but DeepSeek was observed **stalling
part-way through a generation**. Through OpenRouter this model is served by
several different upstream providers (StreamLake, NextBit and Baidu were all
seen in testing), so response quality and stability vary run to run in a way a
single-provider endpoint does not. It keeps its place in the chain as a genuinely
independent provider, just not on the critical path — and carries a request
timeout so a stall fails over instead of hanging the turn.

Gemini is last because it is roughly six times slower to first token.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-provider factories
# ═══════════════════════════════════════════════════════════════════════════════

def deepseek(max_tokens: int, temperature: float = 0.7) -> ChatOpenAI:
    """
    Second fallback, reached through OpenRouter's OpenAI-compatible endpoint.

    Was the primary until it was seen stalling part-way through a generation;
    it now sits behind Mistral and gpt-oss-20b. Still worth keeping — it is a
    genuinely independent provider, and cheap.

    Both `extra_body` keys were measured and neither is optional — deleting
    either one quietly undoes the point of this model being here at all.

    provider.sort = "latency"
        OpenRouter spreads this model across several upstream providers
        (StreamLake, NextBit and Baidu were all seen in testing) and its default
        routing is not latency-aware. Without this key, time-to-first-token
        measured 2.90s; with it, 0.55s.

    reasoning.enabled = False
        Despite being marketed as a non-reasoning model, it reasons by default
        on this route — and those tokens are invisible but still spend the
        budget. On a 600-token cap: reasoning on gave 328 reasoning tokens and
        only ~270 visible (1215 chars); reasoning off gave 0 and 600 (2497
        chars). At the router's 300-token budget, reasoning could plausibly
        consume the entire allowance and return no structured output at all.
    """
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        # This model has been seen to stall mid-generation. As a fallback that
        # matters more, not less: without a bound, a stuck DeepSeek would hang
        # the whole turn rather than failing over to Gemini behind it. Generous
        # enough not to cut a legitimately long answer — a full 2500-token
        # response measured well under 20s.
        timeout=60,
        extra_body={
            "reasoning": {"enabled": False},
            "provider": {"sort": "latency"},
        },
    )


def mistral(max_tokens: int, temperature: float = 0.7) -> ChatMistralAI:
    """
    The primary. Fastest to first token of anything in the chain (0.44s min),
    and — the reason it leads rather than DeepSeek — it does not stall.
    """
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=settings.mistral_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def groq(max_tokens: int, temperature: float = 0.7) -> ChatGroq:
    """
    First fallback — the only reasoning model in the chain.

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
        model="gemini-2.5-flash",
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

    DeepSeek is skipped when OPENROUTER_API_KEY is missing, so a deployment
    without that key simply runs a three-model chain rather than failing over
    to a provider it cannot authenticate with. Nothing else shifts — it has not
    been the primary since it was seen stalling mid-generation.
    """
    models = [
        mistral(light_tokens, temperature),
        groq(reasoning_tokens, temperature),
    ]

    if settings.openrouter_api_key:
        models.append(deepseek(light_tokens, temperature))

    models.append(gemini(light_tokens, temperature))

    return models


def describe_chain() -> str:
    """The chain as a one-line string, for the startup log."""
    names = ["mistral-small-latest", "openai/gpt-oss-20b"]
    if settings.openrouter_api_key:
        names.append(f"{settings.deepseek_model} (OpenRouter)")
    names.append("gemini-2.5-flash")
    return " -> ".join(names)


def log_chain() -> None:
    """
    Say which chain was built, once, at startup.

    Only a note when DeepSeek is absent, not a warning: it is a third-position
    fallback now, so the chain is perfectly healthy without it.
    """
    logger.info("Model chain: %s", describe_chain())

    if not settings.openrouter_api_key:
        logger.info(
            "OPENROUTER_API_KEY is not set — running without the DeepSeek "
            "fallback. Mistral and gpt-oss-20b are unaffected."
        )
