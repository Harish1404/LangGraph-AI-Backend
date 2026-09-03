import logging

import httpx
from langchain_core.tools import tool

from app.core.config import settings

logger = logging.getLogger(__name__)

# One client for the whole process, so the TCP connection and TLS session are
# reused between calls.
#
# This is where the time was going. Opening a fresh connection per call costs a
# DNS lookup, a TCP handshake and a TLS handshake before a single byte of the
# request is sent — measured at ~0.4s to hook.us2.make.com, against ~0.3s for
# the webhook's own work. Keeping the connection alive roughly halves the call.
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Lazily builds the shared client. Created on first use rather than at
    import time, so it binds to the running event loop."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0),
        )
    return _client


async def close_client() -> None:
    """Closes the shared client. Called from the app's shutdown lifespan."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        logger.info("Weather HTTP client closed.")
    _client = None


@tool
async def get_weather(city: str) -> str:
    """Fetch current weather details for a given city."""
    if not settings.weather_webhook_url:
        return "Weather service is not configured."

    try:
        response = await get_client().post(
            settings.weather_webhook_url,
            json={"city": city},
        )

        if response.status_code != 200:
            logger.warning(
                f"Weather webhook returned HTTP {response.status_code} for {city!r}"
            )
            return f"Weather service error (HTTP {response.status_code}): {response.text}"

        try:
            return str(response.json())
        except Exception:
            return response.text

    except httpx.TimeoutException:
        logger.error(f"Weather webhook timed out for {city!r}")
        return "The weather service did not respond in time."
    except Exception as e:
        logger.error(f"Weather webhook failed for {city!r}: {e}")
        return f"Error reaching weather service: {e}"
