"""
Everything that talks to Clerk, in one place.

Clerk is the identity provider only: it answers "who is this person" and holds
the OAuth grants they have given us. It is deliberately *not* consulted on
whether a request may proceed — that is this backend's own session layer, so
the chat hot path never waits on a third-party network call.

`get_oauth_token` is the seam the GitHub MCP connector will use later: Clerk
already stores the user's GitHub OAuth token and refreshes it on read, which is
why that work does not need a credential vault of its own.
"""

import logging
from typing import Any

from clerk_backend_api import Clerk
from clerk_backend_api.security import VerifyTokenOptions, verify_token_async
from clerk_backend_api.security.types import TokenVerificationError

from app.core.config import settings

logger = logging.getLogger(__name__)


class ClerkAuthError(Exception):
    """Raised when a Clerk-issued token cannot be trusted."""


_client: Clerk | None = None


def get_client() -> Clerk:
    """
    One SDK client for the process, built lazily.

    Lazy rather than at import time so a missing secret key surfaces as a clean
    401 on the auth routes instead of taking the whole app down at startup —
    the chat and RAG paths do not need Clerk to work.
    """
    global _client
    if _client is None:
        if not settings.CLERK_SECRET_KEY:
            raise ClerkAuthError("CLERK_SECRET_KEY is not configured")
        _client = Clerk(bearer_auth=settings.CLERK_SECRET_KEY)
    return _client


async def verify_clerk_token(token: str) -> dict[str, Any]:
    """
    Verifies a Clerk session token and returns its claims.

    `authorized_parties` is the part worth not skipping: it pins the `azp`
    claim to our own origins, so a token minted for a different Clerk
    application cannot be replayed against this backend.
    """
    if not settings.CLERK_SECRET_KEY:
        raise ClerkAuthError("CLERK_SECRET_KEY is not configured")

    try:
        payload = await verify_token_async(
            token,
            VerifyTokenOptions(
                secret_key=settings.CLERK_SECRET_KEY,
                authorized_parties=settings.CLERK_AUTHORIZED_PARTIES or None,
            ),
        )
    except TokenVerificationError as e:
        raise ClerkAuthError(f"Clerk token rejected: {e.reason.value[1]}") from e
    except Exception as e:
        raise ClerkAuthError(f"Clerk token verification failed: {e}") from e

    if not payload.get("sub"):
        raise ClerkAuthError("Clerk token has no subject claim")

    return payload


async def get_clerk_user(clerk_user_id: str) -> Any:
    """The full Clerk user record — email, username, avatar."""
    try:
        return await get_client().users.get_async(user_id=clerk_user_id)
    except Exception as e:
        raise ClerkAuthError(f"Could not load Clerk user {clerk_user_id}: {e}") from e


def extract_identity(clerk_user: Any) -> dict[str, str | bool]:
    """
    Flattens a Clerk user into the fields this app stores.

    Clerk models email as a list of addresses with a separate `primary_email_
    address_id` pointer, and username can be null for OAuth-only signups. This
    resolves both, and falls back through username -> first/last -> the email
    local part so a profile is never created with a blank display name.
    """
    emails = getattr(clerk_user, "email_addresses", None) or []
    primary_id = getattr(clerk_user, "primary_email_address_id", None)

    primary = next((e for e in emails if getattr(e, "id", None) == primary_id), None)
    if primary is None and emails:
        primary = emails[0]

    email = getattr(primary, "email_address", "") if primary else ""

    verified = False
    if primary is not None:
        verification = getattr(primary, "verification", None)
        verified = getattr(verification, "status", None) == "verified"

    username = getattr(clerk_user, "username", None)
    if not username:
        first = getattr(clerk_user, "first_name", None) or ""
        last = getattr(clerk_user, "last_name", None) or ""
        username = f"{first} {last}".strip()
    if not username:
        username = email.split("@")[0] if email else "user"

    return {
        "email": email,
        "email_verified": verified,
        "username": username,
        "avatar_url": getattr(clerk_user, "image_url", None) or "",
    }


async def get_oauth_token(clerk_user_id: str, provider: str = "github") -> str | None:
    """
    The user's OAuth access token for a connected provider.

    Clerk mints a fresh one on each read, so there is nothing to cache, encrypt
    or refresh on our side. Returns None when the user has not connected that
    provider, which callers should treat as "not connected" rather than as an
    error.
    """
    try:
        result = await get_client().users.get_o_auth_access_token_async(
            user_id=clerk_user_id, provider=provider
        )
    except Exception as e:
        logger.warning(f"No {provider} OAuth token for {clerk_user_id}: {e}")
        return None

    tokens = getattr(result, "data", None) or result
    if not tokens:
        return None

    first = tokens[0] if isinstance(tokens, list) else tokens
    return getattr(first, "token", None)
