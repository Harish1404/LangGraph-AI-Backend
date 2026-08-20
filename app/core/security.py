"""
Token minting, token hashing, and cookie policy.

Two different kinds of credential live here, and they are deliberately not
treated the same way:

  Access token   A short-lived signed JWT. Verified with no database read,
                 which is what keeps it off the hot path of every request —
                 and also why its lifetime doubles as the worst-case delay
                 before a ban takes effect. Kept to ~15 minutes for that
                 reason. Carries `ver`, checked against the user's
                 token_version so a ban can cut it short.

  Refresh token  Opaque random bytes, never signed, stored only as a hash.
                 Long-lived, rotated on every use, and the only credential
                 that can mint a new access token.

The refresh cookie is scoped to the refresh endpoint's path, so it is simply
not sent on ordinary API calls. That single line removes it from the blast
radius of an XSS-readable response or a leaky proxy log on every other route.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Response

from app.core.config import settings

logger = logging.getLogger(__name__)

# Rejected rather than defaulted: a blank secret would still sign tokens, and
# every one of them would be forgeable.
if not settings.JWT_SECRET:
    logger.warning(
        "JWT_SECRET is not set — /auth endpoints will fail. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )


class TokenError(Exception):
    """Raised when an access token is missing, malformed, expired, or forged."""


# ── Access tokens ────────────────────────────────────────────────────────────

def create_access_token(user: dict) -> str:
    """
    Signs a short-lived access token for a user document.

    `sub` is the app's own user id, not the Clerk id, so every downstream
    ownership check compares the same value it stores.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["_id"],
        "clerk_id": user.get("clerk_user_id"),
        "role": user.get("role", "user"),
        "ver": user.get("token_version", 0),
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_TTL_MIN),
        "typ": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Verifies signature and expiry, and returns the claims.

    `algorithms` is pinned to a single value on purpose: accepting a list the
    caller does not control is how algorithm-confusion attacks get in.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as e:
        raise TokenError("Access token expired") from e
    except jwt.InvalidTokenError as e:
        raise TokenError("Invalid access token") from e

    if payload.get("typ") != "access":
        # Stops a token minted for some other purpose being replayed here.
        raise TokenError("Wrong token type")

    return payload


# ── Refresh tokens ───────────────────────────────────────────────────────────

def generate_refresh_token() -> str:
    """256 bits of CSPRNG output. The raw value is shown to the client once."""
    return secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    """
    The form stored in MongoDB.

    Plain SHA-256, not bcrypt — see the module docstring in
    app/repositories/session_repo.py for why a slow KDF is the wrong tool for a
    high-entropy random token.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Cookies ──────────────────────────────────────────────────────────────────

def set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    """
    Writes both credentials as HttpOnly cookies.

    HttpOnly is the point: no script on the page can read either value, so an
    XSS bug cannot exfiltrate the session the way a localStorage token would.
    """
    response.set_cookie(
        key=settings.ACCESS_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
        path="/",
        max_age=settings.ACCESS_TOKEN_TTL_MIN * 60,
    )
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
        # Scoped: the browser sends this only to /auth/refresh.
        path=settings.REFRESH_COOKIE_PATH,
        max_age=settings.REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60,
    )


def clear_auth_cookies(response: Response) -> None:
    """
    Deletes both cookies.

    The path arguments must match what set_auth_cookies used — a cookie is
    identified by (name, domain, path), so deleting the refresh cookie at "/"
    would silently leave the real one in place.
    """
    response.delete_cookie(
        key=settings.ACCESS_COOKIE_NAME,
        path="/",
        domain=settings.COOKIE_DOMAIN,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path=settings.REFRESH_COOKIE_PATH,
        domain=settings.COOKIE_DOMAIN,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
    )
