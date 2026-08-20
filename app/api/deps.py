"""
The auth dependencies every protected endpoint hangs off.

There is one rule these exist to enforce: **identity is never taken from the
request body**. Before this, `user_id` was a field on ChatRequest, which meant
anyone could act as anyone by editing one line of JSON. Now it comes from a
signed, HttpOnly cookie the client cannot write, and the body field is gone.

`get_current_user` reads the user document on every request rather than
trusting the token's claims alone. That costs one indexed lookup, and buys the
ability to enforce a ban within one access-token lifetime instead of waiting
for it to expire — the `ver` check below is what closes that window.
"""

import logging

from fastapi import Depends, HTTPException, Request, WebSocket, status

from app.core.config import settings
from app.core.security import TokenError, decode_access_token
from app.repositories.user_repo import user_repo

logger = logging.getLogger(__name__)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def _read_token(request: Request) -> str | None:
    """
    Pulls the access token from the cookie, falling back to a bearer header.

    The cookie is the real mechanism — it is HttpOnly, so script on the page
    cannot read it, and it rides the WebSocket handshake for free. The header
    fallback exists so `curl` and the API docs stay usable during development.
    """
    token = request.cookies.get(settings.ACCESS_COOKIE_NAME)
    if token:
        return token

    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip() or None

    return None


async def _resolve_user(token: str | None) -> dict:
    """Shared by the HTTP and WebSocket paths. Raises on any failure."""
    if not token:
        raise _UNAUTHENTICATED

    try:
        payload = decode_access_token(token)
    except TokenError as e:
        logger.debug(f"Access token rejected: {e}")
        raise _UNAUTHENTICATED from e

    user = await user_repo.get_by_id(payload["sub"])
    if user is None:
        raise _UNAUTHENTICATED

    # A ban bumps token_version, so every token minted before it stops
    # verifying here — without waiting for the 15-minute expiry.
    if payload.get("ver") != user.get("token_version", 0):
        logger.info(f"Stale token version for {user['_id']}; forcing re-auth")
        raise _UNAUTHENTICATED

    if user.get("is_banned"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=user.get("ban_reason") or "This account has been suspended",
        )

    return user


async def get_current_user(request: Request) -> dict:
    """The authenticated user document. 401 if absent, 403 if banned."""
    return await _resolve_user(_read_token(request))


async def get_current_user_id(user: dict = Depends(get_current_user)) -> str:
    """
    Just the id, for the many endpoints that need nothing else.

    This is the replacement for the hardcoded "default_user" that used to be
    scattered through the route and schema layers.
    """
    return user["_id"]


async def require_verified(user: dict = Depends(get_current_user)) -> dict:
    """Additionally requires the app-level verification flag."""
    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your account to continue",
        )
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return user


async def get_current_user_ws(websocket: WebSocket) -> dict | None:
    """
    The WebSocket equivalent, for /ws/voice.

    Returns None instead of raising, because a WebSocket cannot answer with an
    HTTP status once the handshake is done — the caller closes with 1008
    instead. Cookies are used here rather than a bearer token because the
    browser WebSocket API cannot set request headers at all; the alternative
    would be a token in the query string, which lands in every access log.
    """
    token = websocket.cookies.get(settings.ACCESS_COOKIE_NAME)
    try:
        return await _resolve_user(token)
    except HTTPException:
        return None
