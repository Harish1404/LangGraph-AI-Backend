"""
Session lifecycle: establish, refresh, inspect, end.

The flow, end to end:

    1. The user signs in through Clerk on the frontend.
    2. The frontend sends that Clerk token once to POST /auth/session.
    3. We verify it, upsert the user and profile, and set two HttpOnly cookies.
    4. Every later request — including the WebSocket — carries those cookies
       automatically, and Clerk is not consulted again.

Tokens never appear in a response body. They are set as cookies the browser
sends but no script can read, which is the whole reason for doing it this way
rather than handing a JWT to localStorage.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import clear_auth_cookies, set_auth_cookies
from app.schemas.auth import (
    MeOut,
    MessageOut,
    ProfileUpdate,
    SessionOut,
    profile_out,
    user_out,
)
from app.services.auth_service import AuthError, auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


def _client_ip(request: Request) -> str:
    """
    Best-effort client IP, for the session audit trail only.

    X-Forwarded-For is spoofable unless a trusted proxy overwrites it, so this
    is recorded for debugging and never used to make an authorization decision.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expected a Clerk session token in the Authorization header",
        )
    token = header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
        )
    return token


@router.post("/session", response_model=SessionOut)
async def create_session(request: Request, response: Response):
    """
    Trade a Clerk session token for this backend's own session cookies.

    Called once by the frontend right after Clerk sign-in. Safe to call again:
    it re-issues cookies and starts a fresh token family, which is what makes
    it usable as a recovery path when cookies are cleared.
    """
    clerk_token = _bearer_token(request)

    try:
        user, profile, access_token, refresh_token = await auth_service.establish_session(
            clerk_token=clerk_token,
            user_agent=request.headers.get("User-Agent"),
            ip=_client_ip(request),
        )
    except AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error during session establishment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal authentication error: {str(e)}",
        ) from e

    set_auth_cookies(response, access_token, refresh_token)
    logger.info(f"Session established for {user['_id']}")

    return SessionOut(
        user=user_out(user),
        profile=profile_out(profile),
        expires_in=settings.ACCESS_TOKEN_TTL_MIN * 60,
    )


@router.post("/refresh", response_model=SessionOut)
async def refresh_session(request: Request, response: Response):
    """
    Rotate the refresh token and mint a new access token.

    This is the only path the refresh cookie is sent to — it is scoped to this
    URL in set_auth_cookies, so it never rides along on ordinary API calls.

    On failure the cookies are cleared as well as rejected, so a client holding
    a dead token is pushed back to sign-in rather than retrying forever.
    """
    raw_refresh = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token"
        )

    try:
        user, access_token, new_refresh = await auth_service.refresh_session(
            raw_refresh=raw_refresh,
            user_agent=request.headers.get("User-Agent"),
            ip=_client_ip(request),
        )
    except AuthError as e:
        clear_auth_cookies(response)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    _, profile = await auth_service.get_me(user["_id"])
    set_auth_cookies(response, access_token, new_refresh)

    return SessionOut(
        user=user_out(user),
        profile=profile_out(profile),
        expires_in=settings.ACCESS_TOKEN_TTL_MIN * 60,
    )


@router.post("/logout", response_model=MessageOut)
async def logout(request: Request, response: Response):
    """
    End this session and clear both cookies.

    Intentionally unauthenticated: logging out with an already-expired access
    token must still work, or a user whose token lapsed could never clear their
    own cookies.
    """
    await auth_service.logout(request.cookies.get(settings.REFRESH_COOKIE_NAME))
    clear_auth_cookies(response)
    return MessageOut(message="Signed out")


@router.post("/logout-all", response_model=MessageOut)
async def logout_all(response: Response, user: dict = Depends(get_current_user)):
    """End every session for this user, on every device."""
    count = await auth_service.logout_all(user["_id"])
    clear_auth_cookies(response)
    return MessageOut(message=f"Signed out of {count} session(s)")


@router.get("/me", response_model=MeOut)
async def get_me(user: dict = Depends(get_current_user)):
    """The signed-in user and their profile."""
    user_doc, profile = await auth_service.get_me(user["_id"])
    return MeOut(user=user_out(user_doc), profile=profile_out(profile))


@router.patch("/me/profile", response_model=MeOut)
async def update_profile(
    body: ProfileUpdate, user: dict = Depends(get_current_user)
):
    """
    Update the editable half of the account.

    `exclude_unset` is what makes this a genuine PATCH: a field the client did
    not send is left alone, rather than being overwritten with None.
    """
    try:
        profile = await auth_service.update_profile(
            user["_id"], body.model_dump(exclude_unset=True)
        )
    except AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    return MeOut(user=user_out(user), profile=profile_out(profile))
