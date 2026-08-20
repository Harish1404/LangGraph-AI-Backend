"""
Clerk webhook receiver.

This is the piece that makes running two session authorities safe. Clerk owns
identity and this backend owns sessions; without a sync channel the two drift,
and a user deleted or revoked in Clerk would keep a working session here until
their refresh token expired. Every handler below exists to close that gap.

Unauthenticated by design — Clerk's servers call this, not a signed-in user.
Trust comes from the Svix signature instead, verified against
CLERK_WEBHOOK_SECRET before the payload is even parsed. Skipping that check
would let anyone on the internet ban or delete accounts by POSTing JSON.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from svix.webhooks import Webhook, WebhookVerificationError

from app.core.config import settings
from app.repositories.profile_repo import profile_repo
from app.repositories.session_repo import session_repo
from app.repositories.user_repo import user_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


def _primary_email(data: dict) -> tuple[str, bool]:
    """Resolves the primary address out of a Clerk webhook payload."""
    addresses = data.get("email_addresses") or []
    primary_id = data.get("primary_email_address_id")

    primary = next((a for a in addresses if a.get("id") == primary_id), None)
    if primary is None and addresses:
        primary = addresses[0]
    if primary is None:
        return "", False

    verified = (primary.get("verification") or {}).get("status") == "verified"
    return primary.get("email_address", ""), verified


def _username(data: dict, email: str) -> str:
    username = data.get("username")
    if not username:
        first = data.get("first_name") or ""
        last = data.get("last_name") or ""
        username = f"{first} {last}".strip()
    if not username:
        username = email.split("@")[0] if email else "user"
    return username


@router.post("/clerk", status_code=status.HTTP_204_NO_CONTENT)
async def clerk_webhook(request: Request):
    """
    Applies a Clerk event to the local user store.

    Returns 204 for events we do not handle rather than an error, so Clerk does
    not retry them forever.
    """
    if not settings.CLERK_WEBHOOK_SECRET:
        logger.error("CLERK_WEBHOOK_SECRET is not set; rejecting webhook")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhooks are not configured",
        )

    payload = await request.body()

    try:
        # Svix verifies the signature over the *raw* body, so this must happen
        # before any parsing — re-serialised JSON would not match the digest.
        event = Webhook(settings.CLERK_WEBHOOK_SECRET).verify(
            payload, dict(request.headers)
        )
    except WebhookVerificationError as e:
        logger.warning(f"Rejected Clerk webhook with bad signature: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature"
        ) from e

    event_type = event.get("type", "")
    data = event.get("data", {}) or {}
    clerk_user_id = data.get("id") or data.get("user_id")

    if not clerk_user_id:
        logger.info(f"Clerk webhook {event_type!r} carried no user id; ignoring")
        return None

    logger.info(f"Clerk webhook: {event_type} for {clerk_user_id}")

    if event_type in ("user.created", "user.updated"):
        email, verified = _primary_email(data)
        user = await user_repo.sync_email(clerk_user_id, email, verified)

        if user is None:
            # A user.created that arrives before their first POST /auth/session.
            # Creating the record now keeps the two paths idempotent — the
            # upsert in establish_session will simply find it already there.
            if not email:
                return None
            user = await user_repo.upsert_from_clerk(clerk_user_id, email, verified)

        await profile_repo.ensure_for_user(
            user_id=user["_id"],
            username=_username(data, email),
            email=email,
            avatar_url=data.get("image_url") or "",
        )

    elif event_type == "user.deleted":
        user = await user_repo.get_by_clerk_id(clerk_user_id)
        if user:
            # Banned rather than deleted: their conversations still reference
            # this user_id, and a hard delete would orphan them. This locks the
            # account out permanently while keeping the history intact.
            await user_repo.set_banned(
                user["_id"], True, reason="Account deleted in Clerk"
            )
            revoked = await session_repo.revoke_all_for_user(user["_id"])
            logger.info(f"Deleted Clerk user {clerk_user_id}: revoked {revoked} session(s)")

    elif event_type in ("session.revoked", "session.removed", "session.ended"):
        user = await user_repo.get_by_clerk_id(clerk_user_id)
        if user:
            await session_repo.revoke_all_for_user(user["_id"])
            # Cuts short any access token already issued, which would otherwise
            # keep working for up to ACCESS_TOKEN_TTL_MIN after the revocation.
            await user_repo.bump_token_version(user["_id"])

    elif event_type == "user.banned":
        user = await user_repo.get_by_clerk_id(clerk_user_id)
        if user:
            await user_repo.set_banned(user["_id"], True, reason="Banned in Clerk")
            await session_repo.revoke_all_for_user(user["_id"])

    elif event_type == "user.unbanned":
        user = await user_repo.get_by_clerk_id(clerk_user_id)
        if user:
            await user_repo.set_banned(user["_id"], False)

    return None
