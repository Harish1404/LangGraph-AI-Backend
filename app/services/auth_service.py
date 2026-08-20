"""
Session policy: establishing, refreshing, and ending a login.

The division of labour across this system is worth stating once, because it is
what makes the two-provider setup safe rather than confusing:

    Clerk       owns identity. It runs the sign-in UI and OAuth, and is the
                source of truth for *who* someone is.
    This layer  owns the session. It decides whether that person may act right
                now (is_verified / is_banned), and issues the credentials that
                every other endpoint checks.
    Webhooks    keep the two in sync. When Clerk revokes or deletes, the
                handler in app/routes/webhooks.py burns the local tokens.

Without that third piece the two drift apart — a user revoked in Clerk would
keep a working session here until it expired. It is not optional.
"""

import logging

from app.core.ids import new_family_id
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_token,
)
from app.core.config import settings
from app.repositories.profile_repo import profile_repo
from app.repositories.session_repo import session_repo
from app.repositories.user_repo import user_repo
from app.services import clerk_service
from app.services.clerk_service import ClerkAuthError

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """A session could not be established or continued."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthService:
    # ── Establishing a session ───────────────────────────────────────────────

    async def establish_session(
        self,
        clerk_token: str,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> tuple[dict, dict, str, str]:
        """
        Trades a verified Clerk token for this app's own session.

        Called once after Clerk sign-in. Returns (user, profile, access, refresh).

        The ban check sits *after* the upsert on purpose: a banned user's record
        still gets its last_login_at touched, so attempted sign-ins by a banned
        account remain visible rather than silently vanishing.
        """
        try:
            claims = await clerk_service.verify_clerk_token(clerk_token)
            clerk_user_id = claims["sub"]
            clerk_user = await clerk_service.get_clerk_user(clerk_user_id)
        except ClerkAuthError as e:
            raise AuthError(str(e), status_code=401) from e

        identity = clerk_service.extract_identity(clerk_user)
        if not identity["email"]:
            raise AuthError("Clerk account has no email address", status_code=400)

        user = await user_repo.upsert_from_clerk(
            clerk_user_id=clerk_user_id,
            email=identity["email"],
            email_verified=identity["email_verified"],
        )

        if user.get("is_banned"):
            raise AuthError(
                user.get("ban_reason") or "This account has been suspended",
                status_code=403,
            )
        if not user.get("is_verified", True):
            raise AuthError("This account is not verified", status_code=403)

        # First sight of this user creates their profile; a returning user's
        # edits are preserved (ensure_for_user uses $setOnInsert only).
        profile = await profile_repo.ensure_for_user(
            user_id=user["_id"],
            username=identity["username"],
            email=identity["email"],
            avatar_url=identity["avatar_url"],
        )

        access_token, refresh_token = await self._issue_pair(
            user, family_id=new_family_id(), user_agent=user_agent, ip=ip
        )
        return user, profile, access_token, refresh_token

    async def _issue_pair(
        self,
        user: dict,
        family_id: str,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> tuple[str, str]:
        """Mints an access/refresh pair and records the refresh token's hash."""
        access_token = create_access_token(user)
        raw_refresh = generate_refresh_token()

        await session_repo.create(
            user_id=user["_id"],
            token_hash=hash_token(raw_refresh),
            family_id=family_id,
            ttl_days=settings.REFRESH_TOKEN_TTL_DAYS,
            user_agent=user_agent,
            ip=ip,
        )
        return access_token, raw_refresh

    # ── Continuing a session ─────────────────────────────────────────────────

    async def refresh_session(
        self,
        raw_refresh: str,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> tuple[dict, str, str]:
        """
        Rotates a refresh token. Returns (user, access, refresh).

        Reuse detection is the reason this is more than a lookup. Tokens are
        single-use, so a token presented twice means one of two things: the
        real holder retried, or someone stole it and the real holder already
        spent it. We cannot tell those apart from here, so the safe response to
        either is to burn the entire family and force a fresh sign-in.
        """
        token_hash = hash_token(raw_refresh)
        doc = await session_repo.get_by_hash(token_hash)

        if doc is None:
            raise AuthError("Invalid refresh token")

        if not session_repo.is_usable(doc):
            # Already spent, revoked, or expired. If it was spent, treat it as
            # a possible theft and kill every sibling token.
            if doc.get("replaced_by") is not None:
                logger.warning(
                    f"Refresh token reuse detected for user {doc['user_id']}; "
                    f"revoking family {doc['family_id']}"
                )
                await session_repo.revoke_family(doc["family_id"])
            raise AuthError("Refresh token is no longer valid")

        user = await user_repo.get_by_id(doc["user_id"])
        if user is None:
            await session_repo.revoke_family(doc["family_id"])
            raise AuthError("Account no longer exists")

        if user.get("is_banned"):
            # A ban takes effect here immediately, regardless of how long the
            # current access token still has to run.
            await session_repo.revoke_all_for_user(user["_id"])
            raise AuthError(
                user.get("ban_reason") or "This account has been suspended",
                status_code=403,
            )

        access_token, new_raw_refresh = await self._issue_pair(
            user, family_id=doc["family_id"], user_agent=user_agent, ip=ip
        )

        # Consume the old token last, and conditionally. If this returns False
        # another request rotated it in the meantime, so the pair we just minted
        # must not be handed out — that would fork the family into two live
        # chains and make genuine reuse undetectable afterwards.
        won = await session_repo.mark_rotated(
            token_hash, replaced_by=hash_token(new_raw_refresh)
        )
        if not won:
            await session_repo.revoke_family(doc["family_id"])
            raise AuthError("Refresh token is no longer valid")

        return user, access_token, new_raw_refresh

    # ── Ending a session ─────────────────────────────────────────────────────

    async def logout(self, raw_refresh: str | None) -> None:
        """Ends one session. Silent when there is nothing to end."""
        if not raw_refresh:
            return
        await session_repo.revoke(hash_token(raw_refresh))

    async def logout_all(self, user_id: str) -> int:
        """Ends every session for a user, on every device."""
        return await session_repo.revoke_all_for_user(user_id)

    # ── Reading ──────────────────────────────────────────────────────────────

    async def get_me(self, user_id: str) -> tuple[dict, dict]:
        """The user plus their profile, creating the profile if it is missing."""
        user = await user_repo.get_by_id(user_id)
        if user is None:
            raise AuthError("Account no longer exists")

        profile = await profile_repo.get_by_user_id(user_id)
        if profile is None:
            # Defensive: a profile can only be absent if a signup was
            # interrupted between the two writes.
            profile = await profile_repo.ensure_for_user(
                user_id=user_id,
                username=(user.get("email") or "user").split("@")[0],
                email=user.get("email", ""),
            )

        return user, profile

    async def update_profile(self, user_id: str, changes: dict) -> dict:
        profile = await profile_repo.update(user_id, changes)
        if profile is None:
            raise AuthError("Profile not found", status_code=404)
        return profile


auth_service = AuthService()
