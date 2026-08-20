"""
The `refresh_tokens` collection: one document per issued refresh token.

Only the SHA-256 of each token is stored, so a database leak does not hand the
attacker usable sessions. SHA-256 rather than bcrypt is deliberate — bcrypt is
slow on purpose to resist brute force against *low-entropy* human passwords,
but a refresh token is 256 bits from `secrets.token_urlsafe(32)` and is not
guessable at any hash speed. Paying ~100ms per call on a hot endpoint would buy
nothing. See app/core/security.py.

Tokens are chained: every rotation records `replaced_by` on the old document
and issues a new one carrying the same `family_id`. That chain is what makes
theft detectable — see `mark_rotated`.
"""

import logging
from datetime import datetime, timedelta, timezone

from app.core.ids import new_session_id
from app.db.mongodb import get_refresh_token_collection

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionRepository:
    async def create(
        self,
        user_id: str,
        token_hash: str,
        family_id: str,
        ttl_days: int,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> dict:
        doc = {
            "_id": new_session_id(),
            "user_id": user_id,
            "token_hash": token_hash,
            "family_id": family_id,
            "issued_at": _now(),
            "expires_at": _now() + timedelta(days=ttl_days),
            "revoked_at": None,
            "replaced_by": None,
            "user_agent": (user_agent or "")[:300],
            "ip": ip or "",
        }
        await get_refresh_token_collection().insert_one(doc)
        return doc

    async def get_by_hash(self, token_hash: str) -> dict | None:
        return await get_refresh_token_collection().find_one({"token_hash": token_hash})

    async def mark_rotated(self, token_hash: str, replaced_by: str) -> bool:
        """
        Consumes a refresh token exactly once. True if we won, False if it was
        already spent.

        The `replaced_by: None` clause in the filter is what makes this work.
        The check and the write are one atomic operation, so two requests
        presenting the same token cannot both succeed — the loser matches
        nothing and gets False. A False here means the token had already been
        rotated, which is either a stolen token being replayed or a client
        retrying; both are handled the same way, by burning the whole family.

        Doing this as a read-then-write would leave a window where both
        requests read `replaced_by == None` and both proceeded.
        """
        result = await get_refresh_token_collection().update_one(
            {"token_hash": token_hash, "replaced_by": None, "revoked_at": None},
            {"$set": {"replaced_by": replaced_by, "revoked_at": _now()}},
        )
        return result.modified_count == 1

    async def revoke(self, token_hash: str) -> None:
        await get_refresh_token_collection().update_one(
            {"token_hash": token_hash, "revoked_at": None},
            {"$set": {"revoked_at": _now()}},
        )

    async def revoke_family(self, family_id: str) -> int:
        """
        Kills every token descended from one login.

        Called when a spent token is presented again: we cannot tell the thief
        from the legitimate holder, so the safe move is to end the whole chain
        and make both re-authenticate.
        """
        result = await get_refresh_token_collection().update_many(
            {"family_id": family_id, "revoked_at": None},
            {"$set": {"revoked_at": _now()}},
        )
        if result.modified_count:
            logger.warning(
                f"Revoked refresh token family {family_id} "
                f"({result.modified_count} token(s))"
            )
        return result.modified_count

    async def revoke_all_for_user(self, user_id: str) -> int:
        """Every session everywhere — logout-all, ban, or a Clerk revocation."""
        result = await get_refresh_token_collection().update_many(
            {"user_id": user_id, "revoked_at": None},
            {"$set": {"revoked_at": _now()}},
        )
        return result.modified_count

    @staticmethod
    def is_usable(doc: dict) -> bool:
        """A token is usable only if it is unrevoked, unrotated, and unexpired."""
        if doc.get("revoked_at") is not None:
            return False
        if doc.get("replaced_by") is not None:
            return False

        expires_at = doc.get("expires_at")
        if expires_at is None:
            return False

        # Motor hands back naive datetimes for BSON dates; treat them as UTC
        # rather than letting a naive/aware comparison raise TypeError.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return expires_at > _now()


session_repo = SessionRepository()
