"""
The `users` collection: who someone is to Clerk, and whether they may act here.

Kept deliberately small. Anything a user can edit about themselves lives in
`profiles` instead (see profile_repo), so this collection only ever changes as
a result of an authorization decision — sign-in, ban, verification, revocation.
"""

import logging
from datetime import datetime, timezone

from pymongo import ReturnDocument

from app.core.ids import new_user_id
from app.db.mongodb import get_user_collection

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UserRepository:
    """Mongo access for user documents. No policy — see app/services/auth_service.py."""

    async def get_by_id(self, user_id: str) -> dict | None:
        return await get_user_collection().find_one({"_id": user_id})

    async def get_by_clerk_id(self, clerk_user_id: str) -> dict | None:
        return await get_user_collection().find_one({"clerk_user_id": clerk_user_id})

    async def upsert_from_clerk(
        self,
        clerk_user_id: str,
        email: str,
        email_verified: bool = False,
    ) -> dict:
        """
        Finds or creates the user behind a Clerk identity.

        Checks first by clerk_user_id. If not found, checks by email to handle
        re-registered Clerk accounts or existing user linking. If no user exists,
        creates a new record safely.
        """
        now = _now()
        coll = get_user_collection()

        # 1. Try finding and updating by clerk_user_id
        user = await coll.find_one_and_update(
            {"clerk_user_id": clerk_user_id},
            {
                "$set": {
                    "email": email,
                    "email_verified": email_verified,
                    "updated_at": now,
                    "last_login_at": now,
                },
            },
            return_document=ReturnDocument.AFTER,
        )
        if user:
            return user

        # 2. If not found by clerk_user_id, check if a user exists with this email
        user = await coll.find_one_and_update(
            {"email": email},
            {
                "$set": {
                    "clerk_user_id": clerk_user_id,
                    "email_verified": email_verified,
                    "updated_at": now,
                    "last_login_at": now,
                },
            },
            return_document=ReturnDocument.AFTER,
        )
        if user:
            logger.info(f"Linked existing user {user['_id']} ({email}) to new clerk_user_id {clerk_user_id}")
            return user

        # 3. Create a new user record if not found by clerk_user_id or email
        new_doc = {
            "_id": new_user_id(),
            "clerk_user_id": clerk_user_id,
            "email": email,
            "email_verified": email_verified,
            "is_verified": True,
            "is_banned": False,
            "ban_reason": None,
            "banned_at": None,
            "role": "user",
            "token_version": 0,
            "created_at": now,
            "updated_at": now,
            "last_login_at": now,
        }
        try:
            await coll.insert_one(new_doc)
            return new_doc
        except Exception as e:
            # Handle potential race condition if another request created it concurrently
            existing = await coll.find_one({
                "$or": [{"clerk_user_id": clerk_user_id}, {"email": email}]
            })
            if existing:
                return existing
            raise e

    async def set_banned(
        self, user_id: str, banned: bool, reason: str | None = None
    ) -> dict | None:
        """
        Bans or unbans, and bumps token_version on the way in.

        The bump is the part that matters: access tokens are verified without a
        database read, so without invalidating them a banned user would keep
        working until their current token expired.
        """
        update: dict = {
            "$set": {
                "is_banned": banned,
                "ban_reason": reason if banned else None,
                "banned_at": _now() if banned else None,
                "updated_at": _now(),
            }
        }
        if banned:
            update["$inc"] = {"token_version": 1}

        return await get_user_collection().find_one_and_update(
            {"_id": user_id}, update, return_document=ReturnDocument.AFTER
        )

    async def set_verified(self, user_id: str, verified: bool) -> dict | None:
        return await get_user_collection().find_one_and_update(
            {"_id": user_id},
            {"$set": {"is_verified": verified, "updated_at": _now()}},
            return_document=ReturnDocument.AFTER,
        )

    async def sync_email(
        self, clerk_user_id: str, email: str, email_verified: bool
    ) -> dict | None:
        """Applies a `user.updated` webhook. No-op if we have never seen them."""
        return await get_user_collection().find_one_and_update(
            {"clerk_user_id": clerk_user_id},
            {
                "$set": {
                    "email": email,
                    "email_verified": email_verified,
                    "updated_at": _now(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def bump_token_version(self, user_id: str) -> dict | None:
        """Invalidates every access token outstanding for this user."""
        return await get_user_collection().find_one_and_update(
            {"_id": user_id},
            {"$inc": {"token_version": 1}, "$set": {"updated_at": _now()}},
            return_document=ReturnDocument.AFTER,
        )


user_repo = UserRepository()
