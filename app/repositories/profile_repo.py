"""
The `profiles` collection: the user-editable half of an account, 1:1 with users.

Split out from `users` on purpose. This document is written by the person it
describes, `users` is written only by authorization decisions — so a profile
update can never touch a ban flag or a role, no matter what the request body
contains.
"""

import logging
from datetime import datetime, timezone

from pymongo import ReturnDocument

from app.core.ids import new_profile_id
from app.db.mongodb import get_profile_collection

logger = logging.getLogger(__name__)

# The only fields a user may write. Anything else in a PATCH body is dropped
# rather than rejected, so a client sending extra keys still succeeds.
EDITABLE_FIELDS = {"username", "avatar_url", "mobile", "address", "bio"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProfileRepository:
    async def get_by_user_id(self, user_id: str) -> dict | None:
        return await get_profile_collection().find_one({"user_id": user_id})

    async def ensure_for_user(
        self,
        user_id: str,
        username: str,
        email: str,
        avatar_url: str = "",
    ) -> dict:
        """
        Creates the profile on first sign-in, or returns the existing one.

        Everything the OAuth provider gave us is written on insert; the rest
        starts as an empty string for the user to fill in from /profile. They
        are `$setOnInsert` rather than `$set` so a later sign-in can never
        overwrite a display name or avatar the user has since changed.
        """
        return await get_profile_collection().find_one_and_update(
            {"user_id": user_id},
            {
                "$setOnInsert": {
                    "_id": new_profile_id(),
                    "user_id": user_id,
                    "username": username,
                    "email": email,
                    "avatar_url": avatar_url,
                    "mobile": "",
                    "address": "",
                    "bio": "",
                    "created_at": _now(),
                    "updated_at": _now(),
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    async def update(self, user_id: str, changes: dict) -> dict | None:
        """
        Applies a partial update, ignoring any field not in EDITABLE_FIELDS.

        The filter is the security boundary: it is what stops a crafted PATCH
        body from setting `role` or `is_banned` on the way through.
        """
        allowed = {
            k: v for k, v in changes.items() if k in EDITABLE_FIELDS and v is not None
        }
        if not allowed:
            return await self.get_by_user_id(user_id)

        allowed["updated_at"] = _now()

        return await get_profile_collection().find_one_and_update(
            {"user_id": user_id},
            {"$set": allowed},
            return_document=ReturnDocument.AFTER,
        )

    async def delete_for_user(self, user_id: str) -> None:
        await get_profile_collection().delete_one({"user_id": user_id})


profile_repo = ProfileRepository()
