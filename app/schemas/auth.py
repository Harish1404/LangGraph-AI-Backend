"""
Request and response models for the auth endpoints.

As in app/schemas/chat.py, these also translate the MongoDB shape (`_id`) into
the API shape, so the database layout is never exposed to a client.

Note what is *not* here: no request model carries a user id. Identity comes
from the session cookie via app/api/deps.py, never from a body a client can
edit.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Requests ──────────────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    """
    A partial profile edit. Every field optional; omitted ones are untouched.

    Only these five fields exist on this model, which is the first of two
    gates — the second is EDITABLE_FIELDS in profile_repo. A body containing
    `role` or `is_banned` is dropped by Pydantic before it reaches the repo.
    """

    username: Optional[str] = Field(default=None, min_length=1, max_length=60)
    avatar_url: Optional[str] = Field(default=None, max_length=500)
    mobile: Optional[str] = Field(default=None, max_length=25)
    address: Optional[str] = Field(default=None, max_length=300)
    bio: Optional[str] = Field(default=None, max_length=500)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("Username cannot be blank")
        return v

    @field_validator("avatar_url")
    @classmethod
    def _check_avatar(cls, v: Optional[str]) -> Optional[str]:
        """
        Only http(s) URLs.

        Rejecting other schemes here stops a stored `javascript:` URL from
        being rendered into an href somewhere down the line.
        """
        if v is None or v == "":
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("Avatar URL must start with http:// or https://")
        return v


# ── Responses ─────────────────────────────────────────────────────────────────

class ProfileOut(BaseModel):
    user_id: str
    username: str
    email: str
    avatar_url: str = ""
    mobile: str = ""
    address: str = ""
    bio: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserOut(BaseModel):
    user_id: str
    email: str
    email_verified: bool = False
    is_verified: bool = True
    is_banned: bool = False
    role: Literal["user", "admin"] = "user"
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class MeOut(BaseModel):
    """What GET /auth/me returns: the account and its profile together."""

    user: UserOut
    profile: ProfileOut


class SessionOut(BaseModel):
    """
    The result of establishing or refreshing a session.

    Deliberately carries no tokens — both credentials are delivered as HttpOnly
    cookies, so putting them in the body too would hand them straight back to
    any script on the page and undo the point of HttpOnly.
    """

    user: UserOut
    profile: ProfileOut
    expires_in: int


class MessageOut(BaseModel):
    message: str


# ── Document -> model ─────────────────────────────────────────────────────────

def user_out(doc: dict) -> UserOut:
    return UserOut(
        user_id=doc["_id"],
        email=doc.get("email", ""),
        email_verified=doc.get("email_verified", False),
        is_verified=doc.get("is_verified", True),
        is_banned=doc.get("is_banned", False),
        role=doc.get("role", "user"),
        created_at=doc.get("created_at"),
        last_login_at=doc.get("last_login_at"),
    )


def profile_out(doc: dict) -> ProfileOut:
    return ProfileOut(
        user_id=doc["user_id"],
        username=doc.get("username", ""),
        email=doc.get("email", ""),
        avatar_url=doc.get("avatar_url", ""),
        mobile=doc.get("mobile", ""),
        address=doc.get("address", ""),
        bio=doc.get("bio", ""),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )
