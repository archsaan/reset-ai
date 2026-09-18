"""
Auth dependencies.

This is intentionally the ONLY file that decides whether a request is
allowed through. Today it's a mock token check for the admin dashboard.
When real member auth (JWT/session tokens from the existing member app)
is ready, only this file needs to change — every router already depends
on the functions below rather than checking headers itself.
"""

from fastapi import Header, HTTPException

from app.core.config import MOCK_ADMIN_TOKEN


def require_admin(authorization: str = Header(None)) -> bool:
    """Placeholder admin auth. Swap this body for a real credential/JWT
    check later — nothing else in the app needs to change."""
    if authorization != f"Bearer {MOCK_ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Admin authentication required")
    return True


def require_member(authorization: str = Header(None)) -> str | None:
    """Placeholder for real member auth (not implemented yet).

    Once the existing member app's login is wired in, this should verify
    the token/session and return the member's stable identifier (e.g.
    their email or member ID), raising HTTPException(401) if invalid.

    For now it's a no-op that lets `/chat` keep working with the
    email-in-the-request-body pattern used during the learning phase.
    """
    return None
