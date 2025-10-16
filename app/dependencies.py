"""FastAPI dependencies."""

from typing import Annotated
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import verify_token, http_bearer
from app.core.exceptions import AuthenticationError
from app.models.user import User
from app.repositories.user_repository import UserRepository


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(http_bearer)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials

    # Verify token
    payload = verify_token(token)
    if not payload:
        raise AuthenticationError("Invalid or expired token")

    # Extract user_id from token
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload")

    # Get user from database
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(int(user_id))

    if not user:
        raise AuthenticationError("User not found")

    if not user.is_active:
        raise AuthenticationError("User account is inactive")

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise AuthenticationError("User account is inactive")
    return current_user


async def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Require admin role."""
    if current_user.role != "admin" and not current_user.is_superuser:
        raise AuthenticationError("Admin privileges required")
    return current_user


async def require_hr_manager(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Require HR manager or admin role."""
    if (
        current_user.role not in ["admin", "hr_manager"]
        and not current_user.is_superuser
    ):
        raise AuthenticationError("HR Manager or Admin privileges required")
    return current_user
