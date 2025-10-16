"""Authentication endpoints."""

from typing import Annotated
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.schemas.user import UserResponse
from app.services.auth_service import AuthService
from app.dependencies import get_current_user
from app.models.user import User

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Login user with username and password.

    Returns JWT access token.
    """
    auth_service = AuthService(db)
    return await auth_service.login(login_data)


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    register_data: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user.

    Creates a new user account with the provided information.
    """
    auth_service = AuthService(db)
    user = await auth_service.register(register_data)
    return user


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Get current authenticated user information.

    Requires valid JWT token in Authorization header.
    """
    return current_user


@router.post("/logout")
async def logout():
    """
    Logout user.

    Note: With JWT, logout is typically handled client-side by removing the token.
    This endpoint is provided for completeness.
    """
    return {"message": "Successfully logged out"}
