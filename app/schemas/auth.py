"""Authentication schemas."""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request schema."""

    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6)


class RegisterRequest(BaseModel):
    """User registration request schema."""

    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=1, max_length=255)


class TokenResponse(BaseModel):
    """JWT token response schema."""

    access_token: str
    token_type: str = "Bearer"


class TokenData(BaseModel):
    """Token payload data."""

    user_id: int
    username: str
    role: str
