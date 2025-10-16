"""Authentication service."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.exceptions import AuthenticationError
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)

    async def authenticate_user(self, login_data: LoginRequest) -> User:
        """Authenticate user with username and password."""
        user = await self.user_repo.get_by_username(login_data.username)

        if not user:
            raise AuthenticationError("Incorrect username or password")

        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        if not verify_password(login_data.password, user.hashed_password):
            raise AuthenticationError("Incorrect username or password")

        return user

    async def login(self, login_data: LoginRequest) -> TokenResponse:
        """Login user and return JWT token."""
        user = await self.authenticate_user(login_data)

        # Create access token
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "username": user.username,
                "role": user.role,
            }
        )

        return TokenResponse(access_token=access_token)

    async def register(self, register_data: RegisterRequest) -> User:
        """Register a new user."""
        # Check if username exists
        existing_user = await self.user_repo.get_by_username(register_data.username)
        if existing_user:
            raise AuthenticationError("Username already registered")

        # Check if email exists
        existing_email = await self.user_repo.get_by_email(register_data.email)
        if existing_email:
            raise AuthenticationError("Email already registered")

        # Create new user
        user = User(
            username=register_data.username,
            email=register_data.email,
            full_name=register_data.full_name,
            hashed_password=get_password_hash(register_data.password),
            role="user",
            is_active=True,
            is_superuser=False,
        )

        return await self.user_repo.create(user)
