"""Database initialization utilities."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base
from app.db.session import engine


async def create_tables():
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Drop all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def init_db(db: AsyncSession) -> None:
    """Initialize database with default data."""
    # Create tables
    await create_tables()

    # Add seed data here if needed
    pass
