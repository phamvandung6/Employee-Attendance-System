"""Script to seed initial data."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.core.security import get_password_hash
from sqlalchemy import select


async def seed_admin_user():
    """Create default admin user if not exists."""
    async with AsyncSessionLocal() as db:
        # Check if admin exists
        result = await db.execute(select(User).where(User.username == "admin"))
        existing_admin = result.scalar_one_or_none()

        if existing_admin:
            print("⚠️  Admin user already exists")
            return

        # Create admin user
        admin = User(
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            hashed_password=get_password_hash("admin123"),
            role="admin",
            is_active=True,
            is_superuser=True,
        )

        db.add(admin)
        await db.commit()

        print("✅ Admin user created successfully!")
        print("   Username: admin")
        print("   Password: admin123")
        print("   ⚠️  Please change the password in production!")


async def seed_demo_users():
    """Create demo users for testing."""
    async with AsyncSessionLocal() as db:
        demo_users = [
            {
                "username": "hr_manager",
                "email": "hr@example.com",
                "full_name": "HR Manager",
                "password": "hr123456",
                "role": "hr_manager",
            },
            {
                "username": "user1",
                "email": "user1@example.com",
                "full_name": "Demo User",
                "password": "user123456",
                "role": "user",
            },
        ]

        for user_data in demo_users:
            # Check if user exists
            result = await db.execute(
                select(User).where(User.username == user_data["username"])
            )
            existing_user = result.scalar_one_or_none()

            if existing_user:
                print(f"⚠️  User '{user_data['username']}' already exists")
                continue

            # Create user
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                full_name=user_data["full_name"],
                hashed_password=get_password_hash(user_data["password"]),
                role=user_data["role"],
                is_active=True,
                is_superuser=False,
            )

            db.add(user)
            await db.commit()

            print(
                f"✅ User '{user_data['username']}' created (password: {user_data['password']})"
            )


async def main():
    print("🌱 Seeding database...")

    # Seed admin user
    await seed_admin_user()

    # Seed demo users
    await seed_demo_users()

    print("\n✅ Database seeding completed!")


if __name__ == "__main__":
    asyncio.run(main())
