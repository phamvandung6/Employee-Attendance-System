"""Script to initialize database."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.init_db import create_tables


async def main():
    print("Creating database tables...")
    await create_tables()
    print("✅ Database tables created successfully!")


if __name__ == "__main__":
    asyncio.run(main())
