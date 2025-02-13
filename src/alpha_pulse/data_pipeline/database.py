"""
Database module for AlphaPulse data pipeline.
"""
from contextlib import contextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Create async engine
engine = create_async_engine(
    'sqlite+aiosqlite:///data.db',
    echo=False,
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()