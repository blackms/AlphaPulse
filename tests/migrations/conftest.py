"""
Pytest fixtures for migration tests.

Provides database session fixtures for testing Alembic migrations.
"""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

# Database URL for testing
DATABASE_URL = "postgresql+asyncpg://alphapulse:alphapulse@localhost:5432/alphapulse"


@pytest_asyncio.fixture
async def db_session():
    """
    Provide async database session for migration tests.

    Creates a new async database session for each test.
    Note: Does not auto-rollback as migrations are meant to persist.
    """
    engine = create_async_engine(
        DATABASE_URL,
        poolclass=NullPool,  # No connection pooling for tests
        echo=False
    )

    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    session = async_session_factory()
    yield session

    await session.close()
    await engine.dispose()
