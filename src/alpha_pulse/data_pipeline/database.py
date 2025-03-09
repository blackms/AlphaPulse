"""
Database module for AlphaPulse data pipeline.

This module provides SQLAlchemy AsyncSession for PostgreSQL database connections.
"""
import os
from contextlib import contextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Default PostgreSQL connection parameters - sync with connection.py
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "alphapulse"
DEFAULT_DB_USER = "testuser"
DEFAULT_DB_PASS = "testpassword"

# Get connection parameters from environment or use defaults
db_host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
db_port = os.environ.get("DB_PORT", DEFAULT_DB_PORT)
db_name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
db_user = os.environ.get("DB_USER", DEFAULT_DB_USER)
db_pass = os.environ.get("DB_PASS", DEFAULT_DB_PASS)

# Create PostgreSQL connection URL
db_url = f"postgresql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Create async engine
engine = create_async_engine(
    db_url,
    echo=False,
    future=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=300
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