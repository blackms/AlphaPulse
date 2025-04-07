"""
Database session management for AlphaPulse data pipeline.

This module provides SQLAlchemy AsyncSession for PostgreSQL database connections.
"""
import os
from contextlib import contextmanager, asynccontextmanager # Import asynccontextmanager
from typing import Optional, AsyncGenerator # Import Optional and AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from loguru import logger

# Store engine globally but initialize lazily
_engine: Optional[AsyncEngine] = None

def get_database_url() -> str:
    """Constructs the database URL from environment variables or defaults."""
    # Default PostgreSQL connection parameters
    DEFAULT_DB_HOST = "localhost"
    DEFAULT_DB_PORT = "5432" # Keep as string for os.environ.get
    DEFAULT_DB_NAME = "alphapulse" # Default, overridden by env vars
    DEFAULT_DB_USER = "testuser"   # Default, overridden by env vars
    DEFAULT_DB_PASS = "testpassword" # Default, overridden by env vars

    # Get connection parameters from environment or use defaults
    db_host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
    db_port = os.environ.get("DB_PORT", DEFAULT_DB_PORT)
    db_name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
    db_user = os.environ.get("DB_USER", DEFAULT_DB_USER)
    db_pass = os.environ.get("DB_PASS", DEFAULT_DB_PASS)

    # Create PostgreSQL connection URL
    db_url = f"postgresql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    logger.debug(f"Constructed DB URL: postgresql+asyncpg://{db_user}:***@{db_host}:{db_port}/{db_name}")
    return db_url

def get_engine() -> AsyncEngine:
    """Returns the singleton async engine, creating it if necessary."""
    global _engine
    if _engine is None:
        db_url = get_database_url()
        logger.info(f"Creating async engine for URL: {db_url.split('@')[1]}") # Log without creds
        _engine = create_async_engine(
            db_url,
            echo=False, # Set to True for debugging SQL
            future=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=300
)
    return _engine

# Create async session factory using the engine getter
# Note: The 'bind' argument expects an Engine instance, not a function.
# We need to ensure get_engine() is called *before* sessionmaker uses it.
# A common pattern is to initialize sessionmaker later or pass the engine explicitly.
# Let's adjust the pattern slightly: create sessionmaker without bind initially,
# and bind it when a session is requested or use the engine directly.

_async_sessionmaker = sessionmaker(
    class_=AsyncSession,
    expire_on_commit=False
)

def async_session() -> AsyncSession:
    """Returns a new async session bound to the engine."""
    engine = get_engine() # Ensure engine is created
    return _async_sessionmaker(bind=engine)

async def get_db() -> AsyncSession:
    """Dependency function to get database session (e.g., for FastAPI)."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Simple context manager for non-FastAPI use cases
@asynccontextmanager # Use async context manager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]: # Use AsyncGenerator
    """Provides an async session within a context manager."""
    session = async_session()
    try:
        yield session
        await session.commit() # Commit on successful exit
    except Exception:
        await session.rollback() # Rollback on error
        raise
    finally:
        await session.close()