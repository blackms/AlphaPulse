"""
Database session management for AlphaPulse data pipeline.

This module provides SQLAlchemy AsyncSession for PostgreSQL database connections.
"""
import os
from contextlib import contextmanager, asynccontextmanager # Import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict # Import Optional, AsyncGenerator, and Dict
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine
from loguru import logger

# Store engine globally but initialize lazily
_engine: Optional[AsyncEngine] = None

def get_database_url(config: Optional[Dict] = None) -> str:
    """
    Constructs the database URL from a config dictionary or environment variables.
    """
    # Default PostgreSQL connection parameters (used if not in config or env)
    DEFAULT_DB_HOST = "localhost"
    DEFAULT_DB_PORT = "5432" # Keep as string for os.environ.get
    DEFAULT_DB_NAME = "alphapulse" # Default, overridden by env vars
    DEFAULT_DB_USER = "testuser"   # Default, overridden by env vars
    DEFAULT_DB_PASS = "testpassword"

    if config:
        db_host = config.get("host", DEFAULT_DB_HOST)
        db_port = str(config.get("port", DEFAULT_DB_PORT)) # Ensure port is string for URL
        db_name = config.get("database", DEFAULT_DB_NAME)
        db_user = config.get("user", DEFAULT_DB_USER)
        db_pass = config.get("password", DEFAULT_DB_PASS)
        logger.debug("Using database config from provided dictionary.")
    else:
        # Get connection parameters from environment or use defaults
        db_host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
        db_port = os.environ.get("DB_PORT", DEFAULT_DB_PORT)
        db_name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
        db_user = os.environ.get("DB_USER", DEFAULT_DB_USER)
        db_pass = os.environ.get("DB_PASS", DEFAULT_DB_PASS)
        logger.debug("Using database config from environment variables or defaults.")

    # Create PostgreSQL connection URL
    db_url = f"postgresql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    logger.debug(f"Constructed DB URL: postgresql+asyncpg://{db_user}:***@{db_host}:{db_port}/{db_name}")
    return db_url

def get_engine(config: Optional[Dict] = None) -> AsyncEngine:
    """
    Returns the singleton async engine, creating it if necessary.
    Uses config if provided, otherwise falls back to environment variables.
    """
    global _engine
    if _engine is None:
        db_url = get_database_url(config) # Pass config to get URL
        logger.info(f"Creating async engine for URL: {db_url.split('@')[1]}") # Log without creds
        _engine = create_async_engine(
            db_url, # Use the potentially config-derived URL
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

def async_session(config: Optional[Dict] = None) -> AsyncSession:
    """
    Returns a new async session bound to the engine.
    Uses config if provided to initialize the engine.
    """
    engine = get_engine(config) # Ensure engine is created, potentially using config
    return _async_sessionmaker(bind=engine)

async def get_db(config: Optional[Dict] = None) -> AsyncGenerator[AsyncSession, None]: # Accept config
    """Dependency function to get database session (e.g., for FastAPI)."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Simple context manager for non-FastAPI use cases
@asynccontextmanager
async def get_session_context(config: Optional[Dict] = None) -> AsyncGenerator[AsyncSession, None]: # Accept config
    """
    Provides an async session within a context manager.
    Uses config if provided to initialize the engine.
    """
    session = async_session(config) # Pass config
    try:
        yield session
        await session.commit() # Commit on successful exit
    except Exception:
        await session.rollback() # Rollback on error
        raise
    finally:
        await session.close()