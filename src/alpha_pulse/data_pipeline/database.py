"""
Database connection module for AlphaPulse.
"""
from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from alpha_pulse.config.settings import settings


def create_session_factory():
    """Create a session factory for database connections."""
    engine = create_engine(settings.DATABASE_URL)
    return sessionmaker(bind=engine)


SessionFactory = create_session_factory()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get a database session using context manager.

    Usage:
        with get_db() as db:
            db.query(...)

    Yields:
        Database session
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        session.close()