from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator
from loguru import logger

from config.settings import settings

# Add debug logging
logger.info(f"Attempting to connect to database with URL: {settings.DATABASE_URL}")

# Enable SQLAlchemy logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

engine = create_engine(
    settings.DATABASE_URL,
    echo=True  # This will log all SQL statements
)

# Add connection debugging
@event.listens_for(engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    logger.info("New database connection established")
    logger.info(f"Connected with parameters: {dbapi_connection.get_dsn_parameters()}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator:
    db = SessionLocal()
    try:
        logger.info("Database session created")
        yield db
    except Exception as e:
        logger.error(f"Database error occurred: {str(e)}")
        raise
    finally:
        logger.info("Closing database session")
        db.close() 