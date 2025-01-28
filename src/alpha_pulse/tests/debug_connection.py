from data_pipeline.database import get_db
from sqlalchemy import text
from loguru import logger

def test_connection():
    logger.info("Starting connection test")
    try:
        with get_db() as db:
            # Try a simple query
            result = db.execute(text("SELECT 1"))
            logger.info(f"Query result: {result.scalar()}")
            logger.info("Connection test successful!")
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_connection() 