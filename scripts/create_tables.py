#!/usr/bin/env python3
"""
Script to create database tables for AlphaPulse.

This script creates all the necessary tables for the AlphaPulse data pipeline
using SQLAlchemy.
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Import the models
sys.path.insert(0, os.path.abspath("migrations"))
from migrations.models import Base

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("create_tables.log", rotation="10 MB", level="DEBUG")

def create_tables():
    """Create all database tables."""
    # Get database connection info from environment variables
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "alphapulse")
    db_user = os.environ.get("DB_USER", "testuser")
    db_pass = os.environ.get("DB_PASS", "testpassword")
    
    # Create database URL
    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    try:
        # Create engine
        logger.info(f"Connecting to database: {db_host}:{db_port}/{db_name}")
        engine = create_engine(db_url)
        
        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        
        # Create a session to verify tables
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get all table names
        table_names = engine.table_names()
        logger.info(f"Created tables: {', '.join(table_names)}")
        
        # Close session
        session.close()
        
        logger.info("Database tables created successfully!")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    # Set environment variables if not already set
    if "DB_TYPE" not in os.environ:
        os.environ["DB_TYPE"] = "postgres"
    
    # Create tables
    success = create_tables()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)