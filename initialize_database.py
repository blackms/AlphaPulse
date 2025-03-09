#!/usr/bin/env python3
"""
Script to initialize the AlphaPulse database tables.

This script creates all the necessary tables for the AlphaPulse data pipeline.
"""
import os
import sys
import asyncio
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("initialize_database.log", rotation="10 MB", level="DEBUG")

# Set environment variables for database connection
os.environ["DB_TYPE"] = "postgres"
os.environ["DB_NAME"] = "alphapulse"
os.environ["DB_USER"] = "testuser"
os.environ["DB_PASS"] = "testpassword"

# Import after setting environment variables
from alpha_pulse.data_pipeline.database.connection import get_pg_connection


async def initialize_tables():
    """Initialize the required database tables."""
    logger.info("Initializing database tables...")
    
    async with get_pg_connection() as conn:
        # Create sync_status table
        logger.info("Creating sync_status table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_status (
                id SERIAL PRIMARY KEY,
                exchange_id TEXT NOT NULL,
                data_type TEXT NOT NULL,
                status TEXT NOT NULL,
                last_sync TIMESTAMP WITH TIME ZONE,
                next_sync TIMESTAMP WITH TIME ZONE,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(exchange_id, data_type)
            )
        """)
        
        # Create balances table
        logger.info("Creating exchange_balances table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_balances (
                id SERIAL PRIMARY KEY,
                exchange_id TEXT NOT NULL,
                currency TEXT NOT NULL,
                available NUMERIC,
                locked NUMERIC,
                total NUMERIC,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(exchange_id, currency)
            )
        """)
        
        # Create positions table
        logger.info("Creating exchange_positions table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_positions (
                id SERIAL PRIMARY KEY,
                exchange_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity NUMERIC,
                entry_price NUMERIC,
                current_price NUMERIC,
                unrealized_pnl NUMERIC,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(exchange_id, symbol)
            )
        """)
        
        # Create orders table
        logger.info("Creating exchange_orders table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_orders (
                id SERIAL PRIMARY KEY,
                exchange_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                order_type TEXT,
                side TEXT,
                price NUMERIC,
                amount NUMERIC,
                filled NUMERIC,
                status TEXT,
                timestamp TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(exchange_id, order_id)
            )
        """)
        
        # Create prices table
        logger.info("Creating exchange_prices table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_prices (
                id SERIAL PRIMARY KEY,
                exchange_id TEXT NOT NULL,
                base_currency TEXT NOT NULL,
                quote_currency TEXT NOT NULL,
                price NUMERIC,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(exchange_id, base_currency, quote_currency)
            )
        """)
    
    logger.info("Database tables initialized successfully!")


async def main():
    """Main function to run the initialization."""
    try:
        await initialize_tables()
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
        return False


if __name__ == "__main__":
    # Run the main function
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)