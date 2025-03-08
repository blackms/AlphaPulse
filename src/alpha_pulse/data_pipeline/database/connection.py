"""
Database connection utilities.

This module provides functions to establish connections to the database.
"""
import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import asyncpg
import sqlite3
from loguru import logger

# Default connection parameters
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "alphapulse"  # Changed from alpha_pulse to alphapulse
DEFAULT_DB_USER = "testuser"
DEFAULT_DB_PASS = "testpassword"

# Database type
DB_TYPE = os.environ.get("DB_TYPE", "postgres").lower()
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "alphapulse.db")

# Connection pool
_pg_pool = None


async def _initialize_tables(conn):
    """
    Initialize the required tables if they don't exist.
    
    Args:
        conn: Database connection
    """
    # Create sync_status table
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
    
    logger.info("Database tables initialized")


async def _init_pg_pool():
    """Initialize the PostgreSQL connection pool."""
    global _pg_pool
    
    if _pg_pool is not None:
        return
    
    # Get connection parameters from environment
    db_host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
    db_port = int(os.environ.get("DB_PORT", DEFAULT_DB_PORT))
    db_name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
    db_user = os.environ.get("DB_USER", DEFAULT_DB_USER)
    db_pass = os.environ.get("DB_PASS", DEFAULT_DB_PASS)
    
    try:
        # Create connection pool
        logger.info(f"Creating PostgreSQL connection pool to {db_host}:{db_port}/{db_name}")
        _pg_pool = await asyncpg.create_pool(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_pass,
            min_size=1,
            max_size=10
        )
        
        # Initialize tables
        async with _pg_pool.acquire() as conn:
            await _initialize_tables(conn)
            
        logger.info(f"Successfully connected to PostgreSQL database: {db_name}")
    except asyncpg.InvalidCatalogNameError:
        logger.error(f"Database '{db_name}' does not exist. Please create it first.")
        raise
    except asyncpg.InvalidPasswordError:
        logger.error(f"Authentication failed for user '{db_user}'. Check your credentials.")
        raise
    except asyncpg.CannotConnectNowError:
        logger.error(f"Cannot connect to PostgreSQL server at {db_host}:{db_port}. The server may be busy.")
        raise
    except asyncpg.PostgresConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        logger.error(f"Check if PostgreSQL is running at {db_host}:{db_port} and the network is configured correctly.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing database connection: {str(e)}")
        raise


@asynccontextmanager
async def get_pg_connection():
    """
    Get a PostgreSQL connection from the pool.
    
    Returns:
        Connection from the pool
    """
    global _pg_pool
    
    # Initialize pool if needed
    if _pg_pool is None:
        await _init_pg_pool()
    
    # Get connection from pool
    async with _pg_pool.acquire() as conn:
        yield conn


def _json_default(obj):
    """Default JSON encoder for non-serializable objects."""
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


async def init_db():
    """
    Initialize the database connection and tables.
    
    This function is used during application startup to ensure
    the database is properly initialized.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        if DB_TYPE == "postgres":
            await _init_pg_pool()
            return True
        elif DB_TYPE == "sqlite":
            # For SQLite, implement initialization here
            logger.warning(f"Database initialization for {DB_TYPE} not fully implemented")
            # TODO: Implement SQLite initialization
            return True
        else:
            logger.warning(f"Unknown database type: {DB_TYPE}")
            logger.warning(f"Supported types are: postgres, sqlite")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        logger.error("The application may not function correctly without database access")
        # Return False but don't re-raise the exception to allow the application to start
        # with degraded functionality
        return False