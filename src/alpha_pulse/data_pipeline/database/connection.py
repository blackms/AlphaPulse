"""
Database connection utilities.

This module provides functions to establish connections to the database.
"""
import os
import json
import asyncio
import threading
import time
import random
from typing import Optional, Dict, Any, List, Tuple
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

# Thread-local storage for connection pools
_thread_local = threading.local()

# Lock for pool creation
_pool_creation_lock = threading.Lock()

# Maximum retry attempts for database operations
MAX_RETRY_ATTEMPTS = 3
# Base delay for exponential backoff (in seconds)
BASE_RETRY_DELAY = 0.5


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


async def _get_thread_pg_pool() -> asyncpg.Pool:
    """
    Get or create a PostgreSQL connection pool for the current thread.
    
    Returns:
        The thread-specific connection pool
    """
    thread_id = threading.get_ident()
    
    # Check if we already have a pool for this thread
    if not hasattr(_thread_local, 'pg_pool'):
        logger.debug(f"[THREAD {thread_id}] Creating new thread-local connection pool")
        _thread_local.pg_pool = None
    
    # If the pool doesn't exist or is closed, create a new one
    if _thread_local.pg_pool is None:
        with _pool_creation_lock:
            # Double-check in case another thread created the pool while we were waiting
            if _thread_local.pg_pool is None:
                # Get connection parameters from environment
                db_host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
                db_port = int(os.environ.get("DB_PORT", DEFAULT_DB_PORT))
                db_name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
                db_user = os.environ.get("DB_USER", DEFAULT_DB_USER)
                db_pass = os.environ.get("DB_PASS", DEFAULT_DB_PASS)
                
                # Create connection pool with improved settings
                logger.info(f"[THREAD {thread_id}] Creating PostgreSQL connection pool to {db_host}:{db_port}/{db_name}")
                
                # Use exponential backoff for connection attempts
                retry_count = 0
                last_error = None
                
                while retry_count < MAX_RETRY_ATTEMPTS:
                    try:
                        # Create the pool with improved settings
                        _thread_local.pg_pool = await asyncpg.create_pool(
                            host=db_host,
                            port=db_port,
                            database=db_name,
                            user=db_user,
                            password=db_pass,
                            min_size=2,           # Ensure we have at least 2 connections
                            max_size=20,          # Increased from 10 to handle more concurrent operations
                            command_timeout=60.0, # Set command timeout to 60 seconds
                            max_inactive_connection_lifetime=300.0,  # 5 minutes max idle time
                            max_queries=50000,    # Maximum queries per connection
                            setup=lambda conn: conn.execute('SET statement_timeout = 30000;')  # 30 second statement timeout
                        )
                        
                        # Initialize tables
                        async with _thread_local.pg_pool.acquire() as conn:
                            await _initialize_tables(conn)
                            
                        logger.info(f"[THREAD {thread_id}] Successfully connected to PostgreSQL database: {db_name}")
                        break
                    except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                        last_error = e
                        retry_count += 1
                        if retry_count < MAX_RETRY_ATTEMPTS:
                            # Calculate backoff time with jitter
                            backoff = BASE_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                            logger.warning(f"[THREAD {thread_id}] Connection attempt {retry_count} failed: {str(e)}. Retrying in {backoff:.2f} seconds...")
                            await asyncio.sleep(backoff)
                        else:
                            logger.error(f"[THREAD {thread_id}] Failed to connect after {MAX_RETRY_ATTEMPTS} attempts: {str(e)}")
                            raise
                    except Exception as e:
                        logger.error(f"[THREAD {thread_id}] Unexpected error creating connection pool: {str(e)}")
                        raise
                
                # If we've exhausted retries, raise the last error
                if _thread_local.pg_pool is None and last_error is not None:
                    raise last_error
    
    return _thread_local.pg_pool


async def _execute_with_retry(operation, max_retries=MAX_RETRY_ATTEMPTS):
    """
    Execute a database operation with retry logic.
    
    Args:
        operation: Async function that performs the database operation
        max_retries: Maximum number of retry attempts
        
    Returns:
        The result of the operation
        
    Raises:
        Exception: If all retry attempts fail
    """
    thread_id = threading.get_ident()
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            return await operation()
        except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as e:
            # These errors indicate connection issues
            last_error = e
            retry_count += 1
            
            error_msg = str(e)
            if "another operation is in progress" in error_msg or "connection was closed" in error_msg:
                if retry_count < max_retries:
                    # Calculate backoff time with jitter
                    backoff = BASE_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[THREAD {thread_id}] Database operation failed (attempt {retry_count}): {error_msg}. Retrying in {backoff:.2f} seconds...")
                    
                    # Reset the thread's connection pool to force new connections
                    if hasattr(_thread_local, 'pg_pool') and _thread_local.pg_pool is not None:
                        try:
                            await _thread_local.pg_pool.close()
                        except Exception as close_error:
                            logger.warning(f"[THREAD {thread_id}] Error closing pool: {str(close_error)}")
                        _thread_local.pg_pool = None
                        logger.info(f"[THREAD {thread_id}] Reset connection pool due to connection error")
                    
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[THREAD {thread_id}] Database operation failed after {max_retries} attempts: {error_msg}")
                    raise
            else:
                # For other interface errors, just re-raise
                raise
        except Exception as e:
            # For other exceptions, don't retry
            logger.error(f"[THREAD {thread_id}] Database operation error: {str(e)}")
            raise
    
    # If we've exhausted retries, raise the last error
    if last_error is not None:
        raise last_error
    
    # This should never happen, but just in case
    raise RuntimeError("Unexpected error in execute_with_retry")


@asynccontextmanager
async def get_pg_connection():
    """
    Get a PostgreSQL connection from the thread-local pool with retry logic.
    
    Returns:
        Connection from the pool
    """
    thread_id = threading.get_ident()
    logger.debug(f"[THREAD {thread_id}] Getting database connection")
    
    # Get the thread-local pool
    pool = await _get_thread_pg_pool()
    
    # Define the operation to acquire a connection
    async def acquire_connection():
        return await pool.acquire()
    
    # Acquire a connection with retry logic
    conn = await _execute_with_retry(acquire_connection)
    
    try:
        # Start a transaction
        tr = conn.transaction()
        await tr.start()
        
        try:
            # Yield the connection
            yield conn
            
            # If we get here, commit the transaction
            await tr.commit()
        except Exception as e:
            # If an exception occurs, rollback the transaction
            try:
                await tr.rollback()
                logger.warning(f"[THREAD {thread_id}] Transaction rolled back due to error: {str(e)}")
            except Exception as rollback_error:
                logger.error(f"[THREAD {thread_id}] Error rolling back transaction: {str(rollback_error)}")
            
            # Re-raise the original exception
            raise
    finally:
        # Always release the connection back to the pool
        try:
            await pool.release(conn)
            logger.debug(f"[THREAD {thread_id}] Released database connection back to pool")
        except Exception as release_error:
            logger.error(f"[THREAD {thread_id}] Error releasing connection: {str(release_error)}")


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
    thread_id = threading.get_ident()
    logger.info(f"[THREAD {thread_id}] Initializing database")
    
    try:
        if DB_TYPE == "postgres":
            # Initialize the thread-local pool
            await _get_thread_pg_pool()
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


async def close_all_connections():
    """
    Close all database connections.
    
    This function should be called during application shutdown to ensure
    all database connections are properly closed.
    """
    thread_id = threading.get_ident()
    
    if hasattr(_thread_local, 'pg_pool') and _thread_local.pg_pool is not None:
        logger.info(f"[THREAD {thread_id}] Closing database connection pool")
        try:
            await _thread_local.pg_pool.close()
            _thread_local.pg_pool = None
            logger.info(f"[THREAD {thread_id}] Database connection pool closed")
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error closing database connection pool: {str(e)}")