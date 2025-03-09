"""
Enhanced database connection manager.

This module provides improved connection pooling and transaction management
specifically designed to handle multiple async event loops and threads.
"""
import asyncio
import os
import threading
import random
import time
from typing import Dict, Optional, Any, Callable, Coroutine
from contextlib import asynccontextmanager

import asyncpg
from loguru import logger

# Import the default connection parameters from the existing connection module
from alpha_pulse.data_pipeline.database.connection import (
    DEFAULT_DB_HOST,
    DEFAULT_DB_PORT,
    DEFAULT_DB_NAME,
    DEFAULT_DB_USER,
    DEFAULT_DB_PASS,
    _initialize_tables
)

# Maximum retry attempts for database operations
MAX_RETRY_ATTEMPTS = 3
# Base delay for exponential backoff (in seconds)
BASE_RETRY_DELAY = 0.5

# Store pools by event loop ID and thread ID
_connection_pools: Dict[str, asyncpg.Pool] = {}
_pool_creation_lock = threading.Lock()

def get_loop_thread_key() -> str:
    """
    Generate a unique key based on the current event loop and thread.
    
    This ensures proper isolation of connection pools between different
    event loops and threads.
    
    Returns:
        Unique key combining loop and thread IDs
    """
    thread_id = threading.get_ident()
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        # No event loop running
        loop_id = "no_loop"
    
    return f"{thread_id}_{loop_id}"


async def get_connection_pool() -> asyncpg.Pool:
    """
    Get or create a connection pool for the current event loop and thread.
    
    This function ensures each combination of event loop and thread
    has its own isolated connection pool.
    
    Returns:
        Connection pool for the current context
    """
    loop_thread_key = get_loop_thread_key()
    
    # Check if we already have a pool for this combination
    if loop_thread_key in _connection_pools:
        pool = _connection_pools[loop_thread_key]
        if not pool.is_closed():
            logger.debug(f"Using existing connection pool for {loop_thread_key}")
            return pool
        else:
            logger.debug(f"Existing pool for {loop_thread_key} is closed, creating new one")
    
    # Create a new pool
    with _pool_creation_lock:
        # Double-check in case another thread created the pool while waiting
        if loop_thread_key in _connection_pools and not _connection_pools[loop_thread_key].is_closed():
            return _connection_pools[loop_thread_key]
        
        # Get connection parameters from environment
        db_host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
        db_port = int(os.environ.get("DB_PORT", DEFAULT_DB_PORT))
        db_name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
        db_user = os.environ.get("DB_USER", DEFAULT_DB_USER)
        db_pass = os.environ.get("DB_PASS", DEFAULT_DB_PASS)
        
        # Create connection pool with improved settings
        logger.info(f"Creating PostgreSQL connection pool for {loop_thread_key} to {db_host}:{db_port}/{db_name}")
        
        # Use exponential backoff for connection attempts
        retry_count = 0
        last_error = None
        
        while retry_count < MAX_RETRY_ATTEMPTS:
            try:
                # Create the pool with optimized settings
                pool = await asyncpg.create_pool(
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_pass,
                    min_size=3,           # Ensure we have enough connections ready
                    max_size=20,          # Handle more concurrent operations
                    command_timeout=60.0, # Set command timeout to 60 seconds
                    max_inactive_connection_lifetime=300.0,  # 5 minutes max idle time
                    # Use server-side statement cache to improve performance
                    server_settings={'statement_timeout': '30s',  # 30 second statement timeout
                                    'idle_in_transaction_session_timeout': '60s'}  # Prevent hung transactions
                )
                
                # Initialize tables using the first connection from the pool
                async with pool.acquire() as conn:
                    await _initialize_tables(conn)
                
                # Store the pool
                _connection_pools[loop_thread_key] = pool
                logger.info(f"Successfully created connection pool for {loop_thread_key}")
                return pool
            
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                last_error = e
                retry_count += 1
                if retry_count < MAX_RETRY_ATTEMPTS:
                    # Calculate backoff time with jitter
                    backoff = BASE_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"Connection attempt {retry_count} failed: {str(e)}. Retrying in {backoff:.2f} seconds...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Failed to connect after {MAX_RETRY_ATTEMPTS} attempts: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error creating connection pool: {str(e)}")
                raise
        
        # If we've exhausted retries, raise the last error
        if last_error is not None:
            raise last_error
        
        # This should never happen, but just in case
        raise RuntimeError("Unexpected error in get_connection_pool")


async def close_pool(loop_thread_key: Optional[str] = None) -> None:
    """
    Close a specific connection pool or the current one.
    
    Args:
        loop_thread_key: Key of the pool to close, or None for current context
    """
    # If no key provided, use the current context
    if loop_thread_key is None:
        loop_thread_key = get_loop_thread_key()
    
    if loop_thread_key in _connection_pools:
        logger.info(f"Closing connection pool for {loop_thread_key}")
        try:
            await _connection_pools[loop_thread_key].close()
            del _connection_pools[loop_thread_key]
            logger.info(f"Connection pool for {loop_thread_key} closed and removed")
        except Exception as e:
            logger.error(f"Error closing connection pool for {loop_thread_key}: {str(e)}")


async def close_all_pools() -> None:
    """
    Close all connection pools.
    """
    logger.info(f"Closing all connection pools ({len(_connection_pools)} pools)")
    
    for key, pool in list(_connection_pools.items()):
        try:
            logger.debug(f"Closing connection pool for {key}")
            await pool.close()
            logger.debug(f"Connection pool for {key} closed")
        except Exception as e:
            logger.error(f"Error closing connection pool for {key}: {str(e)}")
    
    # Clear the pools dictionary
    _connection_pools.clear()
    logger.info("All connection pools closed")


async def execute_with_retry(operation: Callable[[], Coroutine], max_retries: int = MAX_RETRY_ATTEMPTS) -> Any:
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
    retry_count = 0
    last_error = None
    loop_thread_key = get_loop_thread_key()
    
    while retry_count < max_retries:
        try:
            return await operation()
        except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as e:
            last_error = e
            retry_count += 1
            
            error_msg = str(e)
            if "another operation is in progress" in error_msg or "connection was closed" in error_msg:
                if retry_count < max_retries:
                    # Calculate backoff time with jitter
                    backoff = BASE_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[{loop_thread_key}] Database operation failed (attempt {retry_count}): {error_msg}. Retrying in {backoff:.2f} seconds...")
                    
                    # Reset the connection pool for this context to force new connections
                    if loop_thread_key in _connection_pools:
                        try:
                            await _connection_pools[loop_thread_key].close()
                            del _connection_pools[loop_thread_key]
                            logger.info(f"[{loop_thread_key}] Reset connection pool due to connection error")
                        except Exception as close_error:
                            logger.warning(f"[{loop_thread_key}] Error closing pool: {str(close_error)}")
                    
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Database operation failed after {max_retries} attempts: {error_msg}")
                    raise
            else:
                # For other interface errors, just re-raise
                raise
        except Exception as e:
            # For other exceptions, don't retry
            logger.error(f"[{loop_thread_key}] Database operation error: {str(e)}")
            raise
    
    # If we've exhausted retries, raise the last error
    if last_error is not None:
        raise last_error
    
    # This should never happen, but just in case
    raise RuntimeError("Unexpected error in execute_with_retry")


@asynccontextmanager
async def get_db_connection():
    """
    Get a database connection with transaction management.
    
    This context manager handles:
    1. Getting a connection from the pool
    2. Creating and managing a transaction
    3. Error handling and cleanup
    4. Returning the connection to the pool
    
    Yields:
        Connection from the pool with active transaction
    """
    loop_thread_key = get_loop_thread_key()
    logger.debug(f"[{loop_thread_key}] Getting database connection")
    
    # Get the connection pool for this context
    pool = await get_connection_pool()
    conn = None
    tr = None
    
    try:
        # Define the operation to acquire a connection
        async def acquire_connection():
            return await pool.acquire()
        
        # Acquire a connection with retry logic
        conn = await execute_with_retry(acquire_connection)
        
        # Start a transaction
        tr = conn.transaction()
        await tr.start()
        logger.debug(f"[{loop_thread_key}] Started transaction")
        
        try:
            # Yield the connection
            yield conn
            
            # If we get here, commit the transaction
            await tr.commit()
            logger.debug(f"[{loop_thread_key}] Transaction committed")
        except Exception as e:
            # If an exception occurs, rollback the transaction
            if tr and not tr._done:  # Check if transaction is still active
                try:
                    await tr.rollback()
                    logger.warning(f"[{loop_thread_key}] Transaction rolled back due to error: {str(e)}")
                except Exception as rollback_error:
                    logger.error(f"[{loop_thread_key}] Error rolling back transaction: {str(rollback_error)}")
            
            # Re-raise the original exception
            raise
    finally:
        # Always release the connection back to the pool
        if conn:
            try:
                if tr and not tr._done:  # If transaction wasn't committed or rolled back
                    try:
                        await tr.rollback()
                        logger.warning(f"[{loop_thread_key}] Transaction rolled back in finally block")
                    except Exception as final_rollback_error:
                        logger.error(f"[{loop_thread_key}] Error in final transaction rollback: {str(final_rollback_error)}")
                
                await pool.release(conn)
                logger.debug(f"[{loop_thread_key}] Released database connection back to pool")
            except Exception as release_error:
                logger.error(f"[{loop_thread_key}] Error releasing connection: {str(release_error)}")
                
                # If we can't release, the pool may be in a bad state
                # Try to close and recreate it
                try:
                    await close_pool(loop_thread_key)
                    logger.info(f"[{loop_thread_key}] Closed bad connection pool")
                except Exception as close_error:
                    logger.error(f"[{loop_thread_key}] Error closing bad pool: {str(close_error)}")