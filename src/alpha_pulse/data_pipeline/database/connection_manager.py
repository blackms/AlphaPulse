"""
PostgreSQL connection manager.

This module provides robust connection pooling and transaction management
specifically designed to handle multiple async event loops and threads with
PostgreSQL databases.
"""
import asyncio
import os
import threading
import random
import time
from typing import Dict, Optional, Any, Callable, Coroutine
from contextlib import asynccontextmanager

import asyncpg
from asyncpg.pool import Pool
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

# Store PostgreSQL pools by event loop ID and thread ID
_connection_pools: Dict[str, Pool] = {}
_pool_creation_locks: Dict[str, asyncio.Lock] = {}
_active_connections: Dict[str, int] = {}  # Track active connections per pool
_global_lock = threading.RLock()  # Global lock for thread-safe dictionary access
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

def is_pool_closed(pool: Optional[Pool]) -> bool:
    """
    Safely check if a pool is closed.
    
    This function handles the case where the pool might not have an is_closed method.
    
    Args:
        pool: The connection pool to check
        
    Returns:
        True if the pool is closed or None, False otherwise
    """
    if pool is None:
        return True
    
    try:
        return pool.is_closed()
    except (AttributeError, Exception):
        # If the pool doesn't have an is_closed method or any other error occurs,
        # assume it's in an unusable state (effectively closed)
        return True

async def get_connection_pool() -> Pool:
    """
    Get or create a connection pool for the current event loop and thread.
    
    This function ensures each combination of event loop and thread
    has its own isolated connection pool.
    
    Returns:
        Connection pool for the current context
    """
    loop_thread_key = get_loop_thread_key()
    
    # Thread-safe check of the connection pools dictionary
    with _global_lock:
        # Check if we already have a pool for this combination
        if loop_thread_key in _connection_pools:
            pool = _connection_pools[loop_thread_key]
            if not is_pool_closed(pool):
                _active_connections.setdefault(loop_thread_key, 0)  # Initialize counter if not exists
                logger.debug(f"Using existing connection pool for {loop_thread_key}")
                return pool
            else:
                logger.debug(f"Existing pool for {loop_thread_key} is closed, creating new one")
    
        # Get or create a lock for this specific loop/thread combination
        if loop_thread_key not in _pool_creation_locks:
            _pool_creation_locks[loop_thread_key] = asyncio.Lock()
    
    # Get the lock instance first
    lock = _pool_creation_locks[loop_thread_key]
    
    # Use an async lock to prevent multiple concurrent creations of the same pool
    async with lock:
        # Double-check in case another task created the pool while waiting
        if loop_thread_key in _connection_pools and not is_pool_closed(_connection_pools[loop_thread_key]):
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
                # Create the pool with enhanced PostgreSQL-specific optimized settings
                pool = await asyncpg.create_pool(
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_pass,
                    min_size=5,           # Increased from 3 for better performance
                    max_size=30,          # Increased from 20 for higher concurrency
                    command_timeout=60.0, # Set command timeout to 60 seconds
                    max_inactive_connection_lifetime=300.0,  # 5 minutes max idle time
                    # Enhanced server-side PostgreSQL-specific optimizations
                    server_settings={
                        'statement_timeout': '30s',          # 30 second statement timeout
                        'idle_in_transaction_session_timeout': '60s',  # Prevent hung transactions
                        'application_name': 'AlphaPulse',    # Identify app in pg_stat_activity
                        'client_min_messages': 'notice'      # Reduce log noise
                    }
                )
                
                # Initialize tables using the first connection from the pool
                async with pool.acquire() as conn:
                    await _initialize_tables(conn)
                
                # Store the pool
                _connection_pools[loop_thread_key] = pool
                _active_connections[loop_thread_key] = 0  # Initialize connection counter
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
    
    pool_to_close = None
    with _global_lock:
        # Check if there are active connections for this pool
        active_count = _active_connections.get(loop_thread_key, 0)
        if active_count > 0:
            logger.warning(f"Cannot close pool for {loop_thread_key} - {active_count} active connections")
            # Don't close the pool if there are active connections
            # This prevents the "connection released back to pool" errors
            return
            
        elif loop_thread_key in _connection_pools:
            logger.info(f"Closing connection pool for {loop_thread_key}")
            pool_to_close = _connection_pools[loop_thread_key]
            # Don't set to None yet - wait until we've actually closed the pool
    
    if pool_to_close:
        try:
            await pool_to_close.close()
            with _global_lock:
                if loop_thread_key in _connection_pools:
                    del _connection_pools[loop_thread_key]  
                    # Also clean up the active connections counter
                    if loop_thread_key in _active_connections:
                        del _active_connections[loop_thread_key]
                    # Only remove from registry after successful close
            logger.info(f"Connection pool for {loop_thread_key} closed and removed from registry")
        except Exception as e:
            logger.error(f"Error closing connection pool for {loop_thread_key}: {str(e)}")


async def close_all_pools() -> None:
    """
    Close all connection pools.
    """
    logger.info(f"Closing all connection pools ({len(_connection_pools)} pools)")
    
    # Use a thread-safe copy of the pools dictionary
    pools_to_close = {}
    with _global_lock:
        pools_to_close = dict(_connection_pools)
    
    for key, pool in pools_to_close.items():
        try:
            logger.debug(f"Closing connection pool for {key}")
            if pool is not None and not is_pool_closed(pool):
                await pool.close()
            else:
                logger.debug(f"Pool for {key} is already None or closed")
            logger.debug(f"Connection pool for {key} closed")
        except Exception as e:
            logger.error(f"Error closing connection pool for {key}: {str(e)}")
    
    # Clear the pools and locks dictionaries in a thread-safe way
    with _global_lock:
        _connection_pools.clear()
        _active_connections.clear()
        _pool_creation_locks.clear()
    
    logger.info("All connection pools closed")


async def execute_with_retry(operation: Callable[[], Coroutine], max_retries: int = MAX_RETRY_ATTEMPTS) -> Any:
    """
    Execute a database operation with retry logic.
    
    This function handles database operation retries with proper error handling for:
    - InterfaceError (especially "another operation is in progress")
    - ConnectionDoesNotExistError (connection closed during operation)
    - Event loop related errors
    
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
    
    # Verify we're in a valid event loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError("execute_with_retry must be called from an async context with an active event loop")
    
    while retry_count < max_retries:
        try:
            # Execute the operation
            return await operation()
            
        except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError,
                asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError,
                asyncpg.TooManyConnectionsError) as e:
            last_error = e
            retry_count += 1
            
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Check for PostgreSQL-specific error code
            error_code = getattr(e, 'sqlstate', None)
            if error_code:
                logger.warning(f"[{loop_thread_key}] PostgreSQL error code: {error_code}")
                
                # Handle specific PostgreSQL error conditions
                if error_code == '57P01':  # admin_shutdown
                    logger.warning(f"[{loop_thread_key}] Database server is shutting down, retrying...")
                elif error_code == '53300':  # too_many_connections
                    logger.warning(f"[{loop_thread_key}] Too many database connections, waiting for availability...")
                elif error_code in ('08006', '08001', '08004'):  # connection errors
                    logger.warning(f"[{loop_thread_key}] Connection problem detected: {error_msg}")
            
            # Handle specific database connection errors (for backward compatibility)
            if ("another operation is in progress" in error_msg or
                "connection was closed" in error_msg or
                "connection is closed" in error_msg or
                "canceling statement due to statement timeout" in error_msg):
                
                if retry_count < max_retries:
                    # Calculate backoff time with jitter
                    backoff = BASE_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[{loop_thread_key}] Database operation failed (attempt {retry_count}): {error_type}: {error_msg}. Retrying in {backoff:.2f} seconds...")
                    
                    # For the last retry attempt, reset the connection pool as a more aggressive recovery action
                    if retry_count == max_retries - 1 and loop_thread_key in _connection_pools:
                        try:
                            # Only close the pool if it's not already closed
                            if not is_pool_closed(_connection_pools[loop_thread_key]):
                                await _connection_pools[loop_thread_key].close()
                                del _connection_pools[loop_thread_key]
                                logger.info(f"[{loop_thread_key}] Reset connection pool due to persistent connection error")
                        except Exception as close_error:
                            logger.warning(f"[{loop_thread_key}] Error closing pool: {str(close_error)}")
                    
                    # Wait before retry
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Database operation failed after {max_retries} attempts: {error_type}: {error_msg}")
                    raise
            else:
                # For other interface errors, log and re-raise
                logger.error(f"[{loop_thread_key}] Unhandled {error_type}: {error_msg}")
                raise
                
        except RuntimeError as e:
            # Handle event loop related errors
            error_msg = str(e)
            if "got Future" in error_msg and "attached to a different loop" in error_msg:
                last_error = e
                retry_count += 1
                
                if retry_count < max_retries:
                    backoff = BASE_RETRY_DELAY * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[{loop_thread_key}] Event loop error (attempt {retry_count}): {error_msg}. Retrying in {backoff:.2f} seconds...")
                    
                    # For event loop issues, we need to use time.sleep instead of asyncio.sleep
                    # as asyncio.sleep itself might fail with event loop issues
                    time.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Operation failed after {max_retries} attempts due to event loop issues")
                    raise
            else:
                # Other runtime errors
                logger.error(f"[{loop_thread_key}] Runtime error: {error_msg}")
                raise
                
        except Exception as e:
            # For other exceptions, don't retry
            logger.error(f"[{loop_thread_key}] Database operation error: {type(e).__name__}: {str(e)}")
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
    # Get a unique key for the current event loop and thread
    loop_thread_key = get_loop_thread_key()
    # Verify we're in a valid event loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError("get_db_connection must be called from an async context with an active event loop")
        
    logger.debug(f"[{loop_thread_key}] Getting database connection")
    
    # Get the connection pool for this context
    pool = None
    conn = None
    tr = None
    connection_error = False
    
    try:
        # Get pool with retry logic but limit pool creation attempts
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                pool = await get_connection_pool()
                break
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS:
                    backoff = BASE_RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[{loop_thread_key}] Pool retrieval attempt {attempt} failed: {str(e)}. Retrying in {backoff:.2f}s...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Failed to get connection pool after {MAX_RETRY_ATTEMPTS} attempts")
                    raise
        
        # Define the operation to acquire a connection
        async def acquire_connection():
            conn = await pool.acquire()
            # Set the correct isolation level to prevent concurrent operation errors
            await conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            return conn
        
        # Acquire a connection with retry logic - carefully handle connection acquisition
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                if attempt > 1 and is_pool_closed(pool):
                    # Only recreate the pool if it's closed
                    logger.warning(f"[{loop_thread_key}] Pool is closed, recreating for retry attempt {attempt}")
                    pool = await get_connection_pool()
                
                # Get a connection from the pool
                conn = await pool.acquire()
                # Increment active connection counter
                with _global_lock:
                    _active_connections.setdefault(loop_thread_key, 0)
                    _active_connections[loop_thread_key] += 1
                break
            except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as e:
                if attempt < MAX_RETRY_ATTEMPTS:
                    backoff = BASE_RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[{loop_thread_key}] Connection acquisition attempt {attempt} failed: {str(e)}. Retrying in {backoff:.2f}s...")
                    
                    # Only close the pool if we get a connection error
                    if not is_pool_closed(pool):
                        try:
                            # Only close the pool as a last resort
                            if attempt == MAX_RETRY_ATTEMPTS - 1:
                                await close_pool(loop_thread_key)
                        except Exception as close_err:
                            logger.warning(f"[{loop_thread_key}] Error closing pool: {str(close_err)}")
                    
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Failed to acquire connection after {MAX_RETRY_ATTEMPTS} attempts")
                    connection_error = True
                    raise
            except Exception as e:
                logger.error(f"[{loop_thread_key}] Unexpected error acquiring connection: {str(e)}")
                connection_error = True
                raise
        
        # Start a transaction with retry logic and proper error handling
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                # Check if connection is still valid before starting transaction
                if conn.is_closed():
                    raise asyncpg.ConnectionDoesNotExistError("Connection closed before transaction start")
                
                # Set appropriate transaction isolation level to prevent concurrent operation errors
                await conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                
                # Create and start the transaction
                tr = conn.transaction()
                await tr.start()
                logger.debug(f"[{loop_thread_key}] Started transaction (attempt {attempt})")
                break
            except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as e:
                if attempt < MAX_RETRY_ATTEMPTS:
                    backoff = BASE_RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(f"[{loop_thread_key}] Transaction start attempt {attempt} failed: {str(e)}. Retrying in {backoff:.2f}s...")
                    
                    # Release the bad connection and get a new one
                    try:
                        if not conn.is_closed():
                            await pool.release(conn)
                    except Exception:
                        pass  # Ignore errors on release of bad connection
                    
                    await asyncio.sleep(backoff)
                    
                    # Get a new connection without closing the entire pool
                    try:
                        conn = await pool.acquire()
                    except Exception as acquire_error:
                        logger.warning(f"[{loop_thread_key}] Failed to acquire new connection: {str(acquire_error)}")
                        # Only as a last resort, reset the pool
                        if attempt == MAX_RETRY_ATTEMPTS - 1:
                            await close_pool(loop_thread_key)
                            pool = await get_connection_pool()
                            conn = await pool.acquire()
                else:
                    logger.error(f"[{loop_thread_key}] Failed to start transaction after {MAX_RETRY_ATTEMPTS} attempts")
                    connection_error = True
                    raise
            except Exception as e:
                logger.error(f"[{loop_thread_key}] Unexpected error starting transaction: {str(e)}")
                connection_error = True
                raise
        
        try:
            # Yield the connection
            yield conn
            
            # If we get here, commit the transaction
            if tr and not getattr(tr, '_done', True):  # Check if transaction is still active
                try:
                    if not conn.is_closed():  # Check if connection is still valid
                        await tr.commit()
                        logger.debug(f"[{loop_thread_key}] Transaction committed successfully")
                    else:
                        logger.warning(f"[{loop_thread_key}] Cannot commit transaction: connection is closed")
                        connection_error = True
                        raise asyncpg.ConnectionDoesNotExistError("Connection closed before commit")
                except Exception as commit_error:
                    logger.error(f"[{loop_thread_key}] Error committing transaction: {str(commit_error)}")
                    # If commit fails, try to rollback
                    try:
                        if not conn.is_closed():  # Check if connection is still valid
                            await tr.rollback()
                            logger.warning(f"[{loop_thread_key}] Transaction rolled back after commit failure")
                    except Exception:
                        pass  # Suppress rollback errors after commit failure
                    connection_error = True
                    raise commit_error
        except Exception as e:
            # If an exception occurs, rollback the transaction
            if tr and not getattr(tr, '_done', True):  # Check if transaction is still active
                try:
                    if not conn.is_closed():  # Check if connection is still valid
                        await tr.rollback()
                        logger.warning(f"[{loop_thread_key}] Transaction rolled back due to error: {str(e)}")
                except Exception as rollback_error:
                    logger.error(f"[{loop_thread_key}] Error rolling back transaction: {str(rollback_error)}")
            
            # Re-raise the original exception
            connection_error = True
            raise
    finally:
        # Always release the connection back to the pool
        if conn:
            try:
                # One final check to ensure transaction is closed
                if tr and not getattr(tr, '_done', True) and not conn.is_closed():  # If transaction wasn't committed or rolled back
                    try:
                        await tr.rollback()
                        logger.warning(f"[{loop_thread_key}] Transaction rolled back in finally block")
                    except Exception as final_rollback_error:
                        logger.error(f"[{loop_thread_key}] Error in final transaction rollback: {str(final_rollback_error)}")
                
                # Only release if we have a pool and it's still open
                if pool and not is_pool_closed(pool) and conn and not conn.is_closed():
                    try:
                        # Check if the connection is still valid before releasing
                        try:
                            await conn.fetchval("SELECT 1")
                        except Exception as e:
                            logger.warning(f"[{loop_thread_key}] Connection validation failed before release: {str(e)}")
                            raise asyncpg.InterfaceError("Connection validation failed before release")
                        await pool.release(conn)
                        logger.debug(f"[{loop_thread_key}] Released database connection back to pool")
                        # Decrement active connection counter
                        with _global_lock:
                            if loop_thread_key in _active_connections and _active_connections[loop_thread_key] > 0:
                                _active_connections[loop_thread_key] -= 1
                                logger.debug(f"[{loop_thread_key}] Active connections: {_active_connections[loop_thread_key]}")
                    except Exception as release_error:
                        logger.error(f"[{loop_thread_key}] Error releasing connection: {str(release_error)}")
                        connection_error = True
                elif conn:
                    # If the pool is closed, don't try to release the connection
                    pool_closed = is_pool_closed(pool) if pool else True
                    conn_closed = conn.is_closed() if conn else True
                    logger.warning(f"[{loop_thread_key}] Not releasing connection to pool - pool closed: {pool_closed}, conn closed: {conn_closed}")
                    # Decrement active connection counter even if we can't release properly
                    with _global_lock:
                        if loop_thread_key in _active_connections and _active_connections[loop_thread_key] > 0:
                            _active_connections[loop_thread_key] -= 1
                    
                    # If pool is closed but connection is still open, try to close it directly
                    if not conn_closed:
                        try:
                            await conn.close()
                            logger.warning(f"[{loop_thread_key}] Closed connection directly due to closed pool")
                            # Set conn to None to prevent further attempts to use it
                            conn = None
                            logger.debug(f"[{loop_thread_key}] Set connection to None after direct close")
                        except Exception as close_error:
                            logger.error(f"[{loop_thread_key}] Error closing connection directly: {str(close_error)}")
            except Exception as final_error:
                logger.error(f"[{loop_thread_key}] Error in connection cleanup: {str(final_error)}")
                connection_error = True
            
            # If we had connection errors, the pool may be in a bad state
            # Only close the pool if we had a serious error
            if connection_error and loop_thread_key in _connection_pools:
                try:
                    # Check if there are many active connections still in use
                    try:
                        if pool is None or is_pool_closed(pool):
                            logger.warning(f"[{loop_thread_key}] Cannot get pool stats - pool is None or closed")
                            pool_status = "Pool is None or closed"
                        else:
                            pool_status = f"Pool stats: {pool.get_size()}/{pool.get_max_size()} connections"
                    except Exception:
                        pool_status = "Could not get pool stats"
                    
                    logger.warning(f"[{loop_thread_key}] {pool_status}")
                    
                    # Only close the pool if it's not already closed
                    if pool and not is_pool_closed(pool):
                        await close_pool(loop_thread_key)
                        logger.info(f"[{loop_thread_key}] Closed bad connection pool due to errors")
                except Exception as close_error:
                    logger.error(f"[{loop_thread_key}] Error closing bad pool: {str(close_error)}")


async def get_db_stats():
    """
    Get PostgreSQL database statistics for monitoring.
    
    This function queries PostgreSQL system views to provide insight into
    database performance, connections, and table statistics.
    
    Returns:
        dict: Various PostgreSQL statistics for monitoring
    """
    stats = {}
    
    try:
        async with get_db_connection() as conn:
            # Query for active connections
            connections = await conn.fetch("""
                SELECT datname, usename, application_name, state, query,
                       query_start, backend_start
                FROM pg_stat_activity
                WHERE datname = $1
            """, os.environ.get("DB_NAME", DEFAULT_DB_NAME))
            
            # Query for table statistics
            table_stats = await conn.fetch("""
                SELECT relname, n_live_tup, n_dead_tup, last_vacuum, last_analyze
                FROM pg_stat_user_tables
            """)
            
            # Query for database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size($1))
            """, os.environ.get("DB_NAME", DEFAULT_DB_NAME))
            
            # Query for index usage statistics
            index_stats = await conn.fetch("""
                SELECT
                    relname as table_name,
                    indexrelname as index_name,
                    idx_scan as index_scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
                LIMIT 10
            """)
            
            # Return formatted stats
            stats = {
                "connections": [dict(conn) for conn in connections],
                "tables": [dict(row) for row in table_stats],
                "db_size": db_size,
                "index_usage": [dict(idx) for idx in index_stats],
                "pool_info": {
                    "active_pools": len(_connection_pools),
                    "connection_status": "healthy" if len(_connection_pools) > 0 else "no pools"
                }
            }
            logger.info(f"Successfully retrieved PostgreSQL database stats - DB Size: {db_size}")
    except Exception as e:
        logger.error(f"Error retrieving database stats: {str(e)}")
        stats = {"error": str(e), "status": "failed to retrieve stats"}
    
    return stats