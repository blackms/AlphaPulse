"""
PostgreSQL connection manager (fixed version).

This module provides robust connection pooling and transaction management
specifically designed to handle multiple async event loops and threads with
PostgreSQL databases.

The fixed version includes:
1. Better handling of "connection released back to pool" errors
2. More robust connection validation 
3. Timeouts to prevent hanging connections
"""
# Import from original connection_manager
from alpha_pulse.data_pipeline.database.connection_manager import (
    get_db_connection as original_get_db_connection,
    close_pool,
    close_all_pools,
    get_connection_pool,
    get_loop_thread_key,
    get_db_stats,
    execute_with_retry,
    is_pool_closed,
    _connection_pools,
    _active_connections,
    _pool_closing,
    _pool_locks,
    _global_lock
)
import asyncio
import logging
from loguru import logger
from contextlib import asynccontextmanager
import asyncpg
from typing import Any

@asynccontextmanager
async def get_db_connection():
    """
    Get a database connection with improved transaction management.
    
    This version fixes issues with "connection released back to pool" errors
    by adding better validation and handling of connection states.
    
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
        for attempt in range(1, 4):  # MAX_RETRY_ATTEMPTS = 3
            try:
                pool = await get_connection_pool()
                break
            except Exception as e:
                if attempt < 3:  # MAX_RETRY_ATTEMPTS
                    backoff = 0.5 * (2 ** (attempt - 1)) + asyncio.create_task(asyncio.sleep(0)).result()  # Add jitter
                    logger.warning(f"[{loop_thread_key}] Pool retrieval attempt {attempt} failed: {str(e)}. Retrying in {backoff:.2f}s...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Failed to get connection pool after 3 attempts")
                    raise
        
        # Acquire a connection with retry logic - carefully handle connection acquisition
        for attempt in range(1, 4):  # MAX_RETRY_ATTEMPTS = 3
            try:
                if attempt > 1 and is_pool_closed(pool):
                    # Only recreate the pool if it's closed
                    logger.warning(f"[{loop_thread_key}] Pool is closed, recreating for retry attempt {attempt}")
                    pool = await get_connection_pool()
                
                # Get a connection from the pool
                conn = await pool.acquire()
                
                # Get the pool lock
                pool_lock = _pool_locks.get(loop_thread_key)
                if pool_lock:
                    # Acquire the lock to prevent concurrent pool operations
                    try:
                        # Use timeout to prevent deadlocks
                        await asyncio.wait_for(pool_lock.acquire(), timeout=1.0)
                        try:
                            # Check if the pool is in the process of closing
                            with _global_lock:
                                if loop_thread_key in _pool_closing and _pool_closing[loop_thread_key]:
                                    logger.warning(f"[{loop_thread_key}] Acquired connection from a pool that is closing")
                                    # Don't increment the counter for a closing pool
                                else:
                                    # Increment active connection counter
                                    _active_connections.setdefault(loop_thread_key, 0)
                                    _active_connections[loop_thread_key] += 1
                        finally:
                            pool_lock.release()
                    except asyncio.TimeoutError:
                        logger.warning(f"[{loop_thread_key}] Timed out acquiring pool lock, incrementing counter without lock")
                        # Increment without the lock as a fallback
                        with _global_lock:
                            _active_connections.setdefault(loop_thread_key, 0)
                            _active_connections[loop_thread_key] += 1
                break
            except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as e:
                if attempt < 3:  # MAX_RETRY_ATTEMPTS
                    backoff = 0.5 * (2 ** (attempt - 1)) + asyncio.create_task(asyncio.sleep(0)).result()  # Add jitter
                    logger.warning(f"[{loop_thread_key}] Connection acquisition attempt {attempt} failed: {str(e)}. Retrying in {backoff:.2f}s...")
                    
                    # Only close the pool if we get a connection error
                    if not is_pool_closed(pool):
                        try:
                            # Only close the pool as a last resort
                            if attempt == 2:  # MAX_RETRY_ATTEMPTS - 1
                                await close_pool(loop_thread_key)
                        except Exception as close_err:
                            logger.warning(f"[{loop_thread_key}] Error closing pool: {str(close_err)}")
                    
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{loop_thread_key}] Failed to acquire connection after 3 attempts")
                    connection_error = True
                    raise
            except Exception as e:
                logger.error(f"[{loop_thread_key}] Unexpected error acquiring connection: {str(e)}")
                connection_error = True
                raise
        
        # Start a transaction with retry logic and proper error handling
        for attempt in range(1, 4):  # MAX_RETRY_ATTEMPTS = 3
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
                if attempt < 3:  # MAX_RETRY_ATTEMPTS
                    backoff = 0.5 * (2 ** (attempt - 1)) + asyncio.create_task(asyncio.sleep(0)).result()  # Add jitter
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
                        if attempt == 2:  # MAX_RETRY_ATTEMPTS - 1
                            await close_pool(loop_thread_key)
                            pool = await get_connection_pool()
                            conn = await pool.acquire()
                else:
                    logger.error(f"[{loop_thread_key}] Failed to start transaction after 3 attempts")
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
                        connection_valid = True
                        
                        # Check for the special case where connection was already released
                        try:
                            is_closed = conn.is_closed()
                        except Exception as check_error:
                            error_str = str(check_error)
                            if "connection has been released back to the pool" in error_str:
                                logger.warning(f"[{loop_thread_key}] Connection already released to pool: {error_str}")
                                # Connection is already released, so consider it invalid
                                connection_valid = False
                                
                                # Decrement active connection counter manually since we can't release properly
                                with _global_lock:
                                    if loop_thread_key in _active_connections and _active_connections[loop_thread_key] > 0:
                                        _active_connections[loop_thread_key] -= 1
                                        logger.debug(f"[{loop_thread_key}] Manually decremented connection counter after pre-release detection")
                                
                                # Skip further validation and release
                                conn = None
                        
                        # Only do the validation if connection hasn't been determined as invalid
                        if connection_valid and conn:
                            try:
                                # Use a timeout to prevent hanging if the connection is in a bad state
                                async def _test_query():
                                    return await conn.fetchval("SELECT 1")
                                
                                # Skip validation if we already know it's invalid
                                if connection_valid:
                                    try:
                                        # Use a short timeout for validation
                                        result = await asyncio.wait_for(_test_query(), timeout=0.5)
                                        if result != 1:
                                            connection_valid = False
                                            logger.warning(f"[{loop_thread_key}] Connection validation failed: unexpected result {result}")
                                    except Exception as e:
                                        connection_valid = False
                                        logger.warning(f"[{loop_thread_key}] Connection validation query failed: {str(e)}")
                            except Exception as e:
                                logger.warning(f"[{loop_thread_key}] Connection validation failed: {str(e)}")
                                connection_valid = False
                        
                        # Only release if the connection is still valid
                        if connection_valid and conn:
                            try:
                                await pool.release(conn)
                                logger.debug(f"[{loop_thread_key}] Released database connection back to pool")
                                conn = None  # Set to None to prevent further use
                            except Exception as release_error:
                                if "connection has been released back to the pool" in str(release_error):
                                    logger.warning(f"[{loop_thread_key}] Connection was already released: {str(release_error)}")
                                    conn = None  # Set to None to prevent further use
                                else:
                                    logger.error(f"[{loop_thread_key}] Error releasing connection: {str(release_error)}")
                                    connection_error = True
                                    raise
                        
                        # Update the active connection counter
                        pool_lock = _pool_locks.get(loop_thread_key)
                        if pool_lock:
                            try:
                                # Use timeout to prevent deadlocks
                                await asyncio.wait_for(pool_lock.acquire(), timeout=0.5)
                                try:
                                    # Decrement active connection counter and check if we can close the pool
                                    with _global_lock:
                                        if loop_thread_key in _active_connections and _active_connections[loop_thread_key] > 0:
                                            _active_connections[loop_thread_key] -= 1
                                            logger.debug(f"[{loop_thread_key}] Active connections: {_active_connections[loop_thread_key]}")
                                            # If pool was marked for closing and this was the last connection, close it now
                                            if _active_connections[loop_thread_key] == 0 and loop_thread_key in _pool_closing and _pool_closing[loop_thread_key]:
                                                logger.info(f"[{loop_thread_key}] Last connection released, closing pool that was marked for closing")
                                                asyncio.create_task(close_pool(loop_thread_key))
                                finally:
                                    pool_lock.release()
                            except asyncio.TimeoutError:
                                logger.warning(f"[{loop_thread_key}] Timed out acquiring pool lock for counter decrement, decreasing without lock")
                                # Decrement without the lock as a fallback
                                with _global_lock:
                                    if loop_thread_key in _active_connections and _active_connections[loop_thread_key] > 0:
                                        _active_connections[loop_thread_key] -= 1
                    except Exception as e:
                        logger.error(f"[{loop_thread_key}] Error in connection release process: {str(e)}")
                        connection_error = True
                elif conn:
                    # If the pool is closed, don't try to release the connection
                    pool_closed = is_pool_closed(pool) if pool else True
                    conn_closed = False
                    try:
                        conn_closed = conn.is_closed() if conn else True
                    except Exception as e:
                        if "connection has been released back to the pool" in str(e):
                            logger.warning(f"[{loop_thread_key}] Connection already released during cleanup: {str(e)}")
                            conn_closed = True
                        else:
                            logger.warning(f"[{loop_thread_key}] Error checking if connection is closed: {str(e)}")
                    
                    logger.warning(f"[{loop_thread_key}] Not releasing connection to pool - pool closed: {pool_closed}, conn closed: {conn_closed}")
                    
                    # Decrement active connection counter even if we can't release properly
                    with _global_lock:
                        if loop_thread_key in _active_connections and _active_connections[loop_thread_key] > 0:
                            _active_connections[loop_thread_key] -= 1
                    
                    # If pool is closed but connection is still open, try to close it directly
                    if not conn_closed and conn:
                        try:
                            await conn.close()
                            logger.warning(f"[{loop_thread_key}] Closed connection directly due to closed pool")
                            conn = None
                        except Exception as close_error:
                            logger.error(f"[{loop_thread_key}] Error closing connection directly: {str(close_error)}")
            except Exception as final_error:
                logger.error(f"[{loop_thread_key}] Error in connection cleanup: {str(final_error)}")
                connection_error = True
            
            # If we had connection errors, the pool may be in a bad state
            # Only close the pool if we had a serious error
            if connection_error and pool and not is_pool_closed(pool) and loop_thread_key in _connection_pools:
                try:
                    # Only close the pool if it's not already closed
                    if not is_pool_closed(pool):
                        await close_pool(loop_thread_key)
                        logger.info(f"[{loop_thread_key}] Closed bad connection pool due to errors")
                except Exception as close_error:
                    logger.error(f"[{loop_thread_key}] Error closing bad pool: {str(close_error)}")