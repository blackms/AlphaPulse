"""
Task management for the exchange data synchronization system.

This module provides functionality for managing synchronization tasks.
"""
import asyncio
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Union

from loguru import logger

from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.scheduler.sync_module.types import SyncStatus
from alpha_pulse.data_pipeline.database.exchange_cache_fixed import ExchangeCacheRepository
# Import from the new connection manager
from alpha_pulse.data_pipeline.database.connection_manager import get_db_connection, execute_with_retry, get_loop_thread_key
# Import the default connection parameters from the connection module
from alpha_pulse.data_pipeline.database.connection import (
    DEFAULT_DB_HOST,
    DEFAULT_DB_PORT,
    DEFAULT_DB_NAME,
    DEFAULT_DB_USER,
    DEFAULT_DB_PASS
)


class TaskManager:
    """
    Manages synchronization tasks for the exchange data synchronization system.
    
    This class is responsible for scheduling, queuing, and tracking synchronization tasks.
    """
    
    def __init__(self, sync_interval: int = 3600):
        """
        Initialize the task manager.
        
        Args:
            sync_interval: Interval between syncs in seconds (default: 1 hour)
        """
        self._sync_interval = sync_interval
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._last_sync: Dict[str, Dict[str, datetime]] = {}
        self._sync_queue: Set[tuple] = set()  # (exchange_id, data_type)
    
    def add_to_queue(self, exchange_id: str, data_type: Union[DataType, str]) -> bool:
        """
        Add a synchronization task to the queue.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
            
        Returns:
            True if added to queue, False otherwise
        """
        # Convert string to enum if needed
        if isinstance(data_type, str):
            try:
                data_type = DataType(data_type)
            except ValueError:
                try:
                    # Try uppercase version
                    data_type = DataType(data_type.upper())
                except ValueError:
                    logger.error(f"Invalid data type: {data_type}")
                    return False
        
        # Add to sync queue
        logger.info(f"Adding to sync queue: {exchange_id}, type: {data_type}")
        self._sync_queue.add((exchange_id, data_type))
        return True
    
    def get_queue_items(self) -> List[tuple]:
        """
        Get all items in the queue and clear the queue.
        
        Returns:
            List of (exchange_id, data_type) tuples
        """
        queue_items = list(self._sync_queue)
        self._sync_queue.clear()
        return queue_items
    
    def is_task_active(self, exchange_id: str, data_type: DataType) -> bool:
        """
        Check if a task is already active.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
            
        Returns:
            True if task is active, False otherwise
        """
        task_key = f"{exchange_id}_{data_type.value}"
        return task_key in self._active_tasks and not self._active_tasks[task_key].done()
    
    def add_active_task(self, exchange_id: str, data_type: DataType, task: asyncio.Task) -> None:
        """
        Add a task to the active tasks.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
            task: Task to add
        """
        task_key = f"{exchange_id}_{data_type.value}"
        self._active_tasks[task_key] = task
    
    def get_configured_exchanges(self) -> List[str]:
        """
        Get the list of configured exchanges from environment or config.
        
        Returns:
            List of exchange identifiers
        """
        # For now, just return the exchange from environment
        exchange_type = os.environ.get('EXCHANGE_TYPE', 'bybit').lower()
        return [exchange_type]
    async def check_scheduled_syncs(self) -> List[tuple]:
        """
        Check for scheduled syncs that need to run.
        
        This method has been enhanced with improved error handling and connection
        management to prevent database errors during sync status checks.
        
        Returns:
            List of (exchange_id, data_type) tuples that need to be synced
        """
        # Get current loop and thread identifiers for debugging
        loop_thread_key = get_loop_thread_key()
        current_loop = asyncio.get_running_loop()
        loop_id = id(current_loop)
        
        logger.debug(f"[{loop_thread_key}] Checking scheduled syncs (loop: {loop_id})")
        now = datetime.now(timezone.utc)
        to_sync = []
        
        # Get all exchanges from environment or config
        exchanges = self.get_configured_exchanges()
        
        # Define the operation to get sync statuses with dedicated connection
        async def get_sync_statuses():
            # Use a fresh dedicated connection for this operation
            async with get_db_connection() as conn:
                repo = ExchangeCacheRepository(conn)
                # Use a shorter timeout for this query to prevent blocking
                await conn.execute("SET statement_timeout = '15s'")
                return await repo.get_all_sync_status()
        
        try:
            # Use execute_with_retry with increased max retries for important operation
            sync_statuses = await execute_with_retry(get_sync_statuses, max_retries=4)
            
            if not sync_statuses:
                logger.info(f"[{loop_thread_key}] No sync status records found, will schedule initial syncs")
                # If no records exist yet, schedule all exchanges and data types for initial sync
                for exchange_id in exchanges:
                    for data_type in [dt for dt in DataType if dt != DataType.ALL]:
                        if (exchange_id, data_type) not in self._sync_queue:
                            logger.info(f"[{loop_thread_key}] Scheduling initial sync for {exchange_id}, type: {data_type}")
                            to_sync.append((exchange_id, data_type))
                return to_sync
            
            # Create a dict of sync statuses by exchange and data type
            status_dict = {}
            for status in sync_statuses:
                if status["exchange_id"] not in status_dict:
                    status_dict[status["exchange_id"]] = {}
                status_dict[status["exchange_id"]][status["data_type"]] = status
            
            # Check each exchange and data type
            for exchange_id in exchanges:
                for data_type in [dt for dt in DataType if dt != DataType.ALL]:
                    # Skip if already in queue
                    if (exchange_id, data_type) in self._sync_queue:
                        logger.debug(f"[{loop_thread_key}] Skipping {exchange_id}, {data_type} - already in queue")
                        continue
                    
                    # Get the last sync time and next sync time
                    last_sync_time = None
                    next_sync_time = None
                    status_value = None
                    
                    # Check if we have a status record
                    if (exchange_id in status_dict and
                            data_type.value in status_dict[exchange_id]):
                        status = status_dict[exchange_id][data_type.value]
                        last_sync_time = status["last_sync"]
                        next_sync_time = status["next_sync"]
                        status_value = status.get("status")
                        
                        # Log some diagnostic info for in-progress syncs
                        if status_value == SyncStatus.IN_PROGRESS.value:
                            if last_sync_time:
                                # If a sync has been in progress for a long time, it might be stalled
                                time_in_progress = now - last_sync_time
                                if time_in_progress.total_seconds() > 1800:  # 30 minutes
                                    logger.warning(f"[{loop_thread_key}] Sync for {exchange_id}, {data_type} has been "
                                                f"in progress for {time_in_progress.total_seconds():.0f} seconds, "
                                                f"which is unusually long. It may be stalled.")
                    
                    # Schedule sync if:
                    # 1. No status record exists
                    # 2. Next sync time is in the past
                    # 3. Status is FAILED and it's been at least 5 minutes since last attempt
                    need_sync = False
                    
                    if next_sync_time is None:
                        need_sync = True
                        reason = "no previous sync"
                    elif now >= next_sync_time:
                        need_sync = True
                        reason = "scheduled time reached"
                    elif status_value == SyncStatus.FAILED.value and last_sync_time:
                        # For failed syncs, check if we've waited the retry interval
                        time_since_failure = now - last_sync_time
                        # Wait at least 5 minutes before retrying failed syncs
                        if time_since_failure.total_seconds() >= 300:
                            need_sync = True
                            reason = f"retry after failure ({time_since_failure.total_seconds():.0f}s ago)"
                    
                    if need_sync:
                        # Add to sync queue
                        logger.info(f"[{loop_thread_key}] Scheduling sync for {exchange_id}, type: {data_type} ({reason})")
                        to_sync.append((exchange_id, data_type))
                    
        except asyncpg.InterfaceError as e:
            if "another operation is in progress" in str(e):
                logger.warning(f"[{loop_thread_key}] Concurrent operation error during status check. "
                            f"This is a recoverable error. Will retry on next scheduler cycle.")
            else:
                logger.error(f"[{loop_thread_key}] Database interface error checking syncs: {str(e)}")
                
        except Exception as e:
            logger.error(f"[{loop_thread_key}] Error checking scheduled syncs: {str(e)}")
            logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
            logger.error(f"[{loop_thread_key}] Exception details: {repr(e)}")
            
            # If we have a connection pool issue, attempt to reset it
            if isinstance(e, (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError)):
                try:
                    await close_pool(loop_thread_key)
                    logger.info(f"[{loop_thread_key}] Reset connection pool after error in check_scheduled_syncs")
                except Exception as pool_error:
                    logger.warning(f"[{loop_thread_key}] Error resetting connection pool: {str(pool_error)}")
        
        logger.debug(f"[{loop_thread_key}] Found {len(to_sync)} syncs to schedule")
        return to_sync
    
    async def update_sync_status(self, exchange_id: str, data_type: DataType,
                                status: SyncStatus, error_message: Optional[str] = None) -> None:
        """
        Update the sync status in the database.
        
        This method has been enhanced to handle concurrent operation errors and connection
        issues with improved retry logic and transaction isolation.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data
            status: New status
            error_message: Optional error message
        """
        # Get current loop and thread identifiers for debugging
        loop_thread_key = get_loop_thread_key()
        current_loop = asyncio.get_running_loop()
        loop_id = id(current_loop)
        
        logger.debug(f"[{loop_thread_key}] Updating sync status for {exchange_id}, {data_type} to {status} (loop: {loop_id})")
        
        # Calculate next sync time based on the status
        # For failed statuses, schedule sooner to retry
        if status == SyncStatus.FAILED:
            # Schedule retry sooner for failures (1/3 of normal interval)
            retry_interval = max(int(self._sync_interval / 3), 300)  # At least 5 minutes
            next_sync_time = datetime.now(timezone.utc) + timedelta(seconds=retry_interval)
            logger.debug(f"[{loop_thread_key}] Status FAILED, scheduling retry in {retry_interval} seconds")
        else:
            next_sync_time = datetime.now(timezone.utc) + timedelta(seconds=self._sync_interval)
        
        # Define the operation to update sync status with proper isolation
        async def update_operation():
            # Use a dedicated connection for this operation
            async with get_db_connection() as conn:
                repo = ExchangeCacheRepository(conn)
                
                # First check if record exists
                try:
                    # Attempt to read current status first to reduce lock contention
                    current_status = await repo.get_sync_status(exchange_id, data_type.value)
                    
                    # Then perform the update
                    await repo.update_sync_status(
                        exchange_id,
                        data_type.value,
                        status.value,
                        next_sync_time,
                        error_message
                    )
                    return True
                except Exception as inner_e:
                    logger.warning(f"[{loop_thread_key}] Inner update operation failed: {str(inner_e)}")
                    # If the error is related to concurrent operations, it will be handled by the retry mechanism
                    raise
        
        try:
            # Use execute_with_retry with increased max retries for status updates
            # Status updates are critical operations that should succeed
            await execute_with_retry(update_operation, max_retries=5)  # Increased from default 3
            logger.debug(f"[{loop_thread_key}] Successfully updated sync status for {exchange_id}, {data_type} to {status}")
        except asyncpg.InterfaceError as e:
            if "another operation is in progress" in str(e):
                logger.warning(f"[{loop_thread_key}] Concurrent operation error during status update for {exchange_id}. "
                            f"This may indicate high contention on the database connection.")
                # Try one more time with a dedicated connection approach
                try:
                    # Get a fresh connection pool
                    await close_pool(loop_thread_key)
                    await asyncio.sleep(1.0)  # Give time for connections to clean up
                    
                    # Try update with a completely fresh connection
                    async def last_chance_update():
                        async with get_db_connection() as conn:
                            repo = ExchangeCacheRepository(conn)
                            await repo.update_sync_status(
                                exchange_id,
                                data_type.value,
                                status.value,
                                next_sync_time,
                                f"{error_message or ''} (Recovered from concurrent operation error)"
                            )
                    await execute_with_retry(last_chance_update)
                    logger.info(f"[{loop_thread_key}] Recovered from concurrent operation error for {exchange_id} status update")
                except Exception as recovery_e:
                    logger.error(f"[{loop_thread_key}] Failed to recover from concurrent operation error: {str(recovery_e)}")
                    logger.error(f"[{loop_thread_key}] Status update for {exchange_id}, {data_type} to {status} was NOT completed")
            else:
                logger.error(f"[{loop_thread_key}] Database interface error updating sync status: {str(e)}")
        except Exception as e:
            logger.error(f"[{loop_thread_key}] Error updating sync status: {str(e)}")
            logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
            logger.error(f"[{loop_thread_key}] Exception details: {repr(e)}")