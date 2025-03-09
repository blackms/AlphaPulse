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
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository
from alpha_pulse.data_pipeline.database.connection import get_pg_connection
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
        
        Returns:
            List of (exchange_id, data_type) tuples that need to be synced
        """
        thread_id = threading.get_ident()
        now = datetime.now(timezone.utc)
        to_sync = []
        
        # Get all exchanges from environment or config
        exchanges = self.get_configured_exchanges()
        
        # Define the operation to get sync statuses
        async def get_sync_statuses():
            async with get_pg_connection() as conn:
                repo = ExchangeCacheRepository(conn)
                return await repo.get_all_sync_status()
        
        try:
            # Use the execute_with_retry function from connection.py
            from alpha_pulse.data_pipeline.database.connection import _execute_with_retry
            sync_statuses = await _execute_with_retry(get_sync_statuses)
            
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
                        continue
                    
                    # Get the last sync time and next sync time
                    last_sync_time = None
                    next_sync_time = None
                    
                    # Check if we have a status record
                    if (exchange_id in status_dict and
                            data_type.value in status_dict[exchange_id]):
                        status = status_dict[exchange_id][data_type.value]
                        last_sync_time = status["last_sync"]
                        next_sync_time = status["next_sync"]
                    
                    # If no status record, or next_sync_time is in the past
                    if (next_sync_time is None or
                            now >= next_sync_time):
                        # Add to sync queue
                        logger.info(f"[THREAD {thread_id}] Scheduling sync for {exchange_id}, type: {data_type}")
                        to_sync.append((exchange_id, data_type))
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error checking scheduled syncs: {str(e)}")
            logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
            logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
        
        return to_sync
    
    async def update_sync_status(self, exchange_id: str, data_type: DataType,
                                status: SyncStatus, error_message: Optional[str] = None) -> None:
        """
        Update the sync status in the database.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data
            status: New status
            error_message: Optional error message
        """
        thread_id = threading.get_ident()
        next_sync_time = datetime.now(timezone.utc) + timedelta(seconds=self._sync_interval)
        
        # Define the operation to update sync status
        async def update_operation():
            async with get_pg_connection() as conn:
                repo = ExchangeCacheRepository(conn)
                await repo.update_sync_status(
                    exchange_id,
                    data_type.value,
                    status.value,
                    next_sync_time,
                    error_message
                )
        
        try:
            # Use the execute_with_retry function from connection.py
            from alpha_pulse.data_pipeline.database.connection import _execute_with_retry
            await _execute_with_retry(update_operation)
            logger.debug(f"[THREAD {thread_id}] Successfully updated sync status for {exchange_id}, {data_type}")
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error updating sync status: {str(e)}")
            logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
            logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")