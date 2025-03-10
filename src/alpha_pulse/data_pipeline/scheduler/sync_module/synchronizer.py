"""
Main synchronizer module for exchange data.

This module provides the main ExchangeDataSynchronizer class that orchestrates
the synchronization of exchange data.
"""
import asyncio
import threading
import time
import signal
from typing import Dict, List, Optional, Any, Set, Union
import asyncpg

from loguru import logger

from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.scheduler.sync_module.types import SyncStatus
from alpha_pulse.data_pipeline.scheduler.sync_module.exchange_manager import ExchangeManager
from alpha_pulse.data_pipeline.scheduler.sync_module.data_sync import DataSynchronizer
from alpha_pulse.data_pipeline.scheduler.sync_module.task_manager import TaskManager
from alpha_pulse.data_pipeline.scheduler.sync_module.event_loop_manager import EventLoopManager
# Import the new connection manager
from alpha_pulse.data_pipeline.database.connection_manager import (
    get_db_connection,
    execute_with_retry,
    close_all_pools,
    get_loop_thread_key
)
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository


class ExchangeDataSynchronizer:
    """
    Synchronizes exchange data on a regular schedule.
    
    This class schedules and executes periodic data synchronization tasks for
    exchange data such as balances, positions, orders, and prices. It can also
    be triggered manually to force a synchronization.
    """
    
    # Singleton instance
    _instance = None
    _instance_lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._instance_lock:
            if cls._instance is None:
                logger.debug(f"Creating new ExchangeDataSynchronizer instance")
                cls._instance = super(ExchangeDataSynchronizer, cls).__new__(cls)
                cls._instance._initialized = False
            else:
                logger.debug(f"Returning existing ExchangeDataSynchronizer instance")
        return cls._instance
    
    def __init__(self, sync_interval: int = 3600):
        """
        Initialize the synchronizer.
        
        Args:
            sync_interval: Interval between syncs in seconds (default: 1 hour)
        """
        # Only initialize once
        if self._initialized:
            logger.debug(f"ExchangeDataSynchronizer already initialized, skipping initialization")
            return
        
        loop_thread_key = get_loop_thread_key()
        logger.debug(f"[{loop_thread_key}] Initializing ExchangeDataSynchronizer")
        
        # Initialize components
        self._exchange_manager = ExchangeManager()
        self._data_synchronizer = DataSynchronizer()
        self._task_manager = TaskManager(sync_interval=sync_interval)
        
        # Initialize state
        self._should_stop = threading.Event()
        self._worker_thread = None
        self._running = False
        self._initialized = True
        
        # Start the background task
        self.start()
        
        # Register shutdown handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.debug(f"[{loop_thread_key}] ExchangeDataSynchronizer initialization complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("Synchronizer is already running")
            return
        
        logger.info("Starting exchange data synchronizer...")
        self._should_stop.clear()
        self._worker_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._worker_thread.start()
        self._running = True
        logger.info("Exchange data synchronizer started")
    
    def stop(self):
        """Stop the background worker thread."""
        if not self._running:
            logger.warning("Synchronizer is not running")
            return
        
        logger.info("Stopping exchange data synchronizer...")
        self._should_stop.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not terminate gracefully")
        self._running = False
        logger.info("Exchange data synchronizer stopped")
    
    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        # Create a new event loop for this thread
        thread_id = threading.get_ident()
        logger.debug(f"[THREAD {thread_id}] Creating new event loop for thread")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.debug(f"[THREAD {thread_id}] Set event loop for thread")
        
        try:
            # Start the main task
            logger.debug(f"[THREAD {thread_id}] Creating main task")
            main_task = loop.create_task(self._main_loop())
            logger.debug(f"[THREAD {thread_id}] Main task created: {main_task}")
            
            # Run until should_stop is set
            logger.debug(f"[THREAD {thread_id}] Running main task until completion")
            loop.run_until_complete(main_task)
            logger.debug(f"[THREAD {thread_id}] Main task completed")
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error in event loop: {str(e)}")
            logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
            logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
        finally:
            # Clean up
            logger.debug(f"[THREAD {thread_id}] Cleaning up tasks")
            
            # Close the loop
            try:
                logger.debug(f"[THREAD {thread_id}] Getting all pending tasks")
                pending = asyncio.all_tasks(loop)
                pending_count = len(pending)
                logger.debug(f"[THREAD {thread_id}] Pending tasks count: {pending_count}")
                
                if pending:
                    logger.warning(f"[THREAD {thread_id}] Cancelling {pending_count} pending tasks")
                    for i, task in enumerate(pending):
                        logger.debug(f"[THREAD {thread_id}] Cancelling pending task {i+1}/{pending_count}: {task}")
                        task.cancel()
                    
                    # Give tasks a chance to cancel
                    logger.debug(f"[THREAD {thread_id}] Waiting for tasks to cancel")
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    logger.debug(f"[THREAD {thread_id}] All tasks cancelled")
                
                logger.debug(f"[THREAD {thread_id}] Shutting down async generators")
                loop.run_until_complete(loop.shutdown_asyncgens())
                logger.debug(f"[THREAD {thread_id}] Closing event loop")
                loop.close()
                logger.debug(f"[THREAD {thread_id}] Event loop closed")
            except Exception as e:
                logger.error(f"[THREAD {thread_id}] Error closing event loop: {str(e)}")
                logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
                logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
    
    async def _main_loop(self):
        """Main loop that checks for tasks to run."""
        thread_id = threading.get_ident()
        loop_thread_key = get_loop_thread_key()
        logger.info(f"[{loop_thread_key}] Starting main loop for exchange data synchronization")
        
        # Get the current event loop for this thread
        loop = EventLoopManager.get_or_create_event_loop()
        
        iteration = 0
        while not self._should_stop.is_set():
            iteration += 1
            logger.debug(f"[{loop_thread_key}] Main loop iteration {iteration}")
            
            try:
                # Process any queued sync requests
                logger.debug(f"[{loop_thread_key}] Processing sync queue")
                await self._process_sync_queue()
                
                # Check for scheduled syncs
                logger.debug(f"[{loop_thread_key}] Checking scheduled syncs")
                await self._check_scheduled_syncs()
                
                # Sleep for a bit to avoid high CPU usage
                logger.debug(f"[{loop_thread_key}] Sleeping for 5 seconds")
                await asyncio.sleep(5)
                logger.debug(f"[{loop_thread_key}] Async sleep completed")
            except asyncio.CancelledError:
                logger.warning(f"[{loop_thread_key}] Main loop task was cancelled")
                break
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Check for database connection errors
                if "InterfaceError" in error_type or "ConnectionDoesNotExistError" in error_type:
                    logger.warning(f"[{loop_thread_key}] Database connection error: {error_msg}")
                    
                    # Close all connection pools to force reconnection
                    try:
                        await close_all_pools()
                        logger.info(f"[{loop_thread_key}] Closed all database connection pools to force reconnection")
                    except Exception as close_error:
                        logger.error(f"[{loop_thread_key}] Error closing connection pools: {str(close_error)}")
                    
                    # Sleep before retry
                    time.sleep(5)
                # Check for event loop issues
                elif isinstance(e, RuntimeError) and "got Future" in error_msg and "attached to a different loop" in error_msg:
                    logger.warning(f"[{loop_thread_key}] Event loop issue detected, resetting event loop")
                    # Reset the event loop for this thread
                    loop = EventLoopManager.reset_event_loop()
                    loop_thread_key = get_loop_thread_key()  # Get the new key after reset
                    
                    # Close all connection pools to force reconnection with the new loop
                    try:
                        await close_all_pools()
                        logger.info(f"[{loop_thread_key}] Closed all database connection pools after loop reset")
                    except Exception as close_error:
                        logger.error(f"[{loop_thread_key}] Error closing connection pools: {str(close_error)}")
                    
                    # Sleep before retry
                    time.sleep(5)
                else:
                    logger.error(f"[{loop_thread_key}] Error in main loop: {error_msg}")
                    logger.error(f"[{loop_thread_key}] Exception type: {error_type}")
                    logger.error(f"[{loop_thread_key}] Exception details: {repr(e)}")
                    time.sleep(10)  # Sleep on error, but not too long
        
        # Clean up database connections before exiting
        try:
            await close_all_pools()
            logger.info(f"[{loop_thread_key}] Closed all database connection pools before exiting main loop")
        except Exception as close_error:
            logger.error(f"[{loop_thread_key}] Error closing connection pools: {str(close_error)}")
        
        logger.info(f"[{loop_thread_key}] Main loop exited after {iteration} iterations")
    
    async def _process_sync_queue(self):
        """Process any queued sync requests."""
        loop_thread_key = get_loop_thread_key()
        
        # Get all items in the queue
        queue_items = self._task_manager.get_queue_items()
        if not queue_items:
            return
        
        # Process all items in the queue
        for exchange_id, data_type in queue_items:
            # Skip if already syncing this exchange and data type
            if self._task_manager.is_task_active(exchange_id, data_type):
                logger.info(f"[{loop_thread_key}] Skipping already active sync: {exchange_id}_{data_type.value}")
                continue
            
            # Start a new sync task
            logger.info(f"[{loop_thread_key}] Starting queued sync for {exchange_id}, type: {data_type}")
            
            try:
                # Ensure we're using the correct event loop
                current_loop = asyncio.get_running_loop()
                loop_id = id(current_loop)
                logger.debug(f"[{loop_thread_key}] Using event loop {loop_id} for sync operation")
                
                # Define a function that will be executed with retry logic
                async def run_sync():
                    try:
                        return await self._sync_exchange_data(exchange_id, data_type)
                    except Exception as sync_error:
                        error_msg = str(sync_error)
                        # Check for event loop conflicts
                        if isinstance(sync_error, RuntimeError) and "attached to a different loop" in error_msg:
                            logger.error(f"[{loop_thread_key}] Event loop conflict detected: {error_msg}")
                            raise sync_error
                        elif "connection was closed" in error_msg or "cannot perform operation" in error_msg:
                            logger.error(f"[{loop_thread_key}] Database connection error: {error_msg}")
                            raise sync_error
                        else:
                            logger.error(f"[{loop_thread_key}] Error in sync operation: {error_msg}")
                            raise sync_error
                
                # Execute the sync with retry logic
                await execute_with_retry(run_sync)
                
            except Exception as e:
                logger.error(f"[{loop_thread_key}] Error syncing {data_type} for {exchange_id}: {str(e)}")
                logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
                logger.error(f"[{loop_thread_key}] Exception details: {repr(e)}")
                
                # Update sync status to failed if possible
                try:
                    await self._task_manager.update_sync_status(
                        exchange_id,
                        data_type,
                        SyncStatus.FAILED,
                        f"Sync failed: {str(e)}"
                    )
                except Exception as update_error:
                    logger.error(f"[{loop_thread_key}] Could not update sync status after failure: {str(update_error)}")
    
    async def _check_scheduled_syncs(self):
        """Check for scheduled syncs that need to run."""
        loop_thread_key = get_loop_thread_key()
        logger.debug(f"[{loop_thread_key}] Checking scheduled syncs")
        
        try:
            # Get syncs that need to run
            to_sync = await self._task_manager.check_scheduled_syncs()
            
            # Add them to the queue
            for exchange_id, data_type in to_sync:
                self._task_manager.add_to_queue(exchange_id, data_type)
                logger.debug(f"[{loop_thread_key}] Added to queue: {exchange_id}, {data_type}")
        except Exception as e:
            logger.error(f"[{loop_thread_key}] Error checking scheduled syncs: {str(e)}")
            logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
            logger.error(f"[{loop_thread_key}] Exception details: {repr(e)}")
    
    async def _sync_exchange_data(self, exchange_id: str, data_type: DataType):
        """
        Synchronize data for a specific exchange and data type.
        
        This method has been enhanced to handle event loop conflicts and database connection
        issues by using isolated connections for each operation and proper retry logic.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
        """
        # Get loop information for debugging
        loop_thread_key = get_loop_thread_key()
        current_loop = asyncio.get_running_loop()
        loop_id = id(current_loop)
        
        logger.debug(f"[{loop_thread_key}] Starting sync for {exchange_id}, type: {data_type}, loop: {loop_id}")
        
        # Define a function to get a repository with isolated connection
        # This is crucial to prevent database conflicts
        async def get_fresh_repository():
            try:
                async with get_db_connection() as conn:
                    if conn is None or conn.is_closed():
                        logger.warning(f"[{loop_thread_key}] Got None or closed connection from get_db_connection")
                        raise asyncpg.InterfaceError("Connection is None or closed")
                    return ExchangeCacheRepository(conn)
            except asyncpg.InterfaceError as e:
                error_msg = str(e)
                if "connection has been released back to the pool" in error_msg:
                    logger.error(f"[{loop_thread_key}] Database connection error in get_fresh_repository: {error_msg}")
                    logger.error(f"[{loop_thread_key}] This is likely due to a connection pool issue. The operation will be retried.")
                raise
            
        
        # Mark as in progress with proper retry logic
        try:
            async def update_status_op():
                await self._task_manager.update_sync_status(
                    exchange_id,
                    data_type,
                    SyncStatus.IN_PROGRESS
                )
            # Use retry logic for database operations
            await execute_with_retry(update_status_op)
        except Exception as e:
            logger.error(f"[{loop_thread_key}] Error updating initial sync status: {str(e)}")
            logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
            # Continue with the sync despite the status update failure
        
        try:
            # Get exchange instance with retry logic
            async def get_exchange_op():
                return await self._exchange_manager.get_exchange(exchange_id)
            
            exchange = await execute_with_retry(get_exchange_op)
            
            if not exchange:
                logger.error(f"[{loop_thread_key}] Failed to get exchange for {exchange_id}. Troubleshooting steps:")
                logger.error("1. Check API credentials in environment variables or configuration")
                logger.error("2. Verify network connectivity to the exchange API")
                logger.error("3. Confirm testnet/mainnet settings are correct")
                logger.error("4. Check for any IP restrictions on your API key")
                logger.error("5. Run debug_bybit_auth.py or debug_exchange_connection.py for more details")
                
                # Update status to failed with retry logic
                try:
                    async def update_failed_status_op():
                        await self._task_manager.update_sync_status(
                            exchange_id,
                            data_type,
                            SyncStatus.FAILED,
                            "Failed to initialize exchange"
                        )
                    await execute_with_retry(update_failed_status_op)
                except Exception as status_error:
                    logger.error(f"[{loop_thread_key}] Error updating sync status to failed: {str(status_error)}")
                return
            
            # Track sync success
            sync_success = False
            
            try:
                # Handle different data types
                if data_type == DataType.ALL:
                    # Sync all data types individually with separate connections
                    # This prevents "another operation is in progress" errors
                    results = {
                        "balances": False,
                        "positions": False,
                        "orders": False,
                        "prices": False
                    }
                    
                    # Sync balances with proper isolation
                    try:
                        async def sync_balances_op():
                            # Get a fresh repository for this specific operation
                            repo = await get_fresh_repository()
                            await self._data_synchronizer.sync_balances(exchange_id, exchange, repo)
                            return True
                        
                        results["balances"] = await execute_with_retry(sync_balances_op)
                        logger.info(f"[{loop_thread_key}] Successfully synced balances for {exchange_id}")
                        # Small delay to ensure connections are properly released
                        await asyncio.sleep(1.0)  # Increased delay to ensure proper connection cleanup
                    except Exception as e:
                        logger.error(f"[{loop_thread_key}] Error syncing balances for {exchange_id}: {str(e)}")
                        logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
                    
                    # Sync positions with proper isolation
                    try:
                        async def sync_positions_op():
                            # Get a fresh repository for this specific operation
                            repo = await get_fresh_repository()
                            await self._data_synchronizer.sync_positions(exchange_id, exchange, repo)
                            return True
                        
                        results["positions"] = await execute_with_retry(sync_positions_op)
                        logger.info(f"[{loop_thread_key}] Successfully synced positions for {exchange_id}")
                        # Small delay to ensure connections are properly released
                        await asyncio.sleep(1.0)  # Increased delay to ensure proper connection cleanup
                    except Exception as e:
                        logger.error(f"[{loop_thread_key}] Error syncing positions for {exchange_id}: {str(e)}")
                        logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
                    
                    # Sync orders with proper isolation
                    try:
                        async def sync_orders_op():
                            # Get a fresh repository for this specific operation
                            repo = await get_fresh_repository()
                            await self._data_synchronizer.sync_orders(exchange_id, exchange, repo)
                            return True
                        
                        results["orders"] = await execute_with_retry(sync_orders_op)
                        logger.info(f"[{loop_thread_key}] Successfully synced orders for {exchange_id}")
                        # Small delay to ensure connections are properly released
                        await asyncio.sleep(1.0)  # Increased delay to ensure proper connection cleanup
                    except Exception as e:
                        logger.error(f"[{loop_thread_key}] Error syncing orders for {exchange_id}: {str(e)}")
                        logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
                    
                    # Sync prices with proper isolation
                    try:
                        async def sync_prices_op():
                            # Get a fresh repository for this specific operation
                            repo = await get_fresh_repository()
                            await self._data_synchronizer.sync_prices(exchange_id, exchange, repo)
                            return True
                        
                        results["prices"] = await execute_with_retry(sync_prices_op)
                        logger.info(f"[{loop_thread_key}] Successfully synced prices for {exchange_id}")
                    except Exception as e:
                        logger.error(f"[{loop_thread_key}] Error syncing prices for {exchange_id}: {str(e)}")
                        logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
                    
                    # Consider success if at least one type succeeded
                    sync_success = any(results.values())
                    
                    if sync_success:
                        success_count = sum(1 for result in results.values() if result)
                        logger.info(f"[{loop_thread_key}] ALL sync completed with {success_count}/4 successful operations for {exchange_id}")
                    else:
                        logger.error(f"[{loop_thread_key}] ALL sync failed completely for {exchange_id}")
                
                # Handle individual data types with proper isolation
                elif data_type == DataType.BALANCES:
                    async def sync_balances_op():
                        repo = await get_fresh_repository()
                        await self._data_synchronizer.sync_balances(exchange_id, exchange, repo)
                        return True
                    
                    sync_success = await execute_with_retry(sync_balances_op)
                    logger.info(f"[{loop_thread_key}] Successfully synced balances for {exchange_id}")
                
                elif data_type == DataType.POSITIONS:
                    async def sync_positions_op():
                        repo = await get_fresh_repository()
                        await self._data_synchronizer.sync_positions(exchange_id, exchange, repo)
                        return True
                    
                    sync_success = await execute_with_retry(sync_positions_op)
                    logger.info(f"[{loop_thread_key}] Successfully synced positions for {exchange_id}")
                
                elif data_type == DataType.ORDERS:
                    async def sync_orders_op():
                        repo = await get_fresh_repository()
                        await self._data_synchronizer.sync_orders(exchange_id, exchange, repo)
                        return True
                    
                    sync_success = await execute_with_retry(sync_orders_op)
                    logger.info(f"[{loop_thread_key}] Successfully synced orders for {exchange_id}")
                
                elif data_type == DataType.PRICES:
                    async def sync_prices_op():
                        repo = await get_fresh_repository()
                        await self._data_synchronizer.sync_prices(exchange_id, exchange, repo)
                        return True
                    
                    sync_success = await execute_with_retry(sync_prices_op)
                    logger.info(f"[{loop_thread_key}] Successfully synced prices for {exchange_id}")
            
            except Exception as sync_error:
                logger.error(f"[{loop_thread_key}] Error during sync operation: {str(sync_error)}")
                logger.error(f"[{loop_thread_key}] Exception type: {type(sync_error).__name__}")
                logger.error(f"[{loop_thread_key}] Exception details: {repr(sync_error)}")
            
            # Update status to completed if successful
            if sync_success:
                try:
                    async def update_completed_status_op():
                        await self._task_manager.update_sync_status(
                            exchange_id,
                            data_type,
                            SyncStatus.COMPLETED
                        )
                    await execute_with_retry(update_completed_status_op)
                    logger.info(f"[{loop_thread_key}] Successfully completed sync for {exchange_id}, {data_type}")
                except Exception as status_error:
                    logger.error(f"[{loop_thread_key}] Error updating sync status to completed: {str(status_error)}")
            else:
                # Update status to failed
                try:
                    async def update_failed_status_op():
                        await self._task_manager.update_sync_status(
                            exchange_id,
                            data_type,
                            SyncStatus.FAILED,
                            "Sync operation failed - check logs for details"
                        )
                    await execute_with_retry(update_failed_status_op)
                    logger.warning(f"[{loop_thread_key}] Updated sync status to FAILED for {exchange_id}, {data_type}")
                except Exception as status_error:
                    logger.error(f"[{loop_thread_key}] Error updating sync status to failed: {str(status_error)}")
                    
        except Exception as e:
            logger.error(f"[{loop_thread_key}] Error in _sync_exchange_data for {exchange_id}, {data_type}: {str(e)}")
            logger.error(f"[{loop_thread_key}] Exception type: {type(e).__name__}")
            
            # Check for event loop related errors
            if isinstance(e, RuntimeError) and "attached to a different loop" in str(e):
                logger.error(f"[{loop_thread_key}] Event loop conflict detected in sync operation. This needs to be handled by the caller.")
            
            # Update status to failed
            try:
                async def update_error_status_op():
                    await self._task_manager.update_sync_status(
                        exchange_id,
                        data_type,
                        SyncStatus.FAILED,
                        f"Error: {type(e).__name__}: {str(e)}"
                    )
                await execute_with_retry(update_error_status_op)
            except Exception as inner_e:
                logger.error(f"[{loop_thread_key}] Error updating sync status to failed: {str(inner_e)}")
    
    def trigger_sync(self, exchange_id: str, data_type: Union[DataType, str]) -> bool:
        """
        Trigger a manual synchronization.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
            
        Returns:
            True if triggered, False otherwise
        """
        return self._task_manager.add_to_queue(exchange_id, data_type)