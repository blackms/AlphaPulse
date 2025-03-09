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

from loguru import logger

from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.scheduler.sync_module.types import SyncStatus
from alpha_pulse.data_pipeline.scheduler.sync_module.exchange_manager import ExchangeManager
from alpha_pulse.data_pipeline.scheduler.sync_module.data_sync import DataSynchronizer
from alpha_pulse.data_pipeline.scheduler.sync_module.task_manager import TaskManager
from alpha_pulse.data_pipeline.scheduler.sync_module.event_loop_manager import EventLoopManager
from alpha_pulse.data_pipeline.database.connection import get_pg_connection
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
            
        thread_id = threading.get_ident()
        logger.debug(f"[THREAD {thread_id}] Initializing ExchangeDataSynchronizer")
        
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
        
        logger.debug(f"[THREAD {thread_id}] ExchangeDataSynchronizer initialization complete")
    
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
        logger.info(f"[THREAD {thread_id}] Starting main loop for exchange data synchronization")
        
        # Get the current event loop for this thread
        loop = EventLoopManager.get_or_create_event_loop()
        
        # Import the close_all_connections function to ensure proper cleanup
        from alpha_pulse.data_pipeline.database.connection import close_all_connections
        
        iteration = 0
        while not self._should_stop.is_set():
            iteration += 1
            logger.debug(f"[THREAD {thread_id}] Main loop iteration {iteration}")
            
            try:
                # Process any queued sync requests
                logger.debug(f"[THREAD {thread_id}] Processing sync queue")
                await self._process_sync_queue()
                
                # Check for scheduled syncs
                logger.debug(f"[THREAD {thread_id}] Checking scheduled syncs")
                await self._check_scheduled_syncs()
                
                # Sleep for a bit to avoid high CPU usage
                logger.debug(f"[THREAD {thread_id}] Sleeping for 5 seconds")
                await asyncio.sleep(5)
                logger.debug(f"[THREAD {thread_id}] Async sleep completed")
            except asyncio.CancelledError:
                logger.warning(f"[THREAD {thread_id}] Main loop task was cancelled")
                break
            except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as db_error:
                # Handle database connection errors
                error_msg = str(db_error)
                logger.warning(f"[THREAD {thread_id}] Database connection error: {error_msg}")
                
                # Close all connections to force reconnection
                try:
                    await close_all_connections()
                    logger.info(f"[THREAD {thread_id}] Closed all database connections to force reconnection")
                except Exception as close_error:
                    logger.error(f"[THREAD {thread_id}] Error closing connections: {str(close_error)}")
                
                # Sleep before retry
                time.sleep(5)
            except RuntimeError as e:
                error_msg = str(e)
                logger.warning(f"[THREAD {thread_id}] Runtime error in main loop: {error_msg}")
                
                if "got Future" in error_msg and "attached to a different loop" in error_msg:
                    logger.warning(f"[THREAD {thread_id}] Event loop issue detected, resetting event loop")
                    # Reset the event loop for this thread
                    loop = EventLoopManager.reset_event_loop()
                    
                    # Close all connections to force reconnection with the new loop
                    try:
                        await close_all_connections()
                        logger.info(f"[THREAD {thread_id}] Closed all database connections after loop reset")
                    except Exception as close_error:
                        logger.error(f"[THREAD {thread_id}] Error closing connections: {str(close_error)}")
                    
                    # Sleep before retry
                    time.sleep(5)
                else:
                    logger.error(f"[THREAD {thread_id}] Unexpected runtime error in main loop: {error_msg}")
                    logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
                    logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
                    time.sleep(10)  # Sleep on error, but not too long
            except Exception as e:
                logger.error(f"[THREAD {thread_id}] Error in main loop: {str(e)}")
                logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
                logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
                time.sleep(10)  # Sleep on error, but not too long
        
        # Clean up database connections before exiting
        try:
            await close_all_connections()
            logger.info(f"[THREAD {thread_id}] Closed all database connections before exiting main loop")
        except Exception as close_error:
            logger.error(f"[THREAD {thread_id}] Error closing connections: {str(close_error)}")
        
        logger.info(f"[THREAD {thread_id}] Main loop exited after {iteration} iterations")
    
    async def _process_sync_queue(self):
        """Process any queued sync requests."""
        thread_id = threading.get_ident()
        
        # Get all items in the queue
        queue_items = self._task_manager.get_queue_items()
        if not queue_items:
            return
        
        # Process all items in the queue
        for exchange_id, data_type in queue_items:
            # Skip if already syncing this exchange and data type
            if self._task_manager.is_task_active(exchange_id, data_type):
                logger.info(f"[THREAD {thread_id}] Skipping already active sync: {exchange_id}_{data_type.value}")
                continue
            
            # Start a new sync task
            logger.info(f"[THREAD {thread_id}] Starting queued sync for {exchange_id}, type: {data_type}")
            
            try:
                # Run the sync directly instead of creating a task
                await self._sync_exchange_data(exchange_id, data_type)
            except Exception as e:
                logger.error(f"[THREAD {thread_id}] Error syncing {data_type} for {exchange_id}: {str(e)}")
                logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
                logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
    
    async def _check_scheduled_syncs(self):
        """Check for scheduled syncs that need to run."""
        thread_id = threading.get_ident()
        logger.debug(f"[THREAD {thread_id}] Checking scheduled syncs")
        
        try:
            # Get syncs that need to run
            to_sync = await self._task_manager.check_scheduled_syncs()
            
            # Add them to the queue
            for exchange_id, data_type in to_sync:
                self._task_manager.add_to_queue(exchange_id, data_type)
                logger.debug(f"[THREAD {thread_id}] Added to queue: {exchange_id}, {data_type}")
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error checking scheduled syncs: {str(e)}")
            logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
            logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
    
    async def _sync_exchange_data(self, exchange_id: str, data_type: DataType):
        """
        Synchronize data for a specific exchange and data type.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
        """
        thread_id = threading.get_ident()
        logger.debug(f"[THREAD {thread_id}] Starting sync for {exchange_id}, type: {data_type}")
        
        # Mark as in progress and set next sync time
        try:
            await self._task_manager.update_sync_status(
                exchange_id,
                data_type,
                SyncStatus.IN_PROGRESS
            )
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error updating initial sync status: {str(e)}")
        
        # Define a function to get a repository with a new connection
        async def get_repository():
            async with get_pg_connection() as conn:
                return ExchangeCacheRepository(conn)
        
        try:
            # Get exchange instance
            exchange = await self._exchange_manager.get_exchange(exchange_id)
            
            if not exchange:
                logger.error(f"[THREAD {thread_id}] Failed to get exchange for {exchange_id}. Troubleshooting steps:")
                logger.error("1. Check API credentials in environment variables or configuration")
                logger.error("2. Verify network connectivity to the exchange API")
                logger.error("3. Confirm testnet/mainnet settings are correct")
                logger.error("4. Check for any IP restrictions on your API key")
                logger.error("5. Run debug_bybit_auth.py or debug_exchange_connection.py for more details")
                
                # Update status to failed
                try:
                    await self._task_manager.update_sync_status(
                        exchange_id,
                        data_type,
                        SyncStatus.FAILED,
                        "Failed to initialize exchange"
                    )
                except Exception as status_error:
                    logger.error(f"[THREAD {thread_id}] Error updating sync status to failed: {str(status_error)}")
                return
            
            # Import the execute_with_retry function
            from alpha_pulse.data_pipeline.database.connection import _execute_with_retry
            
            # Synchronize based on data type
            sync_success = False
            try:
                if data_type == DataType.ALL:
                    # Sync all data types one by one with individual error handling
                    try:
                        # Get a fresh repository for each operation
                        repo = await _execute_with_retry(get_repository)
                        await self._data_synchronizer.sync_balances(exchange_id, exchange, repo)
                        logger.info(f"[THREAD {thread_id}] Successfully synced balances for {exchange_id}")
                    except Exception as e:
                        logger.error(f"[THREAD {thread_id}] Error syncing balances for {exchange_id}: {str(e)}")
                    
                    try:
                        # Get a fresh repository for each operation
                        repo = await _execute_with_retry(get_repository)
                        await self._data_synchronizer.sync_positions(exchange_id, exchange, repo)
                        logger.info(f"[THREAD {thread_id}] Successfully synced positions for {exchange_id}")
                    except Exception as e:
                        logger.error(f"[THREAD {thread_id}] Error syncing positions for {exchange_id}: {str(e)}")
                    
                    try:
                        # Get a fresh repository for each operation
                        repo = await _execute_with_retry(get_repository)
                        await self._data_synchronizer.sync_orders(exchange_id, exchange, repo)
                        logger.info(f"[THREAD {thread_id}] Successfully synced orders for {exchange_id}")
                    except Exception as e:
                        logger.error(f"[THREAD {thread_id}] Error syncing orders for {exchange_id}: {str(e)}")
                    
                    try:
                        # Get a fresh repository for each operation
                        repo = await _execute_with_retry(get_repository)
                        await self._data_synchronizer.sync_prices(exchange_id, exchange, repo)
                        logger.info(f"[THREAD {thread_id}] Successfully synced prices for {exchange_id}")
                    except Exception as e:
                        logger.error(f"[THREAD {thread_id}] Error syncing prices for {exchange_id}: {str(e)}")
                    
                    # Consider the sync successful if we got here
                    sync_success = True
                elif data_type == DataType.BALANCES:
                    repo = await _execute_with_retry(get_repository)
                    await self._data_synchronizer.sync_balances(exchange_id, exchange, repo)
                    sync_success = True
                elif data_type == DataType.POSITIONS:
                    repo = await _execute_with_retry(get_repository)
                    await self._data_synchronizer.sync_positions(exchange_id, exchange, repo)
                    sync_success = True
                elif data_type == DataType.ORDERS:
                    repo = await _execute_with_retry(get_repository)
                    await self._data_synchronizer.sync_orders(exchange_id, exchange, repo)
                    sync_success = True
                elif data_type == DataType.PRICES:
                    repo = await _execute_with_retry(get_repository)
                    await self._data_synchronizer.sync_prices(exchange_id, exchange, repo)
                    sync_success = True
            except Exception as sync_error:
                logger.error(f"[THREAD {thread_id}] Error during sync operation: {str(sync_error)}")
                logger.error(f"[THREAD {thread_id}] Exception type: {type(sync_error).__name__}")
                logger.error(f"[THREAD {thread_id}] Exception details: {repr(sync_error)}")
            
            # Update status to completed if successful
            if sync_success:
                try:
                    await self._task_manager.update_sync_status(
                        exchange_id,
                        data_type,
                        SyncStatus.COMPLETED
                    )
                    logger.info(f"[THREAD {thread_id}] Successfully completed sync for {exchange_id}, {data_type}")
                except Exception as status_error:
                    logger.error(f"[THREAD {thread_id}] Error updating sync status to completed: {str(status_error)}")
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error syncing {data_type} for {exchange_id}: {str(e)}")
            # Update status to failed
            try:
                await self._task_manager.update_sync_status(
                    exchange_id,
                    data_type,
                    SyncStatus.FAILED,
                    str(e)
                )
            except Exception as inner_e:
                logger.error(f"[THREAD {thread_id}] Error updating sync status to failed: {str(inner_e)}")
    
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