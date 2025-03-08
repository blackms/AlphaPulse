"""
Exchange data synchronization module.

This module provides functionality to synchronize exchange data on a regular schedule
or on demand, serving as a background worker for the portfolio API.
"""
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Union
import threading
import signal
from enum import Enum

from loguru import logger

from alpha_pulse.data_pipeline.database.connection import get_pg_connection
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository
from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.exchanges.types import ExchangeType
from alpha_pulse.data_pipeline.scheduler import DataType

# Type alias for backward compatibility
Exchange = BaseExchange


class SyncStatus(Enum):
    """Status of a synchronization task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExchangeDataSynchronizer:
    """
    Synchronizes exchange data on a regular schedule.
    
    This class schedules and executes periodic data synchronization tasks for 
    exchange data such as balances, positions, orders, and prices. It can also 
    be triggered manually to force a synchronization.
    """
    
    def __init__(self, sync_interval: int = 3600):
        """
        Initialize the synchronizer.
        
        Args:
            sync_interval: Interval between syncs in seconds (default: 1 hour)
        """
        self._sync_interval = sync_interval
        self._should_stop = threading.Event()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._exchange_instances: Dict[str, Exchange] = {}
        self._worker_thread = None
        self._running = False
        self._last_sync: Dict[str, Dict[str, datetime]] = {}
        self._sync_queue: Set[tuple] = set()  # (exchange_id, data_type)
        
        # Start the background task
        self.start()
        
        # Register shutdown handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Start the main task
            main_task = loop.create_task(self._main_loop())
            
            # Run until should_stop is set
            loop.run_until_complete(main_task)
        except Exception as e:
            logger.error(f"Error in event loop: {str(e)}")
        finally:
            # Clean up
            for task_name, task in self._active_tasks.items():
                if not task.done():
                    logger.warning(f"Cancelling task: {task_name}")
                    task.cancel()
            
            # Close the loop
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            except Exception as e:
                logger.error(f"Error closing event loop: {str(e)}")
    
    async def _main_loop(self):
        """Main loop that checks for tasks to run."""
        logger.info("Starting main loop for exchange data synchronization")
        
        while not self._should_stop.is_set():
            try:
                # Process any queued sync requests
                await self._process_sync_queue()
                
                # Check for scheduled syncs
                await self._check_scheduled_syncs()
                
                # Sleep for a bit to avoid high CPU usage
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(30)  # Sleep longer on error
        
        logger.info("Main loop exited")
    
    async def _process_sync_queue(self):
        """Process any queued sync requests."""
        if not self._sync_queue:
            return
        
        # Process all items in the queue
        queue_items = list(self._sync_queue)
        self._sync_queue.clear()
        
        for exchange_id, data_type in queue_items:
            # Skip if already syncing this exchange and data type
            task_key = f"{exchange_id}_{data_type.value}"
            if task_key in self._active_tasks and not self._active_tasks[task_key].done():
                logger.info(f"Skipping already active sync: {task_key}")
                continue
            
            # Start a new sync task
            logger.info(f"Starting queued sync for {exchange_id}, type: {data_type}")
            self._active_tasks[task_key] = asyncio.create_task(
                self._sync_exchange_data(exchange_id, data_type)
            )
    
    async def _check_scheduled_syncs(self):
        """Check for scheduled syncs that need to run."""
        now = datetime.now(timezone.utc)
        
        # Get all exchanges from environment or config
        exchanges = self._get_configured_exchanges()
        
        async with get_pg_connection() as conn:
            repo = ExchangeCacheRepository(conn)
            sync_statuses = await repo.get_all_sync_status()
            
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
                        logger.info(f"Scheduling sync for {exchange_id}, type: {data_type}")
                        self._sync_queue.add((exchange_id, data_type))
    
    def _get_configured_exchanges(self) -> List[str]:
        """Get the list of configured exchanges from environment or config."""
        # For now, just return the exchange from environment
        exchange_type = os.environ.get('EXCHANGE_TYPE', 'bybit').lower()
        return [exchange_type]
    
    async def _get_exchange(self, exchange_id: str) -> Optional[Exchange]:
        """
        Get or create an exchange instance.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Exchange instance or None if failed
        """
        # Return cached instance if available
        if exchange_id in self._exchange_instances:
            return self._exchange_instances[exchange_id]
        
        try:
            # Get API credentials from environment
            api_key = os.environ.get(f'{exchange_id.upper()}_API_KEY', 
                                     os.environ.get('EXCHANGE_API_KEY', ''))
            api_secret = os.environ.get(f'{exchange_id.upper()}_API_SECRET', 
                                       os.environ.get('EXCHANGE_API_SECRET', ''))
            testnet = os.environ.get(f'{exchange_id.upper()}_TESTNET', 
                                    os.environ.get('EXCHANGE_TESTNET', 'true')).lower() == 'true'
            
            # For Bybit, we need special handling for testnet
            if exchange_id.lower() == 'bybit':
                if 'BYBIT_TESTNET' in os.environ:
                    testnet = os.environ.get('BYBIT_TESTNET', '').lower() == 'true'
                    logger.info(f"Using Bybit-specific testnet setting: {testnet}")
                else:
                    # Default to mainnet for Bybit
                    testnet = False
                    logger.info("No testnet setting for Bybit, defaulting to mainnet")
            
            # Determine exchange type
            exchange_type = None
            if exchange_id.lower() == 'binance':
                exchange_type = ExchangeType.BINANCE
            elif exchange_id.lower() == 'bybit':
                exchange_type = ExchangeType.BYBIT
            else:
                logger.warning(f"Unknown exchange type: {exchange_id}, assuming Binance")
                exchange_type = ExchangeType.BINANCE
            
            # Create exchange instance
            logger.info(f"Creating exchange of type {exchange_type} with testnet={testnet}")
            exchange = ExchangeFactory.create_exchange(
                exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
            
            # Initialize the exchange
            await exchange.initialize()
            
            # Cache the instance
            self._exchange_instances[exchange_id] = exchange
            
            return exchange
        except Exception as e:
            logger.error(f"Error creating exchange {exchange_id}: {str(e)}")
            return None
    
    async def _sync_exchange_data(self, exchange_id: str, data_type: DataType):
        """
        Synchronize data for a specific exchange and data type.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
        """
        # Mark as in progress and set next sync time
        next_sync_time = datetime.now(timezone.utc) + timedelta(seconds=self._sync_interval)
        
        try:
            # Update status
            async with get_pg_connection() as conn:
                repo = ExchangeCacheRepository(conn)
                await repo.update_sync_status(
                    exchange_id, 
                    data_type.value, 
                    SyncStatus.IN_PROGRESS.value,
                    next_sync_time
                )
                
                # Get exchange instance
                exchange = await self._get_exchange(exchange_id)
                if not exchange:
                    logger.error(f"Failed to get exchange for {exchange_id}")
                    await repo.update_sync_status(
                        exchange_id, 
                        data_type.value, 
                        SyncStatus.FAILED.value,
                        next_sync_time,
                        "Failed to initialize exchange"
                    )
                    return
                
                # Synchronize based on data type
                if data_type == DataType.ALL:
                    # Sync all data types
                    await self._sync_balances(exchange_id, exchange, repo)
                    await self._sync_positions(exchange_id, exchange, repo)
                    await self._sync_orders(exchange_id, exchange, repo)
                    await self._sync_prices(exchange_id, exchange, repo)
                elif data_type == DataType.BALANCES:
                    await self._sync_balances(exchange_id, exchange, repo)
                elif data_type == DataType.POSITIONS:
                    await self._sync_positions(exchange_id, exchange, repo)
                elif data_type == DataType.ORDERS:
                    await self._sync_orders(exchange_id, exchange, repo)
                elif data_type == DataType.PRICES:
                    await self._sync_prices(exchange_id, exchange, repo)
                
                # Update status to completed
                await repo.update_sync_status(
                    exchange_id, 
                    data_type.value, 
                    SyncStatus.COMPLETED.value,
                    next_sync_time
                )
                
        except Exception as e:
            logger.error(f"Error syncing {data_type} for {exchange_id}: {str(e)}")
            # Update status to failed
            try:
                async with get_pg_connection() as conn:
                    repo = ExchangeCacheRepository(conn)
                    await repo.update_sync_status(
                        exchange_id, 
                        data_type.value, 
                        SyncStatus.FAILED.value,
                        next_sync_time,
                        str(e)
                    )
            except Exception as inner_e:
                logger.error(f"Error updating sync status: {str(inner_e)}")
    
    async def _sync_balances(self, exchange_id: str, exchange: Exchange, repo: ExchangeCacheRepository):
        """Synchronize balance data."""
        logger.info(f"Syncing balances for {exchange_id}")
        
        try:
            # Get balances from exchange
            balances = await exchange.get_balances()
            
            # Store in database
            await repo.store_balances(exchange_id, balances)
            
            logger.info(f"Successfully synced balances for {exchange_id}")
        except Exception as e:
            logger.error(f"Error syncing balances for {exchange_id}: {str(e)}")
            raise
    
    async def _sync_positions(self, exchange_id: str, exchange: Exchange, repo: ExchangeCacheRepository):
        """Synchronize position data."""
        logger.info(f"Syncing positions for {exchange_id}")
        
        try:
            # Get positions from exchange
            positions = await exchange.get_positions()
            
            # Convert positions to dictionary format expected by repository
            positions_dict = {}
            for position in positions:
                # Skip positions with zero quantity
                if position.quantity == 0:
                    continue
                    
                positions_dict[position.symbol] = {
                    "symbol": position.symbol,
                    "quantity": float(position.quantity),
                    "entry_price": float(position.avg_entry_price) if position.avg_entry_price else None,
                    "current_price": float(position.current_price) if hasattr(position, 'current_price') else None,
                    "unrealized_pnl": float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else None,
                }
            
            # Store in database
            await repo.store_positions(exchange_id, positions_dict)
            
            logger.info(f"Successfully synced positions for {exchange_id}")
        except Exception as e:
            logger.error(f"Error syncing positions for {exchange_id}: {str(e)}")
            raise
    
    async def _sync_orders(self, exchange_id: str, exchange: Exchange, repo: ExchangeCacheRepository):
        """Synchronize order data."""
        logger.info(f"Syncing orders for {exchange_id}")
        
        try:
            # Get all symbols with positions
            positions = await exchange.get_positions()
            symbols = {position.symbol for position in positions if position.quantity != 0}
            
            # Add some default symbols
            symbols.add("BTC/USDT")
            symbols.add("ETH/USDT")
            
            # Get orders for each symbol
            all_orders = []
            for symbol in symbols:
                try:
                    # Get orders for this symbol
                    symbol_orders = await exchange.get_orders(symbol)
                    all_orders.extend(symbol_orders)
                    logger.info(f"Got {len(symbol_orders)} orders for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting orders for {symbol}: {str(e)}")
            
            # Store in database
            await repo.store_orders(exchange_id, all_orders)
            
            logger.info(f"Successfully synced {len(all_orders)} orders for {exchange_id}")
        except Exception as e:
            logger.error(f"Error syncing orders for {exchange_id}: {str(e)}")
            raise
    
    async def _sync_prices(self, exchange_id: str, exchange: Exchange, repo: ExchangeCacheRepository):
        """Synchronize price data."""
        logger.info(f"Syncing prices for {exchange_id}")
        
        try:
            # Get all symbols with positions
            positions = await exchange.get_positions()
            symbols = {position.symbol for position in positions if position.quantity != 0}
            
            # Add some default symbols
            symbols.add("BTC/USDT")
            symbols.add("ETH/USDT")
            
            # Get prices for each symbol
            for symbol in symbols:
                try:
                    # Get price for this symbol
                    ticker = await exchange.get_ticker(symbol)
                    if ticker and 'last' in ticker:
                        price = float(ticker['last'])
                        quote_currency = symbol.split('/')[-1] if '/' in symbol else "USDT"
                        await repo.store_price(
                            exchange_id,
                            symbol.split('/')[0] if '/' in symbol else symbol,
                            quote_currency,
                            price
                        )
                        logger.info(f"Got price for {symbol}: {price}")
                except Exception as e:
                    logger.error(f"Error getting price for {symbol}: {str(e)}")
            
            logger.info(f"Successfully synced prices for {exchange_id}")
        except Exception as e:
            logger.error(f"Error syncing prices for {exchange_id}: {str(e)}")
            raise
    
    def trigger_sync(self, exchange_id: str, data_type: Union[DataType, str]) -> bool:
        """
        Trigger a manual synchronization.
        
        Args:
            exchange_id: Exchange identifier
            data_type: Type of data to synchronize
            
        Returns:
            True if triggered, False otherwise
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
        logger.info(f"Manually triggering sync for {exchange_id}, type: {data_type}")
        self._sync_queue.add((exchange_id, data_type))
        return True