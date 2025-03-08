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
from alpha_pulse.exchanges.interfaces import BaseExchange, ConnectionError, ExchangeError
from alpha_pulse.exchanges.types import ExchangeType
from alpha_pulse.data_pipeline.scheduler import DataType

# Import ccxt for AuthenticationError
import ccxt.async_support as ccxt

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
        
        # Get the current event loop for this thread
        loop = asyncio.get_running_loop()
        
        while not self._should_stop.is_set():
            try:
                # Process any queued sync requests
                await self._process_sync_queue()
                
                # Check for scheduled syncs
                await self._check_scheduled_syncs()
                
                # Sleep for a bit to avoid high CPU usage
                await asyncio.sleep(5)
            except RuntimeError as e:
                if "got Future" in str(e) and "attached to a different loop" in str(e):
                    logger.warning("Event loop issue detected, using thread-specific sleep")
                    # Use thread-specific sleep instead of asyncio.sleep
                    time.sleep(5)
                elif "cannot perform operation" in str(e) and "another operation is in progress" in str(e):
                    logger.warning("Concurrent operation issue detected, using thread-specific sleep")
                    time.sleep(5)
                else:
                    logger.error(f"Runtime error in main loop: {str(e)}")
                    time.sleep(30)  # Sleep longer on error
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                try:
                    await asyncio.sleep(30)  # Sleep longer on error
                except Exception:
                    time.sleep(30)  # Fallback to regular sleep
        
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
        # Check if circuit breaker is active
        circuit_breaker_key = f"{exchange_id}_circuit_breaker"
        circuit_breaker_count = getattr(self, circuit_breaker_key, 0)
        circuit_breaker_time_key = f"{circuit_breaker_key}_time"
        circuit_breaker_time = getattr(self, circuit_breaker_time_key, 0)
        
        # If circuit breaker is active and cooldown period hasn't expired, return None
        if circuit_breaker_count >= 5 and circuit_breaker_time > 0:
            cooldown_period = 600  # 10 minutes in seconds
            current_time = time.time()
            if current_time - circuit_breaker_time < cooldown_period:
                remaining_time = int(cooldown_period - (current_time - circuit_breaker_time))
                logger.warning(f"Circuit breaker active for {exchange_id}. Will not attempt initialization for {remaining_time} more seconds.")
                return None
            else:
                # Reset circuit breaker after cooldown period
                logger.info(f"Circuit breaker cooldown period expired for {exchange_id}. Resetting circuit breaker.")
                setattr(self, circuit_breaker_key, 0)
                setattr(self, circuit_breaker_time_key, 0)
        
        # Return cached instance if available
        if exchange_id in self._exchange_instances:
            logger.debug(f"Using cached exchange instance for {exchange_id}")
            return self._exchange_instances[exchange_id]
        
        try:
            # Get API credentials from environment
            api_key = os.environ.get(f'{exchange_id.upper()}_API_KEY', 
                                     os.environ.get('EXCHANGE_API_KEY', ''))
            api_secret = os.environ.get(f'{exchange_id.upper()}_API_SECRET', 
                                       os.environ.get('EXCHANGE_API_SECRET', ''))
            testnet = os.environ.get(f'{exchange_id.upper()}_TESTNET', 
                                    os.environ.get('EXCHANGE_TESTNET', 'true')).lower() == 'true'
            
            # Log credential information (masked for security)
            if api_key:
                masked_key = api_key[:4] + '...' + api_key[-4:] if len(api_key) > 8 else '***'
                masked_secret = api_secret[:4] + '...' + api_secret[-4:] if len(api_secret) > 8 else '***'
                logger.info(f"Initializing {exchange_id} exchange with API key: {masked_key}")
                logger.debug(f"API secret (masked): {masked_secret}")
            else:
                logger.warning(f"No API key found for {exchange_id}. Check your environment variables.")
                logger.info(f"Expected environment variables: {exchange_id.upper()}_API_KEY or EXCHANGE_API_KEY")
            
            logger.info(f"Using testnet mode: {testnet}")
            
            # For Bybit, always use mainnet (testnet=False)
            if exchange_id.lower() == 'bybit':
                testnet = False
                logger.info("Using mainnet mode for Bybit (testnet=False)")
            
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
            logger.info(f"Creating exchange of type {exchange_type.value} with testnet={testnet}")
            exchange = ExchangeFactory.create_exchange(
                exchange_type,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
            
            # Initialize the exchange with retry logic
            max_retries = 3
            base_retry_delay = 2
            retry_delay = base_retry_delay
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Initializing {exchange_id} exchange (attempt {attempt}/{max_retries})")
                    
                    # Add timeout to prevent hanging indefinitely
                    init_timeout = 30  # 30 seconds timeout for initialization
                    
                    # Create a task with timeout
                    try:
                        await asyncio.wait_for(exchange.initialize(), timeout=init_timeout)
                        logger.info(f"Successfully initialized {exchange_id} exchange")
                        break
                    except asyncio.TimeoutError:
                        logger.warning(f"Initialization timed out after {init_timeout} seconds (attempt {attempt}/{max_retries})")
                        raise ConnectionError(f"Initialization timed out after {init_timeout} seconds")
                    
                except ccxt.NetworkError as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Network error initializing {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except ccxt.ExchangeNotAvailable as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Exchange not available error initializing {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except ccxt.RequestTimeout as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"Request timeout initializing {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except Exception as init_error:
                    last_error = init_error
                    if attempt < max_retries:
                        logger.warning(f"Failed to initialize {exchange_id} exchange (attempt {attempt}/{max_retries}): {str(init_error)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            
            # Check if all attempts failed
            if last_error is not None and attempt == max_retries:
                logger.error(f"Failed to initialize {exchange_id} exchange after {max_retries} attempts: {str(last_error)}")
                raise last_error
            
            # Cache the instance
            self._exchange_instances[exchange_id] = exchange
            
            return exchange
            
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication error creating exchange {exchange_id}: {str(e)}")
            logger.info("This may be due to invalid API credentials or insufficient permissions")
            logger.info("Check your API key and secret, and ensure they have the necessary permissions")
            
            # Add more detailed troubleshooting information
            logger.info("Troubleshooting steps for authentication issues:")
            logger.info("1. Verify that your API key and secret are correct")
            logger.info("2. Check if your API key has expired or been revoked")
            logger.info("3. Ensure your API key has the necessary permissions (read access at minimum)")
            logger.info("4. Verify that you're using the correct testnet/mainnet setting")
            logger.info("5. Check if there are IP restrictions on your API key")
            logger.info(f"6. Run 'python debug_bybit_auth.py' to test authentication specifically")
            
            # Add information about environment variables
            logger.info("Environment variables to check:")
            logger.info(f"- {exchange_id.upper()}_API_KEY or EXCHANGE_API_KEY")
            logger.info(f"- {exchange_id.upper()}_API_SECRET or EXCHANGE_API_SECRET")
            logger.info(f"- {exchange_id.upper()}_TESTNET or EXCHANGE_TESTNET")
            
            return None
            
        except ConnectionError as e:
            logger.error(f"Connection error creating exchange {exchange_id}: {str(e)}")
            logger.info("This may be due to network connectivity issues or API endpoint problems")
            logger.info("Check your internet connection and firewall settings")
            
            # Add more detailed troubleshooting information
            logger.info("Troubleshooting steps for connection issues:")
            logger.info("1. Verify that you can access the exchange API in your browser")
            logger.info("2. Check if your network has any firewall rules blocking the exchange API")
            logger.info("3. Try using a different network connection")
            logger.info("4. Check if the exchange API status page reports any outages")
            logger.info(f"5. Run 'python debug_bybit_api.py' to diagnose API connectivity issues")
            
            # Add information about the specific endpoint that failed
            if "query-info" in str(e):
                logger.info("The asset/coin/query-info endpoint is failing, which is used during initialization")
                logger.info("This endpoint may be temporarily unavailable or rate-limited")
            
            return None
            
        except Exception as e:
            error_msg = f"Error creating exchange {exchange_id}: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Exception details: {repr(e)}")
            
            # Implement circuit breaker pattern
            circuit_breaker_key = f"{exchange_id}_circuit_breaker"
            circuit_breaker_count = getattr(self, circuit_breaker_key, 0) + 1
            setattr(self, circuit_breaker_key, circuit_breaker_count)
            
            # If we've failed too many times, implement a circuit breaker
            max_failures = 5
            if circuit_breaker_count >= max_failures:
                circuit_breaker_time = time.time()
                setattr(self, f"{circuit_breaker_key}_time", circuit_breaker_time)
                logger.warning(f"Circuit breaker activated for {exchange_id} after {circuit_breaker_count} failures")
                logger.warning(f"Will not attempt to initialize {exchange_id} for 10 minutes")
                
                # Add detailed troubleshooting information
                logger.info("Troubleshooting steps for persistent exchange errors:")
                logger.info("1. Check the exchange status page for any reported issues")
                logger.info("2. Verify all API credentials and permissions")
                logger.info("3. Check your network connectivity to the exchange API")
                logger.info("4. Run the diagnostic tools in the DEBUG_TOOLS.md file")
                logger.info("5. Consider creating new API credentials if issues persist")
            else:
                # Add general troubleshooting information
                logger.info("General troubleshooting steps:")
                logger.info("1. Check the logs for specific error details")
                logger.info("2. Verify your exchange configuration")
                logger.info("3. Run 'python debug_exchange_connection.py' for more diagnostics")
            
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
            try:
                async with get_pg_connection() as conn:
                    repo = ExchangeCacheRepository(conn)
                    await repo.update_sync_status(
                        exchange_id,
                        data_type.value,
                        SyncStatus.IN_PROGRESS.value,
                        next_sync_time
                    )
            except RuntimeError as e:
                if "attached to a different loop" in str(e):
                    logger.warning(f"Event loop issue detected during status update, skipping status update")
                    # Continue with the sync even if we can't update the status
                else:
                    raise
                
            try:
                # Get exchange instance
                exchange = await self._get_exchange(exchange_id)
                if not exchange:
                    logger.error(f"Failed to get exchange for {exchange_id}. Troubleshooting steps:")
                    logger.error("1. Check API credentials in environment variables or configuration")
                    logger.error("2. Verify network connectivity to the exchange API")
                    logger.error("3. Confirm testnet/mainnet settings are correct")
                    logger.error("4. Check for any IP restrictions on your API key")
                    logger.error("5. Run debug_bybit_auth.py or debug_exchange_connection.py for more details")
                    
                    # Add more detailed troubleshooting information
                    logger.info("Detailed troubleshooting guide:")
                    logger.info(f"1. For {exchange_id.upper()}_API_KEY, ensure it's set correctly:")
                    logger.info(f"   export {exchange_id.upper()}_API_KEY=your_api_key")
                    logger.info(f"2. For {exchange_id.upper()}_API_SECRET, ensure it's set correctly:")
                    logger.info(f"   export {exchange_id.upper()}_API_SECRET=your_api_secret")
                    logger.info(f"3. For {exchange_id.upper()}_TESTNET, set to 'true' for testnet or 'false' for mainnet:")
                    logger.info(f"   export {exchange_id.upper()}_TESTNET=false")
                    logger.info("4. Check if the exchange API is accessible from your network:")
                    logger.info("   python debug_bybit_api.py")
                    logger.info("5. Verify your API key permissions and status in the exchange dashboard")
                    logger.info("6. Check the DEBUG_TOOLS.md file for more debugging options")
                    
                    try:
                        async with get_pg_connection() as conn:
                            repo = ExchangeCacheRepository(conn)
                            await repo.update_sync_status(
                                exchange_id,
                                data_type.value,
                                SyncStatus.FAILED.value,
                                next_sync_time,
                                "Failed to initialize exchange"
                            )
                    except Exception:
                        logger.warning(f"Could not update status for {exchange_id}, {data_type}")
                    return
                
                # Synchronize based on data type
                try:
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
                except RuntimeError as e:
                    if "attached to a different loop" in str(e):
                        logger.warning(f"Event loop issue detected during sync, skipping {data_type} sync for {exchange_id}")
                    else:
                        raise
                
                # Update status to completed
                try:
                    async with get_pg_connection() as conn:
                        repo = ExchangeCacheRepository(conn)
                        await repo.update_sync_status(
                            exchange_id,
                            data_type.value,
                            SyncStatus.COMPLETED.value,
                            next_sync_time
                        )
                except Exception:
                    logger.warning(f"Could not update completion status for {exchange_id}, {data_type}")
            except RuntimeError as e:
                if "attached to a different loop" in str(e):
                    logger.warning(f"Event loop issue detected during exchange operations, skipping sync for {exchange_id}")
                else:
                    raise
                
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
            
            # Convert Balance objects to dictionaries
            balance_dict = {}
            for currency, balance in balances.items():
                balance_dict[currency] = {
                    'available': float(balance.available),
                    'locked': float(balance.locked),
                    'total': float(balance.total)
                }
            
            # Store in database
            await repo.store_balances(exchange_id, balance_dict)
            
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
            
            # Handle different return types
            if isinstance(positions, dict):
                # If positions is already a dictionary
                positions_dict = positions
            else:
                # If positions is a list of objects
                for position in positions:
                    # Skip if position is a string
                    if isinstance(position, str):
                        logger.warning(f"Unexpected position format (string): {position}")
                        continue
                        
                    # Skip positions with zero quantity
                    try:
                        if hasattr(position, 'quantity') and position.quantity == 0:
                            continue
                            
                        if hasattr(position, 'symbol'):
                            symbol = position.symbol
                            positions_dict[symbol] = {
                                "symbol": symbol,
                                "quantity": float(position.quantity) if hasattr(position, 'quantity') else 0,
                                "entry_price": float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') and position.avg_entry_price else None,
                                "current_price": float(position.current_price) if hasattr(position, 'current_price') else None,
                                "unrealized_pnl": float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else None,
                            }
                    except Exception as pos_error:
                        logger.warning(f"Error processing position: {str(pos_error)}")
                        continue
            
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
            symbols = set()
            
            # Handle different return types for positions
            if isinstance(positions, dict):
                symbols = set(positions.keys())
            else:
                for position in positions:
                    if isinstance(position, str):
                        continue
                    if hasattr(position, 'symbol') and hasattr(position, 'quantity'):
                        if position.quantity != 0:
                            symbols.add(position.symbol)
            
            # Add some default symbols
            symbols.add("BTC/USDT")
            symbols.add("ETH/USDT")
            
            # Get orders for each symbol
            all_orders = []
            for symbol in symbols:
                try:
                    # Get orders for this symbol
                    symbol_orders = await exchange.get_orders(symbol)
                    
                    # Handle different return types
                    if isinstance(symbol_orders, list):
                        all_orders.extend(symbol_orders)
                        logger.info(f"Got {len(symbol_orders)} orders for {symbol}")
                    elif isinstance(symbol_orders, dict):
                        all_orders.append(symbol_orders)
                        logger.info(f"Got 1 order for {symbol}")
                    else:
                        logger.warning(f"Unexpected order format for {symbol}: {type(symbol_orders)}")
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