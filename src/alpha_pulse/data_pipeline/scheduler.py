"""
Exchange data synchronization scheduler.

This module provides a background scheduler for fetching and caching exchange data
at regular intervals to improve API performance.
"""
import asyncio
import signal
import sys
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum
import logging
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from alpha_pulse.data_pipeline.database.connection import get_pg_connection
from alpha_pulse.data_pipeline.database.exchange_cache import (
    ExchangeCacheRepository, SyncStatus, ExchangeSync
)
from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.types import ExchangeType
from alpha_pulse.exchanges.interfaces import BaseExchange


class DataType(str, Enum):
    """Types of data to sync."""
    ORDERS = "orders"
    BALANCES = "balances"
    POSITIONS = "positions"
    PRICES = "prices"
    ALL = "all"


class ExchangeDataSynchronizer:
    """Synchronizes exchange data with local cache."""
    
    def __init__(self):
        """Initialize synchronizer."""
        self._exchanges: Dict[str, BaseExchange] = {}
        self._running = False
        self._sync_tasks = set()
        self._manual_trigger = asyncio.Event()
        self._sync_interval_minutes = 60  # Default sync interval in minutes
        
    async def initialize(self):
        """Initialize exchanges."""
        logger.info("Initializing exchange data synchronizer")
        # For now, just initialize Bybit as the default exchange
        # In a real implementation, we would loop through all configured exchanges
        try:
            exchange_type = ExchangeType.BYBIT
            exchange = ExchangeFactory.create_exchange(exchange_type)
            await exchange.initialize()
            self._exchanges[exchange_type.value.lower()] = exchange
            logger.info(f"Initialized exchange: {exchange_type.value}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_type.value}: {e}")
            
    async def close(self):
        """Close all exchange connections."""
        for exchange_id, exchange in self._exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed exchange connection: {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing exchange {exchange_id}: {e}")
    
    async def start(self):
        """Start synchronization scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
            
        self._running = True
        asyncio.create_task(self._scheduler_loop())
        logger.info("Exchange data synchronization scheduler started")
    
    async def stop(self):
        """Stop synchronization scheduler."""
        self._running = False
        
        # Cancel all running sync tasks
        for task in self._sync_tasks:
            task.cancel()
            
        logger.info("Exchange data synchronization scheduler stopped")
    
    def trigger_sync(self, exchange_id: Optional[str] = None, data_type: Optional[DataType] = None):
        """Manually trigger synchronization."""
        self._manual_trigger.set()
        logger.info(f"Manual sync triggered for {exchange_id or 'all exchanges'}, data type: {data_type or 'all'}")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for pending syncs in the database
                await self._check_scheduled_syncs()
                
                # Wait for the next interval or manual trigger
                try:
                    await asyncio.wait_for(
                        self._manual_trigger.wait(),
                        timeout=60  # Check every minute
                    )
                    # Clear the trigger if it was set
                    self._manual_trigger.clear()
                except asyncio.TimeoutError:
                    # This is normal, just continue
                    pass
                    
            except asyncio.CancelledError:
                logger.info("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                # Don't crash the loop - wait and try again
                await asyncio.sleep(10)
    
    async def _check_scheduled_syncs(self):
        """Check for scheduled syncs that need to be run."""
        async with get_pg_connection() as conn:
            repository = ExchangeCacheRepository(conn)
            pending_syncs = await repository.get_pending_syncs()
            
            for sync in pending_syncs:
                # Skip if already in progress
                if sync.status == SyncStatus.IN_PROGRESS:
                    continue
                    
                # Check if exchange is available
                exchange_id = sync.exchange_id
                if exchange_id not in self._exchanges:
                    logger.warning(f"Exchange {exchange_id} not initialized, skipping sync")
                    continue
                
                # Schedule the sync task
                task = asyncio.create_task(self._run_sync(exchange_id, sync.data_type))
                self._sync_tasks.add(task)
                task.add_done_callback(lambda t: self._sync_tasks.remove(t))
    
    async def _run_sync(self, exchange_id: str, data_type: str):
        """Run a synchronization task."""
        logger.info(f"Starting sync for {exchange_id}, data type: {data_type}")
        exchange = self._exchanges.get(exchange_id)
        if not exchange:
            logger.error(f"Exchange {exchange_id} not found")
            return
            
        async with get_pg_connection() as conn:
            repository = ExchangeCacheRepository(conn)
            
            # Update sync status to in progress
            await repository.create_or_update_sync(
                exchange_id=exchange_id,
                data_type=data_type,
                status=SyncStatus.IN_PROGRESS
            )
            
            try:
                record_count = 0
                
                # Perform the appropriate sync based on data type
                if data_type == DataType.ORDERS:
                    record_count = await self._sync_orders(exchange, exchange_id, repository)
                elif data_type == DataType.BALANCES:
                    record_count = await self._sync_balances(exchange, exchange_id, repository)
                elif data_type == DataType.POSITIONS:
                    record_count = await self._sync_positions(exchange, exchange_id, repository)
                elif data_type == DataType.PRICES:
                    record_count = await self._sync_prices(exchange, exchange_id, repository)
                elif data_type == DataType.ALL:
                    # Sync all data types
                    order_count = await self._sync_orders(exchange, exchange_id, repository)
                    balance_count = await self._sync_balances(exchange, exchange_id, repository)
                    position_count = await self._sync_positions(exchange, exchange_id, repository)
                    price_count = await self._sync_prices(exchange, exchange_id, repository)
                    record_count = order_count + balance_count + position_count + price_count
                
                # Update sync status to completed
                next_sync = datetime.now(timezone.utc) + timedelta(minutes=self._sync_interval_minutes)
                await repository.create_or_update_sync(
                    exchange_id=exchange_id,
                    data_type=data_type,
                    status=SyncStatus.COMPLETED,
                    next_sync=next_sync,
                    record_count=record_count
                )
                
                logger.info(f"Completed sync for {exchange_id}, data type: {data_type}, records: {record_count}")
                
            except Exception as e:
                error_msg = f"Error during sync for {exchange_id}, data type: {data_type}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # Update sync status to failed
                await repository.create_or_update_sync(
                    exchange_id=exchange_id,
                    data_type=data_type,
                    status=SyncStatus.FAILED,
                    error_message=error_msg
                )
    
    async def _sync_orders(self, exchange: BaseExchange, exchange_id: str, repository: ExchangeCacheRepository) -> int:
        """Sync order history for exchange."""
        try:
            orders = await exchange.get_order_history()
            count = await repository.save_orders(exchange_id, orders)
            logger.info(f"Synced {count} orders for {exchange_id}")
            return count
        except Exception as e:
            logger.error(f"Error syncing orders for {exchange_id}: {e}")
            return 0
    
    async def _sync_balances(self, exchange: BaseExchange, exchange_id: str, repository: ExchangeCacheRepository) -> int:
        """Sync balances for exchange."""
        try:
            balances = await exchange.get_balances()
            count = await repository.save_balances(exchange_id, balances)
            logger.info(f"Synced {count} balances for {exchange_id}")
            return count
        except Exception as e:
            logger.error(f"Error syncing balances for {exchange_id}: {e}")
            return 0
    
    async def _sync_positions(self, exchange: BaseExchange, exchange_id: str, repository: ExchangeCacheRepository) -> int:
        """Sync positions for exchange."""
        try:
            positions = await exchange.get_positions()
            count = await repository.save_positions(exchange_id, positions)
            logger.info(f"Synced {count} positions for {exchange_id}")
            return count
        except Exception as e:
            logger.error(f"Error syncing positions for {exchange_id}: {e}")
            return 0
    
    async def _sync_prices(self, exchange: BaseExchange, exchange_id: str, repository: ExchangeCacheRepository) -> int:
        """Sync price data for exchange."""
        try:
            # First, get all positions to determine which symbols to fetch prices for
            positions = await exchange.get_positions()
            count = 0
            
            # Fetch and save prices for each symbol
            for symbol in positions.keys():
                try:
                    # Get price for symbol
                    price = await exchange.get_ticker_price(f"{symbol}/USDT")
                    if price:
                        await repository.save_price(exchange_id, symbol, price)
                        count += 1
                except Exception as e:
                    logger.error(f"Error fetching price for {symbol}: {e}")
            
            logger.info(f"Synced {count} prices for {exchange_id}")
            return count
        except Exception as e:
            logger.error(f"Error syncing prices for {exchange_id}: {e}")
            return 0


# Singleton instance
exchange_data_synchronizer = ExchangeDataSynchronizer()


@asynccontextmanager
async def lifespan_handler():
    """Handle application startup and shutdown."""
    # Initialize synchronizer
    await exchange_data_synchronizer.initialize()
    
    # Start scheduler
    await exchange_data_synchronizer.start()
    
    # Yield to FastAPI application
    yield
    
    # Clean up on shutdown
    await exchange_data_synchronizer.stop()
    await exchange_data_synchronizer.close()


def manual_sync(exchange_id: Optional[str] = None, data_type: Optional[DataType] = None):
    """Manually trigger a sync operation."""
    exchange_data_synchronizer.trigger_sync(exchange_id, data_type)