#!/usr/bin/env python3
"""
Test script to verify the database connection pool fixes in the AlphaPulse data pipeline synchronizer.
This script simulates multiple concurrent Bybit exchange synchronization operations to test the robustness
of the new connection pool implementation.
"""
import asyncio
import time
import random
from datetime import datetime, timezone
import sys
from loguru import logger

from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.scheduler.sync_module.synchronizer import ExchangeDataSynchronizer
from alpha_pulse.data_pipeline.database.connection_manager import initialize_connection_pools, close_all_pools, get_loop_thread_key

# Configure logging
logger.remove()
logger.add(
    sys.stdout, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("bybit_sync_test.log", rotation="50 MB", level="DEBUG")

async def trigger_sync(synchronizer, exchange_id, data_type, delay=0):
    """Trigger a sync operation with optional delay."""
    if delay > 0:
        await asyncio.sleep(delay)
    
    loop_thread_key = get_loop_thread_key()
    logger.info(f"[{loop_thread_key}] Triggering sync for {exchange_id}, type: {data_type}")
    synchronizer.trigger_sync(exchange_id, data_type)
    logger.info(f"[{loop_thread_key}] Sync triggered for {exchange_id}, type: {data_type}")

async def run_concurrent_syncs():
    """Run multiple concurrent sync operations to test connection pool handling."""
    # Initialize connection pools
    await initialize_connection_pools()
    
    # Create synchronizer
    synchronizer = ExchangeDataSynchronizer(sync_interval=60)  # Short interval for testing
    
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting concurrent sync test at: {start_time}")
    
    try:
        # Create tasks for different data types with slightly staggered starts
        tasks = [
            asyncio.create_task(trigger_sync(synchronizer, "bybit", DataType.BALANCES, delay=0.1)),
            asyncio.create_task(trigger_sync(synchronizer, "bybit", DataType.POSITIONS, delay=0.2)),
            asyncio.create_task(trigger_sync(synchronizer, "bybit", DataType.ORDERS, delay=0.3)),
            asyncio.create_task(trigger_sync(synchronizer, "bybit", DataType.PRICES, delay=0.4)),
            # Trigger ALL sync which will test handling multiple operations
            asyncio.create_task(trigger_sync(synchronizer, "bybit", DataType.ALL, delay=0.5)),
        ]
        
        # Wait for all trigger tasks to complete
        await asyncio.gather(*tasks)
        logger.info("All sync operations triggered")
        
        # Wait some time for the syncs to complete (they'll run in the background thread)
        test_duration = 30  # seconds
        logger.info(f"Waiting {test_duration} seconds for sync operations to complete...")
        
        for i in range(test_duration):
            if i % 5 == 0:
                logger.info(f"Elapsed time: {i}s")
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"Error during concurrent sync test: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")
    finally:
        # Attempt to shut down the synchronizer
        logger.info("Stopping synchronizer...")
        synchronizer.stop()
        
        # Close all connection pools
        logger.info("Closing all database connection pools...")
        await close_all_pools()
    
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Finished concurrent sync test at: {end_time}")
    logger.info(f"Total test duration: {duration:.2f} seconds")
    logger.info("Check the log file 'bybit_sync_test.log' for detailed logs")

if __name__ == "__main__":
    logger.info("=== Bybit Connection Pool Test ===")
    logger.info("Testing concurrent database operations with the new connection pool")
    
    try:
        asyncio.run(run_concurrent_syncs())
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {repr(e)}")