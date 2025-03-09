#!/usr/bin/env python3
"""
Test script for Bybit synchronization with improved connection handling.

This script tests concurrent synchronization operations to verify
the robustness of the connection handling improvements.
"""
import asyncio
import os
import sys
import time
from datetime import datetime
import threading

from loguru import logger

# Set up logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("bybit_sync_test_{time}.log", level="DEBUG", rotation="100 MB")

# Configure environment
os.environ["EXCHANGE_TYPE"] = "bybit"

# Import after setting environment
from alpha_pulse.data_pipeline.scheduler.sync_module.synchronizer import ExchangeDataSynchronizer
from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.database.connection_manager import close_all_pools


async def run_concurrent_syncs():
    """Run multiple concurrent sync operations to test connection handling."""
    logger.info(f"[Thread {threading.get_ident()}] Starting concurrent sync test")
    
    # Create synchronizer
    logger.info(f"[Thread {threading.get_ident()}] Creating synchronizer")
    synchronizer = ExchangeDataSynchronizer()
    
    # Trigger multiple syncs concurrently
    logger.info(f"[Thread {threading.get_ident()}] Triggering multiple concurrent syncs")
    
    # Trigger sync for different data types
    for data_type in [DataType.BALANCES, DataType.POSITIONS, DataType.ORDERS, DataType.PRICES]:
        synchronizer.trigger_sync("bybit", data_type)
        logger.info(f"[Thread {threading.get_ident()}] Triggered sync for {data_type}")
    
    # Also trigger an ALL sync
    synchronizer.trigger_sync("bybit", DataType.ALL)
    logger.info(f"[Thread {threading.get_ident()}] Triggered ALL sync")
    
    # Wait for syncs to complete
    logger.info(f"[Thread {threading.get_ident()}] Waiting for syncs to complete (up to 2 minutes)")
    for i in range(24):  # 2 minutes (24 * 5 seconds)
        await asyncio.sleep(5)
        logger.info(f"[Thread {threading.get_ident()}] Still waiting... ({i+1}/24)")
    
    logger.info(f"[Thread {threading.get_ident()}] Test completed")


async def run_in_new_thread():
    """Run an additional sync in a new thread to test thread isolation."""
    thread = threading.Thread(target=lambda: asyncio.run(thread_sync()))
    thread.start()
    thread.join()


async def thread_sync():
    """Run a sync operation in a separate thread."""
    thread_id = threading.get_ident()
    logger.info(f"[Thread {thread_id}] Starting sync in separate thread")
    
    # Create a new synchronizer instance in this thread
    synchronizer = ExchangeDataSynchronizer()
    
    # Trigger a sync
    synchronizer.trigger_sync("bybit", DataType.ALL)
    logger.info(f"[Thread {thread_id}] Triggered ALL sync in separate thread")
    
    # Wait for sync to complete
    for i in range(12):  # 1 minute
        await asyncio.sleep(5)
        logger.info(f"[Thread {thread_id}] Thread sync waiting... ({i+1}/12)")
    
    logger.info(f"[Thread {thread_id}] Thread sync completed")


async def main():
    """Main test function."""
    logger.info("=== Starting Bybit Sync Connection Test ===")
    logger.info(f"Test started at: {datetime.now().isoformat()}")
    logger.info(f"Main thread ID: {threading.get_ident()}")
    
    try:
        # Run a sync in the main thread
        task1 = asyncio.create_task(run_concurrent_syncs())
        
        # Wait a bit before starting the thread test
        await asyncio.sleep(10)
        
        # Run another sync in a different thread
        await run_in_new_thread()
        
        # Wait for the main sync to complete
        await task1
        
        # Ensure all pools are cleanly closed
        await close_all_pools()
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
    finally:
        try:
            await close_all_pools()
        except Exception as e:
            logger.error(f"Error closing pools: {str(e)}")
            
        logger.info(f"Test completed at: {datetime.now().isoformat()}")
        logger.info("=== Bybit Sync Connection Test Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")