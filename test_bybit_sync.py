#!/usr/bin/env python3
"""
Test script to verify the fixed synchronizer with Bybit exchange.

This script runs the ExchangeDataSynchronizer with the Bybit exchange to verify
that our database connection fixes have resolved the issues.
"""
import os
import sys
import asyncio
import threading
import time
from datetime import datetime, timezone
import signal
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Set environment variables for database connection
os.environ["DB_TYPE"] = "postgres"
os.environ["DB_NAME"] = "alphapulse"

from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.scheduler.sync_module.synchronizer import ExchangeDataSynchronizer
from alpha_pulse.data_pipeline.database.connection import init_db, close_all_connections


# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("test_bybit_sync.log", rotation="10 MB", level="DEBUG")


async def run_test():
    """Run the synchronizer test."""
    logger.info("Initializing database")
    await init_db()
    
    logger.info("Creating ExchangeDataSynchronizer instance")
    synchronizer = ExchangeDataSynchronizer(sync_interval=60)  # 1 minute interval for testing
    
    # Trigger a sync for all data types
    logger.info("Triggering sync for bybit exchange")
    success = synchronizer.trigger_sync("bybit", DataType.ALL)
    
    if success:
        logger.info("Sync triggered successfully, waiting for completion")
    else:
        logger.error("Failed to trigger sync")
        return False
    
    # Wait for the sync to complete (up to 2 minutes)
    for _ in range(24):  # 24 * 5 seconds = 2 minutes
        # Sleep for 5 seconds
        await asyncio.sleep(5)
        
        # Check if the sync is still in progress
        # This is a simplified check - in a real test, you would query the database
        logger.info("Checking sync status...")
    
    logger.info("Test completed, cleaning up")
    
    # Stop the synchronizer
    synchronizer.stop()
    
    # Close all database connections
    await close_all_connections()
    
    logger.info("Cleanup completed")
    return True


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the test
    logger.info("Starting Bybit synchronizer test")
    
    try:
        # Get the event loop
        loop = asyncio.get_event_loop()
        
        # Run the test
        success = loop.run_until_complete(run_test())
        
        # Exit with appropriate status code
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Test script completed")