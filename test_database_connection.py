#!/usr/bin/env python3
"""
Test script to verify database connection fixes in the AlphaPulse data pipeline.

This script tests the improved connection pooling, transaction management, and error handling
to ensure that the "another operation is in progress" and "connection was closed" errors
are properly handled.
"""
import os
import sys
import asyncio
import threading
import time
from datetime import datetime, timezone
import random
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("src"))

# Set environment variables for database connection
os.environ["DB_TYPE"] = "postgres"
os.environ["DB_NAME"] = "alphapulse"

from alpha_pulse.data_pipeline.database.connection import (
    get_pg_connection,
    _execute_with_retry,
    close_all_connections,
    init_db
)
from alpha_pulse.data_pipeline.scheduler import DataType
from alpha_pulse.data_pipeline.scheduler.sync_module.types import SyncStatus
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository


# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("test_database_connection.log", rotation="10 MB", level="DEBUG")


async def update_sync_status(exchange_id, data_type, status, thread_id):
    """Update sync status with retry logic."""
    next_sync_time = datetime.now(timezone.utc)
    
    async def update_operation():
        async with get_pg_connection() as conn:
            repo = ExchangeCacheRepository(conn)
            await repo.update_sync_status(
                exchange_id,
                data_type,
                status,
                next_sync_time,
                f"Test from thread {thread_id}"
            )
    
    try:
        await _execute_with_retry(update_operation)
        logger.info(f"[THREAD {thread_id}] Successfully updated sync status for {exchange_id}, {data_type}")
        return True
    except Exception as e:
        logger.error(f"[THREAD {thread_id}] Error updating sync status: {str(e)}")
        logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
        logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
        return False


async def get_all_sync_status(thread_id):
    """Get all sync statuses with retry logic."""
    async def get_operation():
        async with get_pg_connection() as conn:
            repo = ExchangeCacheRepository(conn)
            return await repo.get_all_sync_status()
    
    try:
        result = await _execute_with_retry(get_operation)
        logger.info(f"[THREAD {thread_id}] Successfully retrieved all sync statuses")
        return result
    except Exception as e:
        logger.error(f"[THREAD {thread_id}] Error getting sync statuses: {str(e)}")
        logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
        logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
        return []


async def worker(worker_id, exchange_id, iterations=10):
    """Worker function that performs database operations."""
    thread_id = threading.get_ident()
    logger.info(f"Worker {worker_id} started in thread {thread_id}")
    
    # Get the event loop for this thread
    loop = asyncio.get_event_loop()
    logger.info(f"Worker {worker_id} using event loop: {loop}")
    
    success_count = 0
    error_count = 0
    
    for i in range(iterations):
        # Choose a random data type
        data_type = random.choice([
            "balances", "positions", "orders", "prices"
        ])
        
        # Choose a random status
        status = random.choice([
            "pending", "in_progress", "completed", "failed"
        ])
        
        # Update sync status
        logger.info(f"[Worker {worker_id}] Iteration {i+1}/{iterations}: Updating {exchange_id} {data_type} to {status}")
        success = await update_sync_status(exchange_id, data_type, status, thread_id)
        
        if success:
            success_count += 1
        else:
            error_count += 1
        
        # Get all sync statuses
        statuses = await get_all_sync_status(thread_id)
        logger.info(f"[Worker {worker_id}] Retrieved {len(statuses)} sync status records")
        
        # Sleep for a random time to simulate real-world conditions
        sleep_time = random.uniform(0.1, 0.5)
        await asyncio.sleep(sleep_time)
    
    logger.info(f"Worker {worker_id} completed: {success_count} successes, {error_count} errors")
    return success_count, error_count


async def run_concurrent_workers(num_workers=5, iterations=10):
    """Run multiple workers concurrently to test database connection handling."""
    logger.info(f"Starting {num_workers} concurrent workers with {iterations} iterations each")
    
    # Initialize the database
    await init_db()
    
    # Create tasks for each worker
    tasks = []
    for i in range(num_workers):
        exchange_id = f"test_exchange_{i}"
        task = asyncio.create_task(worker(i, exchange_id, iterations))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    total_success = 0
    total_error = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Worker {i} failed with exception: {str(result)}")
            total_error += iterations
        else:
            success, error = result
            total_success += success
            total_error += error
    
    # Close all connections
    await close_all_connections()
    
    logger.info(f"Test completed: {total_success} successful operations, {total_error} errors")
    return total_success, total_error


def run_in_thread(thread_id, num_workers=3, iterations=5):
    """Run workers in a separate thread."""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    logger.info(f"Thread {thread_id} started with event loop: {loop}")
    
    try:
        # Run the concurrent workers
        success, error = loop.run_until_complete(run_concurrent_workers(num_workers, iterations))
        logger.info(f"Thread {thread_id} completed: {success} successes, {error} errors")
    except Exception as e:
        logger.error(f"Thread {thread_id} failed with exception: {str(e)}")
    finally:
        # Close the event loop
        loop.close()
        logger.info(f"Thread {thread_id} closed event loop")


async def main():
    """Main function to run the test."""
    logger.info("Starting database connection test")
    
    # First, run in the main thread
    logger.info("Running test in main thread")
    main_success, main_error = await run_concurrent_workers(3, 5)
    
    # Then, run in multiple threads
    logger.info("Running test in multiple threads")
    threads = []
    for i in range(3):
        thread = threading.Thread(target=run_in_thread, args=(i, 2, 3))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for i, thread in enumerate(threads):
        thread.join()
        logger.info(f"Thread {i} joined")
    
    logger.info("All tests completed")
    
    # Return success if there were no errors
    return main_error == 0


if __name__ == "__main__":
    # Run the main function
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)