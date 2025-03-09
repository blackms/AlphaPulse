#!/usr/bin/env python3
"""
Connection pool management test script for AlphaPulse.

This script tests the database connection pooling and transaction handling
to verify that the fixes for concurrent operation errors are working correctly.
"""
import asyncio
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from alpha_pulse.data_pipeline.database.connection_manager import (
    get_db_connection,
    close_all_pools,
    get_loop_thread_key,
    execute_with_retry
)

# Configure logger
logger.add("connection_pool_test_{time}.log", rotation="500 MB", level="DEBUG")

async def run_database_operations(thread_id, num_operations=5):
    """
    Run a series of database operations to test connection handling.
    
    Args:
        thread_id: Thread identifier for logging
        num_operations: Number of operations to run
    """
    loop_thread_key = get_loop_thread_key()
    logger.info(f"[Thread {thread_id}] Starting database operations (loop: {loop_thread_key})")
    
    try:
        for i in range(num_operations):
            # Run a database operation with proper connection and transaction handling
            async def db_operation():
                async with get_db_connection() as conn:
                    # Wait a random time to increase likelihood of concurrent operations
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    
                    # Execute a query that just returns the current timestamp
                    result = await conn.fetchval("SELECT NOW()")
                    
                    # Add some additional delay to increase chance of concurrent operations
                    await asyncio.sleep(random.uniform(0.2, 0.8))
                    
                    return result
            
            # Execute with retry logic
            timestamp = await execute_with_retry(db_operation)
            
            logger.info(f"[Thread {thread_id}] Operation {i+1}/{num_operations} completed: {timestamp}")
            
            # Small delay between operations
            await asyncio.sleep(random.uniform(0.1, 0.3))
        
        logger.info(f"[Thread {thread_id}] All operations completed successfully")
        return True
    except Exception as e:
        logger.error(f"[Thread {thread_id}] Error in database operations: {str(e)}")
        logger.error(f"[Thread {thread_id}] Exception type: {type(e).__name__}")
        return False

async def run_concurrent_operations(thread_id, num_tasks=3):
    """
    Run multiple concurrent operations within the same thread/event loop.
    
    Args:
        thread_id: Thread identifier for logging
        num_tasks: Number of concurrent tasks to run
    """
    loop_thread_key = get_loop_thread_key()
    logger.info(f"[Thread {thread_id}] Starting {num_tasks} concurrent operations (loop: {loop_thread_key})")
    
    # Create multiple concurrent tasks
    tasks = []
    for i in range(num_tasks):
        task_id = f"{thread_id}_{i+1}"
        task = asyncio.create_task(run_database_operations(task_id, num_operations=3))
        tasks.append(task)
        logger.debug(f"[Thread {thread_id}] Created task {task_id}")
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check for exceptions
    success_count = sum(1 for result in results if result is True)
    error_count = len(results) - success_count
    
    logger.info(f"[Thread {thread_id}] Concurrent operations completed: {success_count} succeeded, {error_count} failed")
    
    # Close the connection pools at the end
    try:
        await close_all_pools()
        logger.info(f"[Thread {thread_id}] Closed all connection pools")
    except Exception as e:
        logger.error(f"[Thread {thread_id}] Error closing connection pools: {str(e)}")

def run_thread(thread_id):
    """
    Runner for a new thread with its own event loop.
    
    Args:
        thread_id: Thread identifier for logging
    """
    logger.info(f"Thread {thread_id} starting")
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run concurrent operations in this thread
        loop.run_until_complete(run_concurrent_operations(thread_id))
    except Exception as e:
        logger.error(f"Thread {thread_id} error: {str(e)}")
    finally:
        # Close the event loop
        loop.close()
        logger.info(f"Thread {thread_id} completed")

async def main():
    """Main function to run the tests."""
    logger.info("Starting connection pool test")
    
    # First, run concurrent operations in the main thread
    await run_concurrent_operations("main")
    
    # Then, run operations in multiple threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(run_thread, i+1) for i in range(3)]
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    logger.info("Connection pool test completed")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())