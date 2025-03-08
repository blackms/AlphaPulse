"""
Event loop management for the exchange data synchronization system.

This module provides utilities for managing event loops in a multi-threaded environment.
"""
import asyncio
import threading
import time
from typing import Optional, Callable, Any, Coroutine
from loguru import logger


class EventLoopManager:
    """
    Manages event loops for the exchange data synchronization system.
    
    This class provides utilities for creating, managing, and resetting event loops
    in a multi-threaded environment.
    """
    
    @staticmethod
    def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
        """
        Get the current event loop or create a new one if none exists.
        
        Returns:
            The current or new event loop
        """
        thread_id = threading.get_ident()
        try:
            loop = asyncio.get_event_loop()
            logger.debug(f"[THREAD {thread_id}] Got existing event loop: {loop}")
            return loop
        except RuntimeError:
            logger.debug(f"[THREAD {thread_id}] No event loop found, creating new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug(f"[THREAD {thread_id}] Created and set new event loop: {loop}")
            return loop
    
    @staticmethod
    def reset_event_loop() -> asyncio.AbstractEventLoop:
        """
        Reset the event loop for the current thread.
        
        Returns:
            The new event loop
        """
        thread_id = threading.get_ident()
        
        # Try to get the current loop
        current_loop = None
        try:
            current_loop = asyncio.get_event_loop()
            logger.debug(f"[THREAD {thread_id}] Current event loop: {current_loop}")
        except RuntimeError as loop_error:
            logger.warning(f"[THREAD {thread_id}] Error getting current event loop: {str(loop_error)}")
        
        # Close the current loop if it exists
        if current_loop:
            try:
                logger.debug(f"[THREAD {thread_id}] Closing current event loop")
                current_loop.close()
                logger.debug(f"[THREAD {thread_id}] Current event loop closed")
            except Exception as close_error:
                logger.warning(f"[THREAD {thread_id}] Error closing current event loop: {str(close_error)}")
        
        # Get a new event loop
        logger.debug(f"[THREAD {thread_id}] Creating new event loop")
        new_loop = asyncio.new_event_loop()
        logger.debug(f"[THREAD {thread_id}] Setting new event loop")
        asyncio.set_event_loop(new_loop)
        logger.info(f"[THREAD {thread_id}] Reset event loop for thread")
        
        return new_loop
    
    @staticmethod
    async def run_with_timeout(coro: Coroutine, timeout: float) -> Any:
        """
        Run a coroutine with a timeout.
        
        Args:
            coro: The coroutine to run
            timeout: Timeout in seconds
            
        Returns:
            The result of the coroutine
            
        Raises:
            asyncio.TimeoutError: If the coroutine times out
        """
        thread_id = threading.get_ident()
        logger.debug(f"[THREAD {thread_id}] Running coroutine with timeout {timeout}s")
        
        try:
            # Create a task for the coroutine
            task = asyncio.create_task(coro)
            logger.debug(f"[THREAD {thread_id}] Created task: {task}")
            
            # Wait for the task with timeout
            result = await asyncio.wait_for(task, timeout=timeout)
            logger.debug(f"[THREAD {thread_id}] Task completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.warning(f"[THREAD {thread_id}] Task timed out after {timeout} seconds")
            
            # Cancel the task if it's still running
            if not task.done():
                logger.warning(f"[THREAD {thread_id}] Cancelling task")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"[THREAD {thread_id}] Successfully cancelled task")
                except Exception as cancel_error:
                    logger.warning(f"[THREAD {thread_id}] Error while cancelling task: {str(cancel_error)}")
            
            # Re-raise the timeout error
            raise
    
    @staticmethod
    def safe_sleep(seconds: float) -> None:
        """
        Sleep safely, using either asyncio.sleep or time.sleep depending on the context.
        
        Args:
            seconds: Number of seconds to sleep
        """
        thread_id = threading.get_ident()
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, use asyncio.sleep
                logger.debug(f"[THREAD {thread_id}] Using asyncio.sleep for {seconds} seconds")
                asyncio.sleep(seconds)
            else:
                # If we're not in an async context, use time.sleep
                logger.debug(f"[THREAD {thread_id}] Using time.sleep for {seconds} seconds")
                time.sleep(seconds)
        except RuntimeError:
            # If we can't get the event loop, use time.sleep
            logger.debug(f"[THREAD {thread_id}] Using time.sleep for {seconds} seconds (no event loop)")
            time.sleep(seconds)
    
    @staticmethod
    def run_coroutine_in_new_loop(coro_func: Callable[[], Coroutine]) -> Any:
        """
        Run a coroutine function in a new event loop.
        
        This method is useful for running a coroutine in a context where the current
        event loop is causing issues or when you need to run a coroutine from a
        synchronous context.
        
        Args:
            coro_func: A function that returns a coroutine
            
        Returns:
            The result of the coroutine
            
        Raises:
            Exception: Any exception raised by the coroutine
        """
        thread_id = threading.get_ident()
        logger.debug(f"[THREAD {thread_id}] Running coroutine in new event loop")
        
        # Store the original event loop if it exists
        original_loop = None
        try:
            original_loop = asyncio.get_event_loop()
            logger.debug(f"[THREAD {thread_id}] Stored original event loop: {original_loop}")
        except RuntimeError:
            logger.debug(f"[THREAD {thread_id}] No original event loop to store")
        
        # Create a new event loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        logger.debug(f"[THREAD {thread_id}] Created and set new event loop: {new_loop}")
        
        try:
            # Run the coroutine in the new loop
            logger.debug(f"[THREAD {thread_id}] Running coroutine in new loop")
            result = new_loop.run_until_complete(coro_func())
            logger.debug(f"[THREAD {thread_id}] Coroutine completed successfully")
            return result
        except Exception as e:
            logger.error(f"[THREAD {thread_id}] Error running coroutine in new loop: {str(e)}")
            logger.error(f"[THREAD {thread_id}] Exception type: {type(e).__name__}")
            logger.error(f"[THREAD {thread_id}] Exception details: {repr(e)}")
            raise
        finally:
            # Clean up the new loop
            try:
                logger.debug(f"[THREAD {thread_id}] Closing new event loop")
                new_loop.close()
                logger.debug(f"[THREAD {thread_id}] New event loop closed")
            except Exception as e:
                logger.warning(f"[THREAD {thread_id}] Error closing new event loop: {str(e)}")
            
            # Restore the original event loop if it existed
            if original_loop:
                logger.debug(f"[THREAD {thread_id}] Restoring original event loop")
                try:
                    asyncio.set_event_loop(original_loop)
                    logger.debug(f"[THREAD {thread_id}] Original event loop restored")
                except Exception as e:
                    logger.warning(f"[THREAD {thread_id}] Error restoring original event loop: {str(e)}")