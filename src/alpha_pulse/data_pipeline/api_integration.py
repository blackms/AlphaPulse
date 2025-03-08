"""
FastAPI integration for exchange data synchronization.

This module provides functionality to integrate the exchange data
synchronization with a FastAPI application.
"""
import asyncio
from typing import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from alpha_pulse.data_pipeline.database.connection import init_db
from alpha_pulse.data_pipeline.scheduler import exchange_data_synchronizer, DataType


async def startup_exchange_sync() -> None:
    """Initialize database and start exchange synchronization."""
    try:
        # Initialize database (create tables if they don't exist)
        await init_db()
        logger.info("Database initialized for exchange data cache")
        
        # Initialize exchange synchronizer
        try:
            await exchange_data_synchronizer.initialize()
            logger.info("Exchange data synchronizer initialized")
        except AttributeError:
            # Handle case where initialize method doesn't exist (older version)
            logger.warning("Exchange data synchronizer does not have initialize method - using older version")
        
        # Start the scheduler
        await exchange_data_synchronizer.start()
        logger.info("Exchange data synchronization scheduler started")
        
        # Initial sync of all data types
        exchange_data_synchronizer.trigger_sync(data_type=DataType.ALL)
        logger.info("Initial data synchronization triggered")
    except Exception as e:
        logger.error(f"Error during exchange synchronization startup: {e}")
        # Allow the application to start anyway - it will use direct API calls if needed


async def shutdown_exchange_sync() -> None:
    """Shutdown exchange synchronization."""
    try:
        # Stop the scheduler
        await exchange_data_synchronizer.stop()
        logger.info("Exchange data synchronization scheduler stopped")
        
        # Close exchange connections
        await exchange_data_synchronizer.close()
        logger.info("Exchange connections closed")
    except Exception as e:
        logger.error(f"Error during exchange synchronization shutdown: {e}")


@asynccontextmanager
async def exchange_sync_lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for exchange data synchronization.
    
    Use this in FastAPI app creation:
    
    ```python
    from alpha_pulse.data_pipeline.api_integration import exchange_sync_lifespan
    
    app = FastAPI(lifespan=exchange_sync_lifespan)
    ```
    """
    # Startup logic
    await startup_exchange_sync()
    
    yield  # FastAPI runs during this yield
    
    # Shutdown logic
    await shutdown_exchange_sync()


def register_exchange_sync_events(app: FastAPI) -> None:
    """
    Register startup and shutdown events for exchange data synchronization.
    
    Use this in FastAPI app creation if you're not using the lifespan:
    
    ```python
    from alpha_pulse.data_pipeline.api_integration import register_exchange_sync_events
    
    app = FastAPI()
    register_exchange_sync_events(app)
    ```
    """
    @app.on_event("startup")
    async def start_exchange_sync():
        await startup_exchange_sync()
    
    @app.on_event("shutdown")
    async def stop_exchange_sync():
        await shutdown_exchange_sync()


# Export functions to be used in other modules
__all__ = [
    "startup_exchange_sync",
    "shutdown_exchange_sync",
    "exchange_sync_lifespan",
    "register_exchange_sync_events"
]