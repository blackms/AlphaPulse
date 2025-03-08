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
        db_init_success = await init_db()
        if db_init_success:
            logger.info("Database initialized for exchange data cache")
        else:
            logger.warning("Database initialization failed - some features may not work correctly")
        
        # Initialize exchange synchronizer
        try:
            await exchange_data_synchronizer.initialize()
            logger.info("Exchange data synchronizer initialized successfully")
        except AttributeError:
            # Handle case where initialize method doesn't exist (older version)
            logger.warning("ExchangeDataSynchronizer does not have initialize method, continuing without initialization")
        except Exception as e:
            # Handle other exceptions that might occur during initialization
            logger.warning(f"Error initializing exchange data synchronizer: {e}")
            logger.warning("Continuing without initialization")
        
        # Start the scheduler
        try:
            await exchange_data_synchronizer.start()
            logger.info("Exchange data synchronization scheduler started")
            
            # Initial sync of all data types
            exchange_data_synchronizer.trigger_sync(data_type=DataType.ALL)
            logger.info("Initial data synchronization triggered")
        except Exception as e:
            logger.error(f"Failed to start exchange data synchronizer: {e}")
            logger.warning("Exchange data synchronization will not be available")
    except AttributeError as e:
        logger.error(f"Error during exchange synchronization startup: {e}")
        logger.warning("Missing attribute or method - the system will use direct API calls if needed")
        # Allow the application to start anyway - it will use direct API calls if needed
    except Exception as e:
        logger.error(f"Error during exchange synchronization startup: {e}")
        logger.warning("The system will use direct API calls if needed")
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
    except AttributeError as e:
        logger.error(f"Error during exchange synchronization shutdown: {e}")
        logger.warning("Missing attribute or method during shutdown - some resources may not be properly released")
    except Exception as e:
        logger.error(f"Error during exchange synchronization shutdown: {e}")
        logger.warning("Unexpected error during shutdown - some resources may not be properly released")


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