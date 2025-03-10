"""
FastAPI integration for the new exchange_sync module.

This module provides functionality to integrate the exchange_sync module
with a FastAPI application, handling startup, shutdown, and background tasks.
"""
import asyncio
import logging
from typing import Callable, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks

from alpha_pulse.exchange_sync.repository import PortfolioRepository
from alpha_pulse.exchange_sync.scheduler import ExchangeSyncScheduler
from alpha_pulse.exchange_sync.config import configure_logging

# Configure logging
logger = logging.getLogger(__name__)


async def initialize_exchange_sync() -> bool:
    """
    Initialize the exchange_sync module.
    
    This function initializes the database tables needed for the exchange_sync module.
    
    Returns:
        True if initialization was successful
    """
    try:
        # Configure logging for the exchange_sync module
        configure_logging()
        
        # Initialize database tables
        repo = PortfolioRepository()
        await repo.initialize_tables()
        
        logger.info("Exchange sync module initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize exchange sync module: {e}")
        return False


async def startup_exchange_sync(background_tasks: BackgroundTasks = None) -> None:
    """
    Start the exchange synchronization process.
    
    This function initializes the exchange_sync module and starts the scheduler
    either as a background task or directly.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance for background processing
    """
    try:
        # Initialize the exchange_sync module
        init_success = await initialize_exchange_sync()
        if not init_success:
            logger.warning("Exchange sync initialization failed - some features may not work correctly")
            return
        
        # Create the scheduler
        scheduler = ExchangeSyncScheduler()
        
        # Start the scheduler
        if background_tasks:
            # Run as a background task if BackgroundTasks is provided
            logger.info("Starting exchange sync scheduler as a background task")
            background_tasks.add_task(scheduler.start)
        else:
            # Run in a separate task
            logger.info("Starting exchange sync scheduler in a separate task")
            asyncio.create_task(scheduler.start())
        
        # Run an initial sync
        logger.info("Triggering initial data synchronization")
        asyncio.create_task(ExchangeSyncScheduler.run_once())
        
    except Exception as e:
        logger.error(f"Error during exchange sync startup: {e}")
        logger.warning("Exchange data synchronization will not be available")


async def shutdown_exchange_sync() -> None:
    """
    Shutdown the exchange synchronization process.
    
    This function stops the scheduler and performs any necessary cleanup.
    """
    try:
        # Create a scheduler instance to access the stop method
        # This works because the scheduler is designed to manage its own state
        scheduler = ExchangeSyncScheduler()
        
        # Stop the scheduler
        await scheduler.stop()
        
        logger.info("Exchange sync scheduler stopped")
    except Exception as e:
        logger.error(f"Error during exchange sync shutdown: {e}")
        logger.warning("Some resources may not be properly released")


@asynccontextmanager
async def exchange_sync_lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for exchange sync.
    
    Use this in FastAPI app creation:
    
    ```python
    from alpha_pulse.api.exchange_sync_integration import exchange_sync_lifespan
    
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
    Register startup and shutdown events for exchange sync.
    
    Use this in FastAPI app creation if you're not using the lifespan:
    
    ```python
    from alpha_pulse.api.exchange_sync_integration import register_exchange_sync_events
    
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


async def trigger_exchange_sync(exchange_id: str = None) -> Dict[str, Any]:
    """
    Trigger an immediate synchronization for one or all exchanges.
    
    Args:
        exchange_id: Optional exchange ID to sync (all exchanges if None)
        
    Returns:
        Dictionary with operation status and results
    """
    try:
        logger.info(f"Triggering immediate sync for {'all exchanges' if exchange_id is None else exchange_id}")
        
        # Run the sync
        results = await ExchangeSyncScheduler.run_once()
        
        # Filter results if a specific exchange was requested
        if exchange_id:
            if exchange_id in results:
                results = {exchange_id: results[exchange_id]}
            else:
                return {
                    "status": "error",
                    "message": f"Exchange {exchange_id} not found or not configured"
                }
        
        # Process results
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        # Create response
        response = {
            "status": "success" if success_count == total_count else "partial_success" if success_count > 0 else "error",
            "message": f"Synchronized {success_count}/{total_count} exchanges successfully",
            "timestamp": results[next(iter(results))].end_time.isoformat() if results else None,
            "results": {}
        }
        
        # Add detailed results
        for ex_id, result in results.items():
            response["results"][ex_id] = {
                "success": result.success,
                "items_processed": result.items_processed,
                "items_synced": result.items_synced,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors if not result.success else []
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error triggering exchange sync: {e}")
        return {
            "status": "error",
            "message": f"Failed to trigger synchronization: {str(e)}"
        }