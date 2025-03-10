"""
Scheduler for exchange data synchronization.

This module provides functionality to schedule and run exchange data synchronization
at regular intervals.
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from loguru import logger

from .portfolio_service import PortfolioService
from .models import SyncResult
from .config import get_sync_config


class ExchangeSyncScheduler:
    """
    Scheduler for exchange data synchronization.
    
    This class handles scheduling and running exchange data synchronization
    at regular intervals.
    """
    
    # Class-level variables to track the scheduler state
    _running = False
    _task = None
    
    def __init__(self):
        """Initialize the scheduler with configuration."""
        self.config = get_sync_config()
        self.interval_minutes = self.config['interval_minutes']
        self.enabled = self.config['enabled']
        self.exchanges = self.config['exchanges']
        
        logger.info(f"Scheduler initialized with {self.interval_minutes} minute interval")
        
        if not self.enabled:
            logger.warning("Scheduler is disabled by configuration")
    
    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return ExchangeSyncScheduler._running
    
    async def start(self) -> None:
        """
        Start the scheduler.
        
        This method starts the scheduler, which will run exchange data synchronization
        at regular intervals.
        """
        if ExchangeSyncScheduler._running:
            logger.warning("Scheduler is already running")
            return
        
        if not self.enabled:
            logger.warning("Scheduler is disabled by configuration")
            return
        
        logger.info("Starting exchange sync scheduler")
        ExchangeSyncScheduler._running = True
        
        # Create a task for the scheduler loop
        ExchangeSyncScheduler._task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self) -> None:
        """
        Stop the scheduler.
        
        This method stops the scheduler, cancelling any pending synchronization tasks.
        """
        if not ExchangeSyncScheduler._running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping exchange sync scheduler")
        ExchangeSyncScheduler._running = False
        
        # Cancel the scheduler task if it exists
        if ExchangeSyncScheduler._task:
            ExchangeSyncScheduler._task.cancel()
            try:
                await ExchangeSyncScheduler._task
            except asyncio.CancelledError:
                logger.info("Scheduler task cancelled")
            ExchangeSyncScheduler._task = None
        
        logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """
        Scheduler loop.
        
        This method runs in a loop, executing exchange data synchronization
        at regular intervals.
        """
        try:
            while ExchangeSyncScheduler._running:
                logger.info("Executing scheduled exchange sync")
                
                try:
                    # Run the synchronization
                    await self._run_sync()
                except Exception as e:
                    logger.error(f"Error during scheduled sync: {str(e)}")
                
                # Sleep until the next sync
                interval_seconds = self.interval_minutes * 60
                logger.debug(f"Sleeping for {interval_seconds} seconds until next sync")
                
                # Use a loop with small sleep intervals to allow for clean shutdown
                for _ in range(interval_seconds):
                    if not ExchangeSyncScheduler._running:
                        break
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Scheduler task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in scheduler loop: {str(e)}")
            ExchangeSyncScheduler._running = False
            raise
    
    async def _run_sync(self) -> Dict[str, SyncResult]:
        """
        Run exchange data synchronization.
        
        This method runs the exchange data synchronization for all configured exchanges.
        
        Returns:
            Dictionary mapping exchange IDs to sync results
        """
        service = PortfolioService()  # No need to pass exchange_id as we're syncing multiple exchanges
        return await service.sync_portfolio(self.exchanges)
    
    @classmethod
    async def run_once(cls) -> Dict[str, SyncResult]:
        """
        Run exchange data synchronization once.
        
        This class method runs the exchange data synchronization once for all
        configured exchanges.
        
        Returns:
            Dictionary mapping exchange IDs to sync results
        """
        logger.info("Running one-time exchange synchronization")
        
        # Create a scheduler instance to get the configuration
        scheduler = cls()
        
        # Run the synchronization
        results = await scheduler._run_sync()
        
        # Count successful syncs
        success_count = sum(1 for result in results.values() if result.success)
        total_count = len(results)
        
        logger.info(f"One-time sync completed: {success_count}/{total_count} exchanges successful")
        
        return results