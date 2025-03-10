"""
Scheduler for periodic exchange data synchronization.

This module provides a simple scheduler that runs the synchronization process
at regular intervals, ensuring reliable and consistent data updates.
"""
import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable

from .config import get_sync_config, configure_logging
from .portfolio_service import PortfolioService
from .models import SyncResult


class ExchangeSyncScheduler:
    """
    Scheduler for executing exchange data synchronization at regular intervals.
    
    This class provides a simple, reliable scheduler that runs the synchronization
    process periodically, handling errors and shutdown requests gracefully.
    """
    
    def __init__(self, interval_minutes: Optional[int] = None):
        """
        Initialize the scheduler.
        
        Args:
            interval_minutes: Interval between sync operations in minutes
                             (default from config)
        """
        # Configure logging
        configure_logging()
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.sync_config = get_sync_config()
        self.interval_minutes = interval_minutes or self.sync_config['interval_minutes']
        self.enabled = self.sync_config['enabled']
        
        # Internal state
        self.running = False
        self.last_run = datetime.min
        self.task = None
        
        # Statistics
        self.run_count = 0
        self.error_count = 0
        self.last_result = None
        
        self.logger.info(f"Scheduler initialized with {self.interval_minutes} minute interval")
    
    async def start(self) -> None:
        """
        Start the scheduler.
        
        This method begins the periodic execution of the synchronization process.
        """
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        if not self.enabled:
            self.logger.warning("Scheduler is disabled by configuration")
            return
        
        self.running = True
        self.logger.info("Starting exchange sync scheduler")
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        try:
            # Start the main scheduling loop
            await self._scheduling_loop()
        except asyncio.CancelledError:
            self.logger.info("Scheduler task cancelled")
        except Exception as e:
            self.logger.error(f"Scheduler failed with error: {str(e)}")
        finally:
            self.running = False
            self.logger.info("Scheduler stopped")
    
    async def stop(self) -> None:
        """
        Stop the scheduler.
        
        This method gracefully stops the scheduler, ensuring any in-progress
        synchronization tasks are completed.
        """
        if not self.running:
            self.logger.warning("Scheduler is not running")
            return
        
        self.logger.info("Stopping scheduler...")
        self.running = False
        
        # Cancel the scheduling task if it exists
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Scheduler stopped")
    
    def _setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for graceful shutdown.
        
        This method registers signal handlers for SIGINT and SIGTERM
        to ensure graceful shutdown when the process is terminated.
        """
        try:
            # Create an event loop for handling signals
            loop = asyncio.get_event_loop()
            
            # Register signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.stop())
                )
            
            self.logger.debug("Signal handlers installed")
        except (NotImplementedError, RuntimeError):
            # Windows doesn't support add_signal_handler
            self.logger.warning("Could not install signal handlers (not supported)")
    
    async def _scheduling_loop(self) -> None:
        """
        Main scheduling loop.
        
        This method runs the main loop that checks when to execute
        the next synchronization task based on the configured interval.
        """
        while self.running:
            now = datetime.now()
            
            # Check if it's time to run the sync
            if now - self.last_run > timedelta(minutes=self.interval_minutes):
                self.logger.info("Executing scheduled exchange sync")
                
                # Run the sync operation
                try:
                    self.last_result = await PortfolioService.sync_all_exchanges()
                    self.last_run = now
                    self.run_count += 1
                    
                    # Log results
                    success_count = sum(1 for r in self.last_result.values() if r.success)
                    self.logger.info(
                        f"Sync completed: {success_count}/{len(self.last_result)} exchanges successful"
                    )
                except Exception as e:
                    self.logger.error(f"Error during scheduled sync: {str(e)}")
                    self.error_count += 1
            
            # Sleep for 1 minute before checking again
            # This prevents tight CPU usage while still being responsive
            await asyncio.sleep(60)
    
    @classmethod
    async def run_once(cls) -> Dict[str, SyncResult]:
        """
        Run the synchronization process once.
        
        This class method provides a convenient way to run the synchronization
        process a single time without starting the scheduler.
        
        Returns:
            Dictionary mapping exchange IDs to their sync results
        """
        logger = logging.getLogger(__name__)
        logger.info("Running one-time exchange synchronization")
        
        try:
            result = await PortfolioService.sync_all_exchanges()
            
            # Log results
            success_count = sum(1 for r in result.values() if r.success)
            logger.info(
                f"One-time sync completed: {success_count}/{len(result)} exchanges successful"
            )
            
            return result
        except Exception as e:
            logger.error(f"Error during one-time sync: {str(e)}")
            raise


# Standalone entry point
async def main():
    """
    Entry point for running the scheduler as a standalone process.
    """
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting exchange sync scheduler as standalone process")
    
    # Create and start the scheduler
    scheduler = ExchangeSyncScheduler()
    
    try:
        await scheduler.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping scheduler")
    except Exception as e:
        logger.error(f"Unexpected error in scheduler: {str(e)}")
    finally:
        await scheduler.stop()
        logger.info("Scheduler process exited")


if __name__ == "__main__":
    asyncio.run(main())