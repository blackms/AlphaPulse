"""
Runner script for exchange data synchronization.

This module provides a command-line interface for running the exchange
synchronization process, either as a one-time operation or as a scheduled service.
"""
import asyncio
import argparse
import logging
import sys
from datetime import datetime

from .config import configure_logging, get_sync_config
from .scheduler import ExchangeSyncScheduler
from .portfolio_service import PortfolioService
from .repository import PortfolioRepository


async def initialize_database() -> bool:
    """
    Initialize the database tables needed for exchange synchronization.
    
    Returns:
        True if initialization was successful
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing database tables...")
    
    try:
        repo = PortfolioRepository()
        await repo.initialize_tables()
        logger.info("Database initialization complete")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False


async def run_once() -> None:
    """
    Run a one-time synchronization of all configured exchanges.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting one-time synchronization")
    
    try:
        # Initialize database
        if not await initialize_database():
            logger.error("Cannot proceed with synchronization due to database initialization failure")
            return
        
        # Run the sync
        start_time = datetime.now()
        results = await ExchangeSyncScheduler.run_once()
        end_time = datetime.now()
        
        # Log summary
        success_count = sum(1 for r in results.values() if r.success)
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Synchronization completed in {duration:.2f} seconds")
        logger.info(f"Results: {success_count}/{len(results)} exchanges successful")
        
        # Log detailed results
        for exchange_id, result in results.items():
            if result.success:
                logger.info(f"{exchange_id}: Successfully synced {result.items_synced} items")
            else:
                logger.error(f"{exchange_id}: Sync failed with errors: {', '.join(result.errors)}")
    
    except Exception as e:
        logger.error(f"One-time synchronization failed: {str(e)}")


async def run_scheduler(interval_minutes: int) -> None:
    """
    Run the synchronization scheduler as a service.
    
    Args:
        interval_minutes: Time between synchronization runs in minutes
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting scheduler with {interval_minutes} minute interval")
    
    try:
        # Initialize database
        if not await initialize_database():
            logger.error("Cannot start scheduler due to database initialization failure")
            return
        
        # Create and start the scheduler
        scheduler = ExchangeSyncScheduler(interval_minutes=interval_minutes)
        
        try:
            await scheduler.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping scheduler")
        finally:
            await scheduler.stop()
            logger.info("Scheduler stopped")
    
    except Exception as e:
        logger.error(f"Scheduler execution failed: {str(e)}")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Exchange Data Synchronization Runner'
    )
    
    # Mode selection
    parser.add_argument(
        '--one-time',
        action='store_true',
        help='Run a one-time synchronization and exit'
    )
    
    # Scheduler options
    parser.add_argument(
        '--interval',
        type=int,
        help='Scheduler interval in minutes (default from config)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=None,
        help='Logging level (default from config)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory for log files (default from config)'
    )
    
    return parser.parse_args()


async def main():
    """
    Main entry point for the runner.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(log_dir=args.log_dir, log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Exchange Sync Runner starting")
    
    # Get sync configuration
    sync_config = get_sync_config()
    
    # Determine interval (command line args override config)
    interval_minutes = args.interval or sync_config['interval_minutes']
    
    try:
        if args.one_time:
            # Run one-time sync
            await run_once()
        else:
            # Run as a scheduler
            await run_scheduler(interval_minutes)
    
    except Exception as e:
        logger.error(f"Runner failed with error: {str(e)}")
        return 1
    
    logger.info("Runner exiting")
    return 0


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)