"""
Runner for executing exchange synchronization.

This module provides a simple way to run exchange synchronization
as a standalone process.
"""
import asyncio
import sys
import argparse
from typing import List, Optional

from loguru import logger

from .portfolio_service import PortfolioService
from .config import configure_logging


async def run_sync(exchanges: List[str], log_level: str = "INFO") -> None:
    """
    Run exchange synchronization for the specified exchanges.
    
    Args:
        exchanges: List of exchange identifiers to synchronize
        log_level: Logging level
    """
    # Configure logging
    configure_logging(log_level=log_level)
    
    logger.info(f"Starting exchange synchronization for: {', '.join(exchanges)}")
    
    # Create the portfolio service
    service = PortfolioService()  # No need to pass exchange_id as we're syncing multiple exchanges
    
    try:
        # Run the synchronization
        results = await service.sync_portfolio(exchanges)
        
        # Process results
        success_count = sum(1 for result in results.values() if result.success)
        total_count = len(results)
        
        logger.info(f"Synchronization completed: {success_count}/{total_count} exchanges successful")
        
        # Log detailed results
        for exchange_id, result in results.items():
            if result.success:
                logger.info(f"{exchange_id}: Synced {result.items_synced}/{result.items_processed} items in {result.duration_seconds:.2f}s")
            else:
                logger.error(f"{exchange_id}: Failed - {', '.join(result.errors)}")
        
    except Exception as e:
        logger.error(f"Error during synchronization: {str(e)}")
        sys.exit(1)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run exchange synchronization")
    parser.add_argument(
        "--exchanges", 
        type=str, 
        default="bybit",
        help="Comma-separated list of exchanges to synchronize"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Parse exchanges
    exchanges = [e.strip() for e in args.exchanges.split(",")]
    
    # Run the synchronization
    asyncio.run(run_sync(exchanges, args.log_level))


if __name__ == "__main__":
    main()