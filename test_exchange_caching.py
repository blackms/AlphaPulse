"""
Test the exchange data caching functionality.

This script tests the database caching for exchange data and validates
that the portfolio API is correctly using cached data.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules
from alpha_pulse.api.data.portfolio import PortfolioDataAccessor
from alpha_pulse.data_pipeline.scheduler import exchange_data_synchronizer, DataType
from alpha_pulse.data_pipeline.database.connection import get_pg_connection
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository


async def test_exchange_data_caching():
    """Test the exchange data caching functionality."""
    logger.info("Testing exchange data caching...")
    
    # Initialize portfolio accessor
    portfolio_accessor = PortfolioDataAccessor()
    await portfolio_accessor._initialize()
    
    # 1. Force refresh data to ensure we have fresh data in cache
    logger.info("Forcing data refresh to populate cache...")
    refresh_result = await portfolio_accessor.reload_data(DataType.ALL)
    logger.info(f"Refresh result: {refresh_result}")
    
    # Wait for the background sync to complete (give it a few seconds)
    logger.info("Waiting for background sync to complete...")
    await asyncio.sleep(5)
    
    # 2. Get portfolio data from cache
    logger.info("Getting portfolio data from cache...")
    start_time = datetime.now(timezone.utc)
    cached_portfolio = await portfolio_accessor.get_portfolio(use_cache=True)
    cache_query_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    logger.info(f"Cache query time: {cache_query_time:.3f} seconds")
    logger.info(f"Using data source: {cached_portfolio.get('data_source', 'unknown')}")
    
    # 3. Get portfolio data directly from exchange API
    logger.info("Getting portfolio data directly from exchange...")
    start_time = datetime.now(timezone.utc)
    direct_portfolio = await portfolio_accessor.get_portfolio(use_cache=False)
    direct_query_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    logger.info(f"Direct query time: {direct_query_time:.3f} seconds")
    logger.info(f"Using data source: {direct_portfolio.get('data_source', 'unknown')}")
    
    # 4. Compare performance
    speedup = direct_query_time / cache_query_time if cache_query_time > 0 else float('inf')
    logger.info(f"Cache speedup factor: {speedup:.2f}x")
    
    # 5. Get cache statistics
    async with get_pg_connection() as conn:
        repo = ExchangeCacheRepository(conn)
        
        # Get all sync status records
        sync_status = await repo.get_all_sync_status()
        logger.info(f"Sync status records: {len(sync_status)}")
        
        # Get balances
        exchange_id = os.environ.get('EXCHANGE_TYPE', 'bybit').lower()
        balances = await repo.get_balances(exchange_id)
        logger.info(f"Cached balances: {len(balances) if balances else 0}")
        
        # Get positions
        positions = await repo.get_positions(exchange_id)
        logger.info(f"Cached positions: {len(positions) if positions else 0}")
    
    logger.info("Exchange data caching test complete")
    return {
        "cache_query_time": cache_query_time,
        "direct_query_time": direct_query_time,
        "speedup": speedup,
        "cache_data_source": cached_portfolio.get('data_source', 'unknown'),
        "direct_data_source": direct_portfolio.get('data_source', 'unknown')
    }


async def main():
    """Run the exchange data caching test."""
    try:
        test_results = await test_exchange_data_caching()
        logger.info(f"Test results: {test_results}")
    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())