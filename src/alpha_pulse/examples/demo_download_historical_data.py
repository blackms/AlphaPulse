"""
Script to download and store historical market data.

This script demonstrates:
1. Configuring data pipeline components
2. Downloading historical data for multiple timeframes
3. Proper error handling and validation
4. Progress monitoring
"""
import asyncio
from datetime import datetime, timedelta, UTC
from typing import List, Tuple
from loguru import logger

from alpha_pulse.exchanges import ExchangeType
from alpha_pulse.data_pipeline import (
    SQLAlchemyStorage,
    ExchangeFetcher,
    HistoricalDataManager,
    StorageConfig,
    DataFetchConfig,
    validate_timeframe,
    validate_symbol,
    DataFetchError,
    StorageError
)


async def download_historical_data(
    manager: HistoricalDataManager,
    exchange_type: ExchangeType,
    symbol: str,
    timeframes: List[str],
    days: int
) -> List[Tuple[str, int]]:
    """
    Download historical data for multiple timeframes.

    Args:
        manager: Historical data manager
        exchange_type: Type of exchange
        symbol: Trading symbol
        timeframes: List of timeframes to download
        days: Number of days of history

    Returns:
        List of (timeframe, record count) tuples
    """
    results = []
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)
    
    for timeframe in timeframes:
        try:
            # Validate inputs
            validate_timeframe(timeframe)
            validate_symbol(symbol)
            
            logger.info(f"📥 Downloading {timeframe} data for {symbol}...")
            
            # Ensure data is available
            await manager.ensure_data_available(
                exchange_type=exchange_type,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # Get data to verify
            data = manager.get_historical_data(
                exchange_type=exchange_type,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            results.append((timeframe, len(data)))
            logger.info(f"✅ Downloaded {len(data)} {timeframe} candles")
            
        except (DataFetchError, StorageError) as e:
            logger.error(f"❌ Error downloading {timeframe} data: {str(e)}")
            results.append((timeframe, 0))
            
    return results


async def main():
    """Download historical data for multiple timeframes."""
    logger.info("🚀 Starting historical data download...")
    
    # Configuration
    exchange_type = ExchangeType.BINANCE
    symbol = "BTC/USDT"
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    days = 30
    
    try:
        # Initialize components with configuration
        storage = SQLAlchemyStorage(
            config=StorageConfig(
                batch_size=1000,
                max_connections=5,
                timeout=30.0
            )
        )
        
        fetcher = ExchangeFetcher(
            config=DataFetchConfig(
                batch_size=1000,
                max_retries=3,
                retry_delay=1.0
            )
        )
        
        # Create manager
        manager = HistoricalDataManager(storage, fetcher)
        
        # Download data
        results = await download_historical_data(
            manager=manager,
            exchange_type=exchange_type,
            symbol=symbol,
            timeframes=timeframes,
            days=days
        )
        
        # Display summary
        logger.info("\n📊 Download Summary:")
        total_records = 0
        for timeframe, count in results:
            logger.info(f"{timeframe}: {count:,} records")
            total_records += count
            
        if total_records > 0:
            logger.info(f"\n✨ Successfully downloaded {total_records:,} total records")
        else:
            logger.warning("\n⚠️ No data was downloaded")
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())