"""
Example script demonstrating the AlphaPulse data pipeline functionality.

This script shows how to:
1. Initialize data pipeline components
2. Fetch historical data
3. Monitor real-time prices
4. Handle errors properly
"""
import asyncio
from datetime import datetime, timedelta, UTC
from loguru import logger

from alpha_pulse.exchanges import ExchangeType
from alpha_pulse.data_pipeline import (
    SQLAlchemyStorage,
    ExchangeFetcher,
    ExchangeDataProvider,
    HistoricalDataManager,
    RealTimeDataManager,
    DataFetchConfig,
    MarketDataConfig,
    validate_timeframe
)


async def demo_historical_data():
    """Demonstrate historical data fetching."""
    logger.info("🕒 Demonstrating historical data fetching...")
    
    # Initialize components
    storage = SQLAlchemyStorage()
    fetcher = ExchangeFetcher(
        config=DataFetchConfig(
            batch_size=1000,
            max_retries=3
        )
    )
    
    # Create historical data manager
    manager = HistoricalDataManager(storage, fetcher)
    
    try:
        # Fetch last 7 days of data
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=7)
        
        # Validate timeframe
        timeframe = "1h"
        validate_timeframe(timeframe)
        
        # Ensure data is available
        await manager.ensure_data_available(
            exchange_type=ExchangeType.BINANCE,
            symbol="BTC/USDT",
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get the data
        data = manager.get_historical_data(
            exchange_type=ExchangeType.BINANCE,
            symbol="BTC/USDT",
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info(f"✅ Retrieved {len(data)} historical records")
        
        # Show sample of the data
        if data:
            latest = data[-1]
            logger.info(
                f"Latest OHLCV - Time: {latest.timestamp}, "
                f"Close: ${latest.close:,.2f}, "
                f"Volume: {latest.volume:,.2f}"
            )
            
    except Exception as e:
        logger.error(f"❌ Error fetching historical data: {str(e)}")


async def demo_real_time_data():
    """Demonstrate real-time data monitoring."""
    logger.info("\n📊 Demonstrating real-time data monitoring...")
    
    # Initialize components
    storage = SQLAlchemyStorage()
    provider = ExchangeDataProvider(
        exchange_type=ExchangeType.BINANCE,
        testnet=True,
        config=MarketDataConfig(
            update_interval=1.0,
            cache_duration=60
        )
    )
    
    # Create real-time manager
    manager = RealTimeDataManager(provider, storage)
    
    try:
        # Start monitoring symbols
        symbols = ["BTC/USDT", "ETH/USDT"]
        await manager.start(symbols)
        logger.info(f"✅ Started monitoring {len(symbols)} symbols")
        
        # Monitor for 30 seconds
        for _ in range(30):
            for symbol in symbols:
                price = manager.get_price(symbol)
                if price:
                    logger.info(f"{symbol}: ${price:,.2f}")
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"❌ Error monitoring real-time data: {str(e)}")
        
    finally:
        # Clean up
        manager.stop()
        await provider.close()


async def main():
    """Run the data pipeline demonstration."""
    logger.info("🚀 Starting data pipeline demonstration...")
    
    # Run demos
    await demo_historical_data()
    await demo_real_time_data()
    
    logger.info("\n✨ Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())