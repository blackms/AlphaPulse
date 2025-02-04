"""
Script to check historical OHLCV data in storage.

This script demonstrates:
1. Querying historical data using the storage component
2. Validating data integrity
3. Displaying data statistics and sample records
"""
from datetime import datetime, timedelta, UTC
from loguru import logger

from alpha_pulse.exchanges import ExchangeType
from alpha_pulse.data_pipeline import (
    SQLAlchemyStorage,
    StorageConfig,
    MarketDataConfig,
    validate_timeframe,
    validate_symbol
)


def format_price(price: float) -> str:
    """Format price with appropriate precision."""
    return f"${price:,.2f}"


def format_volume(volume: float) -> str:
    """Format volume with appropriate precision."""
    if volume >= 1_000_000:
        return f"{volume/1_000_000:,.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:,.2f}K"
    else:
        return f"{volume:,.2f}"


def main():
    """Check historical data in storage."""
    logger.info("🔍 Checking historical OHLCV data...")
    
    try:
        # Initialize storage with configuration
        storage = SQLAlchemyStorage(
            config=StorageConfig(
                batch_size=1000,
                max_connections=5,
                timeout=30.0
            )
        )
        
        # Parameters for data check
        exchange = ExchangeType.BINANCE
        symbol = "BTC/USDT"
        timeframe = "1d"
        
        # Get valid timeframes from MarketDataConfig
        market_config = MarketDataConfig()
        valid_timeframes = list(market_config.timeframe_durations.keys())
        
        # Validate inputs
        validate_symbol(symbol)
        validate_timeframe(timeframe, valid_timeframes)
        
        # Query data for the last 30 days
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=30)
        
        data = storage.get_historical_data(
            exchange_type=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        if not data:
            logger.warning("❌ No historical data found")
            return
        
        # Display data statistics
        logger.info("\n📊 Data Statistics:")
        logger.info(f"Total Records: {len(data)}")
        logger.info(f"Date Range: {data[0].timestamp} to {data[-1].timestamp}")
        
        # Calculate price statistics
        prices = [candle.close for candle in data]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        logger.info("\n💰 Price Statistics:")
        logger.info(f"Average Price: {format_price(avg_price)}")
        logger.info(f"Min Price: {format_price(min_price)}")
        logger.info(f"Max Price: {format_price(max_price)}")
        
        # Calculate volume statistics
        volumes = [candle.volume for candle in data]
        avg_volume = sum(volumes) / len(volumes)
        min_volume = min(volumes)
        max_volume = max(volumes)
        
        logger.info("\n📈 Volume Statistics:")
        logger.info(f"Average Volume: {format_volume(avg_volume)}")
        logger.info(f"Min Volume: {format_volume(min_volume)}")
        logger.info(f"Max Volume: {format_volume(max_volume)}")
        
        # Show latest records
        logger.info("\n🕒 Latest Records:")
        for candle in data[-5:]:  # Show last 5 records
            logger.info(
                f"Time: {candle.timestamp}, "
                f"Open: {format_price(candle.open)}, "
                f"High: {format_price(candle.high)}, "
                f"Low: {format_price(candle.low)}, "
                f"Close: {format_price(candle.close)}, "
                f"Volume: {format_volume(candle.volume)}"
            )
        
        # Data quality check
        invalid_records = [
            i for i, candle in enumerate(data)
            if (
                candle.high < candle.low or
                candle.open < 0 or
                candle.high < 0 or
                candle.low < 0 or
                candle.close < 0 or
                candle.volume < 0
            )
        ]
        
        if invalid_records:
            logger.warning(
                f"\n⚠️ Found {len(invalid_records)} invalid records at "
                f"indices: {invalid_records}"
            )
        else:
            logger.info("\n✅ All records passed basic validation")
        
        logger.info("\n💡 Note: For high-performance timeseries operations, consider using TimescaleDB")
        
    except Exception as e:
        logger.error(f"❌ Error checking historical data: {str(e)}")
        raise


if __name__ == "__main__":
    main()