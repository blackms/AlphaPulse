"""
Example script demonstrating the AlphaPulse data fetching functionality.
"""
from loguru import logger

from alpha_pulse.data_pipeline import CCXTExchangeFactory


def main():
    """Run data fetching demonstration."""
    logger.info("Starting data fetching demonstration...")

    # Create exchange instance using factory
    factory = CCXTExchangeFactory()
    exchange = factory.create_exchange("binance")
    
    # Get available trading pairs
    pairs = exchange.get_available_pairs()
    logger.info(f"Available trading pairs: {pairs}")
    
    # Get current price for BTC/USDT
    btc_price = exchange.get_current_price("BTC/USDT")
    logger.info(f"Current BTC price: ${btc_price:,.2f}")
    
    # Fetch historical data
    historical_data = exchange.fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1d",
        start_time=None,  # Default to recent data
        end_time=None     # Default to current time
    )
    
    logger.info("\nHistorical Data Sample:")
    for key, values in historical_data.items():
        logger.info(f"{key}: {values[:5]}...")


if __name__ == "__main__":
    main()