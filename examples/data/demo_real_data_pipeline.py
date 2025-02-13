"""
Example script demonstrating the complete real data pipeline.
"""
import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yaml
import pandas as pd
from loguru import logger
import sys

from alpha_pulse.data_pipeline.manager import DataManager
from alpha_pulse.data_pipeline.interfaces import DataFetchError

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/real_data_pipeline_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
           "{name}:{function}:{line} | "
           "{message}",
    level="DEBUG",
    rotation="500 MB"
)


async def main():
    """Main execution function."""
    manager = None
    try:
        # Load environment variables
        load_dotenv()

        # Load configuration
        with open("config/data_pipeline_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize data manager
        manager = DataManager(config=config)
        await manager.initialize()

        # Test symbols
        crypto_symbols = ["BTCUSDT", "ETHUSDT"]  # For market data

        # Time range for historical data (180 days for better technical analysis)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=180)  # Increased from 90 to 180 days

        # 1. Get Market Data (Crypto)
        logger.info("Fetching Market Data (Crypto)")
        try:
            market_data = await manager.get_market_data(
                symbols=crypto_symbols,
                start_time=start_time,
                end_time=end_time,
                interval="1d"
            )
            
            for symbol, data in market_data.items():
                logger.info(f"\n{symbol} Market Data:")
                logger.info("Last 5 days:")
                for entry in data[-5:]:  # Show last 5 days
                    logger.info(
                        f"Date: {entry.timestamp.date()} | "
                        f"Open: ${entry.open:,.2f} | "
                        f"Close: ${entry.close:,.2f} | "
                        f"Volume: {entry.volume:,.2f}"
                    )
                logger.info(f"Total data points: {len(data)}")

            # 2. Get Technical Indicators
            logger.info("\nCalculating Technical Indicators")
            try:
                technical_data = manager.get_technical_indicators(market_data)
                
                for symbol, indicators in technical_data.items():
                    logger.info(f"\n{symbol} Technical Analysis:")
                    
                    # Get the latest non-NaN values
                    def get_latest_value(values):
                        """Get latest non-NaN value from list."""
                        valid_values = [x for x in values if not pd.isna(x)]
                        return valid_values[-1] if valid_values else float('nan')
                    
                    # RSI (14-day)
                    rsi = get_latest_value(indicators.momentum['rsi'])
                    logger.info(f"RSI (14-day): {rsi:.2f}")
                    
                    # MACD (12,26,9)
                    macd = get_latest_value(indicators.trend['macd'])
                    signal = get_latest_value(indicators.trend['macd_signal'])
                    logger.info(f"MACD: {macd:.2f}")
                    logger.info(f"Signal: {signal:.2f}")
                    logger.info(f"MACD Histogram: {(macd - signal):.2f}")
                    
                    # Bollinger Bands (20,2)
                    bb_upper = get_latest_value(indicators.volatility['bb_upper'])
                    bb_middle = get_latest_value(indicators.volatility['bb_middle'])
                    bb_lower = get_latest_value(indicators.volatility['bb_lower'])
                    logger.info(f"Bollinger Bands:")
                    logger.info(f"  Upper: {bb_upper:.2f}")
                    logger.info(f"  Middle: {bb_middle:.2f}")
                    logger.info(f"  Lower: {bb_lower:.2f}")
                    
                    # Additional indicators
                    if 'sma_20' in indicators.trend:
                        sma_20 = get_latest_value(indicators.trend['sma_20'])
                        logger.info(f"SMA (20-day): {sma_20:.2f}")
                    if 'sma_50' in indicators.trend:
                        sma_50 = get_latest_value(indicators.trend['sma_50'])
                        logger.info(f"SMA (50-day): {sma_50:.2f}")
                    if 'atr' in indicators.volatility:
                        atr = get_latest_value(indicators.volatility['atr'])
                        logger.info(f"ATR (14-day): {atr:.2f}")
            except Exception as e:
                logger.error(f"Failed to calculate technical indicators: {str(e)}")

        except DataFetchError as e:
            logger.error(f"Failed to fetch market data: {str(e)}")
            market_data = {}

        logger.info("\nNote: Skipping Fundamental and Sentiment data due to API key issues.")
        logger.info("Please ensure you have valid API keys in your .env file for:")
        logger.info("- Alpha Vantage (fundamental data)")
        logger.info("- Finnhub (sentiment data)")

    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Clean up
        if manager:
            try:
                await manager.__aexit__(None, None, None)
                logger.debug("Successfully cleaned up data manager")
            except Exception as e:
                logger.exception(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.exception(f"Process terminated with error: {str(e)}")