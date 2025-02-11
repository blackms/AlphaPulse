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
        stock_symbols = ["AAPL", "MSFT", "GOOGL"]  # For fundamental and sentiment data
        crypto_symbols = ["BTCUSDT", "ETHUSDT"]  # For market data

        # Time range for historical data (90 days for better technical analysis)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=90)

        # 1. Get Market Data (Crypto)
        logger.info("Fetching Market Data (Crypto)")
        market_data = await manager.get_market_data(
            symbols=crypto_symbols,
            start_time=start_time,
            end_time=end_time,
            interval="1d"
        )
        
        for symbol, data in market_data.items():
            logger.info(f"{symbol} Market Data:")
            for entry in data[-5:]:  # Show last 5 days
                logger.info(
                    f"Date: {entry.timestamp.date()} | "
                    f"Open: ${entry.open:,.2f} | "
                    f"Close: ${entry.close:,.2f} | "
                    f"Volume: {entry.volume:,.2f}"
                )

        # 2. Get Fundamental Data (Stocks)
        logger.info("Fetching Fundamental Data")
        fundamental_data = await manager.get_fundamental_data(symbols=stock_symbols)
        
        for symbol, data in fundamental_data.items():
            logger.info(f"{symbol} Fundamental Data:")
            logger.info(f"Market Cap: ${data.metadata.get('market_cap', 0):,.2f}")
            logger.info(f"P/E Ratio: {data.financial_ratios.get('pe_ratio', 0):.2f}")
            logger.info(f"Revenue: ${data.income_statement.get('revenue', 0):,.2f}")
            logger.info(f"Net Income: ${data.income_statement.get('net_income', 0):,.2f}")
            logger.info(f"Free Cash Flow: ${data.cash_flow.get('free_cash_flow', 0):,.2f}")

        # 3. Get Sentiment Data (Stocks)
        logger.info("Fetching Sentiment Data")
        sentiment_data = await manager.get_sentiment_data(symbols=stock_symbols)
        
        for symbol, data in sentiment_data.items():
            logger.info(f"{symbol} Sentiment Analysis:")
            logger.info(f"News Sentiment: {data.news_sentiment:.1%}")
            logger.info(f"Social Sentiment: {data.social_sentiment:.1%}")
            
            # Show recent news
            if data.source_data.get("news"):
                logger.info("Recent News Headlines:")
                for article in sorted(
                    data.source_data["news"],
                    key=lambda x: x["datetime"],
                    reverse=True
                )[:3]:  # Show 3 most recent
                    logger.info(
                        f"Headline: {article['headline']}\n"
                        f"Source: {article['source']}\n"
                        f"Time: {article['datetime']}\n"
                        f"Sentiment: {article['sentiment_score']:.1%}"
                    )

        # 4. Get Technical Indicators
        logger.info("Calculating Technical Indicators")
        technical_data = manager.get_technical_indicators(market_data)
        
        for symbol, indicators in technical_data.items():
            logger.info(f"{symbol} Technical Analysis:")
            # Get the latest values (last non-NaN value)
            rsi = next((x for x in reversed(indicators.momentum['rsi']) if not pd.isna(x)), float('nan'))
            macd = next((x for x in reversed(indicators.trend['macd']) if not pd.isna(x)), float('nan'))
            signal = next((x for x in reversed(indicators.trend['macd_signal']) if not pd.isna(x)), float('nan'))
            bb_upper = next((x for x in reversed(indicators.volatility['bb_upper']) if not pd.isna(x)), float('nan'))
            bb_lower = next((x for x in reversed(indicators.volatility['bb_lower']) if not pd.isna(x)), float('nan'))
            
            logger.info(f"RSI: {rsi:.2f}")
            logger.info(f"MACD: {macd:.2f}")
            logger.info(f"Signal: {signal:.2f}")
            logger.info(f"BB Upper: {bb_upper:.2f}")
            logger.info(f"BB Lower: {bb_lower:.2f}")

    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Clean up
        if manager:
            try:
                await manager.__aexit__(None, None, None)
            except Exception as e:
                logger.exception(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.exception(f"Process terminated with error: {str(e)}")