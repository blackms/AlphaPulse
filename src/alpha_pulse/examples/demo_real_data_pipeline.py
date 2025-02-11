"""
Example script demonstrating the complete real data pipeline.
"""
import asyncio
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import yaml

from alpha_pulse.data_pipeline.manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main execution function."""
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

        # Time range for historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        # 1. Get Market Data (Crypto)
        logger.info("\nFetching Market Data (Crypto):")
        market_data = await manager.get_market_data(
            symbols=crypto_symbols,
            start_time=start_time,
            end_time=end_time,
            interval="1d"
        )
        
        for symbol, data in market_data.items():
            logger.info(f"\n{symbol} Market Data:")
            for entry in data[-5:]:  # Show last 5 days
                logger.info(
                    f"Date: {entry.timestamp.date()} | "
                    f"Open: ${entry.open:,.2f} | "
                    f"Close: ${entry.close:,.2f} | "
                    f"Volume: {entry.volume:,.2f}"
                )

        # 2. Get Fundamental Data (Stocks)
        logger.info("\nFetching Fundamental Data:")
        fundamental_data = await manager.get_fundamental_data(symbols=stock_symbols)
        
        for symbol, data in fundamental_data.items():
            logger.info(f"\n{symbol} Fundamental Data:")
            logger.info(f"Market Cap: ${data.metadata['market_cap']:,.2f}")
            logger.info(f"P/E Ratio: {data.financial_ratios['pe_ratio']:.2f}")
            logger.info(f"Revenue: ${data.income_statement['revenue']:,.2f}")
            logger.info(f"Net Income: ${data.income_statement['net_income']:,.2f}")
            logger.info(f"Free Cash Flow: ${data.cash_flow['free_cash_flow']:,.2f}")

        # 3. Get Sentiment Data (Stocks)
        logger.info("\nFetching Sentiment Data:")
        sentiment_data = await manager.get_sentiment_data(symbols=stock_symbols)
        
        for symbol, data in sentiment_data.items():
            logger.info(f"\n{symbol} Sentiment Analysis:")
            logger.info(f"News Sentiment: {data.news_sentiment:.1%}")
            logger.info(f"Social Sentiment: {data.social_sentiment:.1%}")
            
            # Show recent news
            if data.source_data.get("news"):
                logger.info("\nRecent News Headlines:")
                for article in sorted(
                    data.source_data["news"],
                    key=lambda x: x["datetime"],
                    reverse=True
                )[:3]:  # Show 3 most recent
                    logger.info(
                        f"\nHeadline: {article['headline']}"
                        f"\nSource: {article['source']}"
                        f"\nTime: {article['datetime']}"
                        f"\nSentiment: {article['sentiment_score']:.1%}"
                    )

        # 4. Get Technical Indicators
        logger.info("\nCalculating Technical Indicators:")
        technical_data = manager.get_technical_indicators(market_data)
        
        for symbol, indicators in technical_data.items():
            logger.info(f"\n{symbol} Technical Analysis:")
            logger.info(f"RSI: {indicators.momentum['rsi'][-1]:.2f}")
            logger.info(f"MACD: {indicators.trend['macd'][-1]:.2f}")
            logger.info(f"Signal: {indicators.trend['macd_signal'][-1]:.2f}")
            logger.info(f"BB Upper: {indicators.volatility['bb_upper'][-1]:.2f}")
            logger.info(f"BB Lower: {indicators.volatility['bb_lower'][-1]:.2f}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        if 'manager' in locals():
            await manager.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())