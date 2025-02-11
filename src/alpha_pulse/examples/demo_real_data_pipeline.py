"""
Example script demonstrating the real data pipeline usage.
"""
import asyncio
import os
from datetime import datetime, timedelta
import yaml
from dotenv import load_dotenv
import logging
from pathlib import Path

from alpha_pulse.data_pipeline.manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from YAML and environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Load configuration file
    config_path = Path("config/data_pipeline_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        )
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Replace environment variable placeholders
    config["market_data"]["binance"]["api_key"] = os.getenv("BINANCE_API_KEY")
    config["market_data"]["binance"]["api_secret"] = os.getenv("BINANCE_API_SECRET")
    config["fundamental_data"]["financial_modeling_prep"]["api_key"] = os.getenv("FMP_API_KEY")
    config["sentiment_data"]["news_api"]["api_key"] = os.getenv("NEWS_API_KEY")
    config["sentiment_data"]["twitter"]["bearer_token"] = os.getenv("TWITTER_BEARER_TOKEN")
    
    return config


async def demo_market_data(data_manager: DataManager):
    """Demonstrate market data fetching."""
    logger.info("Fetching market data...")
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    market_data = await data_manager.get_market_data(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        interval="1d"
    )
    
    for symbol, data in market_data.items():
        logger.info(f"\n{symbol} market data:")
        logger.info(f"Number of data points: {len(data)}")
        if data:
            latest = data[-1]
            logger.info(f"Latest price: ${latest.close:.2f}")
            logger.info(f"24h volume: {latest.volume:.2f}")
            logger.info(f"24h high: ${latest.high:.2f}")
            logger.info(f"24h low: ${latest.low:.2f}")


async def demo_fundamental_data(data_manager: DataManager):
    """Demonstrate fundamental data fetching."""
    logger.info("\nFetching fundamental data...")
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    fundamental_data = await data_manager.get_fundamental_data(symbols)
    
    for symbol, data in fundamental_data.items():
        logger.info(f"\n{symbol} fundamental metrics:")
        logger.info(f"P/E Ratio: {data.financial_ratios['pe_ratio']:.2f}")
        logger.info(f"P/B Ratio: {data.financial_ratios['pb_ratio']:.2f}")
        logger.info(f"ROE: {data.financial_ratios['roe']:.2%}")
        logger.info(f"Debt/Equity: {data.financial_ratios['debt_to_equity']:.2f}")
        logger.info(f"Operating Margin: {data.financial_ratios['operating_margin']:.2%}")


async def demo_sentiment_data(data_manager: DataManager):
    """Demonstrate sentiment data fetching."""
    logger.info("\nFetching sentiment data...")
    
    symbols = ["TSLA", "NVDA", "META"]
    sentiment_data = await data_manager.get_sentiment_data(symbols)
    
    for symbol, data in sentiment_data.items():
        logger.info(f"\n{symbol} sentiment analysis:")
        logger.info(f"News Sentiment: {data.news_sentiment:.2%}")
        logger.info(f"Social Sentiment: {data.social_sentiment:.2%}")
        
        # Show some news samples
        news = data.source_data.get("news", [])
        if news:
            logger.info("\nRecent news headlines:")
            for article in news[:3]:
                logger.info(
                    f"- {article['title']} "
                    f"(Sentiment: {article['sentiment_score']:.2f})"
                )


async def demo_technical_analysis(data_manager: DataManager):
    """Demonstrate technical analysis."""
    logger.info("\nCalculating technical indicators...")
    
    # First get market data
    symbols = ["BTCUSDT", "ETHUSDT"]
    end_time = datetime.now()
    start_time = end_time - timedelta(days=100)  # Need more data for indicators
    
    market_data = await data_manager.get_market_data(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        interval="1d"
    )
    
    # Calculate technical indicators
    technical_data = data_manager.get_technical_indicators(market_data)
    
    for symbol, data in technical_data.items():
        logger.info(f"\n{symbol} technical analysis:")
        
        # Trend indicators
        logger.info("Trend Indicators:")
        for name, value in data.trend_indicators.items():
            logger.info(f"- {name}: {value:.2f}")
        
        # Momentum indicators
        logger.info("\nMomentum Indicators:")
        for name, value in data.momentum_indicators.items():
            logger.info(f"- {name}: {value:.2f}")
        
        # Volatility indicators
        logger.info("\nVolatility Indicators:")
        for name, value in data.volatility_indicators.items():
            logger.info(f"- {name}: {value:.2f}")
        
        # Volume indicators
        logger.info("\nVolume Indicators:")
        for name, value in data.volume_indicators.items():
            logger.info(f"- {name}: {value:.2f}")


async def main():
    """Main execution function."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize data manager
        async with DataManager(config) as data_manager:
            # Run demonstrations
            await demo_market_data(data_manager)
            await demo_fundamental_data(data_manager)
            await demo_sentiment_data(data_manager)
            await demo_technical_analysis(data_manager)
            
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())