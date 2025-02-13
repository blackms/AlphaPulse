"""
Example script demonstrating Finnhub data fetching.
"""
import asyncio
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

from alpha_pulse.data_pipeline.providers.sentiment.finnhub_provider import (
    FinnhubProvider
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_sentiment(score: float) -> str:
    """Format sentiment score with description."""
    if score >= 0.5:
        return f"{score:.1%} (Very Bullish)"
    elif score >= 0.2:
        return f"{score:.1%} (Bullish)"
    elif score >= -0.2:
        return f"{score:.1%} (Neutral)"
    elif score >= -0.5:
        return f"{score:.1%} (Bearish)"
    else:
        return f"{score:.1%} (Very Bearish)"


async def analyze_symbol(provider: FinnhubProvider, symbol: str):
    """Analyze sentiment for a single symbol."""
    try:
        logger.info(f"\nAnalyzing sentiment for {symbol}...")

        # Get sentiment data with shorter lookback periods for demo
        sentiment_data = await provider.get_sentiment_data(
            symbol=symbol,
            lookback_days=7,  # 1 week of news
            lookback_hours=24  # 24 hours of social data
        )
        
        # Print overall sentiment scores
        logger.info("\nOverall Sentiment Analysis:")
        logger.info(f"News Sentiment: {format_sentiment(sentiment_data.news_sentiment)}")
        logger.info(f"Social Sentiment: {format_sentiment(sentiment_data.social_sentiment)}")

        # Print recent news
        news_data = sentiment_data.source_data["news"]
        if news_data:
            logger.info("\nRecent News Headlines:")
            for article in sorted(
                news_data,
                key=lambda x: x["datetime"],
                reverse=True
            )[:5]:  # Show 5 most recent
                logger.info(
                    f"\nHeadline: {article['headline']}"
                    f"\nSource: {article['source']}"
                    f"\nTime: {article['datetime'].strftime('%Y-%m-%d %H:%M:%S')}"
                    f"\nSentiment: {format_sentiment(article['sentiment_score'])}"
                )
                if article.get('sentiment'):
                    logger.info("Sentiment Breakdown:")
                    logger.info(
                        f"- Bullish: {article['sentiment']['bullish']:.1%}"
                        f"\n- Bearish: {article['sentiment']['bearish']:.1%}"
                        f"\n- Neutral: {article['sentiment']['neutral']:.1%}"
                    )
        else:
            logger.info("\nNo recent news found.")

        # Print social sentiment
        social_data = sentiment_data.source_data["social"]
        if social_data:
            logger.info("\nSocial Media Sentiment:")
            platforms = set(item["platform"] for item in social_data)
            
            for platform in platforms:
                platform_data = [
                    item for item in social_data
                    if item["platform"] == platform
                ]
                
                if platform_data:
                    total_mentions = sum(
                        item["mention_count"] for item in platform_data
                    )
                    avg_sentiment = sum(
                        item["sentiment_score"] * item["mention_count"]
                        for item in platform_data
                    ) / total_mentions if total_mentions > 0 else 0
                    
                    logger.info(f"\n{platform.title()} Statistics:")
                    logger.info(f"Total Mentions: {total_mentions:,}")
                    logger.info(f"Average Sentiment: {format_sentiment(avg_sentiment)}")
                    
                    # Calculate positive/negative ratio
                    total_positive = sum(
                        item["positive_score"] * item["mention_count"]
                        for item in platform_data
                    )
                    total_negative = sum(
                        item["negative_score"] * item["mention_count"]
                        for item in platform_data
                    )
                    logger.info(
                        f"Positive/Negative Ratio: {total_positive:,.1f}/{total_negative:,.1f}"
                    )
        else:
            logger.info("\nNo social media data found.")

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")


async def main():
    """Main execution function."""
    provider = None
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not found in environment")

        # Initialize provider
        provider = FinnhubProvider(api_key=api_key)

        # Test symbols (popular tech stocks)
        symbols = ["AAPL", "TSLA", "NVDA"]

        # Analyze each symbol
        for symbol in symbols:
            await analyze_symbol(provider, symbol)
            # Add delay between symbols to respect rate limits
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        if provider:
            await provider.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())