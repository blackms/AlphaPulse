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


async def main():
    """Main execution function."""
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

        for symbol in symbols:
            logger.info(f"\nAnalyzing sentiment for {symbol}...")

            try:
                # Get combined sentiment data
                sentiment_data = await provider.get_sentiment_data(
                    symbol=symbol,
                    lookback_days=7,  # 1 week of news
                    lookback_hours=24  # 24 hours of social data
                )
                
                # Print overall sentiment scores
                logger.info("\nOverall Sentiment Scores:")
                logger.info(f"News Sentiment: {sentiment_data.news_sentiment:.2%}")
                logger.info(f"Social Sentiment: {sentiment_data.social_sentiment:.2%}")

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
                            f"\nTime: {article['datetime']}"
                            f"\nSentiment Score: {article['sentiment_score']:.2f}"
                            f"\nRelevance Score: {article['relevance_score']:.2f}"
                        )
                        logger.info("Sentiment Breakdown:")
                        logger.info(
                            f"- Bullish: {article['sentiment']['bullish']:.1%}"
                            f"\n- Bearish: {article['sentiment']['bearish']:.1%}"
                            f"\n- Neutral: {article['sentiment']['neutral']:.1%}"
                        )

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
                            logger.info(f"Total Mentions: {total_mentions}")
                            logger.info(f"Average Sentiment: {avg_sentiment:.2%}")
                            logger.info(
                                "Positive/Negative Ratio: "
                                f"{sum(item['positive_score'] for item in platform_data):.1f}/"
                                f"{sum(item['negative_score'] for item in platform_data):.1f}"
                            )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

            # Add delay to respect rate limits
            await asyncio.sleep(1)  # Finnhub has 60 API calls/minute limit

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        await provider.__aexit__(None, None, None)


def print_sentiment_summary(sentiment_score: float) -> str:
    """Get a human-readable sentiment summary."""
    if sentiment_score >= 0.5:
        return "Very Bullish"
    elif sentiment_score >= 0.2:
        return "Bullish"
    elif sentiment_score >= -0.2:
        return "Neutral"
    elif sentiment_score >= -0.5:
        return "Bearish"
    else:
        return "Very Bearish"


if __name__ == "__main__":
    asyncio.run(main())