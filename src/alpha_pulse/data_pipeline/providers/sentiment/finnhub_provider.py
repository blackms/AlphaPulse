"""
Finnhub sentiment data provider implementation.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import aiohttp

from ...interfaces import SentimentData, DataFetchError
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class FinnhubProvider(BaseDataProvider, CacheMixin):
    """
    Finnhub sentiment data provider implementation.
    
    Features:
    - News sentiment analysis
    - Social sentiment analysis
    - Company news
    - Market news
    """

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 300,  # 5 minutes cache for sentiment data
        request_timeout: int = 30,
        tcp_connector_limit: int = 100
    ):
        """
        Initialize Finnhub provider.

        Args:
            api_key: Finnhub API key
            cache_ttl: Cache time-to-live in seconds
            request_timeout: Request timeout in seconds
            tcp_connector_limit: Maximum number of concurrent connections
        """
        BaseDataProvider.__init__(
            self,
            provider_name="finnhub",
            provider_type="sentiment",
            base_url="https://finnhub.io/api/v1",
            default_headers={"X-Finnhub-Token": api_key},
            request_timeout=request_timeout,
            tcp_connector_limit=tcp_connector_limit
        )
        CacheMixin.__init__(self, cache_ttl=cache_ttl)

    def _calculate_sentiment_score(self, news_data: List[Dict[str, Any]]) -> float:
        """Calculate sentiment score from news data."""
        if not news_data:
            return 0.0
            
        total_sentiment = 0.0
        count = 0
        
        for article in news_data:
            sentiment = article.get('sentiment')
            if sentiment is not None:
                total_sentiment += sentiment
                count += 1
                
        return (total_sentiment / count) if count > 0 else 0.0

    @retry_on_error(retries=3, delay=2.0)
    async def get_news_sentiment(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get news sentiment data.

        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back

        Returns:
            Dictionary containing news data and sentiment scores
        """
        cache_key = f"news_{symbol}_{lookback_days}"
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            logger.debug(f"Fetching news data for {symbol}")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get company news
            response = await self._execute_request(
                endpoint="company-news",
                params={
                    'symbol': symbol,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d')
                }
            )
            
            # Parse JSON response
            news_data = await self._process_response(response)
            if not isinstance(news_data, list):
                raise DataFetchError(f"Unexpected response format: {news_data}")
            
            # Process news data
            processed_news = []
            for article in news_data:
                # Calculate sentiment score (-1 to 1)
                sentiment_score = article.get('sentiment', 0)
                
                processed_news.append({
                    'datetime': article.get('datetime'),
                    'headline': article.get('headline'),
                    'source': article.get('source'),
                    'url': article.get('url'),
                    'sentiment_score': sentiment_score
                })
            
            result = {
                'news': processed_news,
                'sentiment_score': self._calculate_sentiment_score(processed_news)
            }
            
            # Cache the results
            self._set_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error fetching news data: {str(e)}")
            raise

    @retry_on_error(retries=3, delay=2.0)
    async def get_social_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get social sentiment data.

        Args:
            symbol: Trading symbol
            lookback_hours: Number of hours to look back

        Returns:
            Dictionary containing social sentiment data
        """
        cache_key = f"social_{symbol}_{lookback_hours}"
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            logger.debug(f"Fetching social sentiment for {symbol}")
            
            # Calculate timestamp
            from_time = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp())
            
            response = await self._execute_request(
                endpoint="stock/social-sentiment",
                params={
                    'symbol': symbol,
                    'from': from_time
                }
            )
            
            # Parse JSON response
            sentiment_data = await self._process_response(response)
            if not isinstance(sentiment_data, dict):
                raise DataFetchError(f"Unexpected response format: {sentiment_data}")
            
            # Calculate average sentiment
            reddit = sentiment_data.get('reddit', [])
            twitter = sentiment_data.get('twitter', [])
            
            reddit_score = sum(post.get('sentiment', 0) for post in reddit) / len(reddit) if reddit else 0
            twitter_score = sum(tweet.get('sentiment', 0) for tweet in twitter) / len(twitter) if twitter else 0
            
            result = {
                'reddit_sentiment': reddit_score,
                'twitter_sentiment': twitter_score,
                'overall_sentiment': (reddit_score + twitter_score) / 2 if reddit or twitter else 0,
                'raw_data': sentiment_data
            }
            
            # Cache the results
            self._set_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error fetching social sentiment: {str(e)}")
            raise

    async def get_sentiment_data(
        self,
        symbol: str,
        lookback_days: int = 7,
        lookback_hours: int = 24
    ) -> SentimentData:
        """
        Get combined sentiment data.

        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back for news
            lookback_hours: Number of hours to look back for social data

        Returns:
            SentimentData object
        """
        try:
            # Get news sentiment
            news_data = await self.get_news_sentiment(symbol, lookback_days)
            
            # Get social sentiment
            social_data = await self.get_social_sentiment(symbol, lookback_hours)
            
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
                news_sentiment=news_data['sentiment_score'],
                social_sentiment=social_data['overall_sentiment'],
                source_data={
                    'news': news_data['news'],
                    'social': social_data['raw_data']
                }
            )
            
        except Exception as e:
            logger.exception(f"Error getting sentiment data for {symbol}: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await super().__aexit__(exc_type, exc_val, exc_tb)