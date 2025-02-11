"""
Finnhub provider implementation for market and sentiment data.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import urljoin

from ...interfaces import SentimentData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class FinnhubProvider(BaseDataProvider, CacheMixin):
    """
    Finnhub provider implementation.
    
    Features:
    - Real-time stock data
    - Company news
    - Sentiment analysis
    - Earnings estimates
    - Price targets
    """

    BASE_URL = "https://finnhub.io/api/v1/"

    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 300,  # 5 minutes cache
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize Finnhub provider.

        Args:
            api_key: Finnhub API key
            cache_ttl: Cache time-to-live in seconds
            session: Optional aiohttp session
        """
        super().__init__("finnhub", "sentiment", api_key)
        CacheMixin.__init__(self, cache_ttl)
        self._session = session
        self._headers = {"X-Finnhub-Token": api_key}

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def _execute_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute Finnhub API request."""
        await self._ensure_session()
        url = urljoin(self.BASE_URL, endpoint)
        
        request_headers = {**self._headers}
        if headers:
            request_headers.update(headers)

        async with self._session.request(
            method=method,
            url=url,
            params=params,
            headers=request_headers,
            json=data
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _calculate_sentiment_score(self, news_items: List[Dict]) -> float:
        """Calculate sentiment score from news items."""
        if not news_items:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for item in news_items:
            sentiment = item.get("sentiment", 0)
            # Weight by relevance score
            weight = item.get("relevance", 1)
            total_score += sentiment * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    @retry_on_error(max_retries=3)
    async def get_news_sentiment(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get news sentiment data from Finnhub.

        Args:
            symbol: Company symbol
            lookback_days: Number of days to look back

        Returns:
            List of news items with sentiment scores
        """
        cache_key = f"news_{symbol}_{lookback_days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Fetch company news
        params = {
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d")
        }
        
        news_data = await self._make_request("company-news", params=params)
        
        # Process news items
        processed_news = []
        for item in news_data:
            # Get sentiment for each news item
            sentiment_params = {
                "symbol": symbol,
                "id": item.get("id")
            }
            try:
                sentiment_data = await self._make_request(
                    "news-sentiment",
                    params=sentiment_params
                )
                
                processed_news.append({
                    "id": item.get("id"),
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)),
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "sentiment_score": sentiment_data.get("sentiment", 0),
                    "relevance_score": sentiment_data.get("relevance", 1),
                    "sentiment": {
                        "bullish": sentiment_data.get("bullishPercent", 0),
                        "bearish": sentiment_data.get("bearishPercent", 0),
                        "neutral": sentiment_data.get("neutralPercent", 0)
                    }
                })
            except Exception as e:
                logger.warning(f"Error getting sentiment for news item: {str(e)}")
                continue

        # Cache the processed data
        self._store_in_cache(cache_key, processed_news)
        return processed_news

    @retry_on_error(max_retries=3)
    async def get_social_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get social sentiment data from Finnhub.

        Args:
            symbol: Company symbol
            lookback_hours: Number of hours to look back

        Returns:
            List of social sentiment data points
        """
        cache_key = f"social_{symbol}_{lookback_hours}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Get social sentiment
        params = {"symbol": symbol}
        sentiment_data = await self._make_request("social-sentiment", params=params)

        # Process sentiment data
        processed_data = []
        for platform, data in sentiment_data.items():
            if isinstance(data, list):
                for item in data:
                    if (datetime.now() - datetime.fromtimestamp(item.get("timestamp", 0))
                        ).total_seconds() <= lookback_hours * 3600:
                        processed_data.append({
                            "platform": platform,
                            "timestamp": datetime.fromtimestamp(item.get("timestamp", 0)),
                            "mention_count": item.get("mention", 0),
                            "positive_score": item.get("positiveScore", 0),
                            "negative_score": item.get("negativeScore", 0),
                            "sentiment_score": (
                                item.get("positiveScore", 0) -
                                item.get("negativeScore", 0)
                            ),
                            "engagement_score": item.get("mention", 0) / 100
                        })

        # Cache the processed data
        self._store_in_cache(cache_key, processed_data)
        return processed_data

    async def get_sentiment_data(
        self,
        symbol: str,
        lookback_days: int = 30,
        lookback_hours: int = 24
    ) -> SentimentData:
        """
        Get combined sentiment data.

        Args:
            symbol: Company symbol
            lookback_days: Days to look back for news
            lookback_hours: Hours to look back for social data

        Returns:
            SentimentData object
        """
        # Fetch news and social sentiment concurrently
        news_task = self.get_news_sentiment(symbol, lookback_days)
        social_task = self.get_social_sentiment(symbol, lookback_hours)
        
        news_data, social_data = await asyncio.gather(news_task, social_task)
        
        # Calculate sentiment scores
        news_sentiment = self._calculate_sentiment_score(news_data)
        
        social_sentiment = 0.0
        if social_data:
            social_sentiment = sum(
                item["sentiment_score"] * item["engagement_score"]
                for item in social_data
            ) / sum(item["engagement_score"] for item in social_data)

        return SentimentData(
            symbol=symbol,
            timestamp=datetime.now(),
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            analyst_sentiment=0.0,  # Could be implemented using price targets
            source_data={
                "news": news_data,
                "social": social_data
            }
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._session:
            await self._session.close()