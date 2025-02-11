"""
News and social media sentiment provider implementation.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
from textblob import TextBlob
import tweepy
from collections import defaultdict

from ...interfaces import SentimentData
from ..base import BaseDataProvider, retry_on_error, CacheMixin


class NewsSentimentProvider(BaseDataProvider, CacheMixin):
    """
    Sentiment data provider implementation using NewsAPI and Twitter.
    
    Features:
    - News sentiment analysis
    - Social media sentiment tracking
    - Real-time sentiment updates
    - Source credibility weighting
    - Engagement metrics
    """

    NEWS_API_BASE_URL = "https://newsapi.org/v2/"
    TWITTER_API_BASE_URL = "https://api.twitter.com/2/"

    def __init__(
        self,
        news_api_key: str,
        twitter_bearer_token: Optional[str] = None,
        cache_ttl: int = 900,  # 15 minutes cache for sentiment data
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize sentiment provider.

        Args:
            news_api_key: NewsAPI key
            twitter_bearer_token: Twitter API bearer token
            cache_ttl: Cache time-to-live in seconds
            session: Optional aiohttp session
        """
        super().__init__("news_sentiment", "sentiment", news_api_key)
        CacheMixin.__init__(self, cache_ttl)
        self._twitter_auth = twitter_bearer_token
        self._session = session
        self._source_weights = self._initialize_source_weights()

    def _initialize_source_weights(self) -> Dict[str, float]:
        """Initialize credibility weights for different news sources."""
        return {
            "reuters": 1.0,
            "bloomberg": 1.0,
            "financial-times": 0.9,
            "wall-street-journal": 0.9,
            "cnbc": 0.8,
            "marketwatch": 0.8,
            "seeking-alpha": 0.7,
            "yahoo-finance": 0.7,
            "default": 0.5
        }

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
        """Execute API request."""
        await self._ensure_session()
        
        if endpoint.startswith("news"):
            url = urljoin(self.NEWS_API_BASE_URL, endpoint)
            headers = {"X-Api-Key": self._api_key}
        else:  # Twitter API
            url = urljoin(self.TWITTER_API_BASE_URL, endpoint)
            headers = {"Authorization": f"Bearer {self._twitter_auth}"}

        async with self._session.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text using TextBlob."""
        try:
            analysis = TextBlob(text)
            # Convert polarity (-1 to 1) to sentiment score
            return analysis.sentiment.polarity
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {str(e)}")
            return 0.0

    def _calculate_source_credibility(self, source: str) -> float:
        """Calculate source credibility score."""
        return self._source_weights.get(source.lower(), self._source_weights["default"])

    @retry_on_error(max_retries=3)
    async def get_news_sentiment(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get news sentiment data.

        Args:
            symbol: Company symbol
            lookback_days: Number of days to look back

        Returns:
            List of news articles with sentiment scores
        """
        cache_key = f"news_{symbol}_{lookback_days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Prepare search query
        company_name = await self._get_company_name(symbol)
        query = f"{company_name} OR {symbol} stock"
        
        params = {
            "q": query,
            "from": (datetime.now() - timedelta(days=lookback_days)).isoformat(),
            "language": "en",
            "sortBy": "relevancy"
        }

        news_data = await self._make_request("everything", params=params)
        if not news_data or "articles" not in news_data:
            return []

        processed_articles = []
        for article in news_data["articles"]:
            # Calculate sentiment scores
            title_sentiment = self._calculate_text_sentiment(article["title"])
            content_sentiment = self._calculate_text_sentiment(
                article.get("description", "") + " " + article.get("content", "")
            )
            
            # Calculate source credibility
            source = article.get("source", {}).get("id", "default")
            credibility = self._calculate_source_credibility(source)
            
            # Combine sentiment scores with credibility weighting
            combined_sentiment = (
                (title_sentiment * 0.4 + content_sentiment * 0.6) * credibility
            )

            processed_articles.append({
                "title": article["title"],
                "url": article["url"],
                "source": article["source"]["name"],
                "published_at": article["publishedAt"],
                "sentiment_score": combined_sentiment,
                "credibility_score": credibility,
                "relevance_score": self._calculate_relevance(article, symbol)
            })

        # Sort by relevance and sentiment impact
        processed_articles.sort(
            key=lambda x: (x["relevance_score"], abs(x["sentiment_score"])),
            reverse=True
        )

        # Cache the processed data
        self._store_in_cache(cache_key, processed_articles)
        return processed_articles

    @retry_on_error(max_retries=3)
    async def get_social_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get social media sentiment data.

        Args:
            symbol: Company symbol
            lookback_hours: Number of hours to look back

        Returns:
            List of social media posts with sentiment scores
        """
        if not self._twitter_auth:
            return []

        cache_key = f"social_{symbol}_{lookback_hours}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Search Twitter for cashtag
        params = {
            "query": f"${symbol}",
            "tweet.fields": "created_at,public_metrics",
            "max_results": 100,
            "start_time": (
                datetime.now() - timedelta(hours=lookback_hours)
            ).isoformat()
        }

        tweets_data = await self._make_request("tweets/search/recent", params=params)
        if not tweets_data or "data" not in tweets_data:
            return []

        processed_tweets = []
        for tweet in tweets_data["data"]:
            # Calculate sentiment
            sentiment_score = self._calculate_text_sentiment(tweet["text"])
            
            # Calculate engagement score
            metrics = tweet["public_metrics"]
            engagement_score = (
                metrics.get("retweet_count", 0) * 2 +
                metrics.get("like_count", 0) +
                metrics.get("reply_count", 0) * 1.5 +
                metrics.get("quote_count", 0) * 1.5
            ) / 100  # Normalize

            processed_tweets.append({
                "text": tweet["text"],
                "created_at": tweet["created_at"],
                "sentiment_score": sentiment_score,
                "engagement_score": min(engagement_score, 1.0),
                "metrics": metrics
            })

        # Sort by engagement and sentiment impact
        processed_tweets.sort(
            key=lambda x: (x["engagement_score"], abs(x["sentiment_score"])),
            reverse=True
        )

        # Cache the processed data
        self._store_in_cache(cache_key, processed_tweets)
        return processed_tweets

    def _calculate_relevance(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate article relevance score."""
        relevance = 0.0
        text = (
            article["title"].lower() + " " +
            article.get("description", "").lower()
        )
        
        # Check for symbol mention
        if symbol.lower() in text:
            relevance += 0.5
            
        # Check for company name mention
        if article.get("company_name", "").lower() in text:
            relevance += 0.3
            
        # Check for title mention
        if symbol.lower() in article["title"].lower():
            relevance += 0.2
            
        return min(relevance, 1.0)

    async def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol."""
        # This could be enhanced to use a company info API
        return symbol

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._session:
            await self._session.close()