"""
Finnhub provider implementation for market and sentiment data.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import urljoin
import logging
import re

from ...interfaces import SentimentData
from ..base import BaseDataProvider, retry_on_error, CacheMixin

logger = logging.getLogger(__name__)


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
        self._initialize_sentiment_patterns()

    def _initialize_sentiment_patterns(self):
        """Initialize regex patterns for sentiment analysis."""
        self._patterns = {
            'positive': [
                r'\b(up|rise|gain|positive|bull|growth|higher|increase|improved|profit|success)\b',
                r'\b(beat|exceed|outperform|strong|boost|surge|jump|rally|soar)\b',
                r'\b(upgrade|recommend|buy|accumulate|overweight)\b',
                r'\b(innovation|breakthrough|patent|launch|expansion)\b'
            ],
            'negative': [
                r'\b(down|fall|loss|negative|bear|decline|lower|decrease|reduced|risk|failed)\b',
                r'\b(miss|underperform|weak|drop|plunge|sink|tumble|crash)\b',
                r'\b(downgrade|sell|reduce|underweight)\b',
                r'\b(lawsuit|investigation|recall|delay|problem|issue)\b'
            ],
            'modifiers': {
                'intensifiers': r'\b(very|highly|extremely|significantly|substantially)\b',
                'diminishers': r'\b(slightly|somewhat|marginally|barely)\b',
                'negators': r'\b(not|no|never|none|neither|nor|without)\b'
            }
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
        """Execute Finnhub API request."""
        await self._ensure_session()
        url = urljoin(self.BASE_URL, endpoint)
        
        request_headers = {**self._headers}
        if headers:
            request_headers.update(headers)

        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                headers=request_headers,
                json=data
            ) as response:
                if response.status == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._execute_request(
                        endpoint, method, params, headers, data
                    )
                
                # For debugging
                if response.status != 200:
                    logger.error(
                        f"API Error: Status {response.status}, "
                        f"URL: {url}, "
                        f"Headers: {request_headers}, "
                        f"Params: {params}"
                    )
                    response_text = await response.text()
                    logger.error(f"Response: {response_text}")
                
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    async def _process_response(self, response: Any) -> Any:
        """Process Finnhub API response."""
        if not response:
            return None
            
        # Check for API error messages
        if isinstance(response, dict):
            if "error" in response:
                raise ValueError(f"API Error: {response['error']}")
            if response.get("s") == "no_data":
                logger.warning("No data available")
                return None
            
        return response

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

        try:
            # Fetch company news
            news_data = await self._make_request(
                "company-news",  # Updated endpoint
                params={
                    "symbol": symbol,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d")
                }
            )
            
            if not news_data:
                return []

            # Process news items
            processed_news = []
            for item in news_data:
                # Only include news that mentions the company
                if not self._is_relevant_to_company(item, symbol):
                    continue
                
                # Calculate sentiment with context
                sentiment_score = self._calculate_sentiment(
                    item.get("headline", "") + " " + item.get("summary", ""),
                    symbol
                )
                
                processed_news.append({
                    "id": item.get("id"),
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)),
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "sentiment_score": sentiment_score,
                    "relevance_score": self._calculate_relevance(item, symbol),
                    "sentiment": {
                        "bullish": max(0, sentiment_score),
                        "bearish": abs(min(0, sentiment_score)),
                        "neutral": 1 - abs(sentiment_score)
                    }
                })

            # Sort by relevance and recency
            processed_news.sort(
                key=lambda x: (x["relevance_score"], x["datetime"]),
                reverse=True
            )

            # Cache the processed data
            self._store_in_cache(cache_key, processed_news)
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            raise

    def _is_relevant_to_company(self, article: Dict[str, Any], symbol: str) -> bool:
        """Check if the article is relevant to the company."""
        text = (
            (article.get("headline", "") + " " + 
             article.get("summary", "")).lower()
        )
        # Check for symbol or common variations
        symbol = symbol.lower()
        if symbol in text:
            return True
            
        # Common company name variations
        variations = {
            'aapl': ['apple', 'iphone', 'ipad', 'mac', 'ios'],
            'msft': ['microsoft', 'windows', 'azure', 'xbox'],
            'googl': ['google', 'alphabet', 'android', 'chrome'],
            'amzn': ['amazon', 'aws', 'prime'],
            'tsla': ['tesla', 'musk', 'elon'],
            'nvda': ['nvidia', 'geforce', 'cuda'],
            'meta': ['facebook', 'instagram', 'whatsapp'],
        }
        
        if symbol in variations:
            return any(term in text for term in variations[symbol])
            
        return False

    def _calculate_sentiment(self, text: str, symbol: str) -> float:
        """
        Calculate sentiment score from text with context awareness.
        
        Args:
            text: The text to analyze
            symbol: Company symbol for context

        Returns:
            Sentiment score between -1 and 1
        """
        text = text.lower()
        
        # Initialize sentiment components
        sentiment = 0.0
        sentence_count = 0
        
        # Split into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_count += 1
            sentence_sentiment = 0.0
            
            # Check for negations
            has_negation = bool(
                re.search(self._patterns['modifiers']['negators'], sentence)
            )
            
            # Count positive and negative patterns
            positive_matches = sum(
                bool(re.search(pattern, sentence))
                for pattern in self._patterns['positive']
            )
            negative_matches = sum(
                bool(re.search(pattern, sentence))
                for pattern in self._patterns['negative']
            )
            
            # Calculate base sentiment
            if positive_matches or negative_matches:
                base_sentiment = (positive_matches - negative_matches) / (
                    positive_matches + negative_matches
                )
                
                # Apply negation
                if has_negation:
                    base_sentiment *= -1
                    
                # Check for intensity modifiers
                if re.search(self._patterns['modifiers']['intensifiers'], sentence):
                    base_sentiment *= 1.5
                elif re.search(self._patterns['modifiers']['diminishers'], sentence):
                    base_sentiment *= 0.5
                    
                sentence_sentiment = base_sentiment
            
            sentiment += sentence_sentiment
        
        # Normalize final sentiment
        if sentence_count > 0:
            sentiment = sentiment / sentence_count
            
        # Ensure bounds
        return max(min(sentiment, 1.0), -1.0)

    def _calculate_relevance(self, article: Dict[str, Any], symbol: str) -> float:
        """
        Calculate article relevance score.
        
        Args:
            article: News article
            symbol: Company symbol

        Returns:
            Relevance score between 0 and 1
        """
        relevance = 0.0
        text = (
            article.get("headline", "").lower() + " " +
            article.get("summary", "").lower()
        )
        
        # Direct symbol mention
        if symbol.lower() in text:
            relevance += 0.4
            
        # Company name variations
        variations = {
            'aapl': ['apple', 'iphone', 'ipad', 'mac', 'ios'],
            'msft': ['microsoft', 'windows', 'azure', 'xbox'],
            'googl': ['google', 'alphabet', 'android', 'chrome'],
            'amzn': ['amazon', 'aws', 'prime'],
            'tsla': ['tesla', 'musk', 'elon'],
            'nvda': ['nvidia', 'geforce', 'cuda'],
            'meta': ['facebook', 'instagram', 'whatsapp'],
        }
        
        if symbol.lower() in variations:
            matches = sum(term in text for term in variations[symbol.lower()])
            if matches:
                relevance += 0.3 * (matches / len(variations[symbol.lower()]))
        
        # Title mention is more important
        if symbol.lower() in article.get("headline", "").lower():
            relevance += 0.3
            
        # Consider source credibility
        source = article.get("source", "").lower()
        credible_sources = {
            'reuters': 0.2,
            'bloomberg': 0.2,
            'cnbc': 0.15,
            'wsj': 0.15,
            'financial times': 0.15,
            'marketwatch': 0.1
        }
        relevance += credible_sources.get(source, 0.05)
        
        return min(relevance, 1.0)

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
        # For now, return empty list as social sentiment requires premium API access
        logger.info("Social sentiment data requires Finnhub premium API access")
        return []

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
        try:
            # Fetch news sentiment (social sentiment is premium only)
            news_data = await self.get_news_sentiment(symbol, lookback_days)
            
            # Calculate weighted sentiment scores
            if news_data:
                total_relevance = sum(item["relevance_score"] for item in news_data)
                if total_relevance > 0:
                    news_sentiment = sum(
                        item["sentiment_score"] * item["relevance_score"]
                        for item in news_data
                    ) / total_relevance
                else:
                    news_sentiment = 0.0
            else:
                news_sentiment = 0.0

            return SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
                news_sentiment=news_sentiment,
                social_sentiment=0.0,  # Not available in free tier
                analyst_sentiment=0.0,  # Not implemented
                source_data={
                    "news": news_data,
                    "social": []  # Not available in free tier
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting sentiment data: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        if self._session:
            await self._session.close()