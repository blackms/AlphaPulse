"""
Base classes for data providers implementing common functionality.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from functools import wraps

from ..interfaces import (
    DataProvider,
    DataFetchError,
    DataValidationError,
    ProviderError,
    MarketData,
    FundamentalData,
    SentimentData
)

logger = logging.getLogger(__name__)

T = TypeVar('T', MarketData, FundamentalData, SentimentData)


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}")
            raise last_error
        return wrapper
    return decorator


class BaseDataProvider(DataProvider, ABC):
    """Base class for all data providers."""

    def __init__(
        self,
        name: str,
        provider_type: str,
        api_key: Optional[str] = None,
        rate_limit: Optional[int] = None
    ):
        """Initialize base provider."""
        self._name = name
        self._type = provider_type
        self._api_key = api_key
        self._rate_limit = rate_limit
        self._last_request_time: Dict[str, datetime] = {}
        self._request_semaphore = asyncio.Semaphore(rate_limit if rate_limit else 10)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._name

    @property
    def provider_type(self) -> str:
        """Get provider type."""
        return self._type

    async def _handle_rate_limit(self, endpoint: str):
        """Handle rate limiting for API calls."""
        if self._rate_limit:
            last_time = self._last_request_time.get(endpoint)
            if last_time:
                elapsed = datetime.now() - last_time
                min_interval = timedelta(seconds=1 / self._rate_limit)
                if elapsed < min_interval:
                    await asyncio.sleep((min_interval - elapsed).total_seconds())
        
        self._last_request_time[endpoint] = datetime.now()

    async def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make API request with rate limiting and retries."""
        async with self._request_semaphore:
            await self._handle_rate_limit(endpoint)
            try:
                response = await self._execute_request(
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    headers=headers,
                    data=data
                )
                return await self._process_response(response)
            except Exception as e:
                logger.error(
                    f"Request failed for {endpoint}: {str(e)}",
                    exc_info=True
                )
                raise DataFetchError(f"Failed to fetch data: {str(e)}")

    @abstractmethod
    async def _execute_request(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        data: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute the actual API request."""
        pass

    @abstractmethod
    async def _process_response(self, response: Any) -> Any:
        """Process API response."""
        pass

    def _validate_data(self, data: T) -> bool:
        """Validate data before returning."""
        if not data:
            return False
        
        try:
            if isinstance(data, MarketData):
                return self._validate_market_data(data)
            elif isinstance(data, FundamentalData):
                return self._validate_fundamental_data(data)
            elif isinstance(data, SentimentData):
                return self._validate_sentiment_data(data)
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}", exc_info=True)
            return False

    def _validate_market_data(self, data: MarketData) -> bool:
        """Validate market data."""
        return all([
            data.symbol,
            data.timestamp,
            data.open >= 0,
            data.high >= data.low,
            data.close >= 0,
            data.volume >= 0
        ])

    def _validate_fundamental_data(self, data: FundamentalData) -> bool:
        """Validate fundamental data."""
        return all([
            data.symbol,
            data.timestamp,
            data.financial_ratios is not None,
            data.balance_sheet is not None,
            data.income_statement is not None,
            data.cash_flow is not None
        ])

    def _validate_sentiment_data(self, data: SentimentData) -> bool:
        """Validate sentiment data."""
        return all([
            data.symbol,
            data.timestamp,
            -1 <= data.news_sentiment <= 1,
            -1 <= data.social_sentiment <= 1,
            -1 <= data.analyst_sentiment <= 1,
            data.source_data is not None
        ])

    async def validate_and_clean_data(self, data: Union[T, List[T]]) -> Union[T, List[T]]:
        """Validate and clean data before returning."""
        if isinstance(data, list):
            valid_data = []
            for item in data:
                try:
                    if self._validate_data(item):
                        valid_data.append(item)
                    else:
                        logger.warning(f"Invalid data found for {item.symbol}")
                except Exception as e:
                    logger.error(f"Error validating data: {str(e)}", exc_info=True)
            return valid_data
        else:
            if not self._validate_data(data):
                raise DataValidationError(f"Invalid data for {data.symbol}")
            return data

    def _format_log_message(self, message: str, **kwargs) -> str:
        """Format log message with provider info."""
        return f"[{self.provider_name}] {message} " + \
               " ".join(f"{k}={v}" for k, v in kwargs.items())

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class CacheMixin:
    """Mixin for adding caching functionality to data providers."""

    def __init__(self, cache_ttl: int = 300):
        """Initialize cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = cache_ttl
        self._cache_timestamps: Dict[str, datetime] = {}

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        return f"{args}:{kwargs}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if key in self._cache:
            timestamp = self._cache_timestamps.get(key)
            if timestamp and (datetime.now() - timestamp).seconds < self._cache_ttl:
                return self._cache[key]
        return None

    def _store_in_cache(self, key: str, data: Any):
        """Store data in cache."""
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()

    def _clear_cache(self):
        """Clear expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if (now - timestamp).seconds >= self._cache_ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]


class RateLimitMixin:
    """Mixin for rate limiting functionality."""

    def __init__(self, requests_per_second: float = 1.0):
        """Initialize rate limiter."""
        self._requests_per_second = requests_per_second
        self._last_request_time: Dict[str, datetime] = {}
        self._rate_limit_lock = asyncio.Lock()

    async def _wait_for_rate_limit(self, endpoint: str):
        """Wait for rate limit if needed."""
        async with self._rate_limit_lock:
            last_time = self._last_request_time.get(endpoint)
            if last_time:
                elapsed = datetime.now() - last_time
                min_interval = timedelta(seconds=1 / self._requests_per_second)
                if elapsed < min_interval:
                    await asyncio.sleep((min_interval - elapsed).total_seconds())
            self._last_request_time[endpoint] = datetime.now()