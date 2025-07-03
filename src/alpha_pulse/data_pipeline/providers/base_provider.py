"""
Enhanced base provider implementation for real market data feeds.

Provides common functionality for all data providers including:
- Rate limiting and request management
- Data caching and normalization
- Health monitoring and failover support
- Error handling and retry logic
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
import redis.asyncio as redis
from loguru import logger

from ..interfaces import MarketData, DataFetchError, ProviderError
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class DataQuality(Enum):
    """Data quality indicators."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POOR = "poor"


@dataclass
class ProviderHealth:
    """Provider health metrics."""
    status: ProviderStatus
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    success_rate: float = 0.0
    avg_latency: float = 0.0
    error_count: int = 0
    total_requests: int = 0


@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    current_usage: int = 0
    reset_time: Optional[datetime] = None


@dataclass
class DataNormalizationResult:
    """Result of data normalization."""
    normalized_data: Any
    quality_score: float
    source_metadata: Dict[str, Any]
    normalization_errors: List[str]


class BaseRealDataProvider(ABC):
    """Enhanced base class for real market data providers."""

    def __init__(
        self,
        provider_name: str,
        provider_type: str,
        base_url: str,
        api_key: str,
        rate_limits: RateLimitInfo,
        default_headers: Optional[Dict[str, str]] = None,
        request_timeout: int = 30,
        cache_ttl: int = 300,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize real data provider.

        Args:
            provider_name: Name of the provider
            provider_type: Type of data provided (market, fundamental, etc.)
            base_url: Base URL for API requests
            api_key: API key for authentication
            rate_limits: Rate limiting configuration
            default_headers: Default headers for requests
            request_timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
            redis_url: Redis connection URL for caching
        """
        self.provider_name = provider_name
        self.provider_type = provider_type
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limits = rate_limits
        self._default_headers = default_headers or {}
        self._request_timeout = request_timeout
        self._cache_ttl = cache_ttl
        self._redis_url = redis_url
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Rate limiting
        self._rate_limiter = asyncio.Semaphore(rate_limits.requests_per_second)
        self._request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Health monitoring
        self._health = ProviderHealth(status=ProviderStatus.HEALTHY)
        self._health_lock = asyncio.Lock()
        
        # Redis cache
        self._redis: Optional[redis.Redis] = None
        
        # Audit logging
        self._audit_logger = get_audit_logger()

    async def initialize(self) -> None:
        """Initialize provider connections and resources."""
        try:
            # Initialize Redis connection
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                health_check_interval=30
            )
            await self._redis.ping()
            
            # Initialize HTTP session
            await self._get_session()
            
            # Test provider connectivity
            await self._health_check()
            
            logger.info(f"Initialized {self.provider_name} provider successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider_name} provider: {e}")
            await self._update_health(ProviderStatus.OFFLINE)
            raise ProviderError(f"Provider initialization failed: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper configuration."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    # Prepare headers with API authentication
                    headers = {**self._default_headers}
                    await self._add_auth_headers(headers)
                    
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=20,
                        enable_cleanup_closed=True
                    )
                    
                    timeout = aiohttp.ClientTimeout(total=self._request_timeout)
                    
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers=headers
                    )
                    
        return self._session

    @abstractmethod
    async def _add_auth_headers(self, headers: Dict[str, str]) -> None:
        """Add authentication headers specific to provider."""
        pass

    async def _rate_limited_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> aiohttp.ClientResponse:
        """Execute rate-limited HTTP request."""
        # Rate limiting
        await self._rate_limiter.acquire()
        
        try:
            # Track request timing
            start_time = time.time()
            
            # Clean up old request times
            await self._cleanup_request_times()
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Execute request
            session = await self._get_session()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            request_headers = {**self._default_headers, **(headers or {})}
            
            async with session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                raise_for_status=True
            ) as response:
                # Record request metrics
                latency = time.time() - start_time
                await self._record_request_metrics(latency, True)
                
                # Log successful request
                self._audit_logger.log(
                    event_type=AuditEventType.DATA_ACCESS,
                    event_data={
                        "provider": self.provider_name,
                        "endpoint": endpoint,
                        "method": method,
                        "status_code": response.status,
                        "latency_ms": int(latency * 1000),
                        "response_size": len(await response.read())
                    },
                    severity=AuditSeverity.INFO
                )
                
                return response
                
        except Exception as e:
            # Record failure metrics
            await self._record_request_metrics(time.time() - start_time, False)
            
            # Log failed request
            self._audit_logger.log(
                event_type=AuditEventType.DATA_ACCESS,
                event_data={
                    "provider": self.provider_name,
                    "endpoint": endpoint,
                    "method": method,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                severity=AuditSeverity.ERROR
            )
            
            # Update health status
            await self._update_health(ProviderStatus.DEGRADED)
            
            raise DataFetchError(f"Request failed for {self.provider_name}: {e}")
        
        finally:
            self._rate_limiter.release()

    async def _cleanup_request_times(self) -> None:
        """Clean up old request timestamps for rate limiting."""
        async with self._rate_limit_lock:
            current_time = time.time()
            # Keep only requests from last minute
            self._request_times = [
                t for t in self._request_times 
                if current_time - t < 60
            ]

    async def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        
        async with self._rate_limit_lock:
            # Check requests per second
            recent_requests = [
                t for t in self._request_times 
                if current_time - t < 1
            ]
            
            if len(recent_requests) >= self.rate_limits.requests_per_second:
                sleep_time = 1 - (current_time - min(recent_requests))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Add current request time
            self._request_times.append(current_time)

    async def _record_request_metrics(self, latency: float, success: bool) -> None:
        """Record request metrics for health monitoring."""
        async with self._health_lock:
            self._health.total_requests += 1
            
            if success:
                self._health.last_success = datetime.utcnow()
                # Update rolling average latency
                if self._health.avg_latency == 0:
                    self._health.avg_latency = latency
                else:
                    self._health.avg_latency = (
                        self._health.avg_latency * 0.9 + latency * 0.1
                    )
            else:
                self._health.last_failure = datetime.utcnow()
                self._health.error_count += 1
            
            # Calculate success rate
            self._health.success_rate = (
                (self._health.total_requests - self._health.error_count) / 
                self._health.total_requests
            )
            
            # Update health status based on metrics
            if self._health.success_rate < 0.5:
                self._health.status = ProviderStatus.UNHEALTHY
            elif self._health.success_rate < 0.8:
                self._health.status = ProviderStatus.DEGRADED
            else:
                self._health.status = ProviderStatus.HEALTHY

    async def _update_health(self, status: ProviderStatus) -> None:
        """Update provider health status."""
        async with self._health_lock:
            old_status = self._health.status
            self._health.status = status
            
            if old_status != status:
                logger.warning(
                    f"{self.provider_name} health status changed: "
                    f"{old_status.value} -> {status.value}"
                )

    async def _health_check(self) -> bool:
        """Perform provider health check."""
        try:
            # Override in subclasses with provider-specific health check
            await self._test_connectivity()
            await self._update_health(ProviderStatus.HEALTHY)
            return True
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_name}: {e}")
            await self._update_health(ProviderStatus.UNHEALTHY)
            return False

    @abstractmethod
    async def _test_connectivity(self) -> None:
        """Test provider connectivity - implement in subclasses."""
        pass

    async def get_health(self) -> ProviderHealth:
        """Get current provider health status."""
        async with self._health_lock:
            return ProviderHealth(
                status=self._health.status,
                last_success=self._health.last_success,
                last_failure=self._health.last_failure,
                success_rate=self._health.success_rate,
                avg_latency=self._health.avg_latency,
                error_count=self._health.error_count,
                total_requests=self._health.total_requests
            )

    async def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        if not self._redis:
            return None
            
        try:
            cached_data = await self._redis.get(cache_key)
            if cached_data:
                import json
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None

    async def _set_cached_data(
        self, 
        cache_key: str, 
        data: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """Set data in Redis cache."""
        if not self._redis:
            return
            
        try:
            import json
            cache_ttl = ttl or self._cache_ttl
            await self._redis.setex(
                cache_key, 
                cache_ttl, 
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format across providers."""
        # Remove common suffixes and standardize format
        symbol = symbol.upper().strip()
        
        # Handle common symbol variations
        symbol_mappings = {
            'BTC-USD': 'BTCUSD',
            'ETH-USD': 'ETHUSD',
            'BTC/USD': 'BTCUSD',
            'ETH/USD': 'ETHUSD'
        }
        
        return symbol_mappings.get(symbol, symbol)

    def _normalize_price_data(
        self, 
        raw_data: Dict[str, Any], 
        symbol: str
    ) -> DataNormalizationResult:
        """Normalize price data across different providers."""
        errors = []
        quality_score = 1.0
        
        try:
            # Extract and validate required fields
            normalized = {
                'symbol': self._normalize_symbol(symbol),
                'timestamp': self._parse_timestamp(raw_data.get('timestamp')),
                'open': self._parse_decimal(raw_data.get('open')),
                'high': self._parse_decimal(raw_data.get('high')),
                'low': self._parse_decimal(raw_data.get('low')),
                'close': self._parse_decimal(raw_data.get('close')),
                'volume': self._parse_decimal(raw_data.get('volume', 0)),
                'source': self.provider_name
            }
            
            # Validate data consistency
            if normalized['high'] < normalized['low']:
                errors.append("High price is less than low price")
                quality_score -= 0.2
                
            if normalized['close'] > normalized['high'] or normalized['close'] < normalized['low']:
                errors.append("Close price is outside high-low range")
                quality_score -= 0.2
                
            if normalized['open'] > normalized['high'] or normalized['open'] < normalized['low']:
                errors.append("Open price is outside high-low range")
                quality_score -= 0.2
            
            # Check for missing data
            for field, value in normalized.items():
                if value is None and field not in ['volume']:
                    errors.append(f"Missing required field: {field}")
                    quality_score -= 0.3
            
            source_metadata = {
                'provider': self.provider_name,
                'raw_data_fields': list(raw_data.keys()),
                'normalization_timestamp': datetime.utcnow().isoformat(),
                'data_quality': self._calculate_quality_level(quality_score)
            }
            
            return DataNormalizationResult(
                normalized_data=normalized,
                quality_score=quality_score,
                source_metadata=source_metadata,
                normalization_errors=errors
            )
            
        except Exception as e:
            errors.append(f"Normalization error: {str(e)}")
            logger.error(f"Data normalization failed for {symbol}: {e}")
            
            return DataNormalizationResult(
                normalized_data=None,
                quality_score=0.0,
                source_metadata={'provider': self.provider_name, 'error': str(e)},
                normalization_errors=errors
            )

    def _parse_timestamp(self, timestamp_data: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not timestamp_data:
            return None
            
        try:
            if isinstance(timestamp_data, datetime):
                return timestamp_data
            elif isinstance(timestamp_data, (int, float)):
                return datetime.fromtimestamp(timestamp_data)
            elif isinstance(timestamp_data, str):
                # Try common timestamp formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        return datetime.strptime(timestamp_data, fmt)
                    except ValueError:
                        continue
                # Try ISO format
                return datetime.fromisoformat(timestamp_data.replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp_data}: {e}")
        
        return None

    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        """Parse decimal value safely."""
        if value is None:
            return None
            
        try:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, (int, float, str)):
                return Decimal(str(value))
        except Exception as e:
            logger.warning(f"Failed to parse decimal {value}: {e}")
        
        return None

    def _calculate_quality_level(self, score: float) -> DataQuality:
        """Calculate data quality level from score."""
        if score >= 0.9:
            return DataQuality.HIGH
        elif score >= 0.7:
            return DataQuality.MEDIUM
        elif score >= 0.5:
            return DataQuality.LOW
        else:
            return DataQuality.POOR

    async def cleanup(self) -> None:
        """Clean up provider resources."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                
            if self._redis:
                await self._redis.close()
                
            logger.info(f"Cleaned up {self.provider_name} provider resources")
            
        except Exception as e:
            logger.error(f"Error during {self.provider_name} cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    # Abstract methods that must be implemented by providers
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> MarketData:
        """Get real-time quote for symbol."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """Get historical data for symbol."""
        pass

    @abstractmethod
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for symbol."""
        pass