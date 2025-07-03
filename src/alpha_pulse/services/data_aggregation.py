"""
Data aggregation service for real market data.

Provides:
- Multi-source data aggregation
- Real-time data stream management
- Historical data caching and retrieval
- Data quality monitoring
- Intelligent failover and load balancing
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass
from collections import defaultdict
import redis.asyncio as redis
from loguru import logger

from alpha_pulse.data_pipeline.providers.provider_factory import (
    DataProviderFactory, ProviderType, FailoverStrategy, get_provider_factory
)
from alpha_pulse.models.market_data import (
    MarketDataPoint, TimeSeriesData, OHLCV, DataSource, DataQuality, AssetClass
)
from alpha_pulse.utils.data_validation import get_data_validator, ValidationLevel
from alpha_pulse.config.data_sources import get_data_source_config
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


@dataclass
class SubscriptionRequest:
    """Data subscription request."""
    symbol: str
    data_types: List[str]  # ['quotes', 'trades', 'ohlcv']
    frequency: str  # 'real-time', '1m', '5m', etc.
    callback: Optional[Callable] = None
    quality_level: ValidationLevel = ValidationLevel.STANDARD


@dataclass
class DataSubscription:
    """Active data subscription."""
    request: SubscriptionRequest
    subscriber_id: str
    created_at: datetime
    last_update: Optional[datetime] = None
    update_count: int = 0
    error_count: int = 0


class DataAggregationService:
    """Service for aggregating market data from multiple sources."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        cache_default_ttl: int = 300,
        max_cache_size_mb: int = 1024
    ):
        """
        Initialize data aggregation service.

        Args:
            redis_url: Redis connection URL for caching
            cache_default_ttl: Default cache TTL in seconds
            max_cache_size_mb: Maximum cache size in MB
        """
        self.redis_url = redis_url
        self.cache_default_ttl = cache_default_ttl
        self.max_cache_size_mb = max_cache_size_mb
        
        # Service components
        self._provider_factory: Optional[DataProviderFactory] = None
        self._redis: Optional[redis.Redis] = None
        self._data_validator = get_data_validator()
        self._config = get_data_source_config()
        self._audit_logger = get_audit_logger()
        
        # Subscription management
        self._subscriptions: Dict[str, Dict[str, DataSubscription]] = defaultdict(dict)
        self._subscription_tasks: Dict[str, asyncio.Task] = {}
        
        # Data caching
        self._memory_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Performance monitoring
        self._request_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_requests': 0,
            'validation_failures': 0
        }
        
        # Service state
        self._is_running = False
        self._background_tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize the data aggregation service."""
        try:
            # Initialize Redis connection
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                health_check_interval=30
            )
            await self._redis.ping()
            
            # Initialize provider factory
            self._provider_factory = await get_provider_factory()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._is_running = True
            logger.info("Data aggregation service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data aggregation service: {e}")
            raise

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Cache cleanup task
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._background_tasks.append(cache_cleanup_task)
        
        # Performance monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._background_tasks.append(monitoring_task)

    async def get_real_time_data(
        self, 
        symbol: str,
        use_cache: bool = True,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> MarketDataPoint:
        """
        Get real-time market data for a symbol.

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data
            validation_level: Data validation strictness level

        Returns:
            MarketDataPoint with current market data
        """
        self._request_stats['total_requests'] += 1
        
        # Check cache first
        if use_cache:
            cached_data = await self._get_from_cache(f"realtime:{symbol}")
            if cached_data:
                self._request_stats['cache_hits'] += 1
                return MarketDataPoint.from_dict(cached_data)
        
        self._request_stats['cache_misses'] += 1
        
        try:
            # Get data from provider factory
            market_data = await self._provider_factory.get_real_time_quote(symbol)
            self._request_stats['provider_requests'] += 1
            
            # Convert to enhanced MarketDataPoint
            data_point = self._convert_to_market_data_point(market_data, symbol)
            
            # Validate data
            if validation_level != ValidationLevel.BASIC:
                validation_result = await self._data_validator.validate_data_point(data_point)
                
                if not validation_result.is_valid:
                    self._request_stats['validation_failures'] += 1
                    logger.warning(f"Data validation failed for {symbol}: {validation_result.errors}")
                    
                    if validation_level == ValidationLevel.CRITICAL:
                        raise ValueError(f"Critical validation failure for {symbol}")
                
                # Update data quality
                data_point.data_quality = self._map_quality_score(validation_result.quality_score)
                data_point.validation_errors = validation_result.errors + validation_result.warnings
            
            # Cache the result
            if use_cache:
                await self._set_in_cache(
                    f"realtime:{symbol}", 
                    data_point.to_dict(), 
                    ttl=30  # 30 seconds for real-time data
                )
            
            # Log successful request
            self._audit_logger.log(
                event_type=AuditEventType.DATA_ACCESS,
                event_data={
                    "symbol": symbol,
                    "data_type": "real_time",
                    "provider": data_point.source.provider if data_point.source else "unknown",
                    "data_quality": data_point.data_quality.value,
                    "validation_level": validation_level.value
                },
                severity=AuditSeverity.INFO
            )
            
            return data_point
            
        except Exception as e:
            logger.error(f"Failed to get real-time data for {symbol}: {e}")
            
            # Log failed request
            self._audit_logger.log(
                event_type=AuditEventType.DATA_ACCESS,
                event_data={
                    "symbol": symbol,
                    "data_type": "real_time",
                    "error": str(e),
                    "validation_level": validation_level.value
                },
                severity=AuditSeverity.ERROR
            )
            
            raise

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        use_cache: bool = True,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> TimeSeriesData:
        """
        Get historical market data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1d, 1h, etc.)
            use_cache: Whether to use cached data
            validation_level: Data validation strictness level

        Returns:
            TimeSeriesData with historical market data
        """
        self._request_stats['total_requests'] += 1
        
        # Generate cache key
        cache_key = f"historical:{symbol}:{start_date.date()}:{end_date.date()}:{interval}"
        
        # Check cache first
        if use_cache:
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                self._request_stats['cache_hits'] += 1
                # Reconstruct TimeSeriesData from cached data
                return self._reconstruct_time_series(cached_data)
        
        self._request_stats['cache_misses'] += 1
        
        try:
            # Get data from provider factory
            market_data_list = await self._provider_factory.get_historical_data(
                symbol, start_date, end_date, interval
            )
            self._request_stats['provider_requests'] += 1
            
            # Convert to enhanced MarketDataPoint list
            data_points = []
            for market_data in market_data_list:
                data_point = self._convert_to_market_data_point(market_data, symbol)
                data_points.append(data_point)
            
            # Create time series
            time_series = TimeSeriesData(
                symbol=symbol,
                data_points=data_points,
                start_time=start_date,
                end_time=end_date,
                interval=interval,
                asset_class=AssetClass.EQUITY  # Default, could be determined from symbol
            )
            
            # Validate time series
            if validation_level != ValidationLevel.BASIC and data_points:
                validation_result = await self._data_validator.validate_time_series(time_series)
                
                if not validation_result.is_valid:
                    self._request_stats['validation_failures'] += 1
                    logger.warning(f"Time series validation failed for {symbol}: {validation_result.errors}")
                    
                    if validation_level == ValidationLevel.CRITICAL:
                        raise ValueError(f"Critical time series validation failure for {symbol}")
            
            # Cache the result
            if use_cache:
                cache_data = {
                    'symbol': symbol,
                    'start_time': start_date.isoformat(),
                    'end_time': end_date.isoformat(),
                    'interval': interval,
                    'asset_class': time_series.asset_class.value,
                    'data_points': [dp.to_dict() for dp in data_points]
                }
                await self._set_in_cache(
                    cache_key, 
                    cache_data, 
                    ttl=3600  # 1 hour for historical data
                )
            
            return time_series
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

    def _convert_to_market_data_point(self, market_data, symbol: str) -> MarketDataPoint:
        """Convert basic MarketData to enhanced MarketDataPoint."""
        # Create OHLCV data
        ohlcv = OHLCV(
            open=market_data.open,
            high=market_data.high,
            low=market_data.low,
            close=market_data.close,
            volume=market_data.volume,
            timestamp=market_data.timestamp,
            vwap=market_data.vwap,
            trade_count=market_data.trades
        )
        
        # Create data source
        source = DataSource(
            provider=market_data.source or "unknown",
            timestamp_received=datetime.utcnow(),
            quality=DataQuality.GOOD
        )
        
        return MarketDataPoint(
            symbol=symbol,
            timestamp=market_data.timestamp,
            asset_class=AssetClass.EQUITY,  # Default
            ohlcv=ohlcv,
            source=source
        )

    def _map_quality_score(self, score: float) -> DataQuality:
        """Map validation score to DataQuality enum."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    def _reconstruct_time_series(self, cached_data: Dict[str, Any]) -> TimeSeriesData:
        """Reconstruct TimeSeriesData from cached data."""
        data_points = [
            MarketDataPoint.from_dict(dp_data) 
            for dp_data in cached_data['data_points']
        ]
        
        return TimeSeriesData(
            symbol=cached_data['symbol'],
            data_points=data_points,
            start_time=datetime.fromisoformat(cached_data['start_time']),
            end_time=datetime.fromisoformat(cached_data['end_time']),
            interval=cached_data['interval'],
            asset_class=AssetClass(cached_data['asset_class'])
        )

    async def subscribe_to_real_time_data(
        self, 
        request: SubscriptionRequest,
        subscriber_id: str
    ) -> str:
        """
        Subscribe to real-time data updates.

        Args:
            request: Subscription request details
            subscriber_id: Unique subscriber identifier

        Returns:
            Subscription ID
        """
        subscription = DataSubscription(
            request=request,
            subscriber_id=subscriber_id,
            created_at=datetime.utcnow()
        )
        
        # Store subscription
        self._subscriptions[request.symbol][subscriber_id] = subscription
        
        # Start subscription task if not already running
        task_key = f"{request.symbol}:{request.frequency}"
        if task_key not in self._subscription_tasks:
            task = asyncio.create_task(
                self._subscription_loop(request.symbol, request.frequency)
            )
            self._subscription_tasks[task_key] = task
        
        logger.info(f"Created subscription for {subscriber_id} on {request.symbol}")
        return f"{request.symbol}:{subscriber_id}"

    async def _subscription_loop(self, symbol: str, frequency: str) -> None:
        """Background loop for handling data subscriptions."""
        while True:
            try:
                # Get current subscribers for this symbol
                subscribers = self._subscriptions.get(symbol, {})
                if not subscribers:
                    # No more subscribers, exit loop
                    break
                
                # Get real-time data
                try:
                    data_point = await self.get_real_time_data(symbol, use_cache=True)
                    
                    # Notify all subscribers
                    for subscription in subscribers.values():
                        if subscription.request.callback:
                            try:
                                await subscription.request.callback(data_point)
                                subscription.last_update = datetime.utcnow()
                                subscription.update_count += 1
                            except Exception as e:
                                subscription.error_count += 1
                                logger.error(f"Callback error for subscriber {subscription.subscriber_id}: {e}")
                
                except Exception as e:
                    logger.error(f"Failed to get data for subscription {symbol}: {e}")
                    # Increment error count for all subscribers
                    for subscription in subscribers.values():
                        subscription.error_count += 1
                
                # Wait based on frequency
                if frequency == "real-time":
                    await asyncio.sleep(1)  # 1 second for real-time
                elif frequency.endswith('m'):
                    minutes = int(frequency[:-1])
                    await asyncio.sleep(minutes * 60)
                else:
                    await asyncio.sleep(60)  # Default 1 minute
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in subscription loop for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from data updates.

        Args:
            subscription_id: Subscription ID to cancel

        Returns:
            True if successfully unsubscribed
        """
        try:
            symbol, subscriber_id = subscription_id.split(':', 1)
            
            if symbol in self._subscriptions:
                if subscriber_id in self._subscriptions[symbol]:
                    del self._subscriptions[symbol][subscriber_id]
                    logger.info(f"Unsubscribed {subscriber_id} from {symbol}")
                    
                    # Clean up empty symbol entries
                    if not self._subscriptions[symbol]:
                        del self._subscriptions[symbol]
                    
                    return True
            
            return False
            
        except ValueError:
            logger.error(f"Invalid subscription ID format: {subscription_id}")
            return False

    async def batch_get_quotes(
        self, 
        symbols: List[str],
        use_cache: bool = True,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, MarketDataPoint]:
        """
        Get quotes for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data
            validation_level: Data validation strictness level

        Returns:
            Dictionary mapping symbols to MarketDataPoint
        """
        results = {}
        uncached_symbols = []
        
        # Check cache for all symbols
        if use_cache:
            for symbol in symbols:
                cached_data = await self._get_from_cache(f"realtime:{symbol}")
                if cached_data:
                    results[symbol] = MarketDataPoint.from_dict(cached_data)
                    self._request_stats['cache_hits'] += 1
                else:
                    uncached_symbols.append(symbol)
                    self._request_stats['cache_misses'] += 1
        else:
            uncached_symbols = symbols
        
        # Get uncached data from provider
        if uncached_symbols:
            try:
                provider_results = await self._provider_factory.batch_quotes(uncached_symbols)
                self._request_stats['provider_requests'] += 1
                
                for symbol, market_data in provider_results.items():
                    data_point = self._convert_to_market_data_point(market_data, symbol)
                    
                    # Validate if required
                    if validation_level != ValidationLevel.BASIC:
                        validation_result = await self._data_validator.validate_data_point(data_point)
                        if not validation_result.is_valid and validation_level == ValidationLevel.CRITICAL:
                            continue  # Skip invalid data in critical mode
                        
                        data_point.data_quality = self._map_quality_score(validation_result.quality_score)
                        data_point.validation_errors = validation_result.errors + validation_result.warnings
                    
                    results[symbol] = data_point
                    
                    # Cache the result
                    if use_cache:
                        await self._set_in_cache(f"realtime:{symbol}", data_point.to_dict(), ttl=30)
                        
            except Exception as e:
                logger.error(f"Failed to get batch quotes: {e}")
        
        return results

    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache (Redis or memory)."""
        # Try memory cache first
        if key in self._memory_cache:
            timestamp = self._cache_timestamps.get(key)
            if timestamp and (datetime.utcnow() - timestamp).total_seconds() < self.cache_default_ttl:
                return self._memory_cache[key]
            else:
                # Expired, remove from memory cache
                del self._memory_cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
        
        # Try Redis cache
        if self._redis:
            try:
                cached_data = await self._redis.get(f"alphapulse:data:{key}")
                if cached_data:
                    import json
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        
        return None

    async def _set_in_cache(self, key: str, data: Any, ttl: int = None) -> None:
        """Set data in cache (Redis and memory)."""
        cache_ttl = ttl or self.cache_default_ttl
        
        # Set in memory cache
        self._memory_cache[key] = data
        self._cache_timestamps[key] = datetime.utcnow()
        
        # Set in Redis cache
        if self._redis:
            try:
                import json
                await self._redis.setex(
                    f"alphapulse:data:{key}",
                    cache_ttl,
                    json.dumps(data, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")

    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while self._is_running:
            try:
                # Clean up expired memory cache entries
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, timestamp in self._cache_timestamps.items():
                    if (current_time - timestamp).total_seconds() > self.cache_default_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    if key in self._memory_cache:
                        del self._memory_cache[key]
                    if key in self._cache_timestamps:
                        del self._cache_timestamps[key]
                
                # Check memory cache size
                cache_size_mb = len(str(self._memory_cache).encode('utf-8')) / 1024 / 1024
                if cache_size_mb > self.max_cache_size_mb:
                    # Remove oldest entries
                    sorted_keys = sorted(
                        self._cache_timestamps.items(), 
                        key=lambda x: x[1]
                    )
                    remove_count = len(sorted_keys) // 4  # Remove 25%
                    for key, _ in sorted_keys[:remove_count]:
                        if key in self._memory_cache:
                            del self._memory_cache[key]
                        if key in self._cache_timestamps:
                            del self._cache_timestamps[key]
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while self._is_running:
            try:
                # Log performance statistics
                self._audit_logger.log(
                    event_type=AuditEventType.SYSTEM_PERFORMANCE,
                    event_data={
                        "service": "data_aggregation",
                        "stats": self._request_stats.copy(),
                        "cache_size": len(self._memory_cache),
                        "active_subscriptions": sum(len(subs) for subs in self._subscriptions.values())
                    },
                    severity=AuditSeverity.INFO
                )
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        return {
            "is_running": self._is_running,
            "provider_status": self._provider_factory.get_provider_status() if self._provider_factory else {},
            "request_stats": self._request_stats.copy(),
            "cache_stats": {
                "memory_cache_size": len(self._memory_cache),
                "cache_hit_rate": (
                    self._request_stats['cache_hits'] / 
                    (self._request_stats['cache_hits'] + self._request_stats['cache_misses'])
                    if (self._request_stats['cache_hits'] + self._request_stats['cache_misses']) > 0 else 0
                )
            },
            "subscription_stats": {
                "active_symbols": len(self._subscriptions),
                "total_subscriptions": sum(len(subs) for subs in self._subscriptions.values()),
                "active_tasks": len(self._subscription_tasks)
            }
        }

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self._is_running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Cancel subscription tasks
        for task in self._subscription_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
        
        # Clean up provider factory
        if self._provider_factory:
            await self._provider_factory.cleanup()
        
        logger.info("Data aggregation service cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Global service instance
_data_aggregation_service: Optional[DataAggregationService] = None


async def get_data_aggregation_service() -> DataAggregationService:
    """Get the global data aggregation service instance."""
    global _data_aggregation_service
    
    if _data_aggregation_service is None:
        _data_aggregation_service = DataAggregationService()
        await _data_aggregation_service.initialize()
    
    return _data_aggregation_service