"""
Data provider factory for managing multiple market data sources.

Provides:
- Provider instantiation and configuration
- Failover and fallback mechanisms
- Provider health monitoring
- Load balancing across providers
- Cost optimization
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from .base_provider import BaseRealDataProvider, ProviderHealth, ProviderStatus
from .iex_cloud import IEXCloudProvider
from .polygon_io import PolygonIOProvider
from ..interfaces import MarketData, DataFetchError, ProviderError
from alpha_pulse.config.secure_settings import get_secret
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class ProviderType(Enum):
    """Supported provider types."""
    IEX_CLOUD = "iex_cloud"
    POLYGON_IO = "polygon_io"


class FailoverStrategy(Enum):
    """Failover strategy options."""
    ROUND_ROBIN = "round_robin"
    HEALTH_BASED = "health_based"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    provider_type: ProviderType
    priority: int  # Lower number = higher priority
    enabled: bool = True
    config: Dict[str, Any] = None
    health_check_interval: int = 60  # seconds
    max_consecutive_failures: int = 3


@dataclass
class ProviderPerformance:
    """Provider performance metrics."""
    avg_latency: float
    success_rate: float
    error_count: int
    total_requests: int
    cost_per_request: float
    last_updated: datetime


class DataProviderFactory:
    """Factory for creating and managing data providers."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        failover_strategy: FailoverStrategy = FailoverStrategy.HEALTH_BASED,
        health_check_interval: int = 60
    ):
        """
        Initialize provider factory.

        Args:
            redis_url: Redis connection URL for caching
            failover_strategy: Strategy for provider failover
            health_check_interval: Health check interval in seconds
        """
        self._redis_url = redis_url
        self._failover_strategy = failover_strategy
        self._health_check_interval = health_check_interval
        
        # Provider management
        self._providers: Dict[ProviderType, BaseRealDataProvider] = {}
        self._provider_configs: Dict[ProviderType, ProviderConfig] = {}
        self._provider_performance: Dict[ProviderType, ProviderPerformance] = {}
        
        # Failover tracking
        self._consecutive_failures: Dict[ProviderType, int] = {}
        self._last_successful_provider: Optional[ProviderType] = None
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._audit_logger = get_audit_logger()
        
        # Default configurations
        self._setup_default_configs()

    def _setup_default_configs(self) -> None:
        """Setup default provider configurations."""
        self._provider_configs = {
            ProviderType.IEX_CLOUD: ProviderConfig(
                provider_type=ProviderType.IEX_CLOUD,
                priority=1,  # Primary provider
                enabled=True,
                config={
                    "is_sandbox": False,
                    "cache_ttl": 30
                }
            ),
            ProviderType.POLYGON_IO: ProviderConfig(
                provider_type=ProviderType.POLYGON_IO,
                priority=2,  # Secondary provider
                enabled=True,
                config={
                    "tier": "free",
                    "cache_ttl": 60
                }
            )
        }

    async def initialize(self, provider_configs: Optional[Dict[ProviderType, ProviderConfig]] = None) -> None:
        """
        Initialize all configured providers.

        Args:
            provider_configs: Optional custom provider configurations
        """
        if provider_configs:
            self._provider_configs.update(provider_configs)
        
        # Initialize enabled providers
        for provider_type, config in self._provider_configs.items():
            if config.enabled:
                try:
                    provider = await self._create_provider(provider_type, config)
                    self._providers[provider_type] = provider
                    self._consecutive_failures[provider_type] = 0
                    
                    # Initialize performance tracking
                    self._provider_performance[provider_type] = ProviderPerformance(
                        avg_latency=0.0,
                        success_rate=1.0,
                        error_count=0,
                        total_requests=0,
                        cost_per_request=0.0,
                        last_updated=datetime.utcnow()
                    )
                    
                    logger.info(f"Initialized {provider_type.value} provider")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {provider_type.value} provider: {e}")
                    config.enabled = False
        
        if not self._providers:
            raise ProviderError("No providers could be initialized")
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        logger.info(f"Provider factory initialized with {len(self._providers)} providers")

    async def _create_provider(
        self, 
        provider_type: ProviderType, 
        config: ProviderConfig
    ) -> BaseRealDataProvider:
        """Create a provider instance based on type and configuration."""
        provider_config = config.config or {}
        
        if provider_type == ProviderType.IEX_CLOUD:
            api_token = await get_secret("iex_cloud_api_token")
            if not api_token:
                raise ProviderError("IEX Cloud API token not found in secrets")
            
            return IEXCloudProvider(
                api_token=api_token,
                is_sandbox=provider_config.get("is_sandbox", False),
                cache_ttl=provider_config.get("cache_ttl", 30),
                redis_url=self._redis_url
            )
            
        elif provider_type == ProviderType.POLYGON_IO:
            api_key = await get_secret("polygon_io_api_key")
            if not api_key:
                raise ProviderError("Polygon.io API key not found in secrets")
            
            return PolygonIOProvider(
                api_key=api_key,
                tier=provider_config.get("tier", "free"),
                cache_ttl=provider_config.get("cache_ttl", 60),
                redis_url=self._redis_url
            )
            
        else:
            raise ProviderError(f"Unsupported provider type: {provider_type}")

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Background task for monitoring provider health."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_providers_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

    async def _check_all_providers_health(self) -> None:
        """Check health of all providers."""
        for provider_type, provider in self._providers.items():
            try:
                health = await provider.get_health()
                await self._update_provider_performance(provider_type, health)
                
                # Log health issues
                if health.status in [ProviderStatus.UNHEALTHY, ProviderStatus.OFFLINE]:
                    self._audit_logger.log(
                        event_type=AuditEventType.SYSTEM_HEALTH,
                        event_data={
                            "provider": provider_type.value,
                            "status": health.status.value,
                            "success_rate": health.success_rate,
                            "error_count": health.error_count,
                            "last_failure": health.last_failure.isoformat() if health.last_failure else None
                        },
                        severity=AuditSeverity.WARNING
                    )
                    
            except Exception as e:
                logger.error(f"Health check failed for {provider_type.value}: {e}")

    async def _update_provider_performance(
        self, 
        provider_type: ProviderType, 
        health: ProviderHealth
    ) -> None:
        """Update provider performance metrics."""
        if provider_type in self._provider_performance:
            perf = self._provider_performance[provider_type]
            perf.avg_latency = health.avg_latency
            perf.success_rate = health.success_rate
            perf.error_count = health.error_count
            perf.total_requests = health.total_requests
            perf.last_updated = datetime.utcnow()

    def _get_provider_priority_order(self) -> List[ProviderType]:
        """Get providers ordered by current strategy."""
        available_providers = [
            p_type for p_type, provider in self._providers.items()
            if self._provider_configs[p_type].enabled
        ]
        
        if self._failover_strategy == FailoverStrategy.ROUND_ROBIN:
            # Simple round-robin based on last successful provider
            if self._last_successful_provider and self._last_successful_provider in available_providers:
                idx = available_providers.index(self._last_successful_provider)
                return available_providers[idx+1:] + available_providers[:idx+1]
            return available_providers
            
        elif self._failover_strategy == FailoverStrategy.HEALTH_BASED:
            # Order by health metrics (success rate, latency)
            def health_score(provider_type: ProviderType) -> float:
                perf = self._provider_performance.get(provider_type)
                if not perf:
                    return 0.0
                # Combine success rate and inverse latency
                latency_score = 1.0 / (1.0 + perf.avg_latency) if perf.avg_latency > 0 else 1.0
                return (perf.success_rate * 0.7) + (latency_score * 0.3)
            
            return sorted(available_providers, key=health_score, reverse=True)
            
        elif self._failover_strategy == FailoverStrategy.COST_OPTIMIZED:
            # Order by cost per request
            def cost_score(provider_type: ProviderType) -> float:
                perf = self._provider_performance.get(provider_type)
                return perf.cost_per_request if perf else float('inf')
            
            return sorted(available_providers, key=cost_score)
            
        elif self._failover_strategy == FailoverStrategy.LATENCY_OPTIMIZED:
            # Order by average latency
            def latency_score(provider_type: ProviderType) -> float:
                perf = self._provider_performance.get(provider_type)
                return perf.avg_latency if perf else float('inf')
            
            return sorted(available_providers, key=latency_score)
        
        # Default: priority order
        return sorted(available_providers, key=lambda p: self._provider_configs[p].priority)

    async def get_real_time_quote(self, symbol: str) -> MarketData:
        """
        Get real-time quote with failover support.

        Args:
            symbol: Stock symbol

        Returns:
            MarketData object from the first successful provider
        """
        provider_order = self._get_provider_priority_order()
        last_error = None
        
        for provider_type in provider_order:
            try:
                provider = self._providers[provider_type]
                result = await provider.get_real_time_quote(symbol)
                
                # Reset failure count on success
                self._consecutive_failures[provider_type] = 0
                self._last_successful_provider = provider_type
                
                # Log successful request
                self._audit_logger.log(
                    event_type=AuditEventType.DATA_ACCESS,
                    event_data={
                        "provider": provider_type.value,
                        "symbol": symbol,
                        "request_type": "real_time_quote",
                        "success": True
                    },
                    severity=AuditSeverity.INFO
                )
                
                return result
                
            except Exception as e:
                last_error = e
                self._consecutive_failures[provider_type] += 1
                
                # Disable provider if too many consecutive failures
                max_failures = self._provider_configs[provider_type].max_consecutive_failures
                if self._consecutive_failures[provider_type] >= max_failures:
                    logger.warning(
                        f"Disabling {provider_type.value} provider due to "
                        f"{max_failures} consecutive failures"
                    )
                    self._provider_configs[provider_type].enabled = False
                
                # Log failed request
                self._audit_logger.log(
                    event_type=AuditEventType.DATA_ACCESS,
                    event_data={
                        "provider": provider_type.value,
                        "symbol": symbol,
                        "request_type": "real_time_quote",
                        "success": False,
                        "error": str(e)
                    },
                    severity=AuditSeverity.WARNING
                )
                
                logger.warning(f"Provider {provider_type.value} failed for {symbol}: {e}")
                continue
        
        # All providers failed
        raise DataFetchError(f"All providers failed to get quote for {symbol}. Last error: {last_error}")

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Get historical data with failover support.

        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval

        Returns:
            List of MarketData objects from the first successful provider
        """
        provider_order = self._get_provider_priority_order()
        last_error = None
        
        for provider_type in provider_order:
            try:
                provider = self._providers[provider_type]
                result = await provider.get_historical_data(symbol, start_date, end_date, interval)
                
                # Reset failure count on success
                self._consecutive_failures[provider_type] = 0
                self._last_successful_provider = provider_type
                
                # Log successful request
                self._audit_logger.log(
                    event_type=AuditEventType.DATA_ACCESS,
                    event_data={
                        "provider": provider_type.value,
                        "symbol": symbol,
                        "request_type": "historical_data",
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "interval": interval,
                        "data_points": len(result),
                        "success": True
                    },
                    severity=AuditSeverity.INFO
                )
                
                return result
                
            except Exception as e:
                last_error = e
                self._consecutive_failures[provider_type] += 1
                
                logger.warning(f"Provider {provider_type.value} failed for historical data {symbol}: {e}")
                continue
        
        # All providers failed
        raise DataFetchError(f"All providers failed to get historical data for {symbol}. Last error: {last_error}")

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information with failover support.

        Args:
            symbol: Stock symbol

        Returns:
            Company information dictionary from the first successful provider
        """
        provider_order = self._get_provider_priority_order()
        last_error = None
        
        for provider_type in provider_order:
            try:
                provider = self._providers[provider_type]
                result = await provider.get_company_info(symbol)
                
                # Reset failure count on success
                self._consecutive_failures[provider_type] = 0
                self._last_successful_provider = provider_type
                
                return result
                
            except Exception as e:
                last_error = e
                self._consecutive_failures[provider_type] += 1
                
                logger.warning(f"Provider {provider_type.value} failed for company info {symbol}: {e}")
                continue
        
        # All providers failed
        raise DataFetchError(f"All providers failed to get company info for {symbol}. Last error: {last_error}")

    async def batch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Get batch quotes with provider optimization.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to MarketData
        """
        # Use the best provider based on batch capabilities
        provider_order = self._get_provider_priority_order()
        
        for provider_type in provider_order:
            try:
                provider = self._providers[provider_type]
                
                # Check if provider supports batch operations
                if hasattr(provider, 'batch_quotes'):
                    result = await provider.batch_quotes(symbols)
                    self._last_successful_provider = provider_type
                    return result
                else:
                    # Fall back to individual requests
                    results = {}
                    for symbol in symbols:
                        try:
                            quote = await provider.get_real_time_quote(symbol)
                            results[symbol] = quote
                        except Exception as e:
                            logger.warning(f"Failed to get quote for {symbol}: {e}")
                            continue
                    
                    if results:
                        self._last_successful_provider = provider_type
                        return results
                    
            except Exception as e:
                logger.warning(f"Provider {provider_type.value} failed for batch quotes: {e}")
                continue
        
        raise DataFetchError("All providers failed to get batch quotes")

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get status of all providers.

        Returns:
            Dictionary with provider status information
        """
        status = {
            "total_providers": len(self._providers),
            "enabled_providers": len([p for p in self._provider_configs.values() if p.enabled]),
            "last_successful_provider": self._last_successful_provider.value if self._last_successful_provider else None,
            "failover_strategy": self._failover_strategy.value,
            "providers": {}
        }
        
        for provider_type, provider in self._providers.items():
            config = self._provider_configs[provider_type]
            perf = self._provider_performance.get(provider_type)
            
            status["providers"][provider_type.value] = {
                "enabled": config.enabled,
                "priority": config.priority,
                "consecutive_failures": self._consecutive_failures.get(provider_type, 0),
                "performance": {
                    "avg_latency": perf.avg_latency if perf else 0,
                    "success_rate": perf.success_rate if perf else 0,
                    "total_requests": perf.total_requests if perf else 0,
                    "error_count": perf.error_count if perf else 0
                } if perf else None
            }
        
        return status

    async def cleanup(self) -> None:
        """Clean up all provider resources."""
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all providers
        for provider in self._providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up provider: {e}")
        
        self._providers.clear()
        logger.info("Provider factory cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Global provider factory instance
_provider_factory: Optional[DataProviderFactory] = None


async def get_provider_factory() -> DataProviderFactory:
    """Get the global provider factory instance."""
    global _provider_factory
    
    if _provider_factory is None:
        _provider_factory = DataProviderFactory()
        await _provider_factory.initialize()
    
    return _provider_factory


async def initialize_providers(provider_configs: Optional[Dict[ProviderType, ProviderConfig]] = None) -> None:
    """Initialize the global provider factory."""
    global _provider_factory
    
    if _provider_factory is not None:
        await _provider_factory.cleanup()
    
    _provider_factory = DataProviderFactory()
    await _provider_factory.initialize(provider_configs)