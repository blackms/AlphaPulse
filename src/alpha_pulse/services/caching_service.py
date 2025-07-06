"""Caching service for AlphaPulse trading system."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Type
from contextlib import asynccontextmanager

from ..cache.redis_manager import RedisManager, CacheTier, CacheConfig as RedisConfig
from ..cache.cache_strategies import CacheStrategyFactory, CacheStrategy
from ..cache.distributed_cache import DistributedCacheManager, CacheNode, NodeRole
from ..cache.cache_invalidation import CacheInvalidationManager, InvalidationRule, InvalidationType
from ..cache.cache_decorators import CacheContextManager
from ..config.cache_config import CacheConfig, CachePresets
from ..utils.serialization_utils import OptimizedSerializer, SerializerConfig
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class CachingService:
    """Central caching service for the trading system."""
    
    def __init__(
        self,
        config: CacheConfig,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize caching service."""
        self.config = config
        self.metrics = metrics_collector
        
        # Core components
        self.redis_manager: Optional[RedisManager] = None
        self.distributed_manager: Optional[DistributedCacheManager] = None
        self.invalidation_manager: Optional[CacheInvalidationManager] = None
        
        # Cache strategies
        self._strategies: Dict[str, CacheStrategy] = {}
        
        # Serializer
        self.serializer = OptimizedSerializer(
            SerializerConfig(
                format=config.serialization.default_format,
                compression=config.serialization.compression,
                compression_threshold=config.serialization.compression_threshold,
                max_size=config.serialization.max_serialized_size,
                numpy_optimize=config.serialization.numpy_optimize,
                pandas_optimize=config.serialization.pandas_optimize
            )
        )
        
        # Cache warming
        self._warming_tasks: List[asyncio.Task] = []
        self._warming_handlers: Dict[str, Callable] = {}
        
        # State
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the caching service."""
        if self._is_initialized:
            return
        
        try:
            logger.info("Initializing caching service...")
            
            # Create Redis manager
            redis_config = self._create_redis_config()
            self.redis_manager = RedisManager(redis_config, self.metrics)
            await self.redis_manager.initialize()
            
            # Initialize distributed caching if enabled
            if self.config.distributed.enabled:
                await self._initialize_distributed_cache()
            
            # Initialize invalidation manager
            self.invalidation_manager = CacheInvalidationManager(
                self.redis_manager,
                self.distributed_manager,
                self.metrics
            )
            await self.invalidation_manager.initialize()
            
            # Initialize cache strategies
            self._initialize_strategies()
            
            # Start cache warming if enabled
            if self.config.warming.enabled:
                await self._start_cache_warming()
            
            # Start monitoring
            if self.config.monitoring.enabled:
                await self._start_monitoring()
            
            self._is_initialized = True
            logger.info("Caching service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize caching service: {e}")
            raise
    
    def _create_redis_config(self) -> RedisConfig:
        """Create Redis configuration from cache config."""
        return RedisConfig(
            redis_host=self.config.redis_node.host,
            redis_port=self.config.redis_node.port,
            redis_password=self.config.redis_node.password,
            redis_db=self.config.redis_node.db,
            cluster_enabled=self.config.redis_cluster.enabled,
            cluster_nodes=self.config.redis_cluster.nodes,
            sentinel_enabled=self.config.redis_sentinel.enabled,
            sentinel_hosts=self.config.redis_sentinel.sentinels,
            sentinel_service_name=self.config.redis_sentinel.service_name,
            default_ttl=self.config.invalidation.default_ttl,
            max_connections=self.config.max_connections,
            connection_timeout=self.config.connection_timeout,
            socket_timeout=self.config.socket_timeout,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
    
    async def _initialize_distributed_cache(self) -> None:
        """Initialize distributed cache manager."""
        local_node = CacheNode(
            id=self.config.distributed.node_id,
            host=self.config.redis_node.host,
            port=self.config.redis_node.port,
            role=NodeRole.PRIMARY,
            tags=set()
        )
        
        # Parse remote nodes from config
        remote_nodes = []
        for node_config in self.config.distributed.static_nodes:
            remote_nodes.append(CacheNode(
                id=node_config["id"],
                host=node_config["host"],
                port=node_config["port"],
                role=NodeRole(node_config.get("role", "replica")),
                weight=node_config.get("weight", 1),
                tags=set(node_config.get("tags", []))
            ))
        
        self.distributed_manager = DistributedCacheManager(
            self.redis_manager,
            self.config.distributed.node_id,
            self.config.distributed.sharding_strategy,
            self.config.distributed.replication_factor,
            self.metrics
        )
        
        await self.distributed_manager.initialize(local_node, remote_nodes)
    
    def _initialize_strategies(self) -> None:
        """Initialize cache strategies."""
        # Create default strategy
        self._strategies["default"] = CacheStrategyFactory.create_strategy(
            self.config.default_strategy,
            self.redis_manager,
            self.metrics
        )
        
        # Create additional strategies if enabled
        if self.config.enable_write_through:
            self._strategies["write_through"] = CacheStrategyFactory.create_strategy(
                "write_through",
                self.redis_manager,
                self.metrics
            )
        
        if self.config.enable_write_behind:
            self._strategies["write_behind"] = CacheStrategyFactory.create_strategy(
                "write_behind",
                self.redis_manager,
                self.metrics,
                buffer_size=100,
                flush_interval=60
            )
        
        if self.config.enable_refresh_ahead:
            self._strategies["refresh_ahead"] = CacheStrategyFactory.create_strategy(
                "refresh_ahead",
                self.redis_manager,
                self.metrics,
                refresh_threshold=0.2
            )
    
    # Core cache operations
    
    async def get(
        self,
        key: str,
        fetch_fn: Optional[Callable[[], Any]] = None,
        ttl: Optional[int] = None,
        tier: Optional[CacheTier] = None,
        strategy: Optional[str] = None
    ) -> Any:
        """Get value from cache with optional fetch function."""
        if not self._is_initialized:
            await self.initialize()
        
        # Add prefix
        key = f"{self.config.cache_prefix}:{key}"
        
        # Determine tier
        if tier is None:
            tier = self._determine_tier(key)
        
        # Use distributed cache if available
        if self.distributed_manager and tier == CacheTier.L3_DISTRIBUTED_REDIS:
            return await self.distributed_manager.get(key, tier)
        
        # Use strategy if fetch function provided
        if fetch_fn:
            strategy_name = strategy or "default"
            cache_strategy = self._strategies.get(strategy_name)
            if cache_strategy:
                return await cache_strategy.get(key, fetch_fn, ttl, tier)
        
        # Direct get
        return await self.redis_manager.get(key, tier)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: Optional[CacheTier] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        if not self._is_initialized:
            await self.initialize()
        
        # Add prefix
        key = f"{self.config.cache_prefix}:{key}"
        
        # Determine tier
        if tier is None:
            tier = self._determine_tier(key)
        
        # Apply TTL variance to prevent thundering herd
        if ttl is None:
            ttl = self.config.invalidation.default_ttl
        
        if self.config.invalidation.ttl_variance > 0:
            import random
            variance = int(ttl * self.config.invalidation.ttl_variance)
            ttl = ttl + random.randint(-variance, variance)
        
        # Track metadata if configured
        if self.config.invalidation.track_dependencies and (tags or dependencies):
            await self.invalidation_manager.track_key(
                key,
                tags=set(tags) if tags else None,
                dependencies=set(dependencies) if dependencies else None
            )
        
        # Use distributed cache if available
        if self.distributed_manager and tier == CacheTier.L3_DISTRIBUTED_REDIS:
            return await self.distributed_manager.set(key, value, ttl, tier)
        
        # Direct set
        return await self.redis_manager.set(key, value, ttl, tier)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._is_initialized:
            await self.initialize()
        
        # Add prefix
        key = f"{self.config.cache_prefix}:{key}"
        
        # Use distributed cache if available
        if self.distributed_manager:
            return await self.distributed_manager.delete(key)
        
        return await self.redis_manager.delete(key)
    
    async def invalidate(
        self,
        keys: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> int:
        """Invalidate cache entries."""
        if not self._is_initialized:
            await self.initialize()
        
        # Add prefix to keys
        if keys:
            keys = [f"{self.config.cache_prefix}:{k}" for k in keys]
        
        return await self.invalidation_manager.invalidate(
            keys=keys,
            tags=tags,
            pattern=pattern
        )
    
    # Batch operations
    
    async def mget(self, keys: List[str], tier: Optional[CacheTier] = None) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self._is_initialized:
            await self.initialize()
        
        # Add prefix
        prefixed_keys = [f"{self.config.cache_prefix}:{k}" for k in keys]
        
        # Determine tier
        if tier is None:
            tier = self._determine_tier(prefixed_keys[0])
        
        # Get values
        results = await self.redis_manager.mget(prefixed_keys, tier)
        
        # Remove prefix from results
        return {
            k.replace(f"{self.config.cache_prefix}:", ""): v
            for k, v in results.items()
        }
    
    async def mset(
        self,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        tier: Optional[CacheTier] = None
    ) -> bool:
        """Set multiple values in cache."""
        if not self._is_initialized:
            await self.initialize()
        
        # Add prefix
        prefixed_data = {
            f"{self.config.cache_prefix}:{k}": v
            for k, v in data.items()
        }
        
        # Determine tier
        if tier is None:
            tier = self._determine_tier(list(prefixed_data.keys())[0])
        
        return await self.redis_manager.mset(prefixed_data, ttl, tier)
    
    # Context managers
    
    @asynccontextmanager
    async def cache_context(
        self,
        namespace: str,
        ttl: Optional[int] = None,
        tier: Optional[CacheTier] = None
    ):
        """Context manager for scoped cache operations."""
        if not self._is_initialized:
            await self.initialize()
        
        context = CacheContextManager(
            self.redis_manager,
            f"{self.config.cache_prefix}:{namespace}",
            ttl,
            tier or CacheTier.L2_LOCAL_REDIS
        )
        
        async with context as cache:
            yield cache
    
    # Cache warming
    
    def register_warming_handler(self, strategy: str, handler: Callable) -> None:
        """Register a cache warming handler."""
        self._warming_handlers[strategy] = handler
    
    async def _start_cache_warming(self) -> None:
        """Start cache warming tasks."""
        if "market_open" in self.config.warming.strategies:
            task = asyncio.create_task(self._warm_market_open())
            self._warming_tasks.append(task)
        
        if "predictive" in self.config.warming.strategies:
            task = asyncio.create_task(self._warm_predictive())
            self._warming_tasks.append(task)
        
        if self.config.warming.background_interval > 0:
            task = asyncio.create_task(self._warm_background())
            self._warming_tasks.append(task)
    
    async def _warm_market_open(self) -> None:
        """Warm cache before market open."""
        while True:
            try:
                # Calculate time until market open
                # This is simplified - in practice, you'd use market calendar
                now = datetime.utcnow()
                market_open = now.replace(hour=14, minute=30, second=0)  # 9:30 AM EST
                
                if now > market_open:
                    market_open += timedelta(days=1)
                
                time_until_open = (market_open - now).total_seconds()
                warm_time = time_until_open + self.config.warming.market_open_offset
                
                if warm_time > 0:
                    await asyncio.sleep(warm_time)
                
                # Warm cache with market open data
                logger.info("Starting market open cache warming")
                
                if "market_open" in self._warming_handlers:
                    await self._warming_handlers["market_open"]()
                
                # Wait until next day
                await asyncio.sleep(86400)  # 24 hours
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market open warming error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _warm_predictive(self) -> None:
        """Predictive cache warming based on access patterns."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if "predictive" in self._warming_handlers:
                    await self._warming_handlers["predictive"]()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Predictive warming error: {e}")
    
    async def _warm_background(self) -> None:
        """Background cache warming."""
        while True:
            try:
                await asyncio.sleep(self.config.warming.background_interval)
                
                if "background" in self._warming_handlers:
                    await self._warming_handlers["background"]()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background warming error: {e}")
    
    # Monitoring
    
    async def _start_monitoring(self) -> None:
        """Start cache monitoring."""
        asyncio.create_task(self._monitor_cache_metrics())
    
    async def _monitor_cache_metrics(self) -> None:
        """Monitor cache metrics."""
        while True:
            try:
                await asyncio.sleep(self.config.monitoring.metrics_interval)
                
                # Get cache stats
                stats = await self.get_stats()
                
                # Check thresholds
                if stats["hit_rate"] < self.config.monitoring.min_hit_rate:
                    logger.warning(f"Cache hit rate below threshold: {stats['hit_rate']}")
                
                # Log metrics
                if self.metrics:
                    self.metrics.gauge("cache.hit_rate", stats["hit_rate"])
                    self.metrics.gauge("cache.total_keys", stats["total_keys"])
                    self.metrics.gauge("cache.memory_usage", stats["memory_usage"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
    
    # Utility methods
    
    def _determine_tier(self, key: str) -> CacheTier:
        """Determine appropriate cache tier for a key."""
        # Check tier configurations for use case matching
        for tier_name, tier_config in self.config.tiers.items():
            if not tier_config.enabled:
                continue
            
            for use_case in tier_config.use_cases:
                if use_case in key:
                    if tier_name == "l1_memory":
                        return CacheTier.L1_MEMORY
                    elif tier_name == "l2_local_redis":
                        return CacheTier.L2_LOCAL_REDIS
                    elif tier_name == "l3_distributed_redis":
                        return CacheTier.L3_DISTRIBUTED_REDIS
        
        # Default to L2
        return CacheTier.L2_LOCAL_REDIS
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = await self.redis_manager.get_stats()
        
        # Calculate hit rate
        total_hits = 0
        total_misses = 0
        
        for tier_stats in stats.get("redis_tiers", {}).values():
            total_hits += tier_stats.get("hits", 0)
            total_misses += tier_stats.get("misses", 0)
        
        hit_rate = total_hits / max(total_hits + total_misses, 1)
        
        # Add distributed cache stats if available
        if self.distributed_manager:
            stats["distributed"] = await self.distributed_manager.get_cluster_stats()
        
        # Add invalidation stats
        if self.invalidation_manager:
            stats["invalidation"] = await self.invalidation_manager.get_invalidation_stats()
        
        stats["hit_rate"] = hit_rate
        stats["total_keys"] = sum(
            tier.get("total_keys", 0)
            for tier in stats.get("redis_tiers", {}).values()
        )
        
        # Calculate memory usage
        total_memory = 0
        for tier_stats in stats.get("redis_tiers", {}).values():
            memory_str = tier_stats.get("used_memory", "0")
            # Parse memory string (e.g., "1.5M", "2G")
            if memory_str.endswith("K"):
                total_memory += float(memory_str[:-1]) * 1024
            elif memory_str.endswith("M"):
                total_memory += float(memory_str[:-1]) * 1024 * 1024
            elif memory_str.endswith("G"):
                total_memory += float(memory_str[:-1]) * 1024 * 1024 * 1024
            else:
                total_memory += float(memory_str)
        
        stats["memory_usage"] = total_memory
        
        return stats
    
    async def clear_all(self) -> int:
        """Clear all cache entries."""
        if not self._is_initialized:
            await self.initialize()
        
        return await self.redis_manager.clear(
            pattern=f"{self.config.cache_prefix}:*"
        )
    
    async def close(self) -> None:
        """Close caching service."""
        try:
            # Cancel warming tasks
            for task in self._warming_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Stop background tasks in strategies
            for strategy in self._strategies.values():
                await strategy.stop_background_tasks()
            
            # Close components
            if self.invalidation_manager:
                await self.invalidation_manager.stop_background_tasks()
            
            if self.distributed_manager:
                await self.distributed_manager.stop_background_tasks()
            
            if self.redis_manager:
                await self.redis_manager.close()
            
            self._is_initialized = False
            logger.info("Caching service closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing caching service: {e}")
    
    # Factory methods
    
    @classmethod
    def create_for_trading(
        cls,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> "CachingService":
        """Create caching service optimized for trading."""
        config = CacheConfig()
        
        # Apply trading presets
        preset = CachePresets.real_time_trading()
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return cls(config, metrics_collector)
    
    @classmethod
    def create_for_backtesting(
        cls,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> "CachingService":
        """Create caching service optimized for backtesting."""
        config = CacheConfig()
        
        # Apply backtesting presets
        preset = CachePresets.backtesting()
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return cls(config, metrics_collector)
    
    @classmethod
    def create_for_analytics(
        cls,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> "CachingService":
        """Create caching service optimized for analytics."""
        config = CacheConfig()
        
        # Apply analytics presets
        preset = CachePresets.analytics()
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return cls(config, metrics_collector)