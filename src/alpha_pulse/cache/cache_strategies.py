"""Cache strategies implementation for different caching patterns."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .redis_manager import RedisManager, CacheTier
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    
    key: str
    value: Any
    ttl: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    tier: CacheTier = CacheTier.L2_LOCAL_REDIS


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize cache strategy."""
        self.redis_manager = redis_manager
        self.metrics = metrics_collector
        self.write_buffer: Dict[str, CacheEntry] = {}
        self.refresh_queue: asyncio.Queue = asyncio.Queue()
        self._background_tasks: List[asyncio.Task] = []
    
    @abstractmethod
    async def get(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> Any:
        """Get value using the cache strategy."""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set value using the cache strategy."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value using the cache strategy."""
        pass
    
    async def start_background_tasks(self) -> None:
        """Start background tasks for the strategy."""
        pass
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()


class CacheAsideStrategy(CacheStrategy):
    """Cache-aside (lazy loading) strategy implementation."""
    
    async def get(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> Any:
        """Get value with cache-aside pattern."""
        try:
            # Try to get from cache
            value = await self.redis_manager.get(key, tier)
            
            if value is not None:
                if self.metrics:
                    self.metrics.increment("cache.strategy.hit", {"strategy": "cache_aside"})
                return value
            
            # Cache miss - fetch from source
            if self.metrics:
                self.metrics.increment("cache.strategy.miss", {"strategy": "cache_aside"})
            
            value = await fetch_fn()
            
            # Store in cache
            if value is not None:
                await self.redis_manager.set(key, value, ttl, tier)
            
            return value
            
        except Exception as e:
            logger.error(f"Cache-aside get error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("cache.strategy.error", {"strategy": "cache_aside"})
            # Fallback to direct fetch
            return await fetch_fn()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set value with cache-aside pattern."""
        try:
            # Just update cache
            return await self.redis_manager.set(key, value, ttl, tier)
        except Exception as e:
            logger.error(f"Cache-aside set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value with cache-aside pattern."""
        try:
            return await self.redis_manager.delete(key)
        except Exception as e:
            logger.error(f"Cache-aside delete error for key {key}: {e}")
            return False


class WriteThroughStrategy(CacheStrategy):
    """Write-through strategy implementation."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        metrics_collector: Optional[MetricsCollector] = None,
        write_fn: Optional[Callable[[str, Any], bool]] = None
    ):
        """Initialize write-through strategy."""
        super().__init__(redis_manager, metrics_collector)
        self.write_fn = write_fn
    
    async def get(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> Any:
        """Get value with write-through pattern."""
        try:
            # Try to get from cache
            value = await self.redis_manager.get(key, tier)
            
            if value is not None:
                if self.metrics:
                    self.metrics.increment("cache.strategy.hit", {"strategy": "write_through"})
                return value
            
            # Cache miss - fetch from source
            if self.metrics:
                self.metrics.increment("cache.strategy.miss", {"strategy": "write_through"})
            
            value = await fetch_fn()
            
            # Store in cache
            if value is not None:
                await self.set(key, value, ttl, tier)
            
            return value
            
        except Exception as e:
            logger.error(f"Write-through get error for key {key}: {e}")
            return await fetch_fn()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set value with write-through pattern."""
        try:
            # Write to both cache and backend synchronously
            cache_success = await self.redis_manager.set(key, value, ttl, tier)
            
            backend_success = True
            if self.write_fn:
                backend_success = await self.write_fn(key, value)
            
            if self.metrics:
                self.metrics.increment(
                    "cache.strategy.write",
                    {"strategy": "write_through", "success": cache_success and backend_success}
                )
            
            return cache_success and backend_success
            
        except Exception as e:
            logger.error(f"Write-through set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value with write-through pattern."""
        try:
            # Delete from both cache and backend
            cache_success = await self.redis_manager.delete(key)
            
            backend_success = True
            if self.write_fn:
                backend_success = await self.write_fn(key, None)
            
            return cache_success and backend_success
            
        except Exception as e:
            logger.error(f"Write-through delete error for key {key}: {e}")
            return False


class WriteBehindStrategy(CacheStrategy):
    """Write-behind (write-back) strategy implementation."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        metrics_collector: Optional[MetricsCollector] = None,
        write_fn: Optional[Callable[[str, Any], bool]] = None,
        buffer_size: int = 100,
        flush_interval: int = 60
    ):
        """Initialize write-behind strategy."""
        super().__init__(redis_manager, metrics_collector)
        self.write_fn = write_fn
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._last_flush = datetime.utcnow()
    
    async def get(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> Any:
        """Get value with write-behind pattern."""
        try:
            # Check write buffer first
            if key in self.write_buffer:
                entry = self.write_buffer[key]
                if self.metrics:
                    self.metrics.increment("cache.strategy.hit", {"strategy": "write_behind", "source": "buffer"})
                return entry.value
            
            # Try to get from cache
            value = await self.redis_manager.get(key, tier)
            
            if value is not None:
                if self.metrics:
                    self.metrics.increment("cache.strategy.hit", {"strategy": "write_behind", "source": "cache"})
                return value
            
            # Cache miss - fetch from source
            if self.metrics:
                self.metrics.increment("cache.strategy.miss", {"strategy": "write_behind"})
            
            value = await fetch_fn()
            
            # Store in cache only (not backend yet)
            if value is not None:
                await self.redis_manager.set(key, value, ttl, tier)
            
            return value
            
        except Exception as e:
            logger.error(f"Write-behind get error for key {key}: {e}")
            return await fetch_fn()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set value with write-behind pattern."""
        try:
            # Write to cache immediately
            cache_success = await self.redis_manager.set(key, value, ttl, tier)
            
            # Add to write buffer
            self.write_buffer[key] = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or 300,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                tier=tier
            )
            
            # Check if we need to flush
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()
            
            if self.metrics:
                self.metrics.increment("cache.strategy.write", {"strategy": "write_behind"})
                self.metrics.gauge("cache.write_buffer.size", len(self.write_buffer))
            
            return cache_success
            
        except Exception as e:
            logger.error(f"Write-behind set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value with write-behind pattern."""
        try:
            # Remove from buffer
            if key in self.write_buffer:
                del self.write_buffer[key]
            
            # Delete from cache
            cache_success = await self.redis_manager.delete(key)
            
            # Add deletion marker to buffer
            self.write_buffer[key] = CacheEntry(
                key=key,
                value=None,  # None indicates deletion
                ttl=0,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            
            return cache_success
            
        except Exception as e:
            logger.error(f"Write-behind delete error for key {key}: {e}")
            return False
    
    async def _flush_buffer(self) -> None:
        """Flush write buffer to backend."""
        if not self.write_fn or not self.write_buffer:
            return
        
        try:
            # Get all entries to flush
            entries_to_flush = list(self.write_buffer.items())
            self.write_buffer.clear()
            
            # Write to backend
            success_count = 0
            for key, entry in entries_to_flush:
                try:
                    if await self.write_fn(key, entry.value):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to flush {key} to backend: {e}")
                    # Re-add to buffer for retry
                    self.write_buffer[key] = entry
            
            self._last_flush = datetime.utcnow()
            
            if self.metrics:
                self.metrics.increment(
                    "cache.write_buffer.flush",
                    {"success": success_count, "failed": len(entries_to_flush) - success_count}
                )
            
            logger.info(f"Flushed {success_count}/{len(entries_to_flush)} entries to backend")
            
        except Exception as e:
            logger.error(f"Buffer flush error: {e}")
    
    async def _periodic_flush(self) -> None:
        """Periodically flush write buffer."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if self.write_buffer:
                    await self._flush_buffer()
                    
            except asyncio.CancelledError:
                # Final flush before stopping
                if self.write_buffer:
                    await self._flush_buffer()
                raise
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")
    
    async def start_background_tasks(self) -> None:
        """Start background tasks."""
        task = asyncio.create_task(self._periodic_flush())
        self._background_tasks.append(task)


class RefreshAheadStrategy(CacheStrategy):
    """Refresh-ahead (proactive refresh) strategy implementation."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        metrics_collector: Optional[MetricsCollector] = None,
        refresh_threshold: float = 0.2,  # Refresh when 20% of TTL remains
        max_concurrent_refreshes: int = 10
    ):
        """Initialize refresh-ahead strategy."""
        super().__init__(redis_manager, metrics_collector)
        self.refresh_threshold = refresh_threshold
        self.max_concurrent_refreshes = max_concurrent_refreshes
        self._refresh_semaphore = asyncio.Semaphore(max_concurrent_refreshes)
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
        self._fetch_functions: Dict[str, Callable[[], Any]] = {}
    
    async def get(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> Any:
        """Get value with refresh-ahead pattern."""
        try:
            # Store fetch function for refresh
            self._fetch_functions[key] = fetch_fn
            
            # Try to get from cache
            value = await self.redis_manager.get(key, tier)
            
            if value is not None:
                if self.metrics:
                    self.metrics.increment("cache.strategy.hit", {"strategy": "refresh_ahead"})
                
                # Check if we need to refresh
                await self._check_refresh(key, ttl or 300, tier)
                
                return value
            
            # Cache miss - fetch from source
            if self.metrics:
                self.metrics.increment("cache.strategy.miss", {"strategy": "refresh_ahead"})
            
            value = await fetch_fn()
            
            # Store in cache
            if value is not None:
                await self.redis_manager.set(key, value, ttl, tier)
            
            return value
            
        except Exception as e:
            logger.error(f"Refresh-ahead get error for key {key}: {e}")
            return await fetch_fn()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set value with refresh-ahead pattern."""
        try:
            return await self.redis_manager.set(key, value, ttl, tier)
        except Exception as e:
            logger.error(f"Refresh-ahead set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value with refresh-ahead pattern."""
        try:
            # Cancel any ongoing refresh
            if key in self._refresh_tasks:
                self._refresh_tasks[key].cancel()
                del self._refresh_tasks[key]
            
            # Remove fetch function
            if key in self._fetch_functions:
                del self._fetch_functions[key]
            
            return await self.redis_manager.delete(key)
            
        except Exception as e:
            logger.error(f"Refresh-ahead delete error for key {key}: {e}")
            return False
    
    async def _check_refresh(self, key: str, ttl: int, tier: CacheTier) -> None:
        """Check if key needs refresh and schedule if necessary."""
        try:
            # Skip if already refreshing
            if key in self._refresh_tasks and not self._refresh_tasks[key].done():
                return
            
            # Get remaining TTL from Redis
            conn = self.redis_manager._connections.get(tier)
            if not conn:
                return
            
            remaining_ttl = await conn.ttl(key)
            
            # Check if we need to refresh
            if remaining_ttl > 0 and remaining_ttl < ttl * self.refresh_threshold:
                # Schedule refresh
                task = asyncio.create_task(self._refresh_key(key, ttl, tier))
                self._refresh_tasks[key] = task
                
                if self.metrics:
                    self.metrics.increment("cache.strategy.refresh_scheduled", {"strategy": "refresh_ahead"})
                
        except Exception as e:
            logger.error(f"Check refresh error for key {key}: {e}")
    
    async def _refresh_key(self, key: str, ttl: int, tier: CacheTier) -> None:
        """Refresh a key proactively."""
        async with self._refresh_semaphore:
            try:
                fetch_fn = self._fetch_functions.get(key)
                if not fetch_fn:
                    return
                
                # Fetch fresh value
                value = await fetch_fn()
                
                if value is not None:
                    # Update cache
                    await self.redis_manager.set(key, value, ttl, tier)
                    
                    if self.metrics:
                        self.metrics.increment("cache.strategy.refresh_completed", {"strategy": "refresh_ahead"})
                    
                    logger.debug(f"Refreshed cache key: {key}")
                    
            except Exception as e:
                logger.error(f"Refresh error for key {key}: {e}")
                if self.metrics:
                    self.metrics.increment("cache.strategy.refresh_error", {"strategy": "refresh_ahead"})
            finally:
                # Clean up
                if key in self._refresh_tasks:
                    del self._refresh_tasks[key]
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        # Cancel all refresh tasks
        for task in self._refresh_tasks.values():
            task.cancel()
        
        # Wait for all to complete
        if self._refresh_tasks:
            await asyncio.gather(*self._refresh_tasks.values(), return_exceptions=True)
        
        self._refresh_tasks.clear()
        self._fetch_functions.clear()
        
        await super().stop_background_tasks()


class CacheStrategyFactory:
    """Factory for creating cache strategies."""
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        redis_manager: RedisManager,
        metrics_collector: Optional[MetricsCollector] = None,
        **kwargs
    ) -> CacheStrategy:
        """Create a cache strategy instance."""
        strategies = {
            "cache_aside": CacheAsideStrategy,
            "write_through": WriteThroughStrategy,
            "write_behind": WriteBehindStrategy,
            "refresh_ahead": RefreshAheadStrategy
        }
        
        strategy_class = strategies.get(strategy_type.lower())
        if not strategy_class:
            raise ValueError(f"Unknown cache strategy: {strategy_type}")
        
        return strategy_class(redis_manager, metrics_collector, **kwargs)