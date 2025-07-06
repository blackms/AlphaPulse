"""Cache invalidation strategies and patterns."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Pattern

from .redis_manager import RedisManager, CacheTier
from .distributed_cache import DistributedCacheManager
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class InvalidationType(Enum):
    """Types of cache invalidation."""
    
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    DEPENDENCY_BASED = "dependency_based"
    VERSION_BASED = "version_based"


@dataclass
class InvalidationRule:
    """Defines a cache invalidation rule."""
    
    id: str
    pattern: Optional[Pattern] = None
    keys: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    type: InvalidationType = InvalidationType.IMMEDIATE
    delay: Optional[timedelta] = None
    schedule: Optional[str] = None  # Cron expression
    dependencies: Optional[Set[str]] = None
    version_key: Optional[str] = None
    callback: Optional[Callable[[], None]] = None


@dataclass
class CacheKeyMetadata:
    """Metadata for cache keys."""
    
    key: str
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    invalidation_rules: Set[str] = field(default_factory=set)


class CacheInvalidationManager:
    """Manages cache invalidation strategies."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        distributed_manager: Optional[DistributedCacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize cache invalidation manager."""
        self.redis_manager = redis_manager
        self.distributed_manager = distributed_manager
        self.metrics = metrics_collector
        
        # Invalidation rules
        self._rules: Dict[str, InvalidationRule] = {}
        
        # Key metadata tracking
        self._metadata: Dict[str, CacheKeyMetadata] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of keys
        self._dependency_graph: Dict[str, Set[str]] = {}  # key -> dependent keys
        
        # Event subscriptions
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._invalidation_queue: asyncio.Queue = asyncio.Queue()
        
        # Version tracking
        self._versions: Dict[str, int] = {}
    
    async def initialize(self) -> None:
        """Initialize invalidation manager."""
        try:
            # Start background workers
            await self.start_background_tasks()
            
            # Load existing metadata from Redis
            await self._load_metadata()
            
            logger.info("Cache invalidation manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize invalidation manager: {e}")
            raise
    
    def add_rule(self, rule: InvalidationRule) -> None:
        """Add an invalidation rule."""
        self._rules[rule.id] = rule
        logger.info(f"Added invalidation rule: {rule.id}")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove an invalidation rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed invalidation rule: {rule_id}")
    
    async def track_key(
        self,
        key: str,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        version: Optional[int] = None
    ) -> None:
        """Track metadata for a cache key."""
        metadata = CacheKeyMetadata(
            key=key,
            tags=tags or set(),
            dependencies=dependencies or set(),
            version=version or 1
        )
        
        self._metadata[key] = metadata
        
        # Update tag index
        for tag in metadata.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(key)
        
        # Update dependency graph
        for dep in metadata.dependencies:
            if dep not in self._dependency_graph:
                self._dependency_graph[dep] = set()
            self._dependency_graph[dep].add(key)
        
        # Save to Redis
        await self._save_metadata(key, metadata)
    
    async def invalidate(
        self,
        keys: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        type: InvalidationType = InvalidationType.IMMEDIATE
    ) -> int:
        """Invalidate cache entries."""
        try:
            invalidated_keys = set()
            
            # Collect keys to invalidate
            if keys:
                invalidated_keys.update(keys)
            
            if tags:
                for tag in tags:
                    if tag in self._tag_index:
                        invalidated_keys.update(self._tag_index[tag])
            
            if pattern:
                # Find keys matching pattern
                pattern_re = re.compile(pattern)
                for key in self._metadata.keys():
                    if pattern_re.match(key):
                        invalidated_keys.add(key)
            
            # Apply invalidation based on type
            if type == InvalidationType.IMMEDIATE:
                count = await self._invalidate_immediate(invalidated_keys)
            elif type == InvalidationType.DELAYED:
                count = await self._invalidate_delayed(invalidated_keys)
            else:
                count = await self._invalidate_queued(invalidated_keys, type)
            
            if self.metrics:
                self.metrics.increment(
                    "cache.invalidation",
                    {"type": type.value, "count": count}
                )
            
            logger.info(f"Invalidated {count} cache entries")
            return count
            
        except Exception as e:
            logger.error(f"Invalidation error: {e}")
            if self.metrics:
                self.metrics.increment("cache.invalidation.error")
            return 0
    
    async def _invalidate_immediate(self, keys: Set[str]) -> int:
        """Immediately invalidate cache keys."""
        count = 0
        
        # Add dependent keys
        all_keys = set(keys)
        for key in keys:
            if key in self._dependency_graph:
                all_keys.update(self._dependency_graph[key])
        
        # Delete from cache
        if self.distributed_manager:
            # Use distributed deletion
            for key in all_keys:
                if await self.distributed_manager.delete(key):
                    count += 1
        else:
            # Use local deletion
            for key in all_keys:
                if await self.redis_manager.delete(key):
                    count += 1
        
        # Clean up metadata
        for key in all_keys:
            await self._cleanup_metadata(key)
        
        # Trigger callbacks
        await self._trigger_callbacks(all_keys)
        
        return count
    
    async def _invalidate_delayed(self, keys: Set[str], delay: timedelta = None) -> int:
        """Invalidate cache keys after a delay."""
        delay = delay or timedelta(seconds=5)
        
        async def delayed_invalidate():
            await asyncio.sleep(delay.total_seconds())
            await self._invalidate_immediate(keys)
        
        asyncio.create_task(delayed_invalidate())
        return len(keys)
    
    async def _invalidate_queued(self, keys: Set[str], type: InvalidationType) -> int:
        """Queue keys for invalidation."""
        for key in keys:
            await self._invalidation_queue.put((key, type))
        
        return len(keys)
    
    async def invalidate_by_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Invalidate cache based on an event."""
        try:
            # Find rules matching this event
            matching_rules = [
                rule for rule in self._rules.values()
                if rule.type == InvalidationType.EVENT_DRIVEN
            ]
            
            keys_to_invalidate = set()
            
            for rule in matching_rules:
                if rule.callback:
                    # Let callback determine what to invalidate
                    result = await rule.callback(event, data)
                    if isinstance(result, list):
                        keys_to_invalidate.update(result)
                elif rule.pattern:
                    # Use pattern matching
                    for key in self._metadata.keys():
                        if rule.pattern.match(key):
                            keys_to_invalidate.add(key)
            
            if keys_to_invalidate:
                await self._invalidate_immediate(keys_to_invalidate)
            
            # Trigger event handlers
            if event in self._event_handlers:
                for handler in self._event_handlers[event]:
                    await handler(data)
            
            if self.metrics:
                self.metrics.increment(
                    "cache.invalidation.event",
                    {"event": event, "keys": len(keys_to_invalidate)}
                )
            
        except Exception as e:
            logger.error(f"Event-based invalidation error: {e}")
    
    def subscribe_event(self, event: str, handler: Callable) -> None:
        """Subscribe to invalidation events."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        
        self._event_handlers[event].append(handler)
    
    async def check_version(self, key: str, version: int) -> bool:
        """Check if cache key version is valid."""
        current_version = self._versions.get(key, 0)
        return version >= current_version
    
    async def increment_version(self, key: str) -> int:
        """Increment version for a key, invalidating cache."""
        new_version = self._versions.get(key, 0) + 1
        self._versions[key] = new_version
        
        # Invalidate all cached versions
        await self.invalidate(keys=[key])
        
        # Save version to Redis
        version_key = f"version:{key}"
        await self.redis_manager.set(version_key, new_version, tier=CacheTier.L2_LOCAL_REDIS)
        
        return new_version
    
    async def _cleanup_metadata(self, key: str) -> None:
        """Clean up metadata for a key."""
        if key not in self._metadata:
            return
        
        metadata = self._metadata[key]
        
        # Remove from tag index
        for tag in metadata.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        
        # Remove from dependency graph
        if key in self._dependency_graph:
            del self._dependency_graph[key]
        
        # Remove metadata
        del self._metadata[key]
        
        # Remove from Redis
        metadata_key = f"metadata:{key}"
        await self.redis_manager.delete(metadata_key)
    
    async def _trigger_callbacks(self, keys: Set[str]) -> None:
        """Trigger callbacks for invalidated keys."""
        triggered_rules = set()
        
        for key in keys:
            if key in self._metadata:
                metadata = self._metadata[key]
                triggered_rules.update(metadata.invalidation_rules)
        
        for rule_id in triggered_rules:
            if rule_id in self._rules:
                rule = self._rules[rule_id]
                if rule.callback:
                    try:
                        await rule.callback()
                    except Exception as e:
                        logger.error(f"Callback error for rule {rule_id}: {e}")
    
    async def _save_metadata(self, key: str, metadata: CacheKeyMetadata) -> None:
        """Save metadata to Redis."""
        metadata_key = f"metadata:{key}"
        metadata_dict = {
            "tags": list(metadata.tags),
            "dependencies": list(metadata.dependencies),
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "last_accessed": metadata.last_accessed.isoformat(),
            "invalidation_rules": list(metadata.invalidation_rules)
        }
        
        await self.redis_manager.set(
            metadata_key,
            metadata_dict,
            tier=CacheTier.L2_LOCAL_REDIS
        )
    
    async def _load_metadata(self) -> None:
        """Load metadata from Redis."""
        try:
            # This would scan Redis for metadata keys
            # For now, we'll start with empty metadata
            logger.info("Metadata loading completed")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    async def _process_invalidation_queue(self) -> None:
        """Process queued invalidations."""
        while True:
            try:
                # Get items from queue with timeout
                key, invalidation_type = await asyncio.wait_for(
                    self._invalidation_queue.get(),
                    timeout=1.0
                )
                
                # Process based on type
                if invalidation_type == InvalidationType.SCHEDULED:
                    # Handle scheduled invalidation
                    pass
                else:
                    await self._invalidate_immediate({key})
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    async def start_background_tasks(self) -> None:
        """Start background tasks."""
        # Start queue processor
        queue_task = asyncio.create_task(self._process_invalidation_queue())
        self._background_tasks.append(queue_task)
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._background_tasks.clear()
    
    async def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        return {
            "total_rules": len(self._rules),
            "tracked_keys": len(self._metadata),
            "total_tags": len(self._tag_index),
            "dependency_graph_size": len(self._dependency_graph),
            "queue_size": self._invalidation_queue.qsize(),
            "rules_by_type": {
                type.value: sum(1 for r in self._rules.values() if r.type == type)
                for type in InvalidationType
            }
        }


class SmartInvalidator:
    """Smart cache invalidation with pattern learning."""
    
    def __init__(self, invalidation_manager: CacheInvalidationManager):
        """Initialize smart invalidator."""
        self.invalidation_manager = invalidation_manager
        self._access_patterns: Dict[str, List[datetime]] = {}
        self._invalidation_history: Dict[str, List[datetime]] = {}
    
    async def track_access(self, key: str) -> None:
        """Track cache key access."""
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        
        self._access_patterns[key].append(datetime.utcnow())
        
        # Keep only recent accesses (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._access_patterns[key] = [
            dt for dt in self._access_patterns[key] if dt > cutoff
        ]
    
    async def track_invalidation(self, key: str) -> None:
        """Track cache invalidation."""
        if key not in self._invalidation_history:
            self._invalidation_history[key] = []
        
        self._invalidation_history[key].append(datetime.utcnow())
    
    async def suggest_invalidation_strategy(self, key: str) -> Dict[str, Any]:
        """Suggest optimal invalidation strategy based on patterns."""
        access_times = self._access_patterns.get(key, [])
        invalidation_times = self._invalidation_history.get(key, [])
        
        if not access_times:
            return {
                "strategy": "ttl_based",
                "ttl": 300,  # Default 5 minutes
                "reason": "No access pattern data"
            }
        
        # Calculate access frequency
        access_count = len(access_times)
        time_span = (access_times[-1] - access_times[0]).total_seconds() if len(access_times) > 1 else 3600
        access_frequency = access_count / max(time_span, 1)
        
        # Calculate invalidation frequency
        invalidation_count = len(invalidation_times)
        
        # Determine strategy
        if access_frequency > 1.0:  # More than 1 access per second
            return {
                "strategy": "refresh_ahead",
                "refresh_threshold": 0.2,
                "reason": "High access frequency"
            }
        elif invalidation_count > 10:  # Frequent invalidations
            return {
                "strategy": "event_driven",
                "reason": "Frequent invalidations detected"
            }
        else:
            # Calculate optimal TTL based on access pattern
            if len(access_times) > 1:
                intervals = [
                    (access_times[i+1] - access_times[i]).total_seconds()
                    for i in range(len(access_times) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                suggested_ttl = int(avg_interval * 2)  # 2x average interval
            else:
                suggested_ttl = 600  # Default 10 minutes
            
            return {
                "strategy": "ttl_based",
                "ttl": suggested_ttl,
                "reason": "Standard access pattern"
            }