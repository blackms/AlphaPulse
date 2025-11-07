"""
Tenant-aware cache manager with namespace isolation and quota enforcement.

This module provides multi-tenant caching with:
- Namespace isolation (tenant:{tenant_id}:{resource}:{key})
- Per-tenant quotas with LRU eviction
- Usage tracking via Redis counters and sorted sets
- Cache metrics collection (hit rate, usage)

Usage:
    cache = TenantCacheManager(redis_manager, db_session)

    # Get from tenant's cache
    value = await cache.get(tenant_id, 'signals', 'technical:BTC')

    # Set with quota enforcement
    await cache.set(tenant_id, 'signals', 'technical:BTC', data, ttl=300)

    # Get usage metrics
    metrics = await cache.get_metrics(tenant_id)
"""
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime
from decimal import Decimal

from ..cache.redis_manager import RedisManager
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class QuotaExceededException(Exception):
    """Raised when tenant exceeds cache quota and eviction fails."""

    def __init__(self, tenant_id: UUID, message: str):
        self.tenant_id = tenant_id
        super().__init__(message)


class TenantCacheManager:
    """
    Multi-tenant cache manager with namespace isolation and quota enforcement.

    Provides tenant-scoped cache operations with:
    - Automatic namespace prefixing (tenant:{id}:{resource}:{key})
    - Quota enforcement with LRU eviction per tenant
    - Usage tracking (Redis counters + sorted sets for LRU)
    - Metrics collection (hit/miss rates, usage)

    Attributes:
        redis: Redis manager for cache operations
        db: Database session for quota queries
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        db_session: Any  # AsyncSession, kept as Any to avoid import
    ):
        """
        Initialize tenant cache manager.

        Args:
            redis_manager: Redis manager instance
            db_session: Async database session
        """
        self.redis = redis_manager
        self.db = db_session

        # Quota cache (tenant_id → quota_bytes, timestamp)
        self._quota_cache: Dict[UUID, int] = {}
        self._quota_cache_timestamps: Dict[UUID, datetime] = {}
        self._quota_cache_ttl = 300  # 5 minutes

    async def get(
        self,
        tenant_id: UUID,
        resource: str,
        key: str
    ) -> Optional[Any]:
        """
        Get value from tenant's cache namespace.

        Args:
            tenant_id: Tenant identifier
            resource: Resource type (e.g., 'signals', 'portfolio', 'session')
            key: Cache key (e.g., 'technical:BTC')

        Returns:
            Cached value or None if not found

        Example:
            >>> value = await cache.get(
            ...     tenant_id=UUID('...'),
            ...     resource='signals',
            ...     key='technical:BTC'
            ... )
            >>> # Fetches from: tenant:{tenant_id}:signals:technical:BTC
        """
        namespaced_key = self._build_key(tenant_id, resource, key)

        try:
            raw_value = await self.redis.get(namespaced_key)

            if raw_value:
                logger.debug(f"Cache HIT: {namespaced_key}")
                await self._increment_hit_counter(tenant_id)
                return self._deserialize(raw_value)
            else:
                logger.debug(f"Cache MISS: {namespaced_key}")
                await self._increment_miss_counter(tenant_id)
                return None

        except Exception as e:
            logger.error(f"Error getting cache key {namespaced_key}: {e}")
            await self._increment_miss_counter(tenant_id)
            return None

    async def set(
        self,
        tenant_id: UUID,
        resource: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in tenant's cache namespace with quota enforcement.

        Args:
            tenant_id: Tenant identifier
            resource: Resource type
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)

        Returns:
            True if set successfully, False otherwise

        Raises:
            QuotaExceededException: Hard limit exceeded (after eviction attempt)

        Example:
            >>> await cache.set(
            ...     tenant_id=UUID('...'),
            ...     resource='signals',
            ...     key='technical:BTC',
            ...     value={'price': 67000},
            ...     ttl=300
            ... )
        """
        try:
            # Check quota before writing
            quota = await self._get_quota(tenant_id)
            usage = await self._get_usage(tenant_id)

            # Serialize value to estimate size
            serialized = self._serialize(value)
            payload_size = len(serialized)

            # If quota would be exceeded, try to evict
            if usage + payload_size > quota:
                logger.warning(
                    f"Tenant {tenant_id} approaching quota: "
                    f"{usage + payload_size} / {quota} bytes"
                )

                # Try to evict 10% of quota to make room
                target_size = int(quota * 0.9)
                evicted = await self._evict_tenant_lru(tenant_id, target_size)

                logger.info(f"Evicted {evicted} bytes for tenant {tenant_id}")

                # Re-check usage after eviction
                usage = await self._get_usage(tenant_id)
                if usage + payload_size > quota:
                    raise QuotaExceededException(
                        tenant_id,
                        f"Quota exceeded: {usage + payload_size} / {quota} bytes"
                    )

            # Write to Redis with namespace
            namespaced_key = self._build_key(tenant_id, resource, key)
            success = await self.redis.set(namespaced_key, serialized, ttl=ttl)

            if success:
                # Track usage
                await self._track_usage(tenant_id, namespaced_key, payload_size)
                logger.debug(f"Cache SET: {namespaced_key} ({payload_size} bytes)")

            return success

        except QuotaExceededException:
            raise
        except Exception as e:
            logger.error(f"Error setting cache key: {e}")
            return False

    async def delete(
        self,
        tenant_id: UUID,
        resource: str,
        key: str
    ) -> bool:
        """
        Delete value from tenant's cache namespace.

        Args:
            tenant_id: Tenant identifier
            resource: Resource type
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        namespaced_key = self._build_key(tenant_id, resource, key)

        try:
            # Get size before deleting (for usage tracking)
            raw_value = await self.redis.get(namespaced_key)
            if not raw_value:
                return False

            payload_size = len(raw_value)

            # Delete from Redis
            deleted = await self.redis.delete(namespaced_key)

            if deleted:
                # Update usage counter
                await self._untrack_usage(tenant_id, namespaced_key, payload_size)
                logger.debug(f"Cache DEL: {namespaced_key} ({payload_size} bytes)")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting cache key {namespaced_key}: {e}")
            return False

    async def get_usage(self, tenant_id: UUID) -> int:
        """
        Get current cache usage for tenant in bytes.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Usage in bytes
        """
        try:
            usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
            usage = await self.redis.get(usage_key)
            return int(usage) if usage else 0
        except Exception as e:
            logger.error(f"Error getting usage for tenant {tenant_id}: {e}")
            return 0

    async def get_metrics(self, tenant_id: UUID) -> Dict:
        """
        Get cache metrics for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dict with hits, misses, hit_rate, usage_bytes, quota_bytes

        Example:
            >>> metrics = await cache.get_metrics(tenant_id)
            >>> {
            ...     'hits': 800,
            ...     'misses': 200,
            ...     'hit_rate': 80.0,
            ...     'usage_bytes': 52428800,
            ...     'quota_bytes': 104857600,
            ...     'usage_percent': 50.0
            ... }
        """
        try:
            # Fetch counters from Redis
            hits_key = f"meta:tenant:{tenant_id}:hits"
            misses_key = f"meta:tenant:{tenant_id}:misses"
            usage_key = f"meta:tenant:{tenant_id}:usage_bytes"

            # Use mget for atomic read
            values = await self.redis.mget([hits_key, misses_key, usage_key])

            hits = int(values.get(hits_key, 0)) if values.get(hits_key) else 0
            misses = int(values.get(misses_key, 0)) if values.get(misses_key) else 0
            usage = int(values.get(usage_key, 0)) if values.get(usage_key) else 0

            quota = await self._get_quota(tenant_id)

            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'hits': hits,
                'misses': misses,
                'hit_rate': round(hit_rate, 2),
                'usage_bytes': usage,
                'quota_bytes': quota,
                'usage_percent': round((usage / quota * 100) if quota > 0 else 0, 2)
            }

        except Exception as e:
            logger.error(f"Error getting metrics for tenant {tenant_id}: {e}")
            return {
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0,
                'usage_bytes': 0,
                'quota_bytes': 0,
                'usage_percent': 0.0
            }

    # Private helper methods

    def _build_key(self, tenant_id: UUID, resource: str, key: str) -> str:
        """
        Build namespaced cache key.

        Args:
            tenant_id: Tenant identifier
            resource: Resource type
            key: Cache key

        Returns:
            Namespaced key: tenant:{tenant_id}:{resource}:{key}
        """
        return f"tenant:{tenant_id}:{resource}:{key}"

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value to bytes (JSON).

        Handles special types:
        - Decimal → string
        - datetime → ISO format string

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes
        """
        def json_encoder(obj):
            """Custom JSON encoder for special types."""
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(value, default=json_encoder).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to value (JSON).

        Args:
            data: Serialized bytes

        Returns:
            Deserialized value
        """
        return json.loads(data.decode('utf-8'))

    async def _get_quota(self, tenant_id: UUID) -> int:
        """
        Get quota for tenant (cached for 5 minutes).

        Args:
            tenant_id: Tenant identifier

        Returns:
            Quota in bytes
        """
        # Check cache
        if tenant_id in self._quota_cache_timestamps:
            age = (datetime.utcnow() - self._quota_cache_timestamps[tenant_id]).total_seconds()
            if age < self._quota_cache_ttl:
                return self._quota_cache[tenant_id]

        # Fetch from database
        try:
            result = await self.db.execute(
                "SELECT quota_mb FROM tenant_cache_quotas WHERE tenant_id = :tenant_id",
                {'tenant_id': tenant_id}
            )
            row = result.fetchone()
            quota_mb = row[0] if row else 100  # Default: 100MB

            quota_bytes = quota_mb * 1024 * 1024

            # Cache for 5 minutes
            self._quota_cache[tenant_id] = quota_bytes
            self._quota_cache_timestamps[tenant_id] = datetime.utcnow()

            return quota_bytes

        except Exception as e:
            logger.error(f"Error fetching quota for tenant {tenant_id}: {e}")
            return 100 * 1024 * 1024  # Default 100MB

    async def _get_usage(self, tenant_id: UUID) -> int:
        """
        Get current usage from Redis counter.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Usage in bytes
        """
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
        try:
            usage = await self.redis.get(usage_key)
            return int(usage) if usage else 0
        except Exception:
            return 0

    async def _track_usage(
        self,
        tenant_id: UUID,
        cache_key: str,
        payload_size: int
    ):
        """
        Track cache usage: increment counter + add to LRU sorted set.

        Args:
            tenant_id: Tenant identifier
            cache_key: Full namespaced cache key
            payload_size: Size in bytes
        """
        try:
            pipeline = self.redis.pipeline()

            # Increment usage counter
            usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
            pipeline.incrby(usage_key, payload_size)

            # Add to LRU sorted set (score = timestamp)
            lru_key = f"meta:tenant:{tenant_id}:lru"
            pipeline.zadd(lru_key, {cache_key: datetime.utcnow().timestamp()})

            await pipeline.execute()

        except Exception as e:
            logger.error(f"Error tracking usage for tenant {tenant_id}: {e}")

    async def _untrack_usage(
        self,
        tenant_id: UUID,
        cache_key: str,
        payload_size: int
    ):
        """
        Untrack cache usage: decrement counter + remove from LRU sorted set.

        Args:
            tenant_id: Tenant identifier
            cache_key: Full namespaced cache key
            payload_size: Size in bytes
        """
        try:
            pipeline = self.redis.pipeline()

            # Decrement usage counter
            usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
            pipeline.decrby(usage_key, payload_size)

            # Remove from LRU sorted set
            lru_key = f"meta:tenant:{tenant_id}:lru"
            pipeline.zrem(lru_key, cache_key)

            await pipeline.execute()

        except Exception as e:
            logger.error(f"Error untracking usage for tenant {tenant_id}: {e}")

    async def _increment_hit_counter(self, tenant_id: UUID):
        """Increment cache hit counter."""
        try:
            hits_key = f"meta:tenant:{tenant_id}:hits"
            await self.redis.incr(hits_key)
        except Exception as e:
            logger.error(f"Error incrementing hit counter: {e}")

    async def _increment_miss_counter(self, tenant_id: UUID):
        """Increment cache miss counter."""
        try:
            misses_key = f"meta:tenant:{tenant_id}:misses"
            await self.redis.incr(misses_key)
        except Exception as e:
            logger.error(f"Error incrementing miss counter: {e}")

    async def _evict_tenant_lru(
        self,
        tenant_id: UUID,
        target_size: int
    ) -> int:
        """
        Evict tenant's LRU keys until usage drops below target.

        Args:
            tenant_id: Tenant identifier
            target_size: Target usage in bytes

        Returns:
            Number of bytes evicted
        """
        lru_key = f"meta:tenant:{tenant_id}:lru"
        current_usage = await self._get_usage(tenant_id)
        bytes_to_evict = current_usage - target_size

        if bytes_to_evict <= 0:
            return 0

        logger.info(
            f"Evicting {bytes_to_evict} bytes for tenant {tenant_id} "
            f"(current: {current_usage}, target: {target_size})"
        )

        total_evicted = 0
        batch_size = 10

        try:
            while total_evicted < bytes_to_evict:
                # Get oldest keys (lowest scores in sorted set)
                oldest_keys = await self.redis.zpopmin(lru_key, batch_size)

                if not oldest_keys:
                    break  # No more keys to evict

                for cache_key, _ in oldest_keys:
                    # Decode if bytes
                    if isinstance(cache_key, bytes):
                        cache_key = cache_key.decode('utf-8')

                    # Get size of value
                    raw_value = await self.redis.get(cache_key)
                    if raw_value:
                        payload_size = len(raw_value)

                        # Delete key
                        await self.redis.delete(cache_key)

                        # Update usage counter
                        await self._untrack_usage(tenant_id, cache_key, payload_size)

                        total_evicted += payload_size

                        logger.debug(f"Evicted: {cache_key} ({payload_size} bytes)")

            logger.info(f"Evicted {total_evicted} bytes for tenant {tenant_id}")
            return total_evicted

        except Exception as e:
            logger.error(f"Error during eviction for tenant {tenant_id}: {e}")
            return total_evicted
