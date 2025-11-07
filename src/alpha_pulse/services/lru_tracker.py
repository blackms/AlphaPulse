"""
LRU tracking service using Redis sorted sets.

Tracks cache key access timestamps using sorted sets for efficient LRU eviction.
Implementation follows ADR-004 specifications.
"""

import logging
import time
from typing import List, Dict, Optional
from uuid import UUID
from redis.asyncio import Redis

from alpha_pulse.middleware.lru_metrics import (
    lru_track_operations_total,
    lru_tracked_keys_current,
    lru_errors_total,
)

logger = logging.getLogger(__name__)


class LRUTracker:
    """
    Tracks cache key access times using Redis sorted sets.

    Uses sorted sets (ZADD/ZRANGE) to maintain LRU order per tenant:
    - Key: meta:tenant:{id}:lru
    - Score: Access timestamp (Unix time)
    - Member: Cache key name

    This enables O(log N) insertions and O(1) oldest key retrieval.
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize LRU tracker.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client

    def _get_lru_key(self, tenant_id: UUID) -> str:
        """
        Get Redis key for tenant LRU sorted set.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Redis key string (meta:tenant:{id}:lru)
        """
        return f"meta:tenant:{tenant_id}:lru"

    async def track_access(
        self,
        tenant_id: UUID,
        cache_key: str
    ) -> float:
        """
        Track cache key access with current timestamp.

        Updates or inserts key in sorted set with current Unix timestamp as score.

        Args:
            tenant_id: Tenant identifier
            cache_key: Cache key being accessed

        Returns:
            Timestamp used for tracking (Unix time as float)

        Raises:
            redis.RedisError: If Redis operation fails
        """
        timestamp = time.time()
        lru_key = self._get_lru_key(tenant_id)

        try:
            async with self.redis.pipeline() as pipe:
                pipe.zadd(
                    lru_key,
                    mapping={cache_key: timestamp}
                )
                await pipe.execute()

            # Update metrics
            lru_track_operations_total.labels(tenant_id=str(tenant_id)).inc()

            logger.debug(
                f"LRU tracked: tenant_id={tenant_id}, cache_key={cache_key}, "
                f"timestamp={timestamp}"
            )

            return timestamp

        except Exception as e:
            lru_errors_total.labels(
                operation="track_access",
                error_type=type(e).__name__
            ).inc()

            logger.error(
                f"LRU track failed: tenant_id={tenant_id}, cache_key={cache_key}, "
                f"error={e}"
            )
            raise

    async def remove_key(
        self,
        tenant_id: UUID,
        cache_key: str
    ) -> bool:
        """
        Remove key from LRU tracking.

        Args:
            tenant_id: Tenant identifier
            cache_key: Cache key to remove

        Returns:
            True if key was removed, False if key didn't exist

        Raises:
            redis.RedisError: If Redis operation fails
        """
        lru_key = self._get_lru_key(tenant_id)

        try:
            async with self.redis.pipeline() as pipe:
                pipe.zrem(lru_key, cache_key)
                results = await pipe.execute()

            removed = results[0] > 0

            if removed:
                logger.debug(
                    f"LRU key removed: tenant_id={tenant_id}, cache_key={cache_key}"
                )
            else:
                logger.debug(
                    f"LRU key not found: tenant_id={tenant_id}, cache_key={cache_key}"
                )

            return removed

        except Exception as e:
            logger.error(
                f"LRU remove failed: tenant_id={tenant_id}, cache_key={cache_key}, "
                f"error={e}"
            )
            raise

    async def get_lru_count(self, tenant_id: UUID) -> int:
        """
        Get count of tracked keys for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of keys in LRU tracking

        Raises:
            redis.RedisError: If Redis operation fails
        """
        lru_key = self._get_lru_key(tenant_id)

        try:
            count = await self.redis.zcard(lru_key)

            # Update gauge metric
            lru_tracked_keys_current.labels(tenant_id=str(tenant_id)).set(count)

            logger.debug(
                f"LRU count: tenant_id={tenant_id}, count={count}"
            )

            return count

        except Exception as e:
            lru_errors_total.labels(
                operation="get_lru_count",
                error_type=type(e).__name__
            ).inc()

            logger.error(
                f"LRU count failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def get_oldest_keys(
        self,
        tenant_id: UUID,
        count: int = 10
    ) -> List[str]:
        """
        Get N oldest keys in LRU order.

        Args:
            tenant_id: Tenant identifier
            count: Number of keys to retrieve (default: 10)

        Returns:
            List of cache keys, oldest first

        Raises:
            redis.RedisError: If Redis operation fails
        """
        lru_key = self._get_lru_key(tenant_id)

        try:
            # ZRANGE returns keys in ascending order (oldest first)
            # Range is inclusive: 0 to (count-1)
            keys = await self.redis.zrange(lru_key, 0, count - 1)

            logger.debug(
                f"LRU oldest keys: tenant_id={tenant_id}, count={len(keys)}"
            )

            return keys

        except Exception as e:
            logger.error(
                f"LRU oldest keys failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def get_newest_keys(
        self,
        tenant_id: UUID,
        count: int = 10
    ) -> List[str]:
        """
        Get N newest keys in reverse LRU order.

        Args:
            tenant_id: Tenant identifier
            count: Number of keys to retrieve (default: 10)

        Returns:
            List of cache keys, newest first

        Raises:
            redis.RedisError: If Redis operation fails
        """
        lru_key = self._get_lru_key(tenant_id)

        try:
            # ZREVRANGE returns keys in descending order (newest first)
            keys = await self.redis.zrevrange(lru_key, 0, count - 1)

            logger.debug(
                f"LRU newest keys: tenant_id={tenant_id}, count={len(keys)}"
            )

            return keys

        except Exception as e:
            logger.error(
                f"LRU newest keys failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def get_keys_to_evict(
        self,
        tenant_id: UUID,
        count: int
    ) -> List[str]:
        """
        Get keys that should be evicted (oldest N keys).

        This is an alias for get_oldest_keys with semantic naming
        for eviction use case.

        Args:
            tenant_id: Tenant identifier
            count: Number of keys to evict

        Returns:
            List of cache keys to evict, oldest first
        """
        return await self.get_oldest_keys(tenant_id, count=count)

    async def evict_keys(
        self,
        tenant_id: UUID,
        cache_keys: List[str]
    ) -> int:
        """
        Remove multiple keys from LRU tracking (batch operation).

        Args:
            tenant_id: Tenant identifier
            cache_keys: List of cache keys to remove

        Returns:
            Number of keys actually removed

        Raises:
            redis.RedisError: If Redis operation fails
        """
        if not cache_keys:
            return 0

        lru_key = self._get_lru_key(tenant_id)

        try:
            async with self.redis.pipeline() as pipe:
                for cache_key in cache_keys:
                    pipe.zrem(lru_key, cache_key)

                results = await pipe.execute()

            # Count how many were actually removed
            removed_count = sum(1 for result in results if result > 0)

            logger.debug(
                f"LRU keys evicted: tenant_id={tenant_id}, "
                f"removed={removed_count}/{len(cache_keys)}"
            )

            return removed_count

        except Exception as e:
            logger.error(
                f"LRU evict keys failed: tenant_id={tenant_id}, "
                f"count={len(cache_keys)}, error={e}"
            )
            raise

    async def clear_lru(self, tenant_id: UUID) -> bool:
        """
        Clear all LRU tracking for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if LRU set was deleted, False if it didn't exist

        Raises:
            redis.RedisError: If Redis operation fails
        """
        lru_key = self._get_lru_key(tenant_id)

        try:
            deleted = await self.redis.delete(lru_key)

            if deleted:
                logger.info(
                    f"LRU cleared: tenant_id={tenant_id}"
                )
            else:
                logger.debug(
                    f"LRU already empty: tenant_id={tenant_id}"
                )

            return deleted > 0

        except Exception as e:
            logger.error(
                f"LRU clear failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def get_lru_stats(self, tenant_id: UUID) -> Dict[str, any]:
        """
        Get LRU statistics for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary with LRU statistics:
            - total_keys: Number of tracked keys
            - oldest_key: Oldest cache key
            - newest_key: Newest cache key
            - oldest_timestamp: Timestamp of oldest access
            - newest_timestamp: Timestamp of newest access
            - age_range_seconds: Time span between oldest and newest

        Raises:
            redis.RedisError: If Redis operation fails
        """
        lru_key = self._get_lru_key(tenant_id)

        try:
            # Get count
            total_keys = await self.redis.zcard(lru_key)

            if total_keys == 0:
                return {
                    "total_keys": 0,
                    "oldest_key": None,
                    "newest_key": None,
                    "oldest_timestamp": None,
                    "newest_timestamp": None,
                    "age_range_seconds": 0.0
                }

            # Get oldest and newest keys
            oldest_keys = await self.redis.zrange(lru_key, 0, 0)
            newest_keys = await self.redis.zrevrange(lru_key, 0, 0)

            oldest_key = oldest_keys[0] if oldest_keys else None
            newest_key = newest_keys[0] if newest_keys else None

            # Get their timestamps
            oldest_timestamp = None
            newest_timestamp = None

            if oldest_key:
                oldest_timestamp = await self.redis.zscore(lru_key, oldest_key)

            if newest_key:
                newest_timestamp = await self.redis.zscore(lru_key, newest_key)

            age_range = 0.0
            if oldest_timestamp and newest_timestamp:
                age_range = newest_timestamp - oldest_timestamp

            stats = {
                "total_keys": total_keys,
                "oldest_key": oldest_key,
                "newest_key": newest_key,
                "oldest_timestamp": oldest_timestamp,
                "newest_timestamp": newest_timestamp,
                "age_range_seconds": age_range
            }

            logger.debug(
                f"LRU stats: tenant_id={tenant_id}, total_keys={total_keys}, "
                f"age_range={age_range:.1f}s"
            )

            return stats

        except Exception as e:
            logger.error(
                f"LRU stats failed: tenant_id={tenant_id}, error={e}"
            )
            raise
