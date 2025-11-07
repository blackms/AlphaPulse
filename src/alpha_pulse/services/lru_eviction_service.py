"""
LRU eviction service for cache quota enforcement.

Implements LRU-based eviction when tenant cache usage exceeds quotas.
Integrates LRU tracker, usage tracker, and quota cache for complete workflow.
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID
from redis.asyncio import Redis

from alpha_pulse.models.quota import QuotaConfig
from alpha_pulse.services.lru_tracker import LRUTracker
from alpha_pulse.services.usage_tracker import UsageTracker
from alpha_pulse.services.quota_cache_service import QuotaCacheService

logger = logging.getLogger(__name__)


class LRUEvictionService:
    """
    Manages LRU-based cache eviction for quota enforcement.

    Workflow:
    1. Check if eviction needed (usage > quota)
    2. Calculate target eviction size
    3. Get oldest keys from LRU tracker
    4. Evict keys and update usage counters
    5. Repeat until target reached or no keys left
    """

    def __init__(
        self,
        redis_client: Redis,
        lru_tracker: LRUTracker,
        usage_tracker: UsageTracker,
        quota_cache_service: QuotaCacheService
    ):
        """
        Initialize eviction service.

        Args:
            redis_client: Async Redis client
            lru_tracker: LRU tracking service
            usage_tracker: Usage tracking service
            quota_cache_service: Quota cache service
        """
        self.redis = redis_client
        self.lru_tracker = lru_tracker
        self.usage_tracker = usage_tracker
        self.quota_cache = quota_cache_service

    async def needs_eviction(
        self,
        tenant_id: UUID,
        quota_config: QuotaConfig
    ) -> bool:
        """
        Check if eviction is needed based on current usage.

        Args:
            tenant_id: Tenant identifier
            quota_config: Current quota configuration

        Returns:
            True if usage exceeds base quota (eviction needed)
        """
        needs_evict = quota_config.current_usage_mb > quota_config.quota_mb

        if needs_evict:
            logger.info(
                f"Eviction needed: tenant_id={tenant_id}, "
                f"usage={quota_config.current_usage_mb:.2f}MB, "
                f"quota={quota_config.quota_mb:.2f}MB"
            )

        return needs_evict

    async def calculate_eviction_target(
        self,
        quota_config: QuotaConfig,
        target_percent: float = 0.9
    ) -> float:
        """
        Calculate how much data needs to be evicted.

        Args:
            quota_config: Current quota configuration
            target_percent: Target usage percentage (default: 90%)

        Returns:
            Size in MB to evict to reach target percentage
        """
        target_usage_mb = quota_config.quota_mb * target_percent
        current_usage_mb = quota_config.current_usage_mb

        eviction_needed_mb = current_usage_mb - target_usage_mb

        if eviction_needed_mb < 0:
            return 0.0

        logger.debug(
            f"Eviction target: current={current_usage_mb:.2f}MB, "
            f"target={target_usage_mb:.2f}MB, need_to_evict={eviction_needed_mb:.2f}MB"
        )

        return eviction_needed_mb

    async def get_eviction_candidates(
        self,
        tenant_id: UUID,
        count: int = 10
    ) -> List[str]:
        """
        Get cache keys that are candidates for eviction (oldest keys).

        Args:
            tenant_id: Tenant identifier
            count: Number of candidates to retrieve (default: 10)

        Returns:
            List of cache keys, oldest first

        Raises:
            redis.RedisError: If Redis operation fails
        """
        try:
            candidates = await self.lru_tracker.get_oldest_keys(
                tenant_id=tenant_id,
                count=count
            )

            logger.debug(
                f"Eviction candidates: tenant_id={tenant_id}, count={len(candidates)}"
            )

            return candidates

        except Exception as e:
            logger.error(
                f"Get eviction candidates failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def evict_key(
        self,
        tenant_id: UUID,
        cache_key: str
    ) -> float:
        """
        Evict a single cache key and update tracking.

        Workflow:
        1. Get key size from Redis
        2. Delete key from Redis
        3. Remove from LRU tracker
        4. Decrement usage counter

        Args:
            tenant_id: Tenant identifier
            cache_key: Cache key to evict

        Returns:
            Size of evicted key in MB

        Raises:
            redis.RedisError: If Redis operation fails
        """
        try:
            # Get key value to calculate size
            value = await self.redis.get(cache_key)

            if value is None:
                logger.debug(
                    f"Eviction skipped (key missing): tenant_id={tenant_id}, "
                    f"cache_key={cache_key}"
                )
                return 0.0

            # Calculate size in MB
            size_bytes = len(value)
            size_mb = size_bytes / (1024 * 1024)

            # Delete from Redis
            deleted = await self.redis.delete(cache_key)

            if deleted == 0:
                logger.warning(
                    f"Key deletion failed: tenant_id={tenant_id}, cache_key={cache_key}"
                )
                return 0.0

            # Remove from LRU tracking
            await self.lru_tracker.remove_key(tenant_id, cache_key)

            # Decrement usage counter
            new_usage = await self.usage_tracker.decrement_usage(tenant_id, size_mb)

            logger.info(
                f"Key evicted: tenant_id={tenant_id}, cache_key={cache_key}, "
                f"size={size_mb:.3f}MB, new_usage={new_usage:.2f}MB"
            )

            return size_mb

        except Exception as e:
            logger.error(
                f"Evict key failed: tenant_id={tenant_id}, cache_key={cache_key}, "
                f"error={e}"
            )
            raise

    async def evict_batch(
        self,
        tenant_id: UUID,
        cache_keys: List[str]
    ) -> Dict[str, any]:
        """
        Evict multiple keys in batch.

        Args:
            tenant_id: Tenant identifier
            cache_keys: List of cache keys to evict

        Returns:
            Dictionary with eviction results:
            - evicted_count: Number of keys evicted
            - total_size_mb: Total size evicted
            - failed_count: Number of keys that failed to evict

        Raises:
            redis.RedisError: If Redis operation fails
        """
        if not cache_keys:
            return {
                "evicted_count": 0,
                "total_size_mb": 0.0,
                "failed_count": 0
            }

        evicted_count = 0
        failed_count = 0
        total_size_mb = 0.0

        try:
            # Get sizes for all keys
            async with self.redis.pipeline() as pipe:
                for cache_key in cache_keys:
                    pipe.get(cache_key)

                values = await pipe.execute()

            # Calculate sizes
            key_sizes = {}
            for cache_key, value in zip(cache_keys, values):
                if value is not None:
                    size_mb = len(value) / (1024 * 1024)
                    key_sizes[cache_key] = size_mb
                else:
                    failed_count += 1

            # Delete keys in batch
            if key_sizes:
                async with self.redis.pipeline() as pipe:
                    for cache_key in key_sizes.keys():
                        pipe.delete(cache_key)

                    delete_results = await pipe.execute()

                # Count successful deletions
                for cache_key, deleted in zip(key_sizes.keys(), delete_results):
                    if deleted > 0:
                        evicted_count += 1
                        total_size_mb += key_sizes[cache_key]
                    else:
                        failed_count += 1

            # Remove from LRU tracking (batch)
            if evicted_count > 0:
                evicted_keys = [k for k, deleted in zip(key_sizes.keys(), delete_results) if deleted > 0]
                await self.lru_tracker.evict_keys(tenant_id, evicted_keys)

                # Decrement total usage
                if total_size_mb > 0:
                    await self.usage_tracker.decrement_usage(tenant_id, total_size_mb)

            logger.info(
                f"Batch eviction: tenant_id={tenant_id}, "
                f"evicted={evicted_count}, failed={failed_count}, "
                f"total_size={total_size_mb:.2f}MB"
            )

            return {
                "evicted_count": evicted_count,
                "total_size_mb": total_size_mb,
                "failed_count": failed_count
            }

        except Exception as e:
            logger.error(
                f"Batch eviction failed: tenant_id={tenant_id}, "
                f"count={len(cache_keys)}, error={e}"
            )
            raise

    async def evict_to_target(
        self,
        tenant_id: UUID,
        quota_config: QuotaConfig,
        target_percent: float = 0.9
    ) -> Dict[str, any]:
        """
        Evict keys until target usage percentage is reached.

        Implements iterative eviction:
        1. Calculate target eviction size
        2. Get oldest keys (batch of 10)
        3. Evict batch
        4. Check if target reached
        5. Repeat if needed

        Args:
            tenant_id: Tenant identifier
            quota_config: Current quota configuration
            target_percent: Target usage percentage (default: 90%)

        Returns:
            Dictionary with eviction summary:
            - success: True if target reached
            - evicted_count: Total keys evicted
            - total_size_mb: Total size evicted
            - final_usage_mb: Usage after eviction
            - message: Human-readable summary

        Raises:
            redis.RedisError: If Redis operation fails
        """
        try:
            # Check if eviction needed
            if not await self.needs_eviction(tenant_id, quota_config):
                return {
                    "success": True,
                    "evicted_count": 0,
                    "total_size_mb": 0.0,
                    "final_usage_mb": quota_config.current_usage_mb,
                    "message": "Eviction not needed"
                }

            # Calculate target
            target_eviction_mb = await self.calculate_eviction_target(
                quota_config=quota_config,
                target_percent=target_percent
            )

            total_evicted_count = 0
            total_evicted_size_mb = 0.0
            max_iterations = 100  # Safety limit
            iteration = 0

            target_usage_mb = quota_config.quota_mb * target_percent

            logger.info(
                f"Starting eviction: tenant_id={tenant_id}, "
                f"current_usage={quota_config.current_usage_mb:.2f}MB, "
                f"target_usage={target_usage_mb:.2f}MB"
            )

            while iteration < max_iterations:
                iteration += 1

                # Check current usage
                current_usage = await self.usage_tracker.get_usage(tenant_id)
                if current_usage is None:
                    current_usage = quota_config.current_usage_mb

                # Check if target reached
                if current_usage <= target_usage_mb:
                    logger.info(
                        f"Eviction target reached: tenant_id={tenant_id}, "
                        f"usage={current_usage:.2f}MB, target={target_usage_mb:.2f}MB"
                    )
                    break

                # Get eviction candidates
                candidates = await self.get_eviction_candidates(
                    tenant_id=tenant_id,
                    count=10
                )

                if not candidates:
                    logger.warning(
                        f"No more keys to evict: tenant_id={tenant_id}, "
                        f"usage={current_usage:.2f}MB > target={target_usage_mb:.2f}MB"
                    )
                    return {
                        "success": False,
                        "evicted_count": total_evicted_count,
                        "total_size_mb": total_evicted_size_mb,
                        "final_usage_mb": current_usage,
                        "message": f"Insufficient keys to reach target (evicted {total_evicted_count} keys)"
                    }

                # Evict batch
                result = await self.evict_batch(tenant_id, candidates)

                total_evicted_count += result["evicted_count"]
                total_evicted_size_mb += result["total_size_mb"]

                logger.debug(
                    f"Eviction iteration {iteration}: evicted={result['evicted_count']}, "
                    f"size={result['total_size_mb']:.2f}MB"
                )

            # Final usage check
            final_usage = await self.usage_tracker.get_usage(tenant_id)
            if final_usage is None:
                final_usage = current_usage

            success = final_usage <= target_usage_mb

            logger.info(
                f"Eviction complete: tenant_id={tenant_id}, "
                f"success={success}, evicted={total_evicted_count} keys, "
                f"size={total_evicted_size_mb:.2f}MB, final_usage={final_usage:.2f}MB"
            )

            return {
                "success": success,
                "evicted_count": total_evicted_count,
                "total_size_mb": total_evicted_size_mb,
                "final_usage_mb": final_usage,
                "message": f"Evicted {total_evicted_count} keys ({total_evicted_size_mb:.2f}MB)"
            }

        except Exception as e:
            logger.error(
                f"Evict to target failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def get_eviction_stats(self, tenant_id: UUID) -> Dict[str, any]:
        """
        Get eviction statistics for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary with eviction statistics:
            - lru_tracked_keys: Number of keys in LRU tracking
            - current_usage_mb: Current cache usage
            - quota_mb: Cache quota limit
            - usage_percent: Usage as percentage
            - oldest_key: Oldest cached key
            - newest_key: Newest cached key

        Raises:
            redis.RedisError: If Redis operation fails
        """
        try:
            # Get LRU stats
            lru_stats = await self.lru_tracker.get_lru_stats(tenant_id)

            # Get usage
            usage_mb = await self.usage_tracker.get_usage(tenant_id)
            if usage_mb is None:
                usage_mb = 0.0

            # Get quota
            quota_config = await self.quota_cache.get_quota_config(tenant_id)

            stats = {
                "lru_tracked_keys": lru_stats["total_keys"],
                "current_usage_mb": usage_mb,
                "quota_mb": quota_config.quota_mb if quota_config else 0.0,
                "usage_percent": quota_config.percent_used if quota_config else 0.0,
                "oldest_key": lru_stats["oldest_key"],
                "newest_key": lru_stats["newest_key"],
                "age_range_seconds": lru_stats["age_range_seconds"]
            }

            logger.debug(
                f"Eviction stats: tenant_id={tenant_id}, "
                f"tracked_keys={stats['lru_tracked_keys']}, "
                f"usage={usage_mb:.2f}MB"
            )

            return stats

        except Exception as e:
            logger.error(
                f"Get eviction stats failed: tenant_id={tenant_id}, error={e}"
            )
            raise
