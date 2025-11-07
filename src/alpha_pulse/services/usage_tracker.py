"""
Usage tracking service for quota enforcement.

Provides atomic usage increment/decrement operations using Redis.
"""

import logging
from typing import Optional
from uuid import UUID
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Tracks tenant cache usage with atomic Redis operations.

    Uses Redis INCRBYFLOAT for atomic counter updates to prevent race conditions
    under concurrent requests.
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize usage tracker.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client

    def _get_usage_key(self, tenant_id: UUID) -> str:
        """
        Get Redis key for tenant usage counter.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Redis key string
        """
        return f"quota:cache:{tenant_id}:current_usage_mb"

    async def increment_usage(
        self,
        tenant_id: UUID,
        size_mb: float
    ) -> float:
        """
        Atomically increment tenant usage.

        Args:
            tenant_id: Tenant identifier
            size_mb: Size to increment in megabytes

        Returns:
            New usage value after increment

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self._get_usage_key(tenant_id)

        try:
            new_usage = await self.redis.incrbyfloat(key, size_mb)

            logger.debug(
                f"Usage incremented: tenant_id={tenant_id}, size_mb={size_mb}, new_usage_mb={new_usage}"
            )

            return float(new_usage)

        except Exception as e:
            logger.error(
                f"Usage increment failed: tenant_id={tenant_id}, size_mb={size_mb}, error={e}"
            )
            raise

    async def decrement_usage(
        self,
        tenant_id: UUID,
        size_mb: float
    ) -> float:
        """
        Atomically decrement tenant usage (rollback).

        Args:
            tenant_id: Tenant identifier
            size_mb: Size to decrement in megabytes

        Returns:
            New usage value after decrement

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self._get_usage_key(tenant_id)

        try:
            # Decrement by incrementing with negative value
            new_usage = await self.redis.incrbyfloat(key, -size_mb)

            # Ensure usage doesn't go negative
            if new_usage < 0:
                await self.redis.set(key, 0)
                new_usage = 0.0

            logger.debug(
                f"Usage decremented: tenant_id={tenant_id}, size_mb={size_mb}, new_usage_mb={new_usage}"
            )

            return float(new_usage)

        except Exception as e:
            logger.error(
                f"Usage decrement failed: tenant_id={tenant_id}, size_mb={size_mb}, error={e}"
            )
            raise

    async def get_usage(self, tenant_id: UUID) -> Optional[float]:
        """
        Get current usage for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Current usage in megabytes, or None if not set
        """
        key = self._get_usage_key(tenant_id)

        try:
            value = await self.redis.get(key)
            if value is None:
                return None

            return float(value)

        except Exception as e:
            logger.error(
                f"Usage get failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def set_usage(
        self,
        tenant_id: UUID,
        usage_mb: float,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set usage value (used for cache initialization).

        Args:
            tenant_id: Tenant identifier
            usage_mb: Usage value in megabytes
            ttl_seconds: Optional TTL for the key

        Raises:
            redis.RedisError: If Redis operation fails
        """
        key = self._get_usage_key(tenant_id)

        try:
            if ttl_seconds:
                await self.redis.setex(key, ttl_seconds, str(usage_mb))
            else:
                await self.redis.set(key, str(usage_mb))

            logger.debug(
                f"Usage set: tenant_id={tenant_id}, usage_mb={usage_mb}, ttl_seconds={ttl_seconds}"
            )

        except Exception as e:
            logger.error(
                f"Usage set failed: tenant_id={tenant_id}, usage_mb={usage_mb}, error={e}"
            )
            raise

    async def reset_usage(self, tenant_id: UUID) -> None:
        """
        Reset usage to zero.

        Args:
            tenant_id: Tenant identifier
        """
        await self.set_usage(tenant_id, 0.0)

    async def delete_usage(self, tenant_id: UUID) -> None:
        """
        Delete usage key from Redis.

        Args:
            tenant_id: Tenant identifier
        """
        key = self._get_usage_key(tenant_id)

        try:
            await self.redis.delete(key)

            logger.debug(
                f"Usage deleted: tenant_id={tenant_id}"
            )

        except Exception as e:
            logger.error(
                f"Usage delete failed: tenant_id={tenant_id}, error={e}"
            )
            raise
