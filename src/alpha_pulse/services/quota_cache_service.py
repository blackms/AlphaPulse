"""
Quota cache service for two-tier caching strategy.

Implements Redis (L1) -> PostgreSQL (L2) caching with TTL-based expiration.
"""

import logging
from typing import Optional, Callable, Awaitable
from uuid import UUID
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from alpha_pulse.models.quota import QuotaConfig
from alpha_pulse.models.cache_quota import TenantCacheQuota

logger = logging.getLogger(__name__)


class QuotaCacheService:
    """
    Two-tier caching service for tenant quota configurations.

    Caching Strategy:
    - L1 (Redis): 5-minute TTL, fast access (<3ms)
    - L2 (PostgreSQL): Authoritative source, fallback on cache miss
    """

    def __init__(
        self,
        redis_client: Redis,
        db_session_factory: Callable[[], Awaitable[AsyncSession]],
        cache_ttl_seconds: int = 300  # 5 minutes default
    ):
        """
        Initialize cache service.

        Args:
            redis_client: Async Redis client
            db_session_factory: Factory function to get database session
            cache_ttl_seconds: TTL for Redis cache entries (default 300s)
        """
        self.redis = redis_client
        self.db_session_factory = db_session_factory
        self.cache_ttl = cache_ttl_seconds

    def _get_quota_key(self, tenant_id: UUID) -> str:
        """Get Redis key for quota configuration."""
        return f"quota:cache:{tenant_id}:quota_mb"

    def _get_overage_allowed_key(self, tenant_id: UUID) -> str:
        """Get Redis key for overage_allowed flag."""
        return f"quota:cache:{tenant_id}:overage_allowed"

    def _get_overage_limit_key(self, tenant_id: UUID) -> str:
        """Get Redis key for overage_limit_mb."""
        return f"quota:cache:{tenant_id}:overage_limit_mb"

    def _get_usage_key(self, tenant_id: UUID) -> str:
        """Get Redis key for current_usage_mb."""
        return f"quota:cache:{tenant_id}:current_usage_mb"

    async def get_quota_config(self, tenant_id: UUID) -> Optional[QuotaConfig]:
        """
        Get quota configuration with two-tier caching.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QuotaConfig if found, None if tenant has no quota

        Raises:
            Exception: If both Redis and PostgreSQL fail
        """
        # Try Redis (L1 cache)
        config = await self._get_from_redis(tenant_id)
        if config is not None:
            logger.debug(
                f"Quota cache hit: tenant_id={tenant_id}, source=redis"
            )
            return config

        # Cache miss - fallback to PostgreSQL (L2)
        logger.debug(
            f"Quota cache miss: tenant_id={tenant_id}"
        )

        config = await self._get_from_database(tenant_id)
        if config is None:
            logger.warning(
                f"Quota not found: tenant_id={tenant_id}"
            )
            return None

        # Populate Redis cache for next request
        await self._set_in_redis(config)

        logger.debug(
            f"Quota cache populated: tenant_id={tenant_id}, source=postgresql"
        )

        return config

    async def _get_from_redis(self, tenant_id: UUID) -> Optional[QuotaConfig]:
        """
        Get quota configuration from Redis.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QuotaConfig if found in cache, None on cache miss
        """
        try:
            # Fetch all quota fields from Redis
            quota_mb = await self.redis.get(self._get_quota_key(tenant_id))
            if quota_mb is None:
                return None  # Cache miss

            overage_allowed = await self.redis.get(
                self._get_overage_allowed_key(tenant_id)
            )
            overage_limit_mb = await self.redis.get(
                self._get_overage_limit_key(tenant_id)
            )
            current_usage_mb = await self.redis.get(
                self._get_usage_key(tenant_id)
            )

            # Construct QuotaConfig
            return QuotaConfig(
                tenant_id=tenant_id,
                quota_mb=float(quota_mb),
                current_usage_mb=float(current_usage_mb or 0),
                overage_allowed=overage_allowed == "true",
                overage_limit_mb=float(overage_limit_mb or 0)
            )

        except Exception as e:
            logger.error(
                f"Redis get failed: tenant_id={tenant_id}, error={e}"
            )
            return None  # Treat Redis errors as cache miss

    async def _get_from_database(self, tenant_id: UUID) -> Optional[QuotaConfig]:
        """
        Get quota configuration from PostgreSQL.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QuotaConfig if found, None if tenant has no quota
        """
        try:
            async with self.db_session_factory() as session:
                stmt = select(TenantCacheQuota).where(
                    TenantCacheQuota.tenant_id == tenant_id
                )
                result = await session.execute(stmt)
                quota_row = result.scalar_one_or_none()

                if quota_row is None:
                    return None

                return QuotaConfig(
                    tenant_id=tenant_id,
                    quota_mb=float(quota_row.quota_mb),
                    current_usage_mb=float(quota_row.current_usage_mb),
                    overage_allowed=quota_row.overage_allowed,
                    overage_limit_mb=float(quota_row.overage_limit_mb)
                )

        except Exception as e:
            logger.error(
                f"Database get failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def _set_in_redis(self, config: QuotaConfig) -> None:
        """
        Store quota configuration in Redis with TTL.

        Args:
            config: QuotaConfig to cache
        """
        try:
            # Use pipeline for atomic multi-key set
            pipe = self.redis.pipeline()

            pipe.setex(
                self._get_quota_key(config.tenant_id),
                self.cache_ttl,
                str(config.quota_mb)
            )
            pipe.setex(
                self._get_overage_allowed_key(config.tenant_id),
                self.cache_ttl,
                "true" if config.overage_allowed else "false"
            )
            pipe.setex(
                self._get_overage_limit_key(config.tenant_id),
                self.cache_ttl,
                str(config.overage_limit_mb)
            )
            pipe.setex(
                self._get_usage_key(config.tenant_id),
                self.cache_ttl,
                str(config.current_usage_mb)
            )

            await pipe.execute()

        except Exception as e:
            logger.error(
                f"Redis set failed: tenant_id={config.tenant_id}, error={e}"
            )
            # Don't raise - cache write failures shouldn't block requests

    async def invalidate_cache(self, tenant_id: UUID) -> None:
        """
        Invalidate Redis cache for a tenant.

        Args:
            tenant_id: Tenant identifier
        """
        try:
            keys_to_delete = [
                self._get_quota_key(tenant_id),
                self._get_overage_allowed_key(tenant_id),
                self._get_overage_limit_key(tenant_id),
                self._get_usage_key(tenant_id),
            ]

            await self.redis.delete(*keys_to_delete)

            logger.debug(
                f"Quota cache invalidated: tenant_id={tenant_id}"
            )

        except Exception as e:
            logger.error(
                f"Cache invalidate failed: tenant_id={tenant_id}, error={e}"
            )
            # Don't raise - cache invalidation failures shouldn't block

    async def get_quota(self, tenant_id: UUID) -> Optional[QuotaConfig]:
        """
        Alias for get_quota_config for backward compatibility.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QuotaConfig if found, None if tenant has no quota
        """
        return await self.get_quota_config(tenant_id)

    async def refresh_cache(self, tenant_id: UUID) -> Optional[QuotaConfig]:
        """
        Force refresh cache from database.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QuotaConfig if found, None otherwise
        """
        # Invalidate existing cache
        await self.invalidate_cache(tenant_id)

        # Reload from database
        return await self.get_quota_config(tenant_id)
