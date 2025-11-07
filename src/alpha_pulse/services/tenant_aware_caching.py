"""
Tenant-aware caching service with namespace isolation.

Implements namespace isolation per ADR-004:
- tenant:{tenant_id}:* for tenant-specific data
- shared:market:* for shared market data
- meta:tenant:{tenant_id}:* for tenant metadata

Integrates with:
- UsageTracker (Story 4.3) for quota enforcement
- LRUTracker (Story 4.4) for eviction tracking
- SharedMarketCache (Story 4.5) for shared market data
"""

import logging
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID
from redis.asyncio import Redis

if TYPE_CHECKING:
    from .usage_tracker import UsageTracker
    from .lru_tracker import LRUTracker
    from .shared_market_cache import SharedMarketCache

logger = logging.getLogger(__name__)


class TenantAwareCachingService:
    """
    Tenant-aware caching service with namespace isolation.

    Provides multi-tenant cache access with:
    - Automatic namespace prefixing per tenant
    - Integration with usage tracking and quota enforcement
    - Access to shared market data cache
    - LRU access tracking for eviction
    """

    def __init__(
        self,
        redis_client: Redis,
        usage_tracker: Optional["UsageTracker"] = None,
        lru_tracker: Optional["LRUTracker"] = None,
        shared_cache: Optional["SharedMarketCache"] = None
    ):
        """
        Initialize tenant-aware caching service.

        Args:
            redis_client: Async Redis client
            usage_tracker: Usage tracker for quota enforcement (Story 4.3)
            lru_tracker: LRU tracker for eviction (Story 4.4)
            shared_cache: Shared market cache (Story 4.5)
        """
        self.redis = redis_client
        self.usage_tracker = usage_tracker
        self.lru_tracker = lru_tracker
        self.shared_cache = shared_cache

    def get_tenant_key(self, tenant_id: UUID, key: str) -> str:
        """
        Get tenant-namespaced cache key.

        Args:
            tenant_id: Tenant UUID
            key: Cache key (e.g., "signals:technical:BTC_USDT")

        Returns:
            Namespaced key (e.g., "tenant:{tenant_id}:signals:technical:BTC_USDT")
        """
        return f"tenant:{tenant_id}:{key}"

    def get_metadata_key(self, tenant_id: UUID, key: str) -> str:
        """
        Get metadata cache key.

        Args:
            tenant_id: Tenant UUID
            key: Metadata key (e.g., "quota", "usage")

        Returns:
            Metadata key (e.g., "meta:tenant:{tenant_id}:quota")
        """
        return f"meta:tenant:{tenant_id}:{key}"

    async def get(
        self,
        tenant_id: UUID,
        key: str
    ) -> Optional[Any]:
        """
        Get value from tenant-specific cache.

        Args:
            tenant_id: Tenant UUID
            key: Cache key

        Returns:
            Cached value if found, None if cache miss

        Raises:
            redis.RedisError: If Redis operation fails
        """
        namespaced_key = self.get_tenant_key(tenant_id, key)

        try:
            value = await self.redis.get(namespaced_key)

            if value is None:
                logger.debug(
                    f"Cache miss: tenant_id={tenant_id}, key={key}"
                )
                return None

            # Track LRU access if tracker available
            if self.lru_tracker:
                await self.lru_tracker.track_access(
                    tenant_id=tenant_id,
                    key=namespaced_key
                )

            # Deserialize JSON
            data = json.loads(value)

            logger.debug(
                f"Cache hit: tenant_id={tenant_id}, key={key}"
            )

            return data

        except json.JSONDecodeError as e:
            logger.error(
                f"Cache data corrupt: tenant_id={tenant_id}, key={key}, error={e}"
            )
            raise

        except Exception as e:
            logger.error(
                f"Cache get failed: tenant_id={tenant_id}, key={key}, error={e}"
            )
            raise

    async def set(
        self,
        tenant_id: UUID,
        key: str,
        value: Any,
        ttl: int
    ) -> bool:
        """
        Set value in tenant-specific cache.

        Args:
            tenant_id: Tenant UUID
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful

        Raises:
            redis.RedisError: If Redis operation fails
        """
        namespaced_key = self.get_tenant_key(tenant_id, key)

        try:
            # Serialize to JSON
            serialized = json.dumps(value)
            payload_bytes = len(serialized.encode())

            # Track usage if tracker available
            if self.usage_tracker:
                size_mb = payload_bytes / (1024 * 1024)
                await self.usage_tracker.increment_usage(
                    tenant_id=tenant_id,
                    size_mb=size_mb
                )

            # Store with TTL
            await self.redis.setex(namespaced_key, ttl, serialized)

            # Track LRU access if tracker available
            if self.lru_tracker:
                await self.lru_tracker.track_access(
                    tenant_id=tenant_id,
                    key=namespaced_key
                )

            logger.debug(
                f"Cache set: tenant_id={tenant_id}, key={key}, "
                f"size={payload_bytes} bytes, ttl={ttl}s"
            )

            return True

        except Exception as e:
            logger.error(
                f"Cache set failed: tenant_id={tenant_id}, key={key}, error={e}"
            )
            raise

    async def delete(
        self,
        tenant_id: UUID,
        key: str
    ) -> bool:
        """
        Delete value from tenant-specific cache.

        Args:
            tenant_id: Tenant UUID
            key: Cache key

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            redis.RedisError: If Redis operation fails
        """
        namespaced_key = self.get_tenant_key(tenant_id, key)

        try:
            deleted = await self.redis.delete(namespaced_key)

            if deleted:
                logger.debug(
                    f"Cache deleted: tenant_id={tenant_id}, key={key}"
                )
            else:
                logger.debug(
                    f"Cache key not found: tenant_id={tenant_id}, key={key}"
                )

            return deleted > 0

        except Exception as e:
            logger.error(
                f"Cache delete failed: tenant_id={tenant_id}, key={key}, error={e}"
            )
            raise

    async def mget(
        self,
        tenant_id: UUID,
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        Get multiple values from tenant-specific cache.

        Args:
            tenant_id: Tenant UUID
            keys: List of cache keys

        Returns:
            Dictionary of key -> value for found keys

        Raises:
            redis.RedisError: If Redis operation fails
        """
        # Namespace all keys
        namespaced_keys = [self.get_tenant_key(tenant_id, k) for k in keys]

        try:
            values = await self.redis.mget(namespaced_keys)

            # Build result dictionary (only include non-None values)
            result = {}
            for original_key, value in zip(keys, values):
                if value is not None:
                    result[original_key] = json.loads(value)

                    # Track LRU access if tracker available
                    if self.lru_tracker:
                        await self.lru_tracker.track_access(
                            tenant_id=tenant_id,
                            key=self.get_tenant_key(tenant_id, original_key)
                        )

            logger.debug(
                f"Cache mget: tenant_id={tenant_id}, requested={len(keys)}, "
                f"found={len(result)}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Cache mget failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def mset(
        self,
        tenant_id: UUID,
        data: Dict[str, Any],
        ttl: int
    ) -> bool:
        """
        Set multiple values in tenant-specific cache.

        Args:
            tenant_id: Tenant UUID
            data: Dictionary of key -> value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful

        Raises:
            redis.RedisError: If Redis operation fails
        """
        try:
            # Set each key individually with TTL
            # (Redis MSET doesn't support TTL, so we use multiple SETEX)
            for key, value in data.items():
                await self.set(tenant_id, key, value, ttl)

            logger.debug(
                f"Cache mset: tenant_id={tenant_id}, keys={len(data)}, ttl={ttl}s"
            )

            return True

        except Exception as e:
            logger.error(
                f"Cache mset failed: tenant_id={tenant_id}, error={e}"
            )
            raise

    async def delete_pattern(
        self,
        tenant_id: UUID,
        pattern: str
    ) -> int:
        """
        Delete all keys matching a pattern for a tenant.

        Args:
            tenant_id: Tenant UUID
            pattern: Key pattern (e.g., "signals:technical:*")

        Returns:
            Number of keys deleted

        Raises:
            redis.RedisError: If Redis operation fails
        """
        # Build full pattern with tenant namespace
        full_pattern = self.get_tenant_key(tenant_id, pattern)

        try:
            # Find all matching keys
            keys = []
            async for key in self.redis.scan_iter(match=full_pattern):
                keys.append(key)

            if not keys:
                logger.debug(
                    f"No keys to delete: tenant_id={tenant_id}, pattern={pattern}"
                )
                return 0

            # Delete all keys
            deleted = await self.redis.delete(*keys)

            logger.info(
                f"Cache pattern deleted: tenant_id={tenant_id}, pattern={pattern}, "
                f"keys_deleted={deleted}"
            )

            return deleted

        except Exception as e:
            logger.error(
                f"Cache pattern delete failed: tenant_id={tenant_id}, "
                f"pattern={pattern}, error={e}"
            )
            raise

    # Shared market data operations (delegate to SharedMarketCache)

    async def get_shared_market_data(
        self,
        tenant_id: UUID,
        data_type: str,
        exchange: str,
        symbol: str,
        interval: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get shared market data accessible to all tenants.

        Args:
            tenant_id: Tenant UUID (for access control, not used in key)
            data_type: Type of data ("ohlcv", "ticker", "orderbook")
            exchange: Exchange name
            symbol: Trading symbol
            interval: Time interval (required for OHLCV)

        Returns:
            Market data if found, None if cache miss

        Raises:
            ValueError: If shared_cache not configured or invalid data_type
            redis.RedisError: If Redis operation fails
        """
        if not self.shared_cache:
            raise ValueError("SharedMarketCache not configured")

        try:
            if data_type == "ohlcv":
                if interval is None:
                    raise ValueError("interval required for OHLCV data")
                return await self.shared_cache.get_ohlcv(
                    exchange=exchange,
                    symbol=symbol,
                    interval=interval
                )
            elif data_type == "ticker":
                return await self.shared_cache.get_ticker(
                    exchange=exchange,
                    symbol=symbol
                )
            elif data_type == "orderbook":
                return await self.shared_cache.get_orderbook(
                    exchange=exchange,
                    symbol=symbol
                )
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

        except Exception as e:
            logger.error(
                f"Shared market data get failed: tenant_id={tenant_id}, "
                f"data_type={data_type}, exchange={exchange}, symbol={symbol}, error={e}"
            )
            raise

    async def set_shared_market_data(
        self,
        tenant_id: UUID,
        data_type: str,
        exchange: str,
        symbol: str,
        data: Dict[str, Any],
        ttl: int,
        interval: Optional[str] = None
    ) -> None:
        """
        Set shared market data accessible to all tenants.

        Args:
            tenant_id: Tenant UUID (for access control)
            data_type: Type of data ("ohlcv", "ticker", "orderbook")
            exchange: Exchange name
            symbol: Trading symbol
            data: Market data to cache
            ttl: Time to live in seconds
            interval: Time interval (required for OHLCV)

        Raises:
            ValueError: If shared_cache not configured or invalid data_type
            redis.RedisError: If Redis operation fails
        """
        if not self.shared_cache:
            raise ValueError("SharedMarketCache not configured")

        try:
            if data_type == "ohlcv":
                if interval is None:
                    raise ValueError("interval required for OHLCV data")
                await self.shared_cache.set_ohlcv(
                    exchange=exchange,
                    symbol=symbol,
                    interval=interval,
                    data=data,
                    ttl_seconds=ttl
                )
            elif data_type == "ticker":
                await self.shared_cache.set_ticker(
                    exchange=exchange,
                    symbol=symbol,
                    data=data,
                    ttl_seconds=ttl
                )
            elif data_type == "orderbook":
                await self.shared_cache.set_orderbook(
                    exchange=exchange,
                    symbol=symbol,
                    data=data,
                    ttl_seconds=ttl
                )
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

        except Exception as e:
            logger.error(
                f"Shared market data set failed: tenant_id={tenant_id}, "
                f"data_type={data_type}, exchange={exchange}, symbol={symbol}, error={e}"
            )
            raise
