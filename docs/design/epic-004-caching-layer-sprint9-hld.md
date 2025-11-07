# EPIC-004: Caching Layer - High-Level Design (HLD)

**Epic**: EPIC-004 (#143)
**Sprint**: 9-10
**Story Points**: 29
**Date**: 2025-11-07
**Phase**: Design
**Author**: Tech Lead (via Claude Code)
**Status**: DRAFT

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Design](#component-design)
4. [Data Models](#data-models)
5. [Namespace Design](#namespace-design)
6. [Quota Management](#quota-management)
7. [Performance & Scalability](#performance--scalability)
8. [Monitoring & Observability](#monitoring--observability)
9. [Deployment Architecture](#deployment-architecture)
10. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Purpose
Transform AlphaPulse's existing Redis caching infrastructure from single-tenant to multi-tenant with namespace isolation, per-tenant quotas, and optimized shared market data caching.

### Scope
- **In Scope**: Namespace isolation, quota enforcement, shared market data optimization, per-tenant metrics, L1 cache tenant awareness
- **Out of Scope**: Redis Cluster deployment (DevOps), custom eviction algorithms beyond LRU

### Key Design Decisions
1. **Namespace Pattern**: `tenant:{tenant_id}:{resource}:{key}` for isolation (per ADR-004)
2. **Shared Market Data**: `shared:market:{symbol}:*` namespace (90% memory reduction)
3. **Quota Tracking**: Redis counters + sorted sets (no keyspace scanning)
4. **L1 Cache**: Per-tenant in-memory LRU (max 10MB each)
5. **Existing Infrastructure**: Build on top of `RedisManager`, `CachingService`, `DistributedCache`

### Success Criteria
- ✅ 100% tenant namespace isolation (0% cross-tenant access)
- ✅ >95% cache hit rate for shared market data (top 100 symbols)
- ✅ <2ms P99 latency for cache operations (no regression)
- ✅ <1% quota violations (soft limits prevent hard failures)

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  CacheQuotaMiddleware                                      │  │
│  │  - Extract tenant_id from JWT                              │  │
│  │  - Check quota before allowing cache writes                │  │
│  │  - Set X-Cache-Quota-Exceeded header if over limit        │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                Service Layer (Tenant-Aware Caching)              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         TenantCacheManager (NEW)                           │  │
│  │  - get(tenant_id, resource, key)                           │  │
│  │  - set(tenant_id, resource, key, value, ttl)               │  │
│  │  - delete(tenant_id, resource, key)                        │  │
│  │  - get_usage(tenant_id)                                    │  │
│  │  - get_metrics(tenant_id)                                  │  │
│  └───────────────┬────────────────────────────────────────────┘  │
│                  │                                                │
│  ┌───────────────┴────────────────────────────────────────────┐  │
│  │  SharedMarketDataCache (NEW)                               │  │
│  │  - get_price(symbol)                                       │  │
│  │  - get_ohlcv(symbol, timeframe, limit)                     │  │
│  │  - Single-flight pattern (prevent stampede)                │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│            Existing Infrastructure (ENHANCED)                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  CachingService (UPDATED with tenant context)              │  │
│  │  - L1: Per-tenant in-memory LRU (max 10MB each)            │  │
│  │  - L2: Redis (tenant namespace)                            │  │
│  │  - L3: Database/API (tenant-filtered)                      │  │
│  └───────────────┬────────────────────────────────────────────┘  │
│                  │                                                │
│  ┌───────────────┴────────────────────────────────────────────┐  │
│  │  RedisManager (EXISTING - no changes)                      │  │
│  │  - Connection pooling (max 50 connections)                 │  │
│  │  - Automatic reconnection with backoff                     │  │
│  │  - Prometheus metrics integration                          │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────┐
    │   Redis Cluster (6 nodes)             │
    │   ┌─────────────────────────────────┐ │
    │   │ Namespace: tenant:{id}:*        │ │
    │   │ - signals:{agent}:{symbol}      │ │
    │   │ - portfolio:positions           │ │
    │   │ - session:{session_id}          │ │
    │   │                                 │ │
    │   │ Namespace: shared:market:*      │ │
    │   │ - {symbol}:price                │ │
    │   │ - {symbol}:1m:ohlcv             │ │
    │   │ - {symbol}:1d:ohlcv             │ │
    │   │                                 │ │
    │   │ Namespace: meta:tenant:{id}:*   │ │
    │   │ - quota (int)                   │ │
    │   │ - usage_bytes (counter)         │ │
    │   │ - lru (sorted set)              │ │
    │   └─────────────────────────────────┘ │
    │   Masters: redis-0, redis-1, redis-2  │
    │   Replicas: redis-3, redis-4, redis-5 │
    └───────────────────────────────────────┘

    ┌───────────────────────────────────────┐
    │   PostgreSQL (Metadata)               │
    │   ┌─────────────────────────────────┐ │
    │   │ tenant_cache_quotas             │ │
    │   │ - tenant_id, quota_mb, usage_mb │ │
    │   │                                 │ │
    │   │ tenant_cache_metrics            │ │
    │   │ - tenant_id, date, hit_rate     │ │
    │   └─────────────────────────────────┘ │
    └───────────────────────────────────────┘

    ┌───────────────────────────────────────┐
    │   Background Jobs (APScheduler)       │
    │   ┌─────────────────────────────────┐ │
    │   │ CacheMetricsFlushJob            │ │
    │   │ - Runs every 5 minutes          │ │
    │   │ - Flushes counters to database  │ │
    │   │ - Calculates hit rates          │ │
    │   └─────────────────────────────────┘ │
    │   ┌─────────────────────────────────┐ │
    │   │ CacheAuditJob                   │ │
    │   │ - Runs every 1 hour             │ │
    │   │ - Reconciles usage counters     │ │
    │   │ - Scans keyspace (batched)      │ │
    │   └─────────────────────────────────┘ │
    └───────────────────────────────────────┘
```

### Data Flow Diagrams

#### 1. Tenant Cache Write Flow (with Quota Check)

```
Trading Agent    CacheQuotaMiddleware    TenantCacheManager    RedisManager    PostgreSQL
     │                   │                        │                  │              │
     │ set_cache()       │                        │                  │              │
     ├──────────────────>│                        │                  │              │
     │                   │ check_quota(tenant_id) │                  │              │
     │                   ├───────────────────────>│                  │              │
     │                   │                        │ GET meta:tenant:{id}:usage_bytes │
     │                   │                        ├─────────────────>│              │
     │                   │                        │ usage=5.2MB      │              │
     │                   │                        │<─────────────────┤              │
     │                   │                        │ SELECT quota FROM tenant_cache_quotas
     │                   │                        ├──────────────────────────────────>
     │                   │                        │ quota=10MB       │              │
     │                   │                        │<──────────────────────────────────┤
     │                   │ Under quota (52%)      │                  │              │
     │                   │<───────────────────────┤                  │              │
     │                   │                        │                  │              │
     │                   │ set_with_namespace()   │                  │              │
     │                   ├───────────────────────>│                  │              │
     │                   │                        │ namespaced_key = │              │
     │                   │                        │ "tenant:{id}:signals:technical:BTC"
     │                   │                        │                  │              │
     │                   │                        │ SET with TTL     │              │
     │                   │                        ├─────────────────>│              │
     │                   │                        │                  │ Write to Redis
     │                   │                        │<─────────────────┤              │
     │                   │                        │                  │              │
     │                   │                        │ track_usage()    │              │
     │                   │                        ├─────────────────>│              │
     │                   │                        │ INCRBY meta:tenant:{id}:usage_bytes 100KB
     │                   │                        │ ZADD meta:tenant:{id}:lru {key: timestamp}
     │                   │                        │<─────────────────┤              │
     │                   │ Success               │                  │              │
     │                   │<───────────────────────┤                  │              │
     │ Success           │                        │                  │              │
     │<──────────────────┤                        │                  │              │
```

#### 2. Shared Market Data Flow (Single-Flight Pattern)

```
Tenant A         Tenant B         SharedMarketDataCache    RedisManager    Exchange API
   │                 │                      │                    │               │
   │ get_price(BTC)  │                      │                    │               │
   ├────────────────────────────────────────>                    │               │
   │                 │                      │ Check cache        │               │
   │                 │                      ├───────────────────>│               │
   │                 │                      │ GET shared:market:BTC:price         │
   │                 │                      │ MISS               │               │
   │                 │                      │<───────────────────┤               │
   │                 │                      │                    │               │
   │                 │                      │ Acquire lock       │               │
   │                 │                      │ SETNX lock:BTC     │               │
   │                 │                      ├───────────────────>│               │
   │                 │                      │ Lock acquired      │               │
   │                 │                      │<───────────────────┤               │
   │                 │                      │                    │               │
   │                 │ get_price(BTC)       │                    │               │
   │                 ├─────────────────────>│ Lock exists (wait) │               │
   │                 │                      │                    │               │
   │                 │                      │ Fetch from exchange│               │
   │                 │                      ├────────────────────────────────────>
   │                 │                      │ price=$67,123.45   │               │
   │                 │                      │<────────────────────────────────────┤
   │                 │                      │                    │               │
   │                 │                      │ SET shared:market:BTC:price TTL=60s │
   │                 │                      ├───────────────────>│               │
   │                 │                      │                    │               │
   │                 │                      │ Release lock       │               │
   │                 │                      │ DEL lock:BTC       │               │
   │                 │                      ├───────────────────>│               │
   │ Price (from cache)                     │                    │               │
   │<────────────────────────────────────────                    │               │
   │                 │ Price (from cache)   │                    │               │
   │                 │<─────────────────────┤                    │               │
   │                 │                      │                    │               │
   │ [99 more tenants request BTC price - all cache HITs]        │               │
```

#### 3. Quota Enforcement and Eviction Flow

```
TenantCacheManager    RedisManager    QuotaEnforcer    EvictionEngine
        │                  │                │                 │
        │ set_cache()      │                │                 │
        ├─────────────────>│                │                 │
        │                  │ check_quota()  │                 │
        │                  ├───────────────>│                 │
        │                  │                │ GET meta:tenant:{id}:usage_bytes
        │                  │                ├────────────────>
        │                  │                │ usage=9.8MB     │
        │                  │                │<────────────────┤
        │                  │                │ quota=10MB      │
        │                  │                │ Usage: 98%      │
        │                  │                │ (OVER THRESHOLD)│
        │                  │                │                 │
        │                  │                │ evict_lru()     │
        │                  │                ├────────────────>│
        │                  │                │                 │ ZPOPMIN meta:tenant:{id}:lru 10
        │                  │                │                 │ (Get 10 oldest keys)
        │                  │                │                 │
        │                  │                │                 │ For each key:
        │                  │                │                 │ - GET key (to measure size)
        │                  │                │                 │ - DEL key
        │                  │                │                 │ - DECRBY usage_bytes {size}
        │                  │                │                 │
        │                  │                │ Evicted 1.2MB   │
        │                  │                │<────────────────┤
        │                  │ Quota OK (87%) │                 │
        │                  │<───────────────┤                 │
        │ Proceed with write                │                 │
        │<─────────────────┤                │                 │
```

---

## Component Design

### 1. TenantCacheManager

**Responsibility**: Provide tenant-aware cache interface with namespace isolation and quota enforcement.

**Location**: `src/alpha_pulse/services/tenant_cache_manager.py` (NEW)

**Interface**:
```python
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TenantCacheManager:
    """Multi-tenant cache manager with namespace isolation."""

    def __init__(
        self,
        redis_manager: RedisManager,
        db_session: AsyncSession
    ):
        self.redis = redis_manager
        self.db = db_session
        self._quota_cache: Dict[UUID, int] = {}  # tenant_id → quota_bytes
        self._quota_cache_timestamps: Dict[UUID, datetime] = {}

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
            value = await cache.get(
                tenant_id=UUID('...'),
                resource='signals',
                key='technical:BTC'
            )
            # Fetches from: tenant:{tenant_id}:signals:technical:BTC
        """
        namespaced_key = self._build_key(tenant_id, resource, key)
        raw_value = await self.redis.get(namespaced_key)

        if raw_value:
            logger.debug(f"Cache HIT: {namespaced_key}")
            await self._increment_hit_counter(tenant_id)
            return self._deserialize(raw_value)
        else:
            logger.debug(f"Cache MISS: {namespaced_key}")
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
            True if set, False if quota exceeded

        Raises:
            QuotaExceededException: Hard limit exceeded (after eviction attempt)
        """
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
            evicted = await self._evict_tenant_lru(
                tenant_id,
                target_size=int(quota * 0.9)
            )

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

    async def get_usage(self, tenant_id: UUID) -> int:
        """
        Get current cache usage for tenant in bytes.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Usage in bytes
        """
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
        usage = await self.redis.get(usage_key)
        return int(usage or 0)

    async def get_metrics(self, tenant_id: UUID) -> Dict:
        """
        Get cache metrics for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dict with hits, misses, hit_rate, usage_bytes, quota_bytes
        """
        # Fetch counters from Redis
        hits_key = f"meta:tenant:{tenant_id}:hits"
        misses_key = f"meta:tenant:{tenant_id}:misses"
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"

        hits, misses, usage = await self.redis.mget([hits_key, misses_key, usage_key])

        hits = int(hits or 0)
        misses = int(misses or 0)
        usage = int(usage or 0)
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

    # Private helper methods

    def _build_key(self, tenant_id: UUID, resource: str, key: str) -> str:
        """Build namespaced cache key."""
        return f"tenant:{tenant_id}:{resource}:{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes (JSON or pickle)."""
        import json
        return json.dumps(value).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        import json
        return json.loads(data.decode('utf-8'))

    async def _get_quota(self, tenant_id: UUID) -> int:
        """Get quota for tenant (cached for 5 minutes)."""
        # Check cache
        if tenant_id in self._quota_cache_timestamps:
            age = (datetime.utcnow() - self._quota_cache_timestamps[tenant_id]).total_seconds()
            if age < 300:  # 5 minutes
                return self._quota_cache[tenant_id]

        # Fetch from database
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

    async def _get_usage(self, tenant_id: UUID) -> int:
        """Get current usage from Redis counter."""
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
        usage = await self.redis.get(usage_key)
        return int(usage or 0)

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
        pipeline = self.redis.pipeline()

        # Increment usage counter
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
        pipeline.incrby(usage_key, payload_size)

        # Add to LRU sorted set (score = timestamp)
        lru_key = f"meta:tenant:{tenant_id}:lru"
        pipeline.zadd(lru_key, {cache_key: datetime.utcnow().timestamp()})

        await pipeline.execute()

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
        pipeline = self.redis.pipeline()

        # Decrement usage counter
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"
        pipeline.decrby(usage_key, payload_size)

        # Remove from LRU sorted set
        lru_key = f"meta:tenant:{tenant_id}:lru"
        pipeline.zrem(lru_key, cache_key)

        await pipeline.execute()

    async def _increment_hit_counter(self, tenant_id: UUID):
        """Increment cache hit counter."""
        hits_key = f"meta:tenant:{tenant_id}:hits"
        await self.redis.incr(hits_key)

    async def _increment_miss_counter(self, tenant_id: UUID):
        """Increment cache miss counter."""
        misses_key = f"meta:tenant:{tenant_id}:misses"
        await self.redis.incr(misses_key)

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

        while total_evicted < bytes_to_evict:
            # Get oldest keys (lowest scores in sorted set)
            oldest_keys = await self.redis.zpopmin(lru_key, batch_size)

            if not oldest_keys:
                break  # No more keys to evict

            for cache_key, _ in oldest_keys:
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


class QuotaExceededException(Exception):
    """Raised when tenant exceeds cache quota."""

    def __init__(self, tenant_id: UUID, message: str):
        self.tenant_id = tenant_id
        super().__init__(message)
```

**Dependencies**:
- `RedisManager` (existing: `src/alpha_pulse/services/redis_manager.py`)
- `AsyncSession` (SQLAlchemy async database session)

### 2. SharedMarketDataCache

**Responsibility**: Optimize shared market data caching with single-flight pattern.

**Location**: `src/alpha_pulse/services/shared_market_data_cache.py` (NEW)

**Interface**:
```python
import asyncio
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SharedMarketDataCache:
    """
    Shared market data cache with single-flight pattern.

    Market data is identical across all tenants, so we cache once
    and share safely. Uses single-flight pattern to prevent cache
    stampede (100 tenants requesting BTC price simultaneously).
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        exchange_adapter: CCXTAdapter
    ):
        self.redis = redis_manager
        self.exchange = exchange_adapter
        self._in_flight: Dict[str, asyncio.Lock] = {}  # cache_key → lock

    async def get_price(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[Decimal]:
        """
        Get latest price for symbol (shared across all tenants).

        Args:
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Price as Decimal or None if not available

        Cache:
            Key: shared:market:{exchange}:{symbol}:price
            TTL: 60 seconds
        """
        cache_key = f"shared:market:{exchange}:{symbol}:price"

        # Try cache first
        cached_price = await self.redis.get(cache_key)
        if cached_price:
            logger.debug(f"Shared cache HIT: {cache_key}")
            return Decimal(cached_price)

        # Cache MISS - fetch with single-flight pattern
        return await self._fetch_with_single_flight(
            cache_key=cache_key,
            fetcher=lambda: self.exchange.fetch_ticker(exchange, symbol),
            extractor=lambda ticker: Decimal(str(ticker['last'])),
            ttl=60
        )

    async def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = '1m',
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Get OHLCV candles for symbol (shared across all tenants).

        Args:
            exchange: Exchange name
            symbol: Trading pair
            timeframe: Timeframe (1m, 5m, 1h, 1d)
            limit: Number of candles

        Returns:
            List of OHLCV dicts or None

        Cache:
            Key: shared:market:{exchange}:{symbol}:{timeframe}:ohlcv
            TTL: 60s (1m), 300s (5m/1h), 3600s (1d)
        """
        cache_key = f"shared:market:{exchange}:{symbol}:{timeframe}:ohlcv"

        # Try cache first
        cached_ohlcv = await self.redis.get(cache_key)
        if cached_ohlcv:
            logger.debug(f"Shared cache HIT: {cache_key}")
            return self._deserialize(cached_ohlcv)

        # Cache MISS - fetch with single-flight pattern
        ttl = self._get_ttl_for_timeframe(timeframe)
        return await self._fetch_with_single_flight(
            cache_key=cache_key,
            fetcher=lambda: self.exchange.fetch_ohlcv(exchange, symbol, timeframe, limit),
            extractor=lambda ohlcv: [
                {
                    'timestamp': candle[0],
                    'open': Decimal(str(candle[1])),
                    'high': Decimal(str(candle[2])),
                    'low': Decimal(str(candle[3])),
                    'close': Decimal(str(candle[4])),
                    'volume': Decimal(str(candle[5]))
                }
                for candle in ohlcv
            ],
            ttl=ttl
        )

    async def warm_top_symbols(self, top_symbols: List[str]):
        """
        Pre-warm cache with top N symbols (background job).

        Args:
            top_symbols: List of symbols to warm (e.g., ['BTC/USDT', 'ETH/USDT'])
        """
        logger.info(f"Warming shared cache for {len(top_symbols)} symbols")

        for symbol in top_symbols:
            try:
                await self.get_price('binance', symbol)
                await self.get_ohlcv('binance', symbol, timeframe='1m', limit=100)
            except Exception as e:
                logger.error(f"Failed to warm {symbol}: {e}")

        logger.info("Cache warming complete")

    # Private helper methods

    async def _fetch_with_single_flight(
        self,
        cache_key: str,
        fetcher: callable,
        extractor: callable,
        ttl: int
    ):
        """
        Fetch data with single-flight pattern to prevent cache stampede.

        Args:
            cache_key: Redis cache key
            fetcher: Async function that fetches data
            extractor: Function to extract value from fetched data
            ttl: Time-to-live in seconds

        Returns:
            Extracted value

        Single-Flight Pattern:
            Only one coroutine fetches from API, others wait for result.
        """
        # Check if fetch is already in flight
        if cache_key not in self._in_flight:
            self._in_flight[cache_key] = asyncio.Lock()

        async with self._in_flight[cache_key]:
            # Double-check cache (another coroutine might have populated it)
            cached_value = await self.redis.get(cache_key)
            if cached_value:
                logger.debug(f"Cache populated by another coroutine: {cache_key}")
                return extractor(self._deserialize(cached_value))

            # Fetch from API (only one coroutine reaches here)
            logger.info(f"Fetching from API: {cache_key}")
            raw_data = await fetcher()
            value = extractor(raw_data)

            # Cache the result
            serialized = self._serialize(value)
            await self.redis.set(cache_key, serialized, ttl=ttl)

            logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
            return value

    def _get_ttl_for_timeframe(self, timeframe: str) -> int:
        """Get appropriate TTL based on timeframe."""
        ttl_map = {
            '1m': 60,      # 1 minute
            '5m': 300,     # 5 minutes
            '15m': 300,    # 5 minutes
            '1h': 300,     # 5 minutes
            '4h': 3600,    # 1 hour
            '1d': 3600,    # 1 hour
        }
        return ttl_map.get(timeframe, 300)

    def _serialize(self, value) -> bytes:
        """Serialize value to bytes."""
        import json
        return json.dumps(value, default=str).encode('utf-8')

    def _deserialize(self, data: bytes):
        """Deserialize bytes to value."""
        import json
        return json.loads(data.decode('utf-8'))
```

### 3. CacheQuotaMiddleware

**Responsibility**: Enforce cache quota at API layer.

**Location**: `src/alpha_pulse/api/middleware/cache_quota.py` (NEW)

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from alpha_pulse.services.tenant_cache_manager import TenantCacheManager
import logging

logger = logging.getLogger(__name__)

class CacheQuotaMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check cache quota before allowing cache writes.

    Sets X-Cache-Quota-Exceeded header if tenant over quota.
    """

    def __init__(self, app, cache_manager: TenantCacheManager):
        super().__init__(app)
        self.cache_manager = cache_manager
        self._quota_check_cache = {}  # tenant_id → (usage, timestamp)

    async def dispatch(self, request: Request, call_next):
        # Extract tenant_id from JWT (set by tenant context middleware)
        tenant_id = getattr(request.state, 'tenant_id', None)

        if not tenant_id:
            return await call_next(request)

        # Check quota (cached check, only query every 5 minutes)
        is_over_quota = await self._check_quota(tenant_id)

        if is_over_quota:
            logger.warning(f"Tenant {tenant_id} exceeded cache quota")

        # Proceed with request
        response = await call_next(request)

        # Add header if over quota
        if is_over_quota:
            response.headers['X-Cache-Quota-Exceeded'] = 'true'

        return response

    async def _check_quota(self, tenant_id) -> bool:
        """
        Check if tenant exceeded quota (cached for 5 minutes).

        Returns:
            True if over quota, False otherwise
        """
        # Check cache first
        if tenant_id in self._quota_check_cache:
            cached_result, cached_time = self._quota_check_cache[tenant_id]
            age = (datetime.utcnow() - cached_time).total_seconds()
            if age < 300:  # 5 minutes
                return cached_result

        # Fetch fresh data
        usage = await self.cache_manager.get_usage(tenant_id)
        quota = await self.cache_manager._get_quota(tenant_id)

        is_over_quota = usage > quota

        # Cache result for 5 minutes
        self._quota_check_cache[tenant_id] = (is_over_quota, datetime.utcnow())

        return is_over_quota
```

---

## Data Models

### 1. Database Schema

**Table**: `tenant_cache_quotas`

```sql
CREATE TABLE tenant_cache_quotas (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    quota_mb INT NOT NULL DEFAULT 100,              -- Max cache size in MB
    current_usage_mb DECIMAL(10,2) DEFAULT 0,       -- Current usage (synced from Redis hourly)
    quota_reset_at TIMESTAMP DEFAULT NOW(),         -- Quota reset time (monthly)
    overage_allowed BOOLEAN DEFAULT false,          -- Allow temporary overage?
    overage_limit_mb INT DEFAULT 10,                -- Max overage before hard limit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id)
);

CREATE INDEX idx_tenant_cache_quotas_tenant_id ON tenant_cache_quotas(tenant_id);
CREATE INDEX idx_tenant_cache_quotas_overage ON tenant_cache_quotas(tenant_id)
    WHERE current_usage_mb > quota_mb;

-- Row-Level Security
ALTER TABLE tenant_cache_quotas ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_cache_quotas_isolation ON tenant_cache_quotas
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
```

**Table**: `tenant_cache_metrics`

```sql
CREATE TABLE tenant_cache_metrics (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    total_requests BIGINT DEFAULT 0,
    cache_hits BIGINT DEFAULT 0,
    cache_misses BIGINT DEFAULT 0,
    hit_rate DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE
            WHEN total_requests > 0 THEN (cache_hits::DECIMAL / total_requests * 100)
            ELSE 0
        END
    ) STORED,
    avg_response_time_ms DECIMAL(10,2),
    total_bytes_served BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, metric_date)
);

CREATE INDEX idx_tenant_cache_metrics_tenant_date ON tenant_cache_metrics(tenant_id, metric_date DESC);
```

### 2. Redis Key Structure

**Tenant-Scoped Keys**:
```
tenant:{tenant_id}:signals:{agent_type}:{symbol}
  Example: tenant:00000000-0000-0000-0000-000000000001:signals:technical:BTC
  TTL: 5 minutes
  Size: ~100KB (JSON with indicators)

tenant:{tenant_id}:portfolio:positions
  Example: tenant:00000000-0000-0000-0000-000000000001:portfolio:positions
  TTL: 1 minute
  Size: ~50KB (JSON array of positions)

tenant:{tenant_id}:session:{session_id}
  Example: tenant:00000000-0000-0000-0000-000000000001:session:abc123
  TTL: 1 hour
  Size: ~10KB (user preferences, WebSocket state)
```

**Shared Keys** (No Tenant Isolation):
```
shared:market:{exchange}:{symbol}:price
  Example: shared:market:binance:BTC/USDT:price
  TTL: 60 seconds
  Size: ~100 bytes (Decimal as string)

shared:market:{exchange}:{symbol}:{timeframe}:ohlcv
  Example: shared:market:binance:BTC/USDT:1m:ohlcv
  TTL: 60-3600 seconds (depends on timeframe)
  Size: ~50KB (100 candles × 500 bytes)
```

**Metadata Keys** (Quota Tracking):
```
meta:tenant:{tenant_id}:usage_bytes
  Example: meta:tenant:00000000-0000-0000-0000-000000000001:usage_bytes
  Type: Integer counter
  No TTL (persistent)

meta:tenant:{tenant_id}:lru
  Example: meta:tenant:00000000-0000-0000-0000-000000000001:lru
  Type: Sorted set (score = timestamp, member = cache_key)
  No TTL (persistent)

meta:tenant:{tenant_id}:hits
meta:tenant:{tenant_id}:misses
  Type: Integer counters
  No TTL (flushed to database every 5 minutes, then reset)
```

---

## Namespace Design

### Namespace Pattern

**Format**: `{scope}:{identifier}:{resource}:{key}`

**Scope Types**:
1. `tenant` - Tenant-specific data (isolated)
2. `shared` - Multi-tenant shared data (market data)
3. `meta` - System metadata (quotas, metrics)

**Examples**:
```
tenant:00000000-0000-0000-0000-000000000001:signals:technical:BTC
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^ ^^^^^^^^^ ^^^
       Tenant ID (UUID)                     Type    Agent     Symbol

shared:market:binance:BTC/USDT:price
^^^^^^ ^^^^^^ ^^^^^^^ ^^^^^^^^ ^^^^^
Scope  Type   Exchange Symbol   Data

meta:tenant:00000000-0000-0000-0000-000000000001:usage_bytes
^^^^ ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^
Scope Type   Tenant ID                            Metric
```

### Namespace Isolation Enforcement

**Application Level**:
- `TenantCacheManager` always prefixes keys with `tenant:{tenant_id}:`
- No direct Redis access allowed (must go through `TenantCacheManager`)
- Tenant ID extracted from JWT (validated by middleware)

**Redis Level**:
- Redis ACLs can enforce patterns (future enhancement):
  ```
  user tenant_00000000-0000-0000-0000-000000000001 on >password ~tenant:00000000-0000-0000-0000-000000000001:* +@all
  ```

---

## Quota Management

### Quota Tiers

| Tier | Cache Quota | Use Case |
|------|-------------|----------|
| Starter | 100MB | Hobbyist traders (1-2 symbols) |
| Pro | 500MB | Active traders (5-10 symbols) |
| Enterprise | 2GB | Institutional traders (50+ symbols) |

### Quota Enforcement Strategy

**Soft Limit** (Warning):
- Trigger: Usage > 80% of quota
- Action: Log warning, set header `X-Cache-Quota-Warning: true`
- Allow writes to continue

**Hard Limit** (Eviction):
- Trigger: Usage > 100% of quota
- Action: Evict oldest 10% of keys (LRU)
- Allow write if space freed

**Critical Limit** (Block):
- Trigger: Usage > 110% of quota AND eviction failed
- Action: Raise `QuotaExceededException`, block write
- User must upgrade tier or delete keys

### Eviction Algorithm

**Per-Tenant LRU** (Not Global):
1. Maintain sorted set per tenant: `meta:tenant:{id}:lru`
2. Score = timestamp of last access
3. On quota exceeded:
   - `ZPOPMIN` to get oldest keys
   - Delete keys until usage < 90% of quota
   - Update usage counter

**Complexity**:
- Eviction: O(N log N) where N = keys to evict (typically <100)
- No global keyspace scan required

---

## Performance & Scalability

### Latency Budget

| Operation | Target P99 | Breakdown |
|-----------|------------|-----------|
| get() - cache HIT | <2ms | Redis GET: 1ms + deserialization: 0.5ms |
| get() - cache MISS | <5ms | Redis GET: 1ms + fetcher: 3ms |
| set() - under quota | <5ms | Serialization: 1ms + Redis SET: 2ms + tracking: 1ms |
| set() - quota check | <10ms | Quota check: 2ms + eviction: 5ms + Redis SET: 2ms |
| Shared cache (single-flight) | <300ms | Exchange API: 200ms + Redis SET: 50ms |

### Throughput

**Redis Cluster**:
- Capacity: 100,000 ops/sec per node (6 nodes = 600K ops/sec)
- Expected load: ~500 ops/sec (1000 tenants × 3 req/min / 60 sec)
- **Headroom**: 1200x capacity

**Memory**:
- Total: 64GB (6 nodes × 10GB each + overhead)
- Tenant cache: 1000 tenants × 100MB avg = 100GB (exceeds capacity!)
- **Mitigation**: Strict quota enforcement + eviction keeps actual usage ~30GB

### Caching Strategy

**Multi-Tier**:
```
L1: In-memory LRU per tenant (max 10MB each)
  └─→ Hit rate: 50-60% (hot keys)
  └─→ Latency: <1ms

L2: Redis (tenant namespace)
  └─→ Hit rate: 80-90% (overall)
  └─→ Latency: <2ms

L3: Database / Exchange API
  └─→ Hit rate: N/A (cache miss)
  └─→ Latency: 50-300ms
```

**Shared Cache Optimization**:
- Single-flight pattern prevents stampede (100 requests → 1 API call)
- Pre-warm top 100 symbols on startup
- Expected hit rate: >95% for popular symbols

---

## Monitoring & Observability

### Prometheus Metrics

**Tenant Cache Metrics**:
```
# Total cache operations
alphapulse_cache_operations_total{tenant_id, operation, status}

# Cache hit rate
alphapulse_cache_hit_rate{tenant_id}

# Cache quota usage
alphapulse_cache_quota_usage_bytes{tenant_id}

# Cache quota exceeded events
alphapulse_cache_quota_exceeded_total{tenant_id}

# Cache evictions
alphapulse_cache_evictions_total{tenant_id}
```

**Shared Cache Metrics**:
```
# Shared cache hit rate by symbol
alphapulse_shared_cache_hit_rate{symbol}

# Single-flight lock wait time
alphapulse_shared_cache_single_flight_wait_seconds{symbol}

# API calls saved (due to caching)
alphapulse_shared_cache_api_calls_saved_total{exchange}
```

**Redis Metrics** (via Redis exporter):
```
# Redis memory usage
redis_memory_used_bytes

# Redis operations per second
redis_commands_processed_total

# Redis keyspace (keys per database)
redis_db_keys{db}
```

### Grafana Dashboards

**Dashboard 1: Tenant Cache Overview**
- Cache hit rate by tenant (top 10)
- Quota usage by tenant (bar chart)
- Eviction rate (line chart over time)
- Quota exceeded events (heatmap)

**Dashboard 2: Shared Cache Performance**
- Shared cache hit rate by symbol
- API calls saved (cost savings)
- Single-flight lock contention

**Dashboard 3: Redis Cluster Health**
- Memory usage per node
- Operations per second
- Replication lag
- Cluster state (OK/FAIL)

### Alerts

**Critical**:
- `RedisMasterDown`: Redis master node unreachable (page on-call)
- `CacheEvictionRateHigh`: >10% keys evicted per minute (indicates quota too low)

**Warning**:
- `TenantQuotaExceeded`: Tenant over quota for >1 hour (contact account manager)
- `SharedCacheHitRate < 70%`: Shared cache not effective (investigate)
- `RedisMemoryUsage > 80%`: Approaching memory limit

---

## Deployment Architecture

### Redis Cluster Deployment

**Kubernetes StatefulSet** (6 nodes):
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: production
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        command:
        - redis-server
        - "--cluster-enabled yes"
        - "--cluster-config-file /data/nodes.conf"
        - "--cluster-node-timeout 5000"
        - "--appendonly yes"
        - "--maxmemory 10gb"
        - "--maxmemory-policy allkeys-lru"
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            cpu: 500m
            memory: 10Gi
          limits:
            cpu: 2000m
            memory: 12Gi
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

**Redis Cluster Initialization** (Job):
```bash
redis-cli --cluster create \
  redis-0.redis-cluster:6379 \
  redis-1.redis-cluster:6379 \
  redis-2.redis-cluster:6379 \
  redis-3.redis-cluster:6379 \
  redis-4.redis-cluster:6379 \
  redis-5.redis-cluster:6379 \
  --cluster-replicas 1 \
  --cluster-yes
```

### Database Migrations

**Alembic Migration** (`alembic/versions/xxx_add_cache_quotas.py`):
```python
"""Add tenant cache quota and metrics tables

Revision ID: xxx
Revises: yyy
Create Date: 2025-11-07

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create tenant_cache_quotas table
    op.create_table(
        'tenant_cache_quotas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('quota_mb', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('current_usage_mb', sa.Numeric(10, 2), server_default='0'),
        sa.Column('quota_reset_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.Column('overage_allowed', sa.Boolean(), server_default='false'),
        sa.Column('overage_limit_mb', sa.Integer(), server_default='10'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id')
    )

    op.create_index('idx_tenant_cache_quotas_tenant_id', 'tenant_cache_quotas', ['tenant_id'])

    # Create tenant_cache_metrics table
    op.create_table(
        'tenant_cache_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_date', sa.DATE(), nullable=False),
        sa.Column('total_requests', sa.BigInteger(), server_default='0'),
        sa.Column('cache_hits', sa.BigInteger(), server_default='0'),
        sa.Column('cache_misses', sa.BigInteger(), server_default='0'),
        sa.Column('avg_response_time_ms', sa.Numeric(10, 2)),
        sa.Column('total_bytes_served', sa.BigInteger(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'metric_date')
    )

    op.create_index('idx_tenant_cache_metrics_tenant_date', 'tenant_cache_metrics', ['tenant_id', 'metric_date'])

    # Enable RLS
    op.execute('ALTER TABLE tenant_cache_quotas ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE tenant_cache_metrics ENABLE ROW LEVEL SECURITY')

    # Create RLS policies
    op.execute("""
        CREATE POLICY tenant_cache_quotas_isolation ON tenant_cache_quotas
        USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    """)

    op.execute("""
        CREATE POLICY tenant_cache_metrics_isolation ON tenant_cache_metrics
        USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    """)

def downgrade():
    op.drop_table('tenant_cache_metrics')
    op.drop_table('tenant_cache_quotas')
```

---

## Testing Strategy

### Unit Tests (90% coverage)

**TenantCacheManager Tests** (`tests/services/test_tenant_cache_manager.py`):
```python
import pytest
from unittest.mock import Mock, AsyncMock
from alpha_pulse.services.tenant_cache_manager import TenantCacheManager

@pytest.mark.asyncio
async def test_get_cache_hit():
    """Test cache retrieval with hit."""
    manager = TenantCacheManager(...)
    # Mock Redis GET returns value
    # Assert value returned, hit counter incremented

@pytest.mark.asyncio
async def test_set_under_quota():
    """Test cache set when under quota."""
    manager = TenantCacheManager(...)
    # Mock quota check (under limit)
    # Assert value stored, usage incremented

@pytest.mark.asyncio
async def test_set_over_quota_triggers_eviction():
    """Test eviction when quota exceeded."""
    manager = TenantCacheManager(...)
    # Mock quota check (over limit)
    # Mock eviction (frees space)
    # Assert value stored after eviction

@pytest.mark.asyncio
async def test_quota_exceeded_exception():
    """Test exception when quota cannot be freed."""
    manager = TenantCacheManager(...)
    # Mock quota check (over limit)
    # Mock eviction (fails to free enough space)
    # Assert QuotaExceededException raised
```

**SharedMarketDataCache Tests** (`tests/services/test_shared_market_data_cache.py`):
```python
@pytest.mark.asyncio
async def test_single_flight_pattern():
    """Test single-flight prevents duplicate API calls."""
    cache = SharedMarketDataCache(...)
    # Launch 100 concurrent get_price() calls
    # Mock exchange API call
    # Assert API called only once
    # Assert all 100 calls get same result

@pytest.mark.asyncio
async def test_cache_warming():
    """Test cache warming on startup."""
    cache = SharedMarketDataCache(...)
    # Call warm_top_symbols(['BTC/USDT', 'ETH/USDT'])
    # Assert prices cached
    # Assert OHLCV cached
```

### Integration Tests

**Redis Integration** (`tests/integration/test_redis_namespaces.py`):
```python
@pytest.mark.integration
async def test_namespace_isolation():
    """Test tenants cannot access each other's cache."""
    manager = TenantCacheManager(...)

    # Tenant A sets value
    await manager.set(tenant_a_id, 'signals', 'BTC', {'price': 67000})

    # Tenant B tries to get Tenant A's value
    value = await manager.get(tenant_b_id, 'signals', 'BTC')

    # Assert value is None (namespace isolation)
    assert value is None
```

### Performance Tests

**Latency Benchmark** (`tests/performance/test_cache_latency.py`):
```python
@pytest.mark.performance
async def test_cache_get_latency():
    """Test P99 latency for cache GET operations."""
    manager = TenantCacheManager(...)

    latencies = []
    for _ in range(1000):
        start = time.time()
        await manager.get(tenant_id, 'signals', 'BTC')
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    assert p99 < 2.0, f"P99 latency {p99}ms exceeds 2ms target"
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
