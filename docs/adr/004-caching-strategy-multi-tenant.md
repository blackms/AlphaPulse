# ADR 004: Caching Strategy for Multi-Tenant Workloads

Date: 2025-10-20
Status: Proposed

## Context

AlphaPulse currently uses Redis for caching market data, agent signals, and session state (see `src/alpha_pulse/services/caching_service.py`). The existing caching service is optimized for single-tenant use. As we transition to multi-tenant SaaS, we need a caching strategy that:

- **Isolates tenant data**: No cache key collisions between tenants
- **Prevents noisy neighbors**: One tenant's cache churn doesn't evict another tenant's data
- **Enforces quotas**: Cache usage limits per tenant tier (Starter: 100MB, Pro: 500MB, Enterprise: 2GB)
- **Optimizes shared data**: Market data (OHLCV) is identical across tenants → cache once, share safely
- **Maintains performance**: Cache hit rate >80%, P99 latency <5ms
- **Scales efficiently**: Support 1000+ tenants on single Redis cluster
- **Provides observability**: Per-tenant cache metrics (hit rate, evictions, usage)

### Current Caching Architecture

From `src/alpha_pulse/services/caching_service.py`:
- Single Redis instance (default configuration)
- No namespace isolation (keys like `market_data:BTC_USDT`)
- No quota enforcement
- Multi-tier caching: L1 (in-memory LRU) + L2 (Redis)
- TTL-based expiration (varies by data type)

### Cached Data Categories

1. **Market Data** (shared across tenants):
   - OHLCV (candlestick) data: 1-minute to 1-day intervals
   - Order book snapshots
   - Ticker data (current price, volume)
   - Size: ~10MB per symbol × 100 symbols = 1GB
   - TTL: 1 minute to 1 hour (depends on timeframe)

2. **Agent Signals** (tenant-specific):
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Fundamental analysis results
   - Sentiment scores
   - Size: ~100KB per symbol per agent × 6 agents × 10 symbols = 6MB per tenant
   - TTL: 5 minutes

3. **Portfolio State** (tenant-specific):
   - Current positions
   - Portfolio value history
   - Risk metrics (VaR, CVaR, Sharpe ratio)
   - Size: ~500KB per tenant
   - TTL: 1 minute

4. **Session State** (tenant-specific):
   - WebSocket connection state
   - User preferences
   - Size: ~10KB per tenant
   - TTL: 1 hour

5. **Exchange Credentials** (tenant-specific, high security):
   - Cached from Vault (per ADR-003)
   - Size: ~1KB per tenant
   - TTL: 5 minutes

### Performance Requirements

- **Cache hit rate**: >80% for market data, >60% for agent signals
- **Latency**: P99 <5ms for reads, P99 <10ms for writes
- **Availability**: 99.9% uptime (max 43 minutes downtime/month)
- **Scalability**: Support 1000 tenants with 10 concurrent requests each = 10k req/sec

## Decision

We will implement a **Hybrid Namespace-Based + Shared Cache Pool** strategy with Redis Cluster:

### Architecture: Redis Cluster with Tenant Namespaces + Shared Pool

**Redis Deployment:**
- Redis Cluster (6 nodes: 3 masters + 3 replicas for HA)
- KeyDB (Redis-compatible) for better multi-core performance
- Redis modules: RedisJSON (for complex objects), RedisTimeSeries (for OHLCV data)

**Key Namespace Design:**

```
# Tenant-specific data (isolated)
tenant:{tenant_id}:signals:{agent}:{symbol}
tenant:{tenant_id}:portfolio:positions
tenant:{tenant_id}:portfolio:metrics
tenant:{tenant_id}:session:{session_id}
tenant:{tenant_id}:credentials:{exchange}

# Shared data (multi-tenant)
shared:market:{exchange}:{symbol}:{interval}:ohlcv
shared:market:{exchange}:{symbol}:ticker
shared:market:{exchange}:{symbol}:orderbook

# Tenant metadata (for quota enforcement)
meta:tenant:{tenant_id}:quota
meta:tenant:{tenant_id}:usage
```

**Benefits of Namespace Design:**
- ✅ Clear tenant isolation (impossible to access other tenant's data)
- ✅ Shared market data reduces memory by 90% (1GB vs 100GB for 100 tenants)
- ✅ Easy to implement quota enforcement (scan by namespace prefix)
- ✅ Simple cache invalidation (delete all keys matching pattern)

### Quota Enforcement

**Per-Tier Limits:**

| Tier | Cache Quota | Eviction Policy | Monitoring |
|------|-------------|-----------------|------------|
| Starter | 100MB | LRU per tenant | Alert at 80% |
| Pro | 500MB | LRU per tenant | Alert at 80% |
| Enterprise | 2GB | LRU per tenant | Alert at 90% |

**Implementation:**

1. **Tracking**: Maintain rolling counters instead of scanning the keyspace.
   - On each write we estimate the payload size (`len(serialized_value)`), increment `meta:tenant:{id}:usage_bytes`, and add the namespaced key to a sorted set `meta:tenant:{id}:lru` with `score = current_timestamp`.
   - On delete/expiry we decrement the counter and remove the key from the sorted set (handled via keyspace notifications + background worker to keep counters accurate).
   - A low-frequency audit job (hourly) may run a `SCAN` with pattern `tenant:{id}:*` to reconcile counters; this job batches results (`COUNT=1000`) to avoid blocking.

   ```python
   async def track_usage(tenant_id: UUID, cache_key: str, payload: bytes) -> None:
       pipeline = redis.pipeline()
       pipeline.incrby(f"meta:tenant:{tenant_id}:usage_bytes", len(payload))
       pipeline.zadd(
           f"meta:tenant:{tenant_id}:lru",
           {f"tenant:{tenant_id}:{cache_key}": time.time()}
       )
       await pipeline.execute()
   ```

2. **Enforcement**: Pre-write check uses the cached counter.
   ```python
   async def set_cached(tenant_id: UUID, key: str, value: Any, ttl: int):
       quota = await get_tenant_quota(tenant_id)
       usage = int(await redis.get(f"meta:tenant:{tenant_id}:usage_bytes") or 0)

       if usage >= quota:
           await evict_tenant_keys(tenant_id, target_size=int(quota * 0.9))

       payload = serialize(value)
       await redis.set(namespaced_key(tenant_id, key), payload, ex=ttl)
       await track_usage(tenant_id, key, payload)
   ```

3. **Eviction**: Custom LRU per tenant (not global LRU)
   - Pop the oldest entries from `meta:tenant:{id}:lru` until usage drops below the threshold.
   - For each popped key, delete the value, subtract its recorded size from `usage_bytes`, and log an eviction event.
   - Because the sorted set stores namespace-qualified keys, eviction runs entirely on metadata without issuing `KEYS`/`SORT` commands.

### Shared Market Data Strategy

**Problem**: 100 tenants requesting BTC/USDT 1-minute OHLCV → 100 identical copies (100MB total)

**Solution**: Cache once, reference many

```python
# Write once (first tenant request or data pipeline)
await redis.set(
    "shared:market:binance:BTC_USDT:1m:ohlcv",
    json.dumps(ohlcv_data),
    ex=60  # 1 minute TTL
)

# All tenants read same key (no duplication)
cached = await redis.get("shared:market:binance:BTC_USDT:1m:ohlcv")
```

**Access Control**: Shared cache is read-only for tenants (only data pipeline can write)

**Benefits**:
- 99% memory reduction for market data (1GB vs 100GB)
- Lower cache churn (single TTL expiration vs 100×)
- Consistent data across tenants (no stale cache issues)

### Multi-Tier Caching (L1 + L2)

Preserve existing multi-tier architecture from `caching_service.py`:

**L1 Cache (In-Memory):**
- Python `cachetools.TTLCache` (per-process)
- Size: 50MB per application instance
- TTL: 30 seconds
- Use case: Hot data (current prices, active positions)

**L2 Cache (Redis):**
- Redis Cluster (shared across all app instances)
- Size: Unlimited (enforced by tenant quotas)
- TTL: Varies by data type (1 minute to 1 hour)
- Use case: Warm data (historical OHLCV, computed signals)

**Cache Read Flow:**
```python
async def get_cached(tenant_id: UUID, key: str) -> Optional[Any]:
    # L1: Check in-memory cache
    l1_key = f"{tenant_id}:{key}"
    if l1_key in l1_cache:
        return l1_cache[l1_key]

    # L2: Check Redis
    l2_key = f"tenant:{tenant_id}:{key}"
    value = await redis.get(l2_key)

    if value:
        # Populate L1 for future requests
        l1_cache[l1_key] = value
        return value

    return None  # Cache miss
```

### Cache Invalidation Strategies

**Time-Based (TTL):**
- Market data: 1 minute (real-time updates)
- Agent signals: 5 minutes (recomputed every 5 min)
- Portfolio state: 1 minute (after trades)
- Session state: 1 hour (user inactivity)

**Event-Based:**
- On trade execution → invalidate `tenant:{id}:portfolio:*`
- On credential update → invalidate `tenant:{id}:credentials:*`
- On tenant tier upgrade → update `meta:tenant:{id}:quota`

**Manual:**
- Admin API: `DELETE /admin/tenants/{id}/cache` (flush all tenant cache)
- Emergency cache clear: `redis-cli --scan --pattern "tenant:{id}:*" | xargs redis-cli DEL`

### Observability & Monitoring

**Per-Tenant Metrics** (Prometheus + Grafana):
- Cache hit rate: `cache_hits / (cache_hits + cache_misses)` (target: >80%)
- Cache usage: `current_bytes / quota_bytes` (alert at 80%)
- Eviction rate: `evictions_per_minute` (alert if >100)
- Latency: P50, P95, P99 for cache reads/writes

**Redis Cluster Metrics**:
- Memory usage per node
- Commands per second
- Keyspace hits/misses
- Replication lag

**Alerting**:
- Tenant cache usage >80% → Warn tenant via email
- Cache hit rate <60% for 10 minutes → Page on-call
- Redis cluster memory >90% → Auto-scale (add nodes)
- Replication lag >5 seconds → Page on-call

### High Availability & Disaster Recovery

**HA Setup:**
- Redis Cluster: 3 masters + 3 replicas (6 nodes total)
- Automatic failover (Redis Sentinel or Cluster mode)
- Cross-AZ deployment (replicas in different availability zones)

**Backup Strategy:**
- Redis RDB snapshots every 6 hours
- Redis AOF (Append-Only File) for point-in-time recovery
- Store backups in S3 with 30-day retention

**Degradation Strategy:**
- If Redis unavailable → Fall back to database (slower but functional)
- L1 cache keeps hot data available during brief Redis outages
- Circuit breaker: Stop caching if Redis latency >100ms (avoid cascading failures)

## Consequences

### Positive

✅ **Isolation**: Namespace design prevents cross-tenant data access
✅ **Efficiency**: Shared market data reduces memory by 90%
✅ **Performance**: Multi-tier caching maintains P99 <5ms latency
✅ **Fairness**: Per-tenant quotas prevent noisy neighbor issues
✅ **Scalability**: Redis Cluster supports 10k+ req/sec (sufficient for 1000 tenants)
✅ **Observability**: Per-tenant metrics enable proactive optimization
✅ **Cost**: Shared cache reduces Redis memory requirements by 80% (saves $400/month)

### Negative

⚠️ **Complexity**: Namespace management adds code complexity
⚠️ **Quota Overhead**: Scanning keys to compute usage adds CPU cost (mitigated by background job)
⚠️ **Redis Cluster**: Operational complexity (6 nodes vs 1, cross-slot operations limited)
⚠️ **Migration**: Existing cache keys must be migrated to new namespace format
⚠️ **Monitoring**: Per-tenant metrics increase Prometheus cardinality (100+ tenants = 1000+ metrics)

### Mitigation Strategies

1. **Namespace Abstraction**: Encapsulate namespace logic in `CachingService` (transparent to application code)
2. **Efficient Scanning**: Use Redis SCAN with COUNT=1000 (non-blocking) for quota calculations
3. **Lazy Migration**: Migrate cache keys on-demand (old keys expire naturally via TTL)
4. **Metric Aggregation**: Aggregate per-tenant metrics by tier for dashboards (reduce cardinality)
5. **Cost Optimization**: Use KeyDB (open-source, multi-threaded) instead of Redis Enterprise (saves $500/month)

## Alternatives Considered

### Option A: Separate Redis Instance per Tenant

**Pros:**
- Perfect isolation (no namespace collisions)
- Independent scaling per tenant
- Easy to implement quotas (just limit instance memory)

**Cons:**
- ❌ **Cost**: 100 tenants × $15/month Redis = $1,500/month (vs $150/month for shared cluster)
- ❌ **Operational overhead**: Managing 100+ Redis instances
- ❌ **No shared data**: Market data duplicated 100× (wastes memory)
- ❌ **Connection overhead**: 100 Redis connections per app instance (vs 1)

**Why Rejected:** Cost prohibitive and operationally complex. Shared market data optimization impossible.

### Option B: Single Redis with Global LRU (No Tenant Isolation)

**Pros:**
- Simplest implementation (no namespace logic)
- Best performance (no key prefix overhead)

**Cons:**
- ❌ **No isolation**: One tenant can evict another tenant's cache entries
- ❌ **Unfair**: High-traffic tenant starves low-traffic tenants
- ❌ **Security risk**: Key collisions could leak data

**Why Rejected:** Unacceptable for multi-tenant SaaS. Noisy neighbor problem guaranteed.

### Option C: Application-Level Cache Only (No Redis)

**Pros:**
- No Redis infrastructure to manage
- Lowest latency (no network round-trip)

**Cons:**
- ❌ **No cache sharing**: Each app instance has separate cache (low hit rate)
- ❌ **Memory waste**: 10 app instances × 1GB cache = 10GB (vs 2GB shared Redis)
- ❌ **Stale data**: Cache invalidation across instances is complex

**Why Rejected:** Poor cache hit rate and memory inefficiency. Required for stateless horizontal scaling.

### Option D: Memcached Instead of Redis

**Pros:**
- Simpler than Redis (no persistence, replication)
- Excellent multi-threaded performance

**Cons:**
- ❌ **No persistence**: All cache lost on restart (poor availability)
- ❌ **Limited data structures**: No JSON, TimeSeries, or sorted sets
- ❌ **No pub/sub**: Can't implement cache invalidation events

**Why Rejected:** Redis provides richer features needed for trading system (time-series, pub/sub). Persistence important for session state.

## Implementation Plan

### Phase 1: Redis Cluster Deployment (Sprint 1)

1. Deploy KeyDB cluster (3 masters + 3 replicas)
2. Configure cross-AZ replication
3. Enable RDB snapshots + AOF
4. Set up monitoring (Prometheus + Grafana)

### Phase 2: Namespace Implementation (Sprint 2)

1. Update `CachingService` to add tenant namespace to all keys
2. Implement shared market data cache (read-only for tenants)
3. Add quota enforcement logic
4. Unit tests for namespace isolation

### Phase 3: Quota Management (Sprint 2)

1. Create background job for usage tracking
2. Implement tenant-aware LRU eviction
3. Add admin API for quota updates
4. Integration tests for quota enforcement

### Phase 4: Migration (Sprint 3)

1. Deploy new caching service alongside old (dual-write)
2. Migrate existing keys to new namespace format
3. Switch read traffic to new service
4. Decommission old Redis instance

### Phase 5: Observability (Sprint 3)

1. Add per-tenant cache metrics to Prometheus
2. Create Grafana dashboards (cache hit rate, usage, evictions)
3. Configure alerts (quota exceeded, low hit rate)
4. Document operational runbook

### Phase 6: Optimization (Sprint 4)

1. Analyze cache hit rates per data type
2. Tune TTL values based on access patterns
3. Implement cache warming for cold-start scenarios
4. Load testing with 100 simulated tenants

## Links

- Issue: [To be created - Multi-Tenant Caching Epic]
- Related: ADR-001 (Data Isolation), ADR-002 (Tenant Provisioning), ADR-003 (Credential Management)
- Code: `src/alpha_pulse/services/caching_service.py`
- Reference: [Redis Best Practices for Multi-Tenancy](https://redis.io/docs/manual/patterns/multi-tenancy/)
- Reference: [KeyDB Documentation](https://docs.keydb.dev/)
