# EPIC-004: Caching Layer - Delivery Plan

**Epic**: EPIC-004 (#143)
**Sprint**: 9-10
**Story Points**: 29
**Date**: 2025-11-07
**Phase**: Build & Validate
**Author**: Tech Lead (via Claude Code)
**Status**: READY FOR EXECUTION

---

## Executive Summary

**Duration**: 2 sprints (Weeks 17-20)
**Team**: 1 backend engineer (parallel with EPIC-003)
**Velocity**: 14.5 SP per sprint
**Delivery**: Week of 2025-12-09

---

## Sprint 9 (Weeks 17-18): Core Tenant Layer + Quota

### Week 17 (Sprint 9, Part 1)

#### Story 4.1: Implement Tenant Namespace Layer (5 SP)

**Owner**: Backend Engineer B
**Dependencies**: RedisManager existing code

**Tasks**:
1. **RED** (Day 1):
   - Write `TenantCacheManager` interface tests (`tests/services/test_tenant_cache_manager.py`)
   - Test: `get()` with namespace isolation
   - Test: `set()` with namespace prefix
   - Test: Cross-tenant isolation (tenant A cannot see tenant B's keys)

2. **GREEN** (Days 2-3):
   - Implement `TenantCacheManager` class (`src/alpha_pulse/services/tenant_cache_manager.py`)
   - Methods: `get()`, `set()`, `delete()`, `get_usage()`, `get_metrics()`
   - Namespace builder: `_build_key()` → `tenant:{tenant_id}:{resource}:{key}`
   - Integrate with existing `RedisManager`

3. **REFACTOR** (Day 4):
   - Add serialization/deserialization (JSON)
   - Add cache hit/miss counters
   - Optimize: Batch Redis operations with pipeline

4. **QUALITY** (Day 5):
   - Integration test with real Redis instance
   - Test: Namespace isolation (scan keyspace, verify no cross-tenant keys)
   - Performance test: 1000 operations (<2ms P99)

**Acceptance Criteria**:
- [x] All cache operations use tenant namespace
- [x] Keys include tenant_id prefix automatically
- [x] No cross-tenant key collisions

**Commit Message**: `feat(cache): implement tenant namespace isolation layer (Story 4.1)`

---

#### Story 4.2: Create Quota Management Schema (3 SP)

**Owner**: Backend Engineer B
**Dependencies**: PostgreSQL multi-tenancy (EPIC-001)

**Tasks**:
1. **RED** (Day 1):
   - Write migration tests (`tests/migrations/test_cache_quota_schema.py`)
   - Test: Table creation with all columns
   - Test: Indexes created
   - Test: RLS policy enforced

2. **GREEN** (Day 2):
   - Create Alembic migration (`alembic/versions/xxx_add_cache_quotas.py`)
   - Create `tenant_cache_quotas` table:
     - id, tenant_id, quota_mb, current_usage_mb, quota_reset_at, overage_allowed, overage_limit_mb
   - Create `tenant_cache_metrics` table:
     - id, tenant_id, metric_date, total_requests, cache_hits, cache_misses, hit_rate (computed)

3. **REFACTOR** (Day 3):
   - Create SQLAlchemy ORM models (`src/alpha_pulse/models/cache_quota.py`)
   - Add helper methods: `get_quota_for_tenant()`, `update_usage()`
   - Add validation: quota_mb > 0, current_usage_mb >= 0

4. **QUALITY** (Day 3):
   - Test RLS policy (tenant A cannot see tenant B's quota)
   - Test computed column (hit_rate = hits / total_requests * 100)
   - Test unique constraint (one quota per tenant)

**Acceptance Criteria**:
- [x] Table tracks quota and usage per tenant
- [x] Automatic quota reset monthly
- [x] Overage handling (soft/hard limits)

**Commit Message**: `feat(db): add cache quota and metrics tables with RLS (Story 4.2)`

---

### Week 18 (Sprint 9, Part 2)

#### Story 4.3: Implement Quota Enforcement (5 SP)

**Owner**: Backend Engineer B
**Dependencies**: TenantCacheManager (Story 4.1), Quota schema (Story 4.2)

**Tasks**:
1. **RED** (Day 1):
   - Write quota enforcement tests (`tests/services/test_cache_quota_enforcement.py`)
   - Test: Under quota → set succeeds
   - Test: Over quota → eviction triggered
   - Test: Critical limit → QuotaExceededException raised

2. **GREEN** (Days 2-3):
   - Implement quota enforcement in `TenantCacheManager.set()`
   - Flow:
     1. Check current usage (Redis counter)
     2. Fetch quota from database (cached 5 min)
     3. If over quota → call `_evict_tenant_lru()`
     4. If still over quota → raise exception
   - Implement `_track_usage()` (INCRBY counter + ZADD to LRU sorted set)
   - Implement `_evict_tenant_lru()` (ZPOPMIN oldest keys until under threshold)

4. **REFACTOR** (Day 4):
   - Add middleware (`CacheQuotaMiddleware` in `src/alpha_pulse/api/middleware/cache_quota.py`)
   - Middleware checks quota on API request, sets header `X-Cache-Quota-Exceeded`
   - Optimize: Cache quota checks (5-min TTL)

5. **QUALITY** (Day 5):
   - Load test with quota violations (simulate 110% usage)
   - Verify eviction works correctly (oldest keys removed first)
   - Test latency: Quota check adds <5ms overhead

**Acceptance Criteria**:
- [x] Cache writes blocked when quota exceeded
- [x] X-Cache-Quota-Exceeded header set
- [x] <5ms latency overhead for quota checks

**Commit Message**: `feat(cache): implement quota enforcement with LRU eviction (Story 4.3)`

---

#### Story 4.4: Build Shared Market Data Cache (5 SP)

**Owner**: Backend Engineer B
**Dependencies**: RedisManager existing code, CCXTAdapter existing code

**Tasks**:
1. **RED** (Day 1):
   - Write shared cache tests (`tests/services/test_shared_market_data_cache.py`)
   - Test: `get_price()` cache hit/miss
   - Test: Single-flight pattern (100 concurrent requests → 1 API call)
   - Test: Cache warming on startup

2. **GREEN** (Days 2-3):
   - Implement `SharedMarketDataCache` class (`src/alpha_pulse/services/shared_market_data_cache.py`)
   - Methods: `get_price()`, `get_ohlcv()`, `warm_top_symbols()`
   - Implement single-flight pattern using asyncio.Lock
   - Namespace: `shared:market:{exchange}:{symbol}:*`

3. **REFACTOR** (Day 4):
   - Add TTL logic based on timeframe (1m → 60s, 1d → 3600s)
   - Add cache warming job (runs on startup, warms top 100 symbols)
   - Add Prometheus metrics (API calls saved counter)

4. **QUALITY** (Day 5):
   - Load test: 1000 tenants request BTC price simultaneously
   - Verify: Only 1 exchange API call made (99.9% cache hit)
   - Test: Cache hit rate >95% for top 100 symbols

**Acceptance Criteria**:
- [x] >95% cache hit rate for top 100 symbols
- [x] Single-flight prevents duplicate API calls
- [x] 60-sec TTL for prices, 5-min for OHLCV

**Commit Message**: `feat(cache): add shared market data cache with single-flight (Story 4.4)`

---

## Sprint 10 (Weeks 19-20): Metrics + Multi-Tier Integration

### Week 19 (Sprint 10, Part 1)

#### Story 4.5: Implement Metrics Collection (5 SP)

**Owner**: Backend Engineer B
**Dependencies**: Quota schema (Story 4.2), TenantCacheManager (Story 4.1)

**Tasks**:
1. **RED** (Day 1):
   - Write metrics collection tests (`tests/jobs/test_cache_metrics_flush.py`)
   - Test: Counters flushed to database every 5 minutes
   - Test: Hit rate calculated correctly
   - Test: Counters reset after flush

2. **GREEN** (Days 2-3):
   - Implement metrics collection in `TenantCacheManager`
   - Counters: `meta:tenant:{id}:hits`, `meta:tenant:{id}:misses`
   - Increment on `get()` (hit) or `get()` with miss
   - Create background job (`src/alpha_pulse/jobs/cache_metrics_flush.py`)
   - Job flushes counters to `tenant_cache_metrics` table every 5 minutes

3. **REFACTOR** (Day 4):
   - Add Prometheus metrics export
   - Metrics:
     - `alphapulse_cache_hit_rate{tenant_id}`
     - `alphapulse_cache_quota_usage_bytes{tenant_id}`
     - `alphapulse_cache_evictions_total{tenant_id}`

4. **QUALITY** (Day 5):
   - Verify accuracy: Hit rate ±1% of actual
   - Test: Metrics persist after service restart
   - Test: No memory leak from counter accumulation

**Acceptance Criteria**:
- [x] Metrics collected per tenant per day
- [x] Hit rate calculation accurate (±1%)
- [x] Dashboard shows trends over time

**Commit Message**: `feat(cache): add metrics collection and Prometheus export (Story 4.5)`

---

#### Story 4.6: Multi-Tier Cache Tenant Integration (6 SP)

**Owner**: Backend Engineer B
**Dependencies**: TenantCacheManager (Story 4.1), Existing CachingService

**Tasks**:
1. **RED** (Day 1):
   - Write L1 cache tenant isolation tests (`tests/services/test_multi_tier_cache_tenant.py`)
   - Test: L1 cache isolated per tenant (tenant A cannot access tenant B's L1)
   - Test: L1 cache size limit (max 10MB per tenant)
   - Test: L1 → L2 → L3 fallback with tenant context

2. **GREEN** (Days 2-4):
   - Update `CachingService` to accept `tenant_id` parameter (`src/alpha_pulse/services/caching_service.py`)
   - Change: `get_or_fetch(key, fetcher)` → `get_or_fetch(tenant_id, key, fetcher)`
   - L1 Cache: Per-tenant in-memory LRU (max 10MB each)
   - L2 Cache: Use `TenantCacheManager` (tenant namespace)
   - L3 Cache: Database queries filtered by `tenant_id`

3. **REFACTOR** (Day 5):
   - Add cache statistics per tier (L1 hit rate, L2 hit rate)
   - Optimize: Batch L2 operations with pipeline
   - Add circuit breaker (if Redis down, skip L2, go to L3)

4. **QUALITY** (Day 6):
   - Load test with 1000 tenants (10K req/sec)
   - Verify: No memory exhaustion (L1 size limits enforced)
   - Test: L1 eviction works correctly (LRU per tenant)

**Acceptance Criteria**:
- [x] L1 cache isolated per tenant (max 10MB each)
- [x] L2 (Redis) uses tenant namespace
- [x] L3 (database) queries filtered by tenant_id

**Commit Message**: `feat(cache): integrate tenant-aware caching into multi-tier system (Story 4.6)`

---

## Integration & Testing (Final Days of Sprint 10)

### End-to-End Testing (Day 1-2)

**Scenarios**:
1. **Full Cache Lifecycle with Quota**:
   - Tenant fills 80% of quota → Warning logged
   - Tenant hits 100% quota → Eviction triggered
   - Oldest keys evicted → Usage drops to 90%
   - New writes succeed

2. **Shared Market Data Efficiency**:
   - 100 tenants request BTC price simultaneously
   - Cache MISS on first request → Fetch from exchange
   - Next 99 requests → Cache HIT (no exchange calls)
   - Result: 99% API call reduction

3. **Multi-Tier Cache with Tenant Isolation**:
   - Tenant A writes to L1 (in-memory) → L2 (Redis) → L3 (database)
   - Tenant B reads → Misses Tenant A's L1, gets own data from L2
   - Verify: No cross-tenant data leakage

**Test Locations**:
- `tests/e2e/test_cache_lifecycle_with_quota.py`
- `tests/e2e/test_shared_market_data_efficiency.py`
- `tests/e2e/test_multi_tier_tenant_isolation.py`

### Performance Testing (Day 3)

**Load Tests**:
1. **Cache Operations**: 10,000 req/sec for 5 minutes
   - Target: P99 <2ms (cache GET)
   - Target: P99 <5ms (cache SET with quota check)

2. **Quota Enforcement**: 1000 tenants hitting quota simultaneously
   - Target: Eviction completes <10 seconds
   - Target: No Redis timeouts

3. **Shared Cache**: 1000 tenants requesting same symbol
   - Target: >95% cache hit rate
   - Target: <10 exchange API calls per minute (vs 60,000 without cache)

**Tools**: Locust, Redis benchmarking tools

### Security Audit (Day 4)

**Checklist**:
- [ ] Namespace isolation verified (penetration test)
- [ ] RLS policies prevent cross-tenant access (database)
- [ ] L1 cache cannot be accessed across tenants
- [ ] Redis ACLs configured (optional, future enhancement)
- [ ] No sensitive data in Redis keys (only in values)

**Tools**:
- Custom tenant isolation test script
- Redis keyspace scanner

### Documentation (Day 5)

**User Documentation**:
- `docs/user-guides/cache-quota-management.md`
- `docs/user-guides/optimizing-cache-usage.md`

**Operational Runbooks**:
- `docs/runbooks/redis-cluster-operations.md`
- `docs/runbooks/cache-quota-exceeded-troubleshooting.md`

**Developer Documentation**:
- `docs/dev/tenant-cache-manager-api.md`
- `docs/dev/shared-market-data-cache-api.md`

---

## Deployment Plan

### Pre-Deployment Checklist

**Week before Sprint 9 Kickoff**:
- [ ] DevOps: Redis Cluster deployed (6 nodes: 3 masters + 3 replicas)
- [ ] DevOps: Persistent volumes provisioned (6 × 50GB for Redis)
- [ ] DevOps: Prometheus exporter configured for Redis
- [ ] DevOps: Grafana dashboards created (cache hit rate, quota usage)

### Sprint 9 Deployment (Week 18, Friday)

**Steps**:
1. Deploy database migration (Story 4.2) → Production
2. Deploy `TenantCacheManager` (Story 4.1) → Production (behind feature flag)
3. Deploy quota enforcement (Story 4.3) → Production (soft limits only)
4. Deploy shared market data cache (Story 4.4) → Production
5. Monitor for 24 hours (Redis memory usage, cache hit rate)

### Sprint 10 Deployment (Week 20, Friday)

**Steps**:
1. Deploy metrics collection job (Story 4.5) → Production
2. Deploy multi-tier cache integration (Story 4.6) → Production
3. Enable hard quota limits (after soft limit monitoring)
4. Enable feature flag for all tenants
5. Announce cache quota tiers via email

---

## Rollback Plan

### Rollback Triggers

- Redis Cluster downtime >5 minutes
- Cache hit rate drops <50% (indicates namespace issue)
- Memory leak detected (L1 cache grows unbounded)
- Quota enforcement causes >10% request failures

### Rollback Steps

1. **Disable Feature Flag**: Revert to single-tenant caching
2. **Stop Metrics Job**: Prevent database writes
3. **Drain Redis Traffic**: Bypass L2 cache, use L3 (database) only
4. **Database Rollback**: Alembic downgrade (remove quota tables)
5. **Kubernetes Rollback**: `kubectl rollout undo deployment alphapulse-api`

**Recovery Time Objective (RTO)**: <10 minutes

---

## Risk Management

### High-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Redis Cluster complexity | 30% | HIGH | DevOps pairing, use managed service (AWS ElastiCache) |
| L1 cache memory exhaustion | 25% | MEDIUM | Strict size limits per tenant (max 10MB), LRU eviction |
| Quota enforcement overhead | 20% | MEDIUM | Cache quota checks (5-min TTL), async usage updates |

### Contingency Plans

**Plan A** (Redis Cluster too complex):
- Use single Redis instance with persistence
- Accept lower throughput (10K req/sec vs 100K)
- Scale vertically (64GB RAM instance)

**Plan B** (L1 cache memory issues):
- Disable L1 cache entirely
- Rely on L2 (Redis) only
- Latency increases 1-2ms

---

## Success Metrics

### Sprint 9 Goals

- ✅ Tenant namespace layer operational
- ✅ Quota schema migrated (RLS policies active)
- ✅ Quota enforcement functional (soft limits)
- ✅ Shared market data cache deployed

### Sprint 10 Goals

- ✅ Metrics collection running (5-min flush job)
- ✅ Multi-tier cache tenant-aware (L1/L2/L3)
- ✅ All stories 100% test coverage

### Overall EPIC Success

- ✅ 100% tenant namespace isolation (0% cross-tenant access)
- ✅ >95% cache hit rate for shared market data (top 100 symbols)
- ✅ <2ms P99 latency for cache operations (no regression)
- ✅ <1% quota violations (soft limits prevent hard failures)

---

## Team Communication

### Daily Standups

**Time**: 9:00 AM daily (15 minutes)

**Format**:
- Yesterday: What I completed
- Today: What I'm working on
- Blockers: Any issues

### Sprint Review (End of Each Sprint)

**Attendees**: Product Owner, Tech Lead, Backend Engineer B, DevOps

**Agenda**:
- Demo completed stories
- Review metrics (velocity, quality)
- Discuss feedback

### Retrospective (End of Sprint 10)

**Questions**:
- What went well?
- What could be improved?
- Action items for next sprint

---

## Dependencies

### Upstream (Completed)

- ✅ EPIC-001: Database Multi-Tenancy (tenant_id available)
- ✅ EPIC-002: Application Multi-Tenancy (JWT + tenant context)

### External (To Be Confirmed)

- ⏳ DevOps: Redis Cluster deployment (6 nodes, confirm by Week 16)
- ⏳ DevOps: Prometheus + Grafana dashboards (confirm by Week 17)

### Parallel Execution

- EPIC-003 (Credential Management): Independent, Backend Engineer A

---

## Sign-Off

**Tech Lead**: ✅ Approved (2025-11-07)
**Product Owner**: ⏳ Pending review
**Backend Engineer B**: ⏳ Acknowledged
**DevOps Lead**: ⏳ Acknowledged (Redis Cluster deployment)

**Next Milestone**: Sprint 9 Kickoff (Week 17, Monday)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
