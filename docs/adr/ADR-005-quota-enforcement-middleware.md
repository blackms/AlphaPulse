# ADR-005: Quota Enforcement Middleware for Multi-Tenant Caching

**Status**: Proposed
**Date**: 2025-11-07
**Deciders**: Tech Lead, Engineering Team
**Epic**: EPIC-004 (Caching Layer)
**Story**: 4.3 - Implement Quota Enforcement Middleware (5 SP)

## Context

We have implemented a multi-tenant cache namespace layer (Story 4.1) and database schema for quota management (Story 4.2). We now need to enforce per-tenant cache quotas at the API request level to:

1. Prevent tenants from exceeding their allocated cache quotas
2. Provide real-time quota violation feedback
3. Enable graceful degradation (allow overages with warnings)
4. Track usage metrics for billing and capacity planning

### Requirements

**Functional:**
- Enforce per-tenant cache quotas before cache write operations
- Support overage allowance (soft limits with warnings)
- Reject writes when hard limit exceeded
- Track quota usage in real-time
- Provide quota status in API responses (headers)

**Non-Functional:**
- Latency impact: < 10ms per request (p99)
- Availability: Must not fail open (deny on error)
- Scalability: Support 1000+ concurrent requests per tenant
- Observability: Metrics on quota checks, rejections, overages

### Current Architecture

```
┌─────────────────┐
│   API Request   │
│  (FastAPI)      │
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│ TenantCacheManager      │ ← Story 4.1
│ (Namespace isolation)   │
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│ Redis                   │
│ (tenant:{id}:*)         │
└─────────────────────────┘

┌─────────────────────────┐
│ PostgreSQL              │
│ - tenant_cache_quotas   │ ← Story 4.2
│ - tenant_cache_metrics  │
└─────────────────────────┘
```

## Decision

We will implement **FastAPI middleware** that:

1. **Intercepts cache write requests** before they reach `TenantCacheManager`
2. **Checks quota** using a two-tier approach:
   - **Tier 1 (Fast Path)**: Redis cached quota (TTL 5 minutes)
   - **Tier 2 (Slow Path)**: PostgreSQL authoritative quota (cache miss only)
3. **Enforces limits** with three-level response:
   - **Allow**: Usage < quota
   - **Warn**: Usage > quota but < (quota + overage_limit)
   - **Reject**: Usage > (quota + overage_limit)
4. **Updates usage atomically** using Redis INCR operations
5. **Returns quota status** in custom response headers

### Architecture

```
┌─────────────────┐
│   API Request   │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────┐
│ QuotaEnforcementMiddleware      │ ← NEW (This ADR)
│                                  │
│ 1. Extract tenant_id            │
│ 2. Check quota (Redis cache)    │
│ 3. Enforce limit decision        │
│ 4. Update usage (atomic)         │
│ 5. Add quota headers             │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────────────┐
│ TenantCacheManager      │
│ (Only if quota allows)  │
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│ Redis Cache             │
└─────────────────────────┘

Redis Keys:
- quota:cache:{tenant_id}:quota_mb         (TTL 5m)
- quota:cache:{tenant_id}:current_usage_mb (TTL 5m)
- quota:cache:{tenant_id}:overage_allowed  (TTL 5m)
- quota:cache:{tenant_id}:overage_limit_mb (TTL 5m)
```

### Implementation Details

**Middleware Class**: `QuotaEnforcementMiddleware`
- Location: `src/alpha_pulse/middleware/quota_enforcement.py`
- Pattern: ASGI middleware (async)
- Scope: Cache write operations only (not reads)

**Quota Check Algorithm:**
```python
async def check_quota(tenant_id: UUID, write_size_mb: float) -> QuotaDecision:
    # 1. Get quota from Redis (or DB if cache miss)
    quota = await get_quota_cached(tenant_id)

    # 2. Atomic usage update
    new_usage = await redis.incrbyfloat(
        f"quota:cache:{tenant_id}:current_usage_mb",
        write_size_mb
    )

    # 3. Decision logic
    if new_usage <= quota.quota_mb:
        return QuotaDecision.ALLOW
    elif quota.overage_allowed and new_usage <= (quota.quota_mb + quota.overage_limit_mb):
        return QuotaDecision.WARN
    else:
        # Rollback usage increment
        await redis.decrbyfloat(
            f"quota:cache:{tenant_id}:current_usage_mb",
            write_size_mb
        )
        return QuotaDecision.REJECT
```

**Response Headers:**
```
X-Cache-Quota-Limit: 500         # MB
X-Cache-Quota-Used: 387.5        # MB
X-Cache-Quota-Remaining: 112.5   # MB
X-Cache-Quota-Percent: 77.5      # %
X-Cache-Quota-Status: ok|warning|exceeded
```

**HTTP Status Codes:**
- `200 OK` + `X-Cache-Quota-Status: ok` → Normal operation
- `200 OK` + `X-Cache-Quota-Status: warning` → Over quota but within overage
- `429 Too Many Requests` + `Retry-After: 3600` → Hard limit exceeded

## Alternatives Considered

### Alternative 1: Application-Level Quota Check (Not Middleware)

**Approach**: Check quota inside `TenantCacheManager.set()` method.

**Pros:**
- Simpler integration (single location)
- No middleware configuration needed

**Cons:**
- ❌ Violates separation of concerns (cache manager shouldn't handle quotas)
- ❌ Can't reject request early (data already processed)
- ❌ Harder to test independently
- ❌ Can't easily disable/enable quota enforcement

**Decision**: Rejected - Middleware provides better separation and control

### Alternative 2: Database-Only Quota Check (No Redis Cache)

**Approach**: Query PostgreSQL on every request for authoritative quota.

**Pros:**
- Always accurate (no cache staleness)
- Simpler implementation (no cache invalidation)

**Cons:**
- ❌ Adds 5-15ms latency per request (database query)
- ❌ Database becomes bottleneck at scale
- ❌ Violates <10ms performance SLO

**Decision**: Rejected - Performance impact too high

### Alternative 3: Redis-Only Quota Management (No Database)

**Approach**: Store quota configuration in Redis only, no PostgreSQL.

**Pros:**
- Lowest latency (<1ms)
- Simplest implementation

**Cons:**
- ❌ No audit trail for quota changes
- ❌ Quota data lost on Redis failure
- ❌ Can't leverage PostgreSQL RLS for tenant isolation
- ❌ No persistent metrics history

**Decision**: Rejected - Lacks durability and auditability

### Alternative 4: Event-Driven Quota Sync (Pub/Sub)

**Approach**: Use Redis Pub/Sub to propagate quota updates from PostgreSQL.

**Pros:**
- Near-instant cache invalidation
- Lower cache staleness

**Cons:**
- ❌ Adds complexity (Pub/Sub infrastructure)
- ❌ Requires message delivery guarantees
- ❌ Overkill for 5-minute TTL use case

**Decision**: Rejected - Over-engineered for requirement

## Consequences

### Positive

✅ **Fast quota checks**: Redis cache keeps latency <5ms (p99)
✅ **Graceful degradation**: Overage allowance prevents hard failures
✅ **Clear feedback**: Response headers inform clients of quota status
✅ **Atomic operations**: Redis INCR prevents race conditions
✅ **Testable**: Middleware can be tested independently
✅ **Feature flag ready**: Easy to enable/disable per tenant

### Negative

⚠️ **Cache staleness**: Quota changes take up to 5 minutes to propagate
⚠️ **Redis dependency**: Quota enforcement requires Redis availability
⚠️ **Complexity**: Two-tier caching adds operational overhead

### Neutral

🔄 **Observability**: Need to add metrics for quota checks, rejections
🔄 **Documentation**: API clients need to handle 429 responses
🔄 **Testing**: Need load tests for concurrent quota enforcement

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cache stampede on quota reload | Medium | Use single-flight pattern (only one goroutine loads) |
| Redis failure breaks quota enforcement | High | Fallback to PostgreSQL (degraded performance) |
| Race condition on quota boundary | Low | Redis atomic INCR prevents double-counting |
| Middleware misconfiguration blocks all requests | Medium | Feature flag + comprehensive testing + rollback plan |

## Validation Criteria

✅ **Performance**: Latency impact < 10ms (p99) under 1000 req/s load
✅ **Correctness**: No tenant can exceed hard limit under concurrent load
✅ **Reliability**: 99.9% quota check success rate
✅ **Observability**: Metrics show quota checks, rejections, cache hit rate

## Implementation Plan

**Story 4.3 Phases (TDD):**
1. **RED**: Write tests for middleware (quota scenarios)
2. **GREEN**: Implement middleware class
3. **REFACTOR**: Optimize cache strategy, add metrics
4. **QUALITY**: Load testing, concurrent request validation

**Estimated Effort**: 5 SP (1.5 days)

**Dependencies:**
- ✅ Story 4.1: TenantCacheManager
- ✅ Story 4.2: Database schema
- ⏳ Story 4.3: This implementation

**Rollout Strategy:**
- Phase 1: Deploy with feature flag disabled (shadow mode)
- Phase 2: Enable for internal tenant (monitoring)
- Phase 3: Gradual rollout to 10% → 50% → 100%

## References

- Story 4.1: Tenant Namespace Layer (PR #211)
- Story 4.2: Quota Management Schema (PR #212)
- FastAPI Middleware Docs: https://fastapi.tiangolo.com/tutorial/middleware/
- Redis INCR Atomicity: https://redis.io/commands/incr/

## Review Status

- [ ] Architecture Review (Tech Lead + Senior Engineers)
- [ ] Security Review (if handling sensitive data)
- [ ] Performance Review (baseline established)

---

**Next Steps:**
1. Architecture review meeting
2. Create HLD for Story 4.3
3. Begin RED phase (write failing tests)
4. Implement middleware (GREEN phase)
