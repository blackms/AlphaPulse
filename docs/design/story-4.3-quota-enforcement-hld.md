# Story 4.3: Quota Enforcement Middleware - High-Level Design (HLD)

**Epic**: EPIC-004 (Caching Layer)
**Story**: 4.3 - Quota Enforcement Middleware (5 SP)
**Status**: Design
**Date**: 2025-11-07
**Author**: Tech Lead
**Reviewers**: Senior Engineers, Platform Team

## Executive Summary

This HLD defines the architecture for a FastAPI middleware component that enforces per-tenant cache quotas in real-time with <10ms latency impact. The design uses a two-tier caching strategy (Redis + PostgreSQL) to achieve high performance while maintaining data consistency.

**Key Decisions:**
- ✅ FastAPI ASGI middleware pattern for request interception
- ✅ Redis as primary quota cache (5-min TTL) with PostgreSQL fallback
- ✅ Atomic usage tracking via Redis INCR operations
- ✅ Three-level enforcement: ALLOW / WARN / REJECT
- ✅ Response headers for quota visibility

**References:**
- ADR-005: Quota Enforcement Middleware (architectural decision)
- Story 4.1: TenantCacheManager (PR #211)
- Story 4.2: Database Schema (PR #212)

---

## 1. Context and Goals

### 1.1 Problem Statement

**Current State**: Tenants can write unlimited data to cache, causing resource exhaustion and unpredictable costs.

**Desired State**: Every cache write is quota-checked, enforced, and metered in <10ms with zero false denials.

**Gap**: No enforcement mechanism at API layer.

### 1.2 Design Goals

**Primary Goals:**
1. **Performance**: Quota check adds <10ms latency (p99)
2. **Accuracy**: 100% of cache writes are quota-checked
3. **Reliability**: Zero false rejections (legitimate writes not blocked)
4. **Observability**: Real-time quota metrics per tenant

**Non-Goals:**
- ❌ Read operation quotas (reads are free)
- ❌ Network bandwidth quotas (only storage)
- ❌ Quota reset scheduling (Story 4.5)
- ❌ Billing integration (future epic)

### 1.3 Success Criteria

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Quota enforcement accuracy | 100% | Integration tests |
| Latency impact (p99) | <10ms | Load testing (1000 req/s) |
| False rejection rate | 0% | Quota boundary tests |
| Quota check success rate | 99.9% | 7-day production monitoring |
| Cache hit rate on quota | >90% | Redis MONITOR analysis |

---

## 2. Architecture Overview

### 2.1 System Context (C4 Level 1)

```
┌─────────────────────────────────────────────────────────────┐
│                     AlphaPulse API                          │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────┐     │
│  │  API Client  │───▶│  QuotaEnforcementMiddleware  │     │
│  │  (Tenant)    │    │  (NEW - This HLD)            │     │
│  └──────────────┘    └──────────┬───────────────────┘     │
│                                  │                          │
│                                  ▼                          │
│                      ┌──────────────────────┐              │
│                      │ TenantCacheManager   │              │
│                      │ (Story 4.1)          │              │
│                      └──────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                               │           │
                               │           │
                     ┌─────────┘           └─────────┐
                     ▼                               ▼
            ┌────────────────┐            ┌──────────────────┐
            │     Redis      │            │   PostgreSQL     │
            │  (Cache Store) │            │  (Quota Config)  │
            └────────────────┘            └──────────────────┘
```

### 2.2 Container View (C4 Level 2)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ASGI Middleware Stack                        │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────┐        │  │
│  │  │  1. TenantContextMiddleware                  │        │  │
│  │  │     (Extracts tenant_id from JWT)            │        │  │
│  │  └──────────────────┬───────────────────────────┘        │  │
│  │                     │                                     │  │
│  │                     ▼                                     │  │
│  │  ┌──────────────────────────────────────────────┐        │  │
│  │  │  2. QuotaEnforcementMiddleware (NEW)         │        │  │
│  │  │     - Extract write size from request        │        │  │
│  │  │     - Check quota (Redis → PostgreSQL)       │        │  │
│  │  │     - Enforce limit (ALLOW/WARN/REJECT)      │        │  │
│  │  │     - Update usage atomically                │        │  │
│  │  │     - Inject response headers                │        │  │
│  │  └──────────────────┬───────────────────────────┘        │  │
│  │                     │                                     │  │
│  │                     ▼                                     │  │
│  │  ┌──────────────────────────────────────────────┐        │  │
│  │  │  3. Application Routes                        │        │  │
│  │  │     (Cache write endpoints)                   │        │  │
│  │  └──────────────────────────────────────────────┘        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
┌──────────────────┐              ┌─────────────────────────┐
│  Redis Cluster   │              │  PostgreSQL Database    │
│                  │              │                         │
│ Keys:            │              │ Tables:                 │
│ quota:cache:     │              │ - tenant_cache_quotas   │
│   {tid}:quota_mb │              │ - tenant_cache_metrics  │
│   {tid}:usage_mb │              │                         │
│   {tid}:overage* │              │                         │
│                  │              │                         │
│ TTL: 5 minutes   │              │ (Authoritative source)  │
└──────────────────┘              └─────────────────────────┘
```

### 2.3 Component View (C4 Level 3)

```
┌───────────────────────────────────────────────────────────────────┐
│         QuotaEnforcementMiddleware Component                      │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  QuotaEnforcementMiddleware (ASGI Middleware)           │    │
│  │                                                           │    │
│  │  async def __call__(request: Request) -> Response:       │    │
│  │    1. tenant_id = extract_tenant_id(request)            │    │
│  │    2. write_size = extract_write_size(request)          │    │
│  │    3. decision = await check_quota(tenant_id, size)     │    │
│  │    4. if decision == REJECT:                            │    │
│  │         return Response(429, headers={...})             │    │
│  │    5. response = await call_next(request)               │    │
│  │    6. inject_quota_headers(response, quota_status)      │    │
│  │    7. return response                                   │    │
│  └────────────────────┬──────────────────────────────────────┘    │
│                       │                                           │
│                       ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  QuotaChecker (Business Logic)                          │    │
│  │                                                           │    │
│  │  async def check_quota(tenant_id, size) -> Decision:    │    │
│  │    quota = await get_quota_cached(tenant_id)            │────┐│
│  │    new_usage = await update_usage(tenant_id, size)      │    ││
│  │    return enforce_limit(quota, new_usage)               │    ││
│  └─────────────────────────────────────────────────────────┘    ││
│                                                                   ││
│  ┌─────────────────────────────────────────────────────────┐    ││
│  │  QuotaCacheService (Caching Layer)                      │◀───┘│
│  │                                                           │    │
│  │  async def get_quota_cached(tenant_id) -> QuotaConfig:  │    │
│  │    # Try Redis first (5-min TTL)                        │    │
│  │    if cached := await redis.get(key):                   │    │
│  │      return cached                                      │    │
│  │    # Fallback to PostgreSQL                             │    │
│  │    quota = await db.get_quota(tenant_id)                │────┐│
│  │    await redis.setex(key, 300, quota)                   │    ││
│  │    return quota                                         │    ││
│  └─────────────────────────────────────────────────────────┘    ││
│                                                                   ││
│  ┌─────────────────────────────────────────────────────────┐    ││
│  │  UsageTracker (Atomic Counter)                          │    ││
│  │                                                           │    ││
│  │  async def update_usage(tenant_id, size) -> float:      │    ││
│  │    key = f"quota:cache:{tenant_id}:usage_mb"            │    ││
│  │    return await redis.incrbyfloat(key, size)            │    ││
│  │                                                           │    ││
│  │  async def rollback_usage(tenant_id, size):             │    ││
│  │    await redis.decrbyfloat(key, size)                   │    ││
│  └─────────────────────────────────────────────────────────┘    ││
└───────────────────────────────────────────────────────────────────┘│
                                                                     │
                                  ┌──────────────────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────────┐
                   │  PostgreSQL Repository           │
                   │                                  │
                   │  async def get_quota(tid):       │
                   │    return await session.get(     │
                   │      TenantCacheQuota,           │
                   │      tenant_id=tid               │
                   │    )                             │
                   └──────────────────────────────────┘
```

---

## 3. Detailed Design

### 3.1 Middleware Flow

**Request Processing Sequence:**

```
┌─────────────┐
│   Request   │
│  (Tenant X) │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│ 1. Extract tenant_id from JWT    │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 2. Detect cache write operation  │
│    - POST /cache/*                │
│    - PUT /cache/*                 │
│    - Check request body size      │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 3. Get quota from cache/DB       │
│    - Try Redis (90% hit rate)    │
│    - Fallback PostgreSQL         │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 4. Update usage atomically       │
│    new_usage = INCRBYFLOAT(...)  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 5. Enforce limit                 │
│    if new_usage > hard_limit:    │
│      - DECRBYFLOAT (rollback)    │
│      - return 429                │
└──────┬───────────────────────────┘
       │ (ALLOW or WARN)
       ▼
┌──────────────────────────────────┐
│ 6. Call next middleware          │
│    response = await call_next()  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 7. Inject quota headers          │
│    X-Cache-Quota-*               │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────┐
│   Response   │
└──────────────┘
```

### 3.2 Quota Check Algorithm

**Pseudocode:**

```python
async def check_quota(
    tenant_id: UUID,
    write_size_mb: float
) -> QuotaDecision:
    """
    Check and enforce cache quota for tenant.

    Returns:
        QuotaDecision.ALLOW: Write proceeds normally
        QuotaDecision.WARN: Write proceeds with warning header
        QuotaDecision.REJECT: Write blocked, HTTP 429
    """

    # Step 1: Get quota configuration (cached)
    quota = await get_quota_cached(tenant_id)

    # Step 2: Atomic usage increment
    usage_key = f"quota:cache:{tenant_id}:current_usage_mb"
    new_usage = await redis.incrbyfloat(usage_key, write_size_mb)

    # Step 3: Enforce limits
    hard_limit = quota.quota_mb + (
        quota.overage_limit_mb if quota.overage_allowed else 0
    )

    if new_usage > hard_limit:
        # Rollback usage increment
        await redis.decrbyfloat(usage_key, write_size_mb)

        # Log rejection
        logger.warning(
            "quota_exceeded",
            tenant_id=tenant_id,
            usage=new_usage,
            limit=hard_limit
        )

        # Emit metric
        quota_rejections_total.labels(tenant_id=tenant_id).inc()

        return QuotaDecision.REJECT

    elif new_usage > quota.quota_mb:
        # Over quota but within overage allowance
        logger.info(
            "quota_warning",
            tenant_id=tenant_id,
            usage=new_usage,
            quota=quota.quota_mb
        )

        quota_warnings_total.labels(tenant_id=tenant_id).inc()

        return QuotaDecision.WARN

    else:
        # Within quota
        return QuotaDecision.ALLOW
```

### 3.3 Caching Strategy

**Two-Tier Cache Architecture:**

```
┌──────────────────────────────────────────────────────────┐
│                   Quota Lookup Flow                      │
└──────────────────────────────────────────────────────────┘

Request: get_quota(tenant_id)
    │
    ▼
┌─────────────────────────────────────┐
│ Redis Cache (Tier 1 - Fast)        │
│                                     │
│ Key: quota:cache:{tid}:quota_mb     │
│ TTL: 300 seconds (5 minutes)       │
│                                     │
│ Cache Hit Rate: ~90%                │
│ Latency: ~1ms                       │
└───────────┬─────────────────────────┘
            │
            ├──[HIT]──▶ Return cached quota
            │
            └──[MISS]──▶
                        │
                        ▼
            ┌────────────────────────────────────┐
            │ PostgreSQL (Tier 2 - Authoritative)│
            │                                    │
            │ Table: tenant_cache_quotas         │
            │ Query: SELECT * WHERE tenant_id=?  │
            │                                    │
            │ Cache Miss Rate: ~10%              │
            │ Latency: ~10ms                     │
            └───────────┬────────────────────────┘
                        │
                        ▼
            ┌────────────────────────────────────┐
            │ Write back to Redis                │
            │ SETEX key 300 value                │
            └───────────┬────────────────────────┘
                        │
                        ▼
                   Return quota
```

**Cache Keys:**

```
# Quota configuration (cached from PostgreSQL)
quota:cache:{tenant_id}:quota_mb          → "500"      (TTL 5m)
quota:cache:{tenant_id}:overage_allowed   → "true"     (TTL 5m)
quota:cache:{tenant_id}:overage_limit_mb  → "50"       (TTL 5m)

# Current usage (real-time counter)
quota:cache:{tenant_id}:current_usage_mb  → "387.5"    (TTL 5m)
```

### 3.4 Data Models

**QuotaConfig (In-Memory Model):**

```python
from dataclasses import dataclass
from uuid import UUID

@dataclass
class QuotaConfig:
    """Tenant cache quota configuration."""
    tenant_id: UUID
    quota_mb: int
    current_usage_mb: float
    overage_allowed: bool
    overage_limit_mb: int

    @property
    def hard_limit_mb(self) -> int:
        """Calculate hard limit (quota + overage)."""
        if self.overage_allowed:
            return self.quota_mb + self.overage_limit_mb
        return self.quota_mb

    @property
    def usage_percent(self) -> float:
        """Calculate usage as percentage."""
        if self.quota_mb == 0:
            return 0.0
        return (self.current_usage_mb / self.quota_mb) * 100

    @property
    def remaining_mb(self) -> float:
        """Calculate remaining quota."""
        return self.quota_mb - self.current_usage_mb
```

**QuotaDecision (Enum):**

```python
from enum import Enum

class QuotaDecision(str, Enum):
    """Quota enforcement decision."""
    ALLOW = "allow"      # Usage <= quota
    WARN = "warn"        # quota < Usage <= hard_limit
    REJECT = "reject"    # Usage > hard_limit
```

**QuotaStatus (Response Model):**

```python
from pydantic import BaseModel

class QuotaStatus(BaseModel):
    """Quota status for API response headers."""
    limit_mb: int
    used_mb: float
    remaining_mb: float
    percent: float
    status: QuotaDecision  # ok, warning, exceeded
```

### 3.5 Response Headers

**Header Format:**

```http
HTTP/1.1 200 OK
X-Cache-Quota-Limit: 500
X-Cache-Quota-Used: 387.5
X-Cache-Quota-Remaining: 112.5
X-Cache-Quota-Percent: 77.5
X-Cache-Quota-Status: ok
```

**Header Scenarios:**

| Scenario | Status Code | X-Cache-Quota-Status | Description |
|----------|-------------|----------------------|-------------|
| Within quota | 200 OK | `ok` | Normal operation |
| Over quota (overage) | 200 OK | `warning` | Warning issued |
| Exceeded hard limit | 429 Too Many Requests | `exceeded` | Write blocked |

**429 Response Example:**

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 3600
X-Cache-Quota-Limit: 500
X-Cache-Quota-Used: 550.0
X-Cache-Quota-Remaining: 0.0
X-Cache-Quota-Percent: 110.0
X-Cache-Quota-Status: exceeded

{
  "error": "quota_exceeded",
  "message": "Cache quota exceeded. Current usage: 550.0 MB, Limit: 500 MB + 50 MB overage.",
  "quota": {
    "limit_mb": 500,
    "used_mb": 550.0,
    "overage_limit_mb": 50,
    "reset_at": "2025-12-01T00:00:00Z"
  },
  "retry_after_seconds": 3600
}
```

---

## 4. Interface Specifications

### 4.1 Middleware Configuration

**FastAPI Application Setup:**

```python
from fastapi import FastAPI
from alpha_pulse.middleware.quota_enforcement import QuotaEnforcementMiddleware

app = FastAPI()

# Add quota enforcement middleware
app.add_middleware(
    QuotaEnforcementMiddleware,
    enabled=True,              # Feature flag
    cache_ttl_seconds=300,     # Quota cache TTL
    redis_client=redis_client,
    db_session_factory=get_db,
    exclude_paths=[            # Paths to skip
        "/health",
        "/metrics",
        "/docs"
    ]
)
```

### 4.2 Metrics Interface

**Prometheus Metrics:**

```python
from prometheus_client import Counter, Histogram

# Quota check metrics
quota_checks_total = Counter(
    "quota_checks_total",
    "Total number of quota checks",
    ["tenant_id", "decision"]  # decision: allow|warn|reject
)

quota_check_latency_ms = Histogram(
    "quota_check_latency_ms",
    "Quota check latency in milliseconds",
    ["cache_hit"]  # cache_hit: true|false
)

# Quota enforcement metrics
quota_rejections_total = Counter(
    "quota_rejections_total",
    "Total number of quota rejections",
    ["tenant_id"]
)

quota_warnings_total = Counter(
    "quota_warnings_total",
    "Total number of quota warnings",
    ["tenant_id"]
)

# Cache performance metrics
quota_cache_hits_total = Counter(
    "quota_cache_hits_total",
    "Quota cache hits (Redis)"
)

quota_cache_misses_total = Counter(
    "quota_cache_misses_total",
    "Quota cache misses (fallback to PostgreSQL)"
)
```

### 4.3 Logging Interface

**Structured Logging:**

```python
import structlog

logger = structlog.get_logger(__name__)

# Quota check logging
logger.info(
    "quota_check",
    tenant_id=str(tenant_id),
    write_size_mb=write_size,
    decision=decision.value,
    latency_ms=latency
)

# Quota rejection logging
logger.warning(
    "quota_exceeded",
    tenant_id=str(tenant_id),
    current_usage_mb=new_usage,
    quota_mb=quota.quota_mb,
    hard_limit_mb=quota.hard_limit_mb,
    overage_allowed=quota.overage_allowed
)

# Cache miss logging
logger.debug(
    "quota_cache_miss",
    tenant_id=str(tenant_id),
    fallback_to_db=True
)
```

---

## 5. Non-Functional Characteristics

### 5.1 Performance

**Targets:**

| Metric | Target | Validation |
|--------|--------|------------|
| Quota check latency (p50) | <3ms | Load testing |
| Quota check latency (p99) | <10ms | Load testing |
| Cache hit rate | >90% | Redis MONITOR |
| Throughput | 1000 req/s per tenant | Load testing |

**Optimization Strategies:**

1. **Redis Caching**: 90% cache hit rate reduces DB queries
2. **Atomic Operations**: Redis INCR is O(1) constant time
3. **Connection Pooling**: Reuse Redis connections (max 100 pool)
4. **Pipeline Optimization**: Batch Redis commands when possible

**Performance Testing Plan:**

```bash
# Load test: 1000 req/s with quota enforcement
locust -f tests/performance/test_quota_enforcement.py \
       --users 1000 \
       --spawn-rate 100 \
       --run-time 5m

# Metrics to collect:
# - p50, p95, p99 latency
# - Error rate
# - Cache hit rate
# - Redis/PostgreSQL query latency
```

### 5.2 Scalability

**Horizontal Scaling:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  API Node 1 │    │  API Node 2 │    │  API Node N │
│  (Stateless)│    │  (Stateless)│    │  (Stateless)│
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Redis Cluster        │
              │   (Shared State)       │
              │   - Atomic counters    │
              │   - Quota cache        │
              └────────────────────────┘
```

**Capacity Planning:**

- **Per Node**: 5000 req/s sustained
- **Redis Cluster**: 100k ops/s across 3 nodes
- **PostgreSQL**: 10k queries/s (but 90% cached)

### 5.3 Reliability

**Failure Modes and Handling:**

| Failure | Impact | Mitigation | Recovery Time |
|---------|--------|------------|---------------|
| Redis unavailable | Quota checks fail | Fallback to PostgreSQL | <100ms |
| PostgreSQL unavailable | Can't load quota | Return cached quota or fail-open | <5s |
| Quota cache stale | Incorrect enforcement | 5-min TTL limits staleness | <5 min |
| Network partition | Middleware unreachable | Load balancer failover | <10s |

**Fallback Strategy:**

```python
async def get_quota_cached(tenant_id: UUID) -> QuotaConfig:
    """Get quota with fallback chain."""
    try:
        # Try Redis first
        return await redis_cache.get(tenant_id)
    except RedisConnectionError:
        logger.warning("redis_unavailable", fallback="postgres")

        try:
            # Fallback to PostgreSQL
            return await postgres_repo.get_quota(tenant_id)
        except DatabaseError:
            logger.error("postgres_unavailable", fallback="default")

            # Last resort: Return default quota
            return QuotaConfig(
                tenant_id=tenant_id,
                quota_mb=100,  # Default quota
                current_usage_mb=0,
                overage_allowed=False,
                overage_limit_mb=0
            )
```

### 5.4 Security

**Threat Model:**

| Threat | Risk | Mitigation |
|--------|------|------------|
| Tenant ID spoofing | High | JWT validation upstream (TenantContextMiddleware) |
| Quota bypass | High | Middleware runs before all routes |
| Cache poisoning | Medium | Redis AUTH, network isolation |
| DOS via quota checks | Medium | Rate limiting at ingress |

**Security Controls:**

1. ✅ **Authentication**: JWT validated before middleware
2. ✅ **Authorization**: Tenant context extracted from JWT (trusted)
3. ✅ **Input Validation**: Write size validated (max 100MB)
4. ✅ **Audit Logging**: All quota rejections logged with tenant_id
5. ✅ **Network Isolation**: Redis accessible only from API nodes

### 5.5 Observability

**Monitoring Dashboard (Grafana):**

```yaml
dashboard:
  title: "Cache Quota Enforcement"
  panels:
    - title: "Quota Check Latency (p99)"
      query: histogram_quantile(0.99, quota_check_latency_ms)
      threshold: 10ms
      alert: if > 15ms for 5 minutes

    - title: "Quota Rejection Rate"
      query: rate(quota_rejections_total[5m])
      threshold: <1% of requests
      alert: if > 5% for 10 minutes

    - title: "Cache Hit Rate"
      query: quota_cache_hits / (quota_cache_hits + quota_cache_misses)
      threshold: >90%
      alert: if < 80% for 5 minutes

    - title: "Tenants Over Quota"
      query: count(quota_warnings_total > 0)
      threshold: <10 tenants
      alert: if > 50 tenants
```

**Alerting Rules:**

```yaml
alerts:
  - name: "QuotaCheckLatencyHigh"
    condition: p99_latency > 15ms for 5 minutes
    severity: warning
    action: Page on-call engineer

  - name: "QuotaRejectionRateHigh"
    condition: rejection_rate > 5% for 10 minutes
    severity: critical
    action: Page on-call + escalate

  - name: "QuotaCacheHitRateLow"
    condition: cache_hit_rate < 80% for 5 minutes
    severity: warning
    action: Investigate Redis performance
```

---

## 6. Risk Analysis

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Severity | Mitigation | Owner |
|------|------------|--------|----------|------------|-------|
| **RISK-001**: Quota check exceeds 10ms latency | Medium | High | **HIGH** | - Redis caching (90% hit rate)<br>- Load testing validation<br>- Performance profiling | Tech Lead |
| **RISK-002**: Race condition on quota boundary | Low | Medium | **MEDIUM** | - Redis atomic INCR operations<br>- Concurrent request testing | Senior Dev |
| **RISK-003**: Cache stampede on quota reload | Low | Medium | **MEDIUM** | - Single-flight pattern<br>- Staggered TTL expiration | Platform Team |
| **RISK-004**: Middleware misconfiguration blocks all requests | Low | Critical | **HIGH** | - Feature flag for gradual rollout<br>- Comprehensive integration tests<br>- Rollback plan | Tech Lead |
| **RISK-005**: Redis failure degrades performance | Medium | Medium | **MEDIUM** | - Fallback to PostgreSQL<br>- Health check monitoring<br>- Redis cluster HA | SRE Team |

### 6.2 Operational Risks

| Risk | Likelihood | Impact | Severity | Mitigation | Owner |
|------|------------|--------|----------|------------|-------|
| **RISK-006**: Quota staleness causes incorrect enforcement | Medium | Low | **LOW** | - 5-min TTL acceptable for business<br>- Manual cache invalidation API | Product Team |
| **RISK-007**: Tenants complain about quota rejections | Medium | Medium | **MEDIUM** | - Clear error messages<br>- Gradual enforcement (warnings first)<br>- Support runbook | Support Team |
| **RISK-008**: Capacity planning based on stale metrics | Low | Medium | **LOW** | - Real-time usage dashboard<br>- Daily metrics sync (Story 4.5) | Data Team |

### 6.3 Mitigation Strategies

**RISK-001: Latency Mitigation**

```python
# Strategy: Optimize Redis queries with pipelining
async def get_quota_cached_optimized(tenant_id: UUID) -> QuotaConfig:
    """Get quota with optimized pipelining."""
    pipe = redis.pipeline()

    # Batch all Redis queries
    pipe.get(f"quota:cache:{tenant_id}:quota_mb")
    pipe.get(f"quota:cache:{tenant_id}:overage_allowed")
    pipe.get(f"quota:cache:{tenant_id}:overage_limit_mb")
    pipe.get(f"quota:cache:{tenant_id}:current_usage_mb")

    # Execute pipeline (single network round-trip)
    results = await pipe.execute()

    if all(results):
        # All keys present, construct QuotaConfig
        return QuotaConfig(
            tenant_id=tenant_id,
            quota_mb=int(results[0]),
            overage_allowed=bool(results[1]),
            overage_limit_mb=int(results[2]),
            current_usage_mb=float(results[3])
        )
    else:
        # Cache miss, fallback to PostgreSQL
        return await postgres_repo.get_quota(tenant_id)
```

**RISK-002: Race Condition Mitigation**

```python
# Strategy: Use Redis INCRBYFLOAT (atomic operation)
async def update_usage_atomic(
    tenant_id: UUID,
    write_size_mb: float
) -> float:
    """Atomically increment usage counter."""
    key = f"quota:cache:{tenant_id}:current_usage_mb"

    # INCRBYFLOAT is atomic - no race conditions
    # Returns new value after increment
    new_usage = await redis.incrbyfloat(key, write_size_mb)

    return new_usage
```

---

## 7. Testing Strategy

### 7.1 Test Levels

**Unit Tests:**
- Quota check logic (all decision paths)
- Response header injection
- Cache key generation
- Quota configuration serialization

**Integration Tests:**
- End-to-end quota enforcement flow
- Redis cache hit/miss scenarios
- PostgreSQL fallback on Redis failure
- Middleware integration with FastAPI

**Load Tests:**
- 1000 req/s sustained load per tenant
- Concurrent writes to same tenant
- Cache hit rate validation
- Latency distribution (p50, p95, p99)

**Chaos Tests:**
- Redis connection failures
- PostgreSQL connection failures
- Network latency simulation
- Cache invalidation storms

### 7.2 Test Scenarios

**Scenario 1: Normal Operation (Within Quota)**

```python
@pytest.mark.asyncio
async def test_quota_check_allow():
    """Test normal operation within quota."""
    # Setup
    tenant_id = UUID("...")
    setup_quota(tenant_id, quota_mb=100, usage_mb=50)

    # Execute
    response = await client.post(
        "/cache/write",
        json={"key": "test", "value": "data", "size_mb": 10}
    )

    # Assert
    assert response.status_code == 200
    assert response.headers["X-Cache-Quota-Status"] == "ok"
    assert response.headers["X-Cache-Quota-Used"] == "60.0"
```

**Scenario 2: Warning (Overage Allowance)**

```python
@pytest.mark.asyncio
async def test_quota_check_warn():
    """Test warning on overage."""
    # Setup
    tenant_id = UUID("...")
    setup_quota(
        tenant_id,
        quota_mb=100,
        usage_mb=95,
        overage_allowed=True,
        overage_limit_mb=10
    )

    # Execute
    response = await client.post(
        "/cache/write",
        json={"key": "test", "value": "data", "size_mb": 10}
    )

    # Assert
    assert response.status_code == 200
    assert response.headers["X-Cache-Quota-Status"] == "warning"
    assert response.headers["X-Cache-Quota-Used"] == "105.0"
```

**Scenario 3: Rejection (Hard Limit)**

```python
@pytest.mark.asyncio
async def test_quota_check_reject():
    """Test rejection at hard limit."""
    # Setup
    tenant_id = UUID("...")
    setup_quota(
        tenant_id,
        quota_mb=100,
        usage_mb=105,
        overage_allowed=True,
        overage_limit_mb=10
    )

    # Execute
    response = await client.post(
        "/cache/write",
        json={"key": "test", "value": "data", "size_mb": 10}
    )

    # Assert
    assert response.status_code == 429
    assert response.headers["X-Cache-Quota-Status"] == "exceeded"
    assert "Retry-After" in response.headers

    # Verify usage was NOT incremented (rollback)
    usage = await get_current_usage(tenant_id)
    assert usage == 105.0  # Unchanged
```

**Scenario 4: Concurrent Requests (Race Condition)**

```python
@pytest.mark.asyncio
async def test_quota_concurrent_requests():
    """Test atomic usage tracking under concurrent load."""
    # Setup
    tenant_id = UUID("...")
    setup_quota(tenant_id, quota_mb=100, usage_mb=0)

    # Execute 100 concurrent 1MB writes
    tasks = [
        client.post("/cache/write", json={"size_mb": 1})
        for _ in range(100)
    ]
    responses = await asyncio.gather(*tasks)

    # Assert
    # All 100 should succeed (100MB total, within quota)
    assert all(r.status_code == 200 for r in responses)

    # Verify usage is exactly 100MB (no double-counting)
    usage = await get_current_usage(tenant_id)
    assert usage == 100.0
```

**Scenario 5: Performance (Latency Target)**

```python
@pytest.mark.asyncio
@pytest.mark.performance
async def test_quota_check_latency():
    """Test quota check meets <10ms p99 latency target."""
    # Setup
    tenant_id = UUID("...")
    setup_quota(tenant_id, quota_mb=1000, usage_mb=0)

    # Warm up cache
    await client.post("/cache/write", json={"size_mb": 1})

    # Execute 1000 requests and measure latency
    latencies = []
    for _ in range(1000):
        start = time.time()
        await client.post("/cache/write", json={"size_mb": 1})
        latencies.append((time.time() - start) * 1000)  # ms

    # Assert p99 < 10ms
    p99_latency = np.percentile(latencies, 99)
    assert p99_latency < 10, f"p99 latency {p99_latency:.2f}ms exceeds 10ms target"
```

---

## 8. Deployment Strategy

### 8.1 Rollout Plan

**Phase 1: Shadow Mode (1 week)**
- Deploy with `enabled=False` (feature flag off)
- Middleware runs but doesn't enforce (logs decisions only)
- Validate metrics, latency impact, cache hit rate

**Phase 2: Internal Tenant (1 week)**
- Enable for single internal test tenant
- Monitor quota rejections, support tickets
- Validate response headers, 429 handling

**Phase 3: Gradual Rollout (2 weeks)**
- 10% of tenants (selected randomly)
- 25% of tenants
- 50% of tenants
- 100% of tenants

**Phase 4: Optimization (ongoing)**
- Tune cache TTL based on metrics
- Adjust overage limits per tenant tier
- Optimize latency based on profiling

### 8.2 Rollback Plan

**Rollback Triggers:**
- p99 latency >15ms sustained
- Quota rejection rate >10% sustained
- Critical bugs (false rejections)
- SLO violations

**Rollback Steps:**
1. Set feature flag `enabled=False` (instant)
2. Restart API pods to clear middleware state
3. Verify latency returns to baseline
4. Investigate root cause
5. Fix and re-deploy

**Rollback SLA:** <5 minutes from trigger to disabled

### 8.3 Configuration Management

**Environment Variables:**

```bash
# Feature flag
QUOTA_ENFORCEMENT_ENABLED=true

# Cache TTL
QUOTA_CACHE_TTL_SECONDS=300

# Fallback behavior
QUOTA_FAIL_OPEN=false  # Deny on error (fail-closed)

# Performance tuning
QUOTA_REDIS_POOL_SIZE=100
QUOTA_CACHE_PIPELINE_ENABLED=true
```

**Per-Tenant Configuration:**

```yaml
# Database: tenant_cache_quotas table
tenant_id: 00000000-0000-0000-0000-000000000001
quota_mb: 500
overage_allowed: true
overage_limit_mb: 50
```

---

## 9. Dependencies

### 9.1 Internal Dependencies

| Dependency | Version | Status | Impact |
|------------|---------|--------|--------|
| Story 4.1: TenantCacheManager | PR #211 | ✅ Merged | Required for cache write detection |
| Story 4.2: Database Schema | PR #212 | ✅ Merged | Required for quota configuration |
| TenantContextMiddleware | Existing | ✅ Available | Required for tenant_id extraction |

### 9.2 External Dependencies

| Dependency | Version | Purpose | Criticality |
|------------|---------|---------|-------------|
| Redis | 6.0+ | Quota cache, atomic counters | **HIGH** |
| PostgreSQL | 14+ | Authoritative quota source | **HIGH** |
| FastAPI | 0.100+ | Async middleware support | **HIGH** |
| Prometheus Client | 0.18+ | Metrics export | **MEDIUM** |

### 9.3 Dependency Risks

| Dependency | Risk | Mitigation |
|------------|------|------------|
| Redis unavailable | Quota checks fail | Fallback to PostgreSQL |
| PostgreSQL slow | Increased latency | 5-min cache TTL reduces DB load |
| FastAPI version incompatibility | Middleware breaks | Pin FastAPI version in pyproject.toml |

---

## 10. Success Metrics

### 10.1 Technical Metrics

| Metric | Target | Measurement | Baseline |
|--------|--------|-------------|----------|
| Quota enforcement accuracy | 100% | Integration tests | N/A (new feature) |
| Latency impact (p99) | <10ms | Load testing | +5ms avg |
| Cache hit rate | >90% | Redis MONITOR | N/A (new cache) |
| False rejection rate | 0% | Test suite + monitoring | N/A |
| Quota check success rate | 99.9% | Prometheus metrics | N/A |

### 10.2 Business Metrics

| Metric | Target | Measurement | Timeline |
|--------|--------|-------------|----------|
| Tenants exceeding quota | <5% | Monthly report | 1 month |
| Support tickets (quota) | <10/month | Support system | 1 month |
| Redis cost reduction | 30% | Infrastructure bills | 3 months |
| Usage-based billing enabled | Yes | Product launch | 3 months |

### 10.3 Operational Metrics

| Metric | Target | Measurement | Alert Threshold |
|--------|--------|-------------|-----------------|
| Quota rejection rate | <1% | Prometheus | >5% for 10 min |
| Middleware errors | <0.1% | Error logs | >1% for 5 min |
| Cache stampede incidents | 0 | Redis logs | >1/week |
| Rollback incidents | 0 | Incident reports | >1/month |

---

## 11. Open Questions

| # | Question | Stakeholder | Priority | Status |
|---|----------|-------------|----------|--------|
| Q1 | Should read operations count against quota? | Product | **HIGH** | ⏳ Open |
| Q2 | What happens when quota is reduced below current usage? | Product + Tech Lead | **MEDIUM** | ⏳ Open |
| Q3 | Should we support per-resource quotas (not just total)? | Product | **LOW** | ⏳ Open |
| Q4 | What's the retry strategy for 429 responses? | Product | **MEDIUM** | ⏳ Open |
| Q5 | Should quota reset be automatic or manual? | Product | **MEDIUM** | Handled in Story 4.5 |

---

## 12. Appendices

### Appendix A: Related Documents

- **ADR-005**: Quota Enforcement Middleware (architectural decision)
- **Story 4.1**: Tenant Namespace Layer (PR #211)
- **Story 4.2**: Database Schema (PR #212)
- **Story 4.5**: Metrics Collection Job (quota reset, usage sync)

### Appendix B: Glossary

- **Quota**: Maximum cache storage allocation per tenant (in MB)
- **Overage**: Amount tenant can exceed quota before hard rejection
- **Hard Limit**: quota_mb + overage_limit_mb (absolute maximum)
- **Soft Limit**: quota_mb (warning threshold)
- **Cache Stampede**: Many requests simultaneously reloading cached data
- **Single-Flight**: Pattern where only one goroutine loads, others wait

### Appendix C: Architecture Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-07 | Use FastAPI middleware | Standard pattern, easy to test/disable |
| 2025-11-07 | Two-tier caching (Redis + PostgreSQL) | Balance performance and consistency |
| 2025-11-07 | Redis atomic INCR for usage | Prevents race conditions |
| 2025-11-07 | 5-minute cache TTL | Balance staleness vs. DB load |
| 2025-11-07 | Fail-closed (deny on error) | Security over availability |

---

## Review Sign-Off

- [ ] **Architecture Review**: Tech Lead + Senior Engineers
- [ ] **Security Review**: Security team (if handling PII)
- [ ] **Performance Review**: Platform team (baseline established)
- [ ] **Product Review**: Product Manager (assumptions validated)

**Status**: ⏳ **READY FOR REVIEW**

**Next Phase**: Phase 3 - Build & Validate (TDD implementation)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Tech Lead
**Reviewers**: TBD
