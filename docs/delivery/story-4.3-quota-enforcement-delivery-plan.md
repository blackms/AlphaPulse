# Story 4.3: Quota Enforcement Middleware - Delivery Plan

**Epic**: EPIC-004 (Caching Layer)
**Story**: 4.3 - Quota Enforcement Middleware (5 SP)
**Status**: Planning
**Date**: 2025-11-07
**Owner**: Tech Lead

## Executive Summary

This delivery plan sequences the implementation of quota enforcement middleware following TDD methodology (RED-GREEN-REFACTOR-QUALITY). Total effort: **5 SP (~1.5 days)** with 4 sequential phases and clear handoff criteria.

**Key Milestones:**
- Phase 1 (RED): Tests written - 0.5 days
- Phase 2 (GREEN): Implementation complete - 0.5 days
- Phase 3 (REFACTOR): Optimizations done - 0.25 days
- Phase 4 (QUALITY): Validation complete - 0.25 days

**Risk Level**: **LOW** (standard middleware pattern, dependencies met)

---

## 1. Delivery Phases

### Phase 1: RED - Write Failing Tests (0.5 days, 2 SP)

**Objective**: Create comprehensive test suite that validates all quota enforcement scenarios.

**Tasks:**

| Task | Description | Effort | Owner | Dependencies |
|------|-------------|--------|-------|--------------|
| T1.1 | Create test file structure | 0.1 SP | Dev | None |
| T1.2 | Write unit tests for QuotaChecker | 0.5 SP | Dev | T1.1 |
| T1.3 | Write middleware integration tests | 0.5 SP | Dev | T1.1 |
| T1.4 | Write load/performance tests | 0.4 SP | Dev | T1.1 |
| T1.5 | Write race condition tests | 0.5 SP | Dev | T1.1 |

**Deliverables:**
- ✅ `tests/middleware/test_quota_enforcement.py` (unit tests)
- ✅ `tests/integration/test_quota_middleware_e2e.py` (integration tests)
- ✅ `tests/performance/test_quota_latency.py` (load tests)
- ✅ All tests FAIL with clear error messages

**Acceptance Criteria:**
- [ ] All 8 acceptance criteria from discovery doc have corresponding tests
- [ ] Tests cover: ALLOW, WARN, REJECT decision paths
- [ ] Tests cover: cache hit/miss scenarios
- [ ] Tests cover: Redis failure fallback
- [ ] Tests cover: concurrent request race conditions
- [ ] Tests cover: performance target (<10ms p99)
- [ ] Test execution time: <30 seconds for unit tests

**Test Scenarios (Minimum):**

```python
# Unit Tests (tests/middleware/test_quota_enforcement.py)
def test_check_quota_allow()          # Within quota
def test_check_quota_warn()           # Over quota, within overage
def test_check_quota_reject()         # Over hard limit
def test_usage_rollback_on_reject()   # Atomic rollback
def test_cache_hit()                  # Redis cache hit
def test_cache_miss_fallback_to_db()  # PostgreSQL fallback
def test_response_headers()           # Header injection
def test_feature_flag_disabled()      # Skip when flag off

# Integration Tests (tests/integration/test_quota_middleware_e2e.py)
async def test_e2e_quota_enforcement()      # Full request flow
async def test_e2e_429_response()           # Rejection response
async def test_e2e_concurrent_writes()      # Race condition
async def test_e2e_redis_failure()          # Fallback scenario

# Performance Tests (tests/performance/test_quota_latency.py)
async def test_latency_within_target()      # <10ms p99
async def test_throughput_1000_req_per_sec()# Throughput target
async def test_cache_hit_rate()             # >90% hit rate
```

**Quality Gates:**
- [ ] Code coverage setup configured (pytest-cov)
- [ ] All tests fail with descriptive error messages
- [ ] Test fixtures for Redis/PostgreSQL mocking
- [ ] Performance test baseline established

**Handoff Criteria:**
- ✅ Test suite runs successfully (all FAIL as expected)
- ✅ Test coverage report generated
- ✅ Tech Lead review: Test scenarios comprehensive

---

### Phase 2: GREEN - Implement Middleware (0.5 days, 2 SP)

**Objective**: Implement minimum viable middleware to pass all tests.

**Tasks:**

| Task | Description | Effort | Owner | Dependencies |
|------|-------------|--------|-------|--------------|
| T2.1 | Create middleware file structure | 0.1 SP | Dev | Phase 1 complete |
| T2.2 | Implement QuotaEnforcementMiddleware class | 0.6 SP | Dev | T2.1 |
| T2.3 | Implement QuotaChecker (business logic) | 0.5 SP | Dev | T2.1 |
| T2.4 | Implement QuotaCacheService (Redis + PostgreSQL) | 0.4 SP | Dev | T2.1 |
| T2.5 | Implement UsageTracker (atomic counters) | 0.3 SP | Dev | T2.1 |
| T2.6 | Implement response header injection | 0.1 SP | Dev | T2.2 |

**Deliverables:**
- ✅ `src/alpha_pulse/middleware/quota_enforcement.py` (middleware)
- ✅ `src/alpha_pulse/services/quota_checker.py` (business logic)
- ✅ `src/alpha_pulse/services/quota_cache_service.py` (caching)
- ✅ `src/alpha_pulse/services/usage_tracker.py` (atomic counters)
- ✅ All Phase 1 tests PASS

**Implementation Files:**

```
src/alpha_pulse/
├── middleware/
│   ├── __init__.py
│   └── quota_enforcement.py          # QuotaEnforcementMiddleware
├── services/
│   ├── quota_checker.py              # QuotaChecker (business logic)
│   ├── quota_cache_service.py        # QuotaCacheService (Redis + DB)
│   └── usage_tracker.py              # UsageTracker (atomic INCR)
└── models/
    └── quota.py                      # QuotaConfig, QuotaDecision, QuotaStatus
```

**Key Implementation Components:**

**1. QuotaEnforcementMiddleware (ASGI Middleware)**

```python
# src/alpha_pulse/middleware/quota_enforcement.py

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

class QuotaEnforcementMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for cache quota enforcement.

    Intercepts cache write requests, checks quota, enforces limits,
    and injects quota status into response headers.
    """

    def __init__(
        self,
        app,
        enabled: bool = True,
        cache_ttl_seconds: int = 300,
        redis_client=None,
        db_session_factory=None,
        exclude_paths: list[str] = None
    ):
        super().__init__(app)
        self.enabled = enabled
        self.cache_ttl = cache_ttl_seconds
        self.redis = redis_client
        self.db_factory = db_session_factory
        self.exclude_paths = exclude_paths or []

        # Initialize services
        self.quota_checker = QuotaChecker(...)
        self.cache_service = QuotaCacheService(...)
        self.usage_tracker = UsageTracker(...)

    async def dispatch(
        self,
        request: Request,
        call_next
    ) -> Response:
        """Process request with quota enforcement."""

        # Skip if disabled or excluded path
        if not self.enabled or request.url.path in self.exclude_paths:
            return await call_next(request)

        # Extract tenant context
        tenant_id = request.state.tenant_id  # Set by TenantContextMiddleware

        # Detect cache write operation
        if not self._is_cache_write(request):
            return await call_next(request)

        # Extract write size
        write_size_mb = await self._extract_write_size(request)

        # Check quota
        decision = await self.quota_checker.check_quota(
            tenant_id,
            write_size_mb
        )

        # Enforce limit
        if decision == QuotaDecision.REJECT:
            return self._create_429_response(tenant_id)

        # Proceed with request
        response = await call_next(request)

        # Inject quota headers
        self._inject_quota_headers(response, tenant_id, decision)

        return response

    def _is_cache_write(self, request: Request) -> bool:
        """Detect if request is a cache write operation."""
        return (
            request.method in ["POST", "PUT"] and
            request.url.path.startswith("/cache/")
        )

    async def _extract_write_size(self, request: Request) -> float:
        """Extract write size from request body."""
        # Implementation: Parse request body for size_mb field
        pass

    def _inject_quota_headers(
        self,
        response: Response,
        tenant_id: UUID,
        decision: QuotaDecision
    ):
        """Inject X-Cache-Quota-* headers into response."""
        # Implementation: Add headers to response
        pass

    def _create_429_response(self, tenant_id: UUID) -> Response:
        """Create 429 Too Many Requests response."""
        # Implementation: Return 429 with quota details
        pass
```

**2. QuotaChecker (Business Logic)**

```python
# src/alpha_pulse/services/quota_checker.py

from uuid import UUID
from alpha_pulse.models.quota import QuotaDecision

class QuotaChecker:
    """Business logic for quota checking and enforcement."""

    def __init__(
        self,
        cache_service: QuotaCacheService,
        usage_tracker: UsageTracker
    ):
        self.cache_service = cache_service
        self.usage_tracker = usage_tracker

    async def check_quota(
        self,
        tenant_id: UUID,
        write_size_mb: float
    ) -> QuotaDecision:
        """
        Check quota and enforce limits.

        Returns:
            QuotaDecision.ALLOW: Write allowed
            QuotaDecision.WARN: Write allowed with warning
            QuotaDecision.REJECT: Write rejected
        """

        # Get quota configuration
        quota = await self.cache_service.get_quota(tenant_id)

        # Atomic usage increment
        new_usage = await self.usage_tracker.increment(
            tenant_id,
            write_size_mb
        )

        # Calculate limits
        hard_limit = quota.quota_mb + (
            quota.overage_limit_mb if quota.overage_allowed else 0
        )

        # Enforce limits
        if new_usage > hard_limit:
            # Rollback usage
            await self.usage_tracker.decrement(tenant_id, write_size_mb)

            # Log rejection
            logger.warning(
                "quota_exceeded",
                tenant_id=str(tenant_id),
                usage=new_usage,
                limit=hard_limit
            )

            # Emit metric
            quota_rejections_total.labels(tenant_id=str(tenant_id)).inc()

            return QuotaDecision.REJECT

        elif new_usage > quota.quota_mb:
            # Over quota but within overage
            logger.info(
                "quota_warning",
                tenant_id=str(tenant_id),
                usage=new_usage,
                quota=quota.quota_mb
            )

            quota_warnings_total.labels(tenant_id=str(tenant_id)).inc()

            return QuotaDecision.WARN

        else:
            # Within quota
            return QuotaDecision.ALLOW
```

**3. QuotaCacheService (Two-Tier Caching)**

```python
# src/alpha_pulse/services/quota_cache_service.py

from uuid import UUID
from alpha_pulse.models.quota import QuotaConfig

class QuotaCacheService:
    """Two-tier caching for quota configuration."""

    def __init__(
        self,
        redis_client,
        db_session_factory,
        cache_ttl: int = 300
    ):
        self.redis = redis_client
        self.db_factory = db_session_factory
        self.cache_ttl = cache_ttl

    async def get_quota(self, tenant_id: UUID) -> QuotaConfig:
        """Get quota with Redis cache + PostgreSQL fallback."""

        # Try Redis first
        cached = await self._get_from_redis(tenant_id)
        if cached:
            quota_cache_hits_total.inc()
            return cached

        # Fallback to PostgreSQL
        quota_cache_misses_total.inc()

        quota = await self._get_from_db(tenant_id)

        # Write back to Redis
        await self._cache_to_redis(tenant_id, quota)

        return quota

    async def _get_from_redis(self, tenant_id: UUID) -> QuotaConfig | None:
        """Get quota from Redis cache."""
        # Implementation: Pipeline GET for all quota fields
        pass

    async def _get_from_db(self, tenant_id: UUID) -> QuotaConfig:
        """Get quota from PostgreSQL."""
        # Implementation: Query tenant_cache_quotas table
        pass

    async def _cache_to_redis(self, tenant_id: UUID, quota: QuotaConfig):
        """Cache quota to Redis with TTL."""
        # Implementation: Pipeline SETEX for all quota fields
        pass
```

**4. UsageTracker (Atomic Counters)**

```python
# src/alpha_pulse/services/usage_tracker.py

from uuid import UUID

class UsageTracker:
    """Atomic usage tracking with Redis INCR."""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def increment(
        self,
        tenant_id: UUID,
        amount_mb: float
    ) -> float:
        """Atomically increment usage counter."""
        key = f"quota:cache:{tenant_id}:current_usage_mb"
        return await self.redis.incrbyfloat(key, amount_mb)

    async def decrement(
        self,
        tenant_id: UUID,
        amount_mb: float
    ) -> float:
        """Atomically decrement usage counter (rollback)."""
        key = f"quota:cache:{tenant_id}:current_usage_mb"
        return await self.redis.decrbyfloat(key, amount_mb)

    async def get_usage(self, tenant_id: UUID) -> float:
        """Get current usage."""
        key = f"quota:cache:{tenant_id}:current_usage_mb"
        usage = await self.redis.get(key)
        return float(usage) if usage else 0.0
```

**Acceptance Criteria:**
- [ ] All Phase 1 tests pass (100% success rate)
- [ ] No linting errors (ruff, black, mypy clean)
- [ ] Code follows existing project structure
- [ ] All imports resolve correctly
- [ ] No obvious bugs or logic errors

**Quality Gates:**
- [ ] Unit test coverage: >=90%
- [ ] Integration tests: All passing
- [ ] Linting: Zero errors
- [ ] Type checking: mypy passes

**Handoff Criteria:**
- ✅ All tests pass
- ✅ Code review by peer (logic correctness)
- ✅ Tech Lead review: Implementation meets design

---

### Phase 3: REFACTOR - Optimize Implementation (0.25 days, 0.5 SP)

**Objective**: Optimize performance, improve code quality, add observability.

**Tasks:**

| Task | Description | Effort | Owner | Dependencies |
|------|-------------|--------|-------|--------------|
| T3.1 | Extract constants and configuration | 0.05 SP | Dev | Phase 2 complete |
| T3.2 | Optimize Redis queries with pipelining | 0.15 SP | Dev | T3.1 |
| T3.3 | Add Prometheus metrics | 0.15 SP | Dev | T3.1 |
| T3.4 | Add structured logging | 0.10 SP | Dev | T3.1 |
| T3.5 | Improve error handling | 0.05 SP | Dev | T3.1 |

**Refactoring Improvements:**

**1. Extract Configuration Constants**

```python
# src/alpha_pulse/middleware/quota_enforcement.py

# Configuration constants
DEFAULT_QUOTA_CACHE_TTL = 300  # 5 minutes
MAX_WRITE_SIZE_MB = 100        # Maximum single write
REDIS_POOL_SIZE = 100
PIPELINE_ENABLED = True

# Paths to exclude from quota enforcement
EXCLUDED_PATHS = [
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json"
]
```

**2. Optimize Redis Queries with Pipelining**

```python
# Before (multiple round-trips)
quota_mb = await redis.get(f"quota:cache:{tid}:quota_mb")
overage = await redis.get(f"quota:cache:{tid}:overage_allowed")
limit = await redis.get(f"quota:cache:{tid}:overage_limit_mb")

# After (single round-trip)
pipe = redis.pipeline()
pipe.get(f"quota:cache:{tid}:quota_mb")
pipe.get(f"quota:cache:{tid}:overage_allowed")
pipe.get(f"quota:cache:{tid}:overage_limit_mb")
results = await pipe.execute()
```

**3. Add Prometheus Metrics**

```python
# src/alpha_pulse/middleware/quota_enforcement.py

from prometheus_client import Counter, Histogram

# Define metrics
quota_checks_total = Counter(
    "quota_checks_total",
    "Total quota checks",
    ["tenant_id", "decision"]
)

quota_check_latency_ms = Histogram(
    "quota_check_latency_ms",
    "Quota check latency",
    ["cache_hit"]
)

# Instrument code
start = time.time()
decision = await check_quota(...)
latency = (time.time() - start) * 1000

quota_checks_total.labels(
    tenant_id=str(tenant_id),
    decision=decision.value
).inc()

quota_check_latency_ms.labels(
    cache_hit="true"
).observe(latency)
```

**4. Add Structured Logging**

```python
import structlog

logger = structlog.get_logger(__name__)

# Structured log entries
logger.info(
    "quota_check",
    tenant_id=str(tenant_id),
    write_size_mb=write_size,
    decision=decision.value,
    latency_ms=latency,
    cache_hit=cache_hit
)
```

**Deliverables:**
- ✅ Constants extracted to configuration
- ✅ Redis queries optimized (pipelining)
- ✅ Prometheus metrics added
- ✅ Structured logging implemented
- ✅ Error handling improved (try/except blocks)
- ✅ All tests still pass

**Quality Gates:**
- [ ] Performance tests show latency improvement (if applicable)
- [ ] Code complexity reduced (cyclomatic complexity <10)
- [ ] No duplicate code (DRY principle)
- [ ] Metrics endpoint (/metrics) shows new quota metrics

**Handoff Criteria:**
- ✅ Refactoring complete, all tests pass
- ✅ Code review: No obvious performance issues
- ✅ Metrics validated in Grafana

---

### Phase 4: QUALITY - Validate and Document (0.25 days, 0.5 SP)

**Objective**: Run comprehensive validation tests, performance benchmarks, and finalize documentation.

**Tasks:**

| Task | Description | Effort | Owner | Dependencies |
|------|-------------|--------|-------|--------------|
| T4.1 | Run full test suite | 0.05 SP | Dev | Phase 3 complete |
| T4.2 | Run load/performance tests | 0.15 SP | Dev | T4.1 |
| T4.3 | Run concurrent request tests | 0.10 SP | Dev | T4.1 |
| T4.4 | Generate coverage report | 0.05 SP | Dev | T4.1 |
| T4.5 | Update documentation | 0.10 SP | Dev | T4.1 |
| T4.6 | Create operational runbook | 0.05 SP | Dev | T4.1 |

**Validation Tests:**

**1. Full Test Suite Execution**

```bash
# Run all tests with coverage
PYTHONPATH=. poetry run pytest \
    tests/middleware/ \
    tests/integration/ \
    tests/performance/ \
    -v --cov=src/alpha_pulse/middleware \
    --cov=src/alpha_pulse/services \
    --cov-report=html \
    --cov-report=term

# Expected: >90% coverage, all tests pass
```

**2. Load Testing (1000 req/s)**

```bash
# Run locust load test
locust -f tests/performance/test_quota_enforcement_load.py \
       --users 1000 \
       --spawn-rate 100 \
       --run-time 5m \
       --host http://localhost:8000

# Validate metrics:
# - p99 latency < 10ms
# - Error rate < 0.1%
# - Throughput >= 1000 req/s
```

**3. Concurrent Request Test (Race Conditions)**

```bash
# Run concurrent writes to same tenant
PYTHONPATH=. poetry run pytest \
    tests/performance/test_quota_concurrent.py \
    -v --count=100

# Expected: No race conditions, usage = sum of writes
```

**4. Security Validation**

```bash
# Run security scan
poetry run bandit -r src/alpha_pulse/middleware/

# Expected: 0 critical, 0 high severity issues
```

**Documentation Updates:**

```markdown
# Files to update:
1. API documentation (response headers)
   - docs/api/quota-enforcement.md

2. Operational runbook (troubleshooting)
   - docs/operations/quota-troubleshooting.md

3. Architecture diagrams (C4 model)
   - docs/architecture/c4-container-view.md

4. CHANGELOG.md (release notes)
   - Added quota enforcement middleware
```

**Deliverables:**
- ✅ All tests pass (unit, integration, load, concurrent)
- ✅ Coverage report: >=90% lines, >=80% branches
- ✅ Performance validated: <10ms p99 latency
- ✅ Security scan clean: 0 critical, 0 high
- ✅ Documentation updated (API docs, runbook, CHANGELOG)

**Quality Gates:**
- [ ] Test coverage: >=90% lines, >=80% branches
- [ ] Performance: p99 < 10ms sustained
- [ ] Security: 0 critical vulnerabilities
- [ ] Documentation: Complete and accurate
- [ ] Code review: 2 approvals (1 senior)

**Handoff Criteria:**
- ✅ All quality gates passed
- ✅ Tech Lead approval for PR creation
- ✅ Ready for code review

---

## 2. Resource Allocation

### 2.1 Team Assignment

| Role | Name | Responsibility | Availability |
|------|------|----------------|--------------|
| **Tech Lead** | TBD | Architecture review, design approval, code review | 20% (spot check) |
| **Senior Developer** | TBD | Implementation (Phases 2-3), mentoring | 100% (1.5 days) |
| **QA Engineer** | TBD | Test review (Phase 1), validation (Phase 4) | 25% (spot check) |
| **Platform Engineer** | TBD | Infrastructure support (Redis, PostgreSQL) | On-demand |

### 2.2 Capacity Planning

**Total Effort**: 5 SP = 1.5 days

**Sprint Capacity**: Assuming 10 SP per sprint (2 weeks), this story consumes **50%** of one developer's sprint capacity.

**Parallel Work**: Can run in parallel with other stories (e.g., Story 4.4) as long as dependencies are met.

---

## 3. Dependencies and Blockers

### 3.1 Internal Dependencies

| Dependency | Status | Impact | Mitigation |
|------------|--------|--------|------------|
| Story 4.1: TenantCacheManager | ✅ Merged (PR #211) | None | N/A |
| Story 4.2: Database Schema | ✅ Merged (PR #212) | None | N/A |
| TenantContextMiddleware | ✅ Existing | Required for tenant_id | Verify working |
| Redis connection pool | ✅ Existing | Required for caching | Verify configuration |
| PostgreSQL async driver | ✅ Existing | Required for DB queries | Verify asyncpg installed |

### 3.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| Redis 6.0+ availability | ✅ Production | Low | Health check before deployment |
| PostgreSQL 14+ availability | ✅ Production | Low | Health check before deployment |
| FastAPI 0.100+ compatibility | ✅ Installed | Low | Pin version in pyproject.toml |

### 3.3 Blockers

**No current blockers identified.**

**Potential Blockers:**
- Infrastructure outage (Redis/PostgreSQL) - **Mitigation**: Test in staging first
- Feature freeze for production - **Mitigation**: Deploy behind feature flag
- Stakeholder sign-off delay - **Mitigation**: Proceed with implementation while awaiting

---

## 4. Risk Management

### 4.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation Strategy | Owner |
|------|------------|--------|---------------------|-------|
| Latency exceeds 10ms target | Medium | High | - Load test early (Phase 4)<br>- Optimize Redis pipelining<br>- Profile bottlenecks | Tech Lead |
| Race condition in concurrent writes | Low | Medium | - Atomic Redis INCR operations<br>- Comprehensive concurrent tests | Senior Dev |
| Redis failure impacts availability | Medium | Medium | - Fallback to PostgreSQL<br>- Health check monitoring<br>- Circuit breaker pattern | Platform Team |
| Feature flag misconfiguration | Low | Critical | - Default to disabled<br>- Staging validation<br>- Rollback plan | Tech Lead |

### 4.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Estimation too optimistic | Low | Low | - Buffer of 0.5 days built in<br>- Daily stand-up check-ins |
| Developer unavailability | Low | Medium | - Knowledge transfer to backup dev<br>- Documentation comprehensive |
| Performance tuning takes longer | Medium | Low | - Defer optimization to future story if needed<br>- Ship with basic implementation |

### 4.3 Mitigation Actions

**Pre-emptive Actions:**
1. ✅ Validate Redis/PostgreSQL connectivity before starting
2. ✅ Set up load testing environment early (Phase 1)
3. ✅ Create feature flag configuration before deployment
4. ✅ Schedule architecture review early (after Phase 1)

**Contingency Plans:**
1. **If latency >10ms**: Deploy with higher threshold (15ms), create optimization story
2. **If Redis fails**: Fallback to PostgreSQL (degraded performance acceptable short-term)
3. **If schedule slips**: Cut scope (Phase 3 REFACTOR becomes future story)

---

## 5. Quality Assurance

### 5.1 Testing Strategy

**Test Pyramid:**

```
                    ▲
                   / \
                  /   \
                 /  E2E  \         5% - End-to-end (integration)
                /---------\
               /           \
              / Integration \      15% - Service integration
             /---------------\
            /                 \
           /    Unit Tests     \   80% - Unit tests
          /---------------------\
```

**Test Coverage Targets:**

| Test Type | Target | Tool | Phase |
|-----------|--------|------|-------|
| Unit tests | >=90% lines, >=80% branches | pytest-cov | Phase 1 |
| Integration tests | All critical paths | pytest | Phase 1 |
| Load tests | 1000 req/s sustained | locust | Phase 4 |
| Concurrent tests | 100 simultaneous writes | pytest-xdist | Phase 4 |

### 5.2 Code Quality Gates

**Automated Checks (CI/CD):**

```yaml
quality_gates:
  linting:
    tool: ruff
    target: 0 errors, 0 warnings
    blocking: true

  formatting:
    tool: black
    target: All files formatted
    blocking: true

  type_checking:
    tool: mypy
    target: 0 errors
    blocking: true

  security:
    tool: bandit
    target: 0 critical, 0 high
    blocking: true

  test_coverage:
    tool: pytest-cov
    target: >=90% lines, >=80% branches
    blocking: true

  performance:
    tool: pytest-benchmark
    target: p99 < 10ms
    blocking: false  # Warning only
```

### 5.3 Review Process

**Code Review Checklist:**

- [ ] **Logic Correctness**: Algorithm matches HLD design
- [ ] **Error Handling**: All error paths handled gracefully
- [ ] **Performance**: No obvious bottlenecks (N+1 queries, etc.)
- [ ] **Security**: Input validation, no injection vulnerabilities
- [ ] **Testing**: All acceptance criteria covered by tests
- [ ] **Documentation**: Code comments, docstrings, README updated
- [ ] **Observability**: Metrics and logging added
- [ ] **Maintainability**: Code is readable, follows conventions

**Review Levels:**

1. **Peer Review** (Required): Another developer reviews logic and tests
2. **Tech Lead Review** (Required): Validates architecture alignment
3. **Security Review** (If applicable): For sensitive operations

**Approval Criteria**: Minimum 2 approvals (1 must be senior/tech lead)

---

## 6. Deployment Strategy

### 6.1 Rollout Plan

**Pre-Deployment Checklist:**

- [ ] All quality gates passed
- [ ] Code review approvals obtained
- [ ] Staging validation complete
- [ ] Feature flag configured (default: disabled)
- [ ] Monitoring dashboards ready
- [ ] Rollback plan documented
- [ ] On-call team briefed

**Rollout Phases:**

| Phase | Scope | Duration | Success Criteria | Rollback Trigger |
|-------|-------|----------|------------------|------------------|
| **Phase 1: Shadow Mode** | All tenants, logging only | 1 week | No errors, metrics baseline established | N/A |
| **Phase 2: Internal Tenant** | 1 internal test tenant | 1 week | No 429s, latency <10ms | >5% error rate |
| **Phase 3: Canary (10%)** | 10% of tenants | 1 week | <1% rejection rate | >5% rejection rate |
| **Phase 4: Gradual (50%)** | 50% of tenants | 1 week | Metrics stable | SLO violations |
| **Phase 5: Full (100%)** | All tenants | Ongoing | All tenants enforced | Critical bugs |

### 6.2 Feature Flag Configuration

**Environment Variables:**

```bash
# Development
QUOTA_ENFORCEMENT_ENABLED=true
QUOTA_ENFORCEMENT_LOG_ONLY=true  # Shadow mode

# Staging
QUOTA_ENFORCEMENT_ENABLED=true
QUOTA_ENFORCEMENT_LOG_ONLY=false

# Production (initial)
QUOTA_ENFORCEMENT_ENABLED=false  # Start disabled

# Production (Phase 1 - Shadow)
QUOTA_ENFORCEMENT_ENABLED=true
QUOTA_ENFORCEMENT_LOG_ONLY=true

# Production (Phase 2+)
QUOTA_ENFORCEMENT_ENABLED=true
QUOTA_ENFORCEMENT_LOG_ONLY=false
```

### 6.3 Monitoring and Alerts

**Grafana Dashboard Panels:**

1. **Quota Check Latency** (p50, p95, p99)
2. **Quota Rejection Rate** (per tenant)
3. **Cache Hit Rate** (quota cache)
4. **Middleware Errors** (error rate)
5. **Tenants Over Quota** (count)

**PagerDuty Alerts:**

```yaml
alerts:
  - name: QuotaCheckLatencyHigh
    condition: p99 > 15ms for 5 minutes
    severity: warning
    action: Notify on-call

  - name: QuotaRejectionRateHigh
    condition: rejection_rate > 5% for 10 minutes
    severity: critical
    action: Page on-call + escalate

  - name: MiddlewareErrors
    condition: error_rate > 1% for 5 minutes
    severity: critical
    action: Page on-call
```

### 6.4 Rollback Plan

**Rollback Triggers:**
- p99 latency >20ms sustained (>10ms SLO)
- Error rate >5% sustained
- Critical bug (false rejections, data loss)
- Stakeholder request

**Rollback Steps:**

```bash
# Step 1: Disable feature flag (instant)
kubectl set env deployment/alphapulse-api \
  QUOTA_ENFORCEMENT_ENABLED=false

# Step 2: Restart pods (< 2 minutes)
kubectl rollout restart deployment/alphapulse-api

# Step 3: Verify latency returns to baseline
# Check Grafana dashboard

# Step 4: Investigate root cause
# Review logs, metrics, profiling data

# Step 5: Fix and re-deploy
# Create hotfix PR, re-test, re-deploy
```

**Rollback SLA:** <5 minutes from trigger to disabled

---

## 7. Success Metrics

### 7.1 Technical Metrics

| Metric | Target | Measurement | Baseline |
|--------|--------|-------------|----------|
| Quota enforcement accuracy | 100% | Integration tests | N/A (new) |
| Latency impact (p99) | <10ms | Prometheus | +5ms expected |
| Test coverage | >=90% lines | pytest-cov | Current: ~85% |
| Cache hit rate | >90% | Redis MONITOR | N/A (new cache) |
| False rejection rate | 0% | Test suite + monitoring | N/A (new) |

### 7.2 Operational Metrics

| Metric | Target | Timeline | Owner |
|--------|--------|----------|-------|
| Zero production incidents | 0 incidents | First 30 days | Tech Lead |
| Quota rejection rate | <1% | First 7 days | Product |
| Support tickets (quota) | <5 tickets | First 30 days | Support |
| Developer satisfaction | >=8/10 | Post-deployment survey | Team Lead |

### 7.3 Business Metrics

| Metric | Target | Timeline | Owner |
|--------|--------|----------|-------|
| Enable usage-based billing | Ready for launch | +3 months | Product |
| Redis cost reduction | -30% | +3 months | Finance |
| Tenant satisfaction (no complaints) | >95% | +1 month | Product |

---

## 8. Handoff and Closure

### 8.1 Definition of Done

- [ ] All 4 phases complete (RED-GREEN-REFACTOR-QUALITY)
- [ ] All tests passing (unit, integration, load, concurrent)
- [ ] Code coverage >=90% lines, >=80% branches
- [ ] Performance validated: p99 <10ms sustained
- [ ] Security scan clean: 0 critical, 0 high
- [ ] Code review approved (2 approvals, 1 senior)
- [ ] Documentation updated (API docs, runbook, CHANGELOG)
- [ ] PR created and merged
- [ ] Feature flag configured (default: disabled)
- [ ] Monitoring dashboards created
- [ ] Rollout plan approved
- [ ] Tech Lead sign-off

### 8.2 Handoff Artifacts

**Code Artifacts:**
- ✅ Middleware implementation (`src/alpha_pulse/middleware/quota_enforcement.py`)
- ✅ Business logic services (`quota_checker.py`, `quota_cache_service.py`, `usage_tracker.py`)
- ✅ Test suites (unit, integration, performance)

**Documentation Artifacts:**
- ✅ API documentation (quota headers, 429 response)
- ✅ Operational runbook (troubleshooting guide)
- ✅ Architecture diagrams (updated C4 model)
- ✅ CHANGELOG.md (release notes)
- ✅ ADR-005 (architecture decision)
- ✅ HLD (this document)
- ✅ Delivery Plan (this document)

**Operational Artifacts:**
- ✅ Grafana dashboard (quota enforcement metrics)
- ✅ PagerDuty alerts (latency, rejection rate, errors)
- ✅ Feature flag configuration
- ✅ Rollback runbook

### 8.3 Knowledge Transfer

**Team Briefing Session (1 hour):**
1. Architecture overview (20 min)
2. Code walkthrough (20 min)
3. Operational runbook review (10 min)
4. Q&A (10 min)

**Attendees:**
- Development team
- SRE/Platform team
- Support team (quota troubleshooting)
- Product team (business context)

---

## 9. Timeline and Milestones

### 9.1 Gantt Chart

```
Story 4.3: Quota Enforcement Middleware (5 SP, 1.5 days)

Day 0   Day 0.5   Day 1.0   Day 1.25   Day 1.5
|-------|---------|---------|----------|---------|
[  RED  ][  GREEN  ][ REFACTOR ][ QUALITY ]
                                           [DONE]

Milestones:
▼ Phase 1 Complete: Tests written (Day 0.5)
▼ Phase 2 Complete: Implementation done (Day 1.0)
▼ Phase 3 Complete: Optimizations done (Day 1.25)
▼ Phase 4 Complete: Validation done (Day 1.5)
```

### 9.2 Critical Path

```
Phase 1 (RED) → Phase 2 (GREEN) → Phase 3 (REFACTOR) → Phase 4 (QUALITY)
     ↓               ↓                    ↓                    ↓
Tests Written   All Tests Pass   Optimized        PR Ready
(0.5 days)      (0.5 days)       (0.25 days)      (0.25 days)
```

**Total Duration**: 1.5 days (sequential phases)

**No parallel work**: Phases must be sequential per TDD methodology.

---

## 10. Appendices

### Appendix A: Related Documents

- **ADR-005**: Quota Enforcement Middleware (architectural decision)
- **HLD**: Story 4.3 High-Level Design
- **Discovery**: Story 4.3 Problem Statement and Requirements
- **Story 4.1**: TenantCacheManager (PR #211)
- **Story 4.2**: Database Schema (PR #212)

### Appendix B: Glossary

- **TDD**: Test-Driven Development (RED-GREEN-REFACTOR)
- **SP**: Story Points (Fibonacci scale: 1, 2, 3, 5, 8)
- **p99**: 99th percentile (99% of requests faster than this value)
- **TTL**: Time To Live (cache expiration)
- **INCR**: Redis atomic increment command

### Appendix C: Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-07 | 1.0 | Initial delivery plan created | Tech Lead |

---

## Approval Sign-Off

- [ ] **Tech Lead**: Delivery plan approved
- [ ] **Product Manager**: Scope and timeline approved
- [ ] **Engineering Manager**: Resource allocation approved

**Status**: ⏳ **READY FOR EXECUTION**

**Next Action**: Begin Phase 1 (RED) - Write failing tests

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Tech Lead
