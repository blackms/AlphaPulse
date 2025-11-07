# Story 4.3: Quota Enforcement Middleware - Discovery

**Epic**: EPIC-004 (Caching Layer)
**Story Points**: 5 SP
**Status**: Discovery
**Date**: 2025-11-07
**Owner**: Tech Lead

## Problem Statement

### Current State

We have implemented:
- ✅ **Story 4.1**: Multi-tenant cache namespace isolation (`TenantCacheManager`)
- ✅ **Story 4.2**: Database schema for quota tracking (`tenant_cache_quotas`, `tenant_cache_metrics`)

**Gap**: There is **no enforcement mechanism** at the API layer to prevent tenants from exceeding their cache quotas. Tenants can currently write unlimited data to cache, causing:

1. **Cost overruns**: Excessive Redis memory usage
2. **Resource exhaustion**: Large tenants can starve small tenants
3. **No billing accountability**: Can't charge for actual usage
4. **Poor capacity planning**: No visibility into quota violations

### User Impact

**Affected Personas:**
- **System Administrators**: Can't control tenant resource consumption
- **Finance Team**: Can't bill accurately for cache usage
- **Platform Engineers**: Redis capacity planning is guesswork
- **Tenant Users**: Large tenants can degrade performance for all

**Pain Points:**
- "We have no way to stop a runaway tenant from filling Redis" - SRE Team
- "We can't bill fairly when we don't enforce quotas" - Finance
- "Small tenants complain about slow cache because large tenants hog memory" - Support

## Desired Outcome

### Vision

**"Every API request that writes to cache is quota-checked, enforced, and metered in <10ms with zero false denials."**

### Success Metrics

**Primary Metrics:**
1. **Quota Enforcement Accuracy**: 100% of cache writes checked against quota
2. **Performance**: Quota check latency <10ms (p99)
3. **Reliability**: Zero false rejections (legitimate writes blocked)
4. **Observability**: Real-time quota usage visible per tenant

**Secondary Metrics:**
1. **Overage Rate**: <5% of tenants exceed quota monthly
2. **Quota Rejection Rate**: <1% of requests rejected (well-tuned quotas)
3. **Cache Hit Rate**: Maintained >80% (quota doesn't degrade caching)
4. **Tenant Satisfaction**: No complaints about quota enforcement

### Business Value

**Revenue Impact:**
- Enable usage-based billing for cache tier ($1000/mo potential)
- Reduce Redis infrastructure costs by 30% (prevent waste)

**Operational Impact:**
- Capacity planning accuracy improves to ±10%
- Support tickets related to "slow cache" reduce by 50%

**Risk Mitigation:**
- Prevent tenant abuse (runaway scripts filling cache)
- Fair resource allocation (no single tenant dominates)

## Requirements

### Functional Requirements

**FR-1: Quota Check Before Write**
- **Description**: Every cache write operation must check tenant quota before proceeding
- **Acceptance Criteria**:
  - ✅ Middleware intercepts cache write requests
  - ✅ Quota retrieved from Redis cache (5-min TTL) or PostgreSQL
  - ✅ Write proceeds only if quota allows

**FR-2: Three-Level Enforcement**
- **Description**: Support soft limits (warn) and hard limits (reject)
- **Acceptance Criteria**:
  - ✅ **ALLOW**: Usage ≤ quota → Write proceeds normally
  - ✅ **WARN**: quota < Usage ≤ (quota + overage) → Write proceeds with warning header
  - ✅ **REJECT**: Usage > (quota + overage) → HTTP 429, write blocked

**FR-3: Real-Time Usage Tracking**
- **Description**: Update tenant usage atomically after each write
- **Acceptance Criteria**:
  - ✅ Redis INCR used for atomic counter updates
  - ✅ Usage synced to PostgreSQL every 5 minutes
  - ✅ Rollback on rejection (usage not counted)

**FR-4: Quota Status Headers**
- **Description**: API responses include quota status
- **Acceptance Criteria**:
  - ✅ Headers: `X-Cache-Quota-Limit`, `X-Cache-Quota-Used`, `X-Cache-Quota-Remaining`
  - ✅ Header: `X-Cache-Quota-Status` (ok|warning|exceeded)
  - ✅ `Retry-After` header on 429 responses

**FR-5: Graceful Degradation**
- **Description**: Quota enforcement failure should not break API
- **Acceptance Criteria**:
  - ✅ Fallback to PostgreSQL if Redis unavailable
  - ✅ Allow writes on quota check failure (fail-open with logging)
  - ✅ Feature flag to disable enforcement

### Non-Functional Requirements

**NFR-1: Performance**
- **Target**: Quota check adds <10ms latency (p99)
- **Validation**: Load testing at 1000 req/s per tenant

**NFR-2: Scalability**
- **Target**: Support 1000+ concurrent requests per tenant
- **Validation**: Concurrent write test with quota enforcement

**NFR-3: Reliability**
- **Target**: 99.9% quota check success rate
- **Validation**: Monitor quota check failures over 7 days

**NFR-4: Observability**
- **Target**: Real-time metrics on quota checks, rejections, overages
- **Validation**: Grafana dashboard showing quota metrics

**NFR-5: Security**
- **Target**: Quota enforcement can't be bypassed
- **Validation**: Penetration test attempts to bypass middleware

## Technical Feasibility

### Assessment

✅ **FEASIBLE** - Standard middleware pattern with existing components

**Confidence Level**: High (90%)

**Evidence:**
- FastAPI middleware is well-documented and tested
- Redis atomic operations (INCR) proven at scale
- Similar pattern used successfully in API rate limiting

**Dependencies:**
- ✅ Story 4.1: TenantCacheManager (complete)
- ✅ Story 4.2: Database schema (complete)
- ✅ Redis available with INCR support
- ✅ FastAPI version supports async middleware

### Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Quota check adds >10ms latency | High | Medium | Redis caching, load testing before rollout |
| Race condition on quota boundary | Medium | Low | Redis atomic INCR operations |
| Cache stampede on quota reload | Medium | Low | Single-flight pattern for cache misses |
| Middleware misconfiguration blocks all requests | High | Low | Feature flag, comprehensive testing, staged rollout |

### ADR Reference

📄 **ADR-005: Quota Enforcement Middleware** (docs/adr/ADR-005-quota-enforcement-middleware.md)

**Decision**: Use FastAPI middleware with two-tier caching (Redis + PostgreSQL)

**Key Trade-offs:**
- ✅ Performance: Redis cache keeps latency <10ms
- ⚠️ Staleness: Quota changes take up to 5 minutes to propagate
- ✅ Reliability: Fallback to PostgreSQL on Redis failure

## Scope

### In Scope

✅ **Middleware Implementation**:
- FastAPI ASGI middleware class
- Quota check logic (three-level enforcement)
- Redis cache integration for quota
- Atomic usage tracking with Redis INCR
- Response header injection

✅ **Configuration**:
- Feature flag for quota enforcement
- Configurable cache TTL (default 5 minutes)
- Configurable enforcement thresholds

✅ **Testing**:
- Unit tests for middleware logic
- Integration tests with Redis and PostgreSQL
- Load tests (1000 req/s concurrent writes)
- Quota boundary tests (race conditions)

✅ **Observability**:
- Metrics: quota_checks_total, quota_rejections_total, quota_warnings_total
- Metrics: quota_check_latency_ms (histogram)
- Logging: Quota violations with tenant_id

### Out of Scope

❌ **Quota Reset Scheduling**: Handled in Story 4.5 (Metrics Collection Job)
❌ **Billing Integration**: Future epic (pricing tier management)
❌ **Admin UI for Quota Management**: Future epic (tenant portal)
❌ **Read Operation Quotas**: Only write operations enforced (reads are free)
❌ **Network Bandwidth Quotas**: Only cache storage quotas enforced

## Acceptance Criteria

**AC-1: Quota Check on Every Write**
- GIVEN a tenant with quota_mb=100 and current_usage_mb=90
- WHEN the tenant attempts a 5MB cache write
- THEN quota is checked and write is allowed (95 < 100)

**AC-2: Warning on Overage**
- GIVEN a tenant with quota_mb=100, overage_limit_mb=10, and current_usage_mb=95
- WHEN the tenant attempts a 10MB cache write
- THEN write is allowed but response has `X-Cache-Quota-Status: warning`

**AC-3: Rejection on Hard Limit**
- GIVEN a tenant with quota_mb=100, overage_limit_mb=10, and current_usage_mb=105
- WHEN the tenant attempts a 10MB cache write
- THEN write is rejected with HTTP 429 and usage is NOT incremented

**AC-4: Response Headers Populated**
- GIVEN any cache write request
- WHEN quota check completes
- THEN response includes X-Cache-Quota-* headers with current quota status

**AC-5: Performance SLO Met**
- GIVEN 1000 concurrent cache write requests
- WHEN quota enforcement is enabled
- THEN p99 latency increase is <10ms

**AC-6: Atomic Usage Tracking**
- GIVEN 100 concurrent writes to same tenant
- WHEN all writes are within quota
- THEN usage counter increments by exactly sum of write sizes (no double-counting)

**AC-7: Graceful Degradation**
- GIVEN Redis is unavailable
- WHEN quota check is attempted
- THEN fallback to PostgreSQL succeeds OR write is allowed with error logging

**AC-8: Feature Flag Control**
- GIVEN feature flag `quota_enforcement_enabled=false`
- WHEN cache write is attempted
- THEN quota check is skipped and write proceeds normally

## Estimation

### Effort Breakdown

**Story Points**: 5 SP (~1.5 days)

**Tasks:**
1. **Middleware Implementation** (2 SP)
   - Create `QuotaEnforcementMiddleware` class
   - Implement quota check logic
   - Redis cache integration
   - Response header injection

2. **Testing** (2 SP)
   - Unit tests (quota scenarios)
   - Integration tests (Redis + PostgreSQL)
   - Load tests (concurrent writes)
   - Race condition tests

3. **Observability** (0.5 SP)
   - Add Prometheus metrics
   - Logging for quota violations
   - Grafana dashboard update

4. **Documentation** (0.5 SP)
   - API documentation (quota headers)
   - Operational runbook (quota troubleshooting)
   - Update ADR-005 with implementation notes

### Three-Point Estimation

- **Optimistic**: 4 SP (1 day) - No issues, tests pass first try
- **Likely**: 5 SP (1.5 days) - Normal development flow
- **Pessimistic**: 7 SP (2 days) - Performance tuning needed, race conditions found

**Confidence**: 80% (standard middleware pattern, low unknowns)

## Dependencies

**Upstream (Blockers):**
- ✅ Story 4.1: TenantCacheManager (merged in PR #211)
- ✅ Story 4.2: Database schema (merged in PR #212)

**Downstream (Dependent Stories):**
- ⏳ Story 4.4: Shared Market Data Cache (needs quota enforcement)
- ⏳ Story 4.5: Metrics Collection Job (syncs usage to DB)

**External Dependencies:**
- ✅ Redis 6.0+ (INCR, INCRBYFLOAT support)
- ✅ PostgreSQL 14+ (tenant_cache_quotas table)
- ✅ FastAPI 0.100+ (async middleware support)

## Questions & Assumptions

### Questions

❓ **Q1**: Should read operations (cache hits) count against quota?
- **Answer Needed From**: Product Manager
- **Impact**: High (changes scope significantly)
- **Assumption**: NO - Only writes count (reads are free)

❓ **Q2**: What happens when tenant quota is reduced below current usage?
- **Answer Needed From**: Product + Tech Lead
- **Impact**: Medium (affects enforcement logic)
- **Assumption**: Existing cache remains, new writes rejected until usage drops

❓ **Q3**: Should we support per-resource quotas (not just total cache)?
- **Answer Needed From**: Product Manager
- **Impact**: High (changes data model)
- **Assumption**: NO - Single total quota per tenant (Story 4.3 scope)

### Assumptions

✅ **A1**: Quota changes propagate within 5 minutes (acceptable for business)
✅ **A2**: Redis availability is >99.9% (fallback to PostgreSQL rare)
✅ **A3**: Tenants are billed monthly (quota resets handled in Story 4.5)
✅ **A4**: Overage allowance is 10-20% of base quota (configurable per tenant)

## Next Steps

### Immediate Actions (This Phase)

1. ✅ **Create ADR-005** - Quota Enforcement Middleware
2. ✅ **Create Discovery Document** - This document
3. ⏳ **Architecture Review** - Tech Lead + Senior Engineers
4. ⏳ **Stakeholder Alignment** - Confirm assumptions with Product
5. ⏳ **Quality Gate Check** - Problem statement validated, feasibility confirmed

### Phase 2: Design the Solution

After **Phase 1 Quality Gates** pass:
1. Create HLD (High-Level Design) document
2. Create Delivery Plan (sequencing, capacity allocation)
3. Architecture review meeting
4. Design approval sign-off

---

## Quality Gates (Phase 1: Discover & Frame)

- [x] **Problem statement validated** - Clear success metrics, bounded scope
- [x] **Technical feasibility confirmed** - No blockers, risks documented
- [x] **ADR created** - ADR-005 documents architectural decision
- [ ] **Stakeholder alignment** - Product confirms assumptions (Q1, Q2, Q3)
- [ ] **Estimation approved** - Tech Lead confirms 5 SP estimate
- [ ] **Go/No-Go for Design Phase** - Awaiting sign-off

**Status**: ⏳ **READY FOR REVIEW**

**Tech Lead Sign-Off**: _______________
**Product Manager Sign-Off**: _______________

---

**References:**
- ADR-005: docs/adr/ADR-005-quota-enforcement-middleware.md
- Story 4.1 (PR #211): Tenant Namespace Layer
- Story 4.2 (PR #212): Quota Management Schema
- LIFECYCLE-ORCHESTRATOR-ENHANCED-PROTO.yaml: Phase 1 (Discover & Frame)
