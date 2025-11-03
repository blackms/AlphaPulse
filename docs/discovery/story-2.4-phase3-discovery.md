# Story 2.4 Phase 3: Tenant Context Integration - Discovery Document

**Epic:** EPIC-002 - Multi-Tenant SaaS Infrastructure
**Story:** Story 2.4 - Update API routes with tenant context
**Phase:** Phase 3 (P2 Priority - High)
**Created:** 2025-11-03
**Status:** Draft

---

## Executive Summary

**Problem Statement:**
11 P2-priority API endpoints across 2 routers (ensemble, hedging) lack tenant context integration, creating security vulnerabilities and tenant data isolation risks. Critical security gap: hedging.py has NO authentication whatsoever.

**Proposed Solution:**
Integrate `get_current_tenant_id` dependency injection following the proven Phase 1 & 2 pattern. Add missing authentication to hedging.py endpoints.

**RICE Score: 64.0 (CRITICAL)**
- **Reach:** 8/10 (Affects ensemble management + hedging operations)
- **Impact:** 10/10 (Security gap + core multi-tenant functionality)
- **Confidence:** 80% (Proven pattern from Phases 1 & 2)
- **Effort:** 1 sprint (11 endpoints, estimated 6.5 SP)

---

## 1. Problem Discovery

### 1.1 Current State Analysis

**Completed Work:**
- ✅ Phase 0 (P0): 14 critical endpoints (risk, risk_budget, portfolio)
- ✅ Phase 1 (P1): 11 high-priority endpoints (alerts, metrics, system, trades, correlation)
- **Total completed:** 25/43 endpoints (58%)

**Remaining Work - Phase 3 (P2 Priority):**
| Router | Endpoints | Current State | Security Risk |
|--------|-----------|---------------|---------------|
| **ensemble.py** | 8 | Has `get_current_user`, missing tenant context | MEDIUM - Data isolation only |
| **hedging.py** | 3 | **NO AUTHENTICATION AT ALL** | **CRITICAL - Unauthenticated access** |
| **TOTAL** | **11** | Mixed authentication state | **CRITICAL** |

### 1.2 Security Gaps Identified

**CRITICAL - hedging.py:**
```python
# Current state - NO AUTHENTICATION
@router.get("/analysis")
async def analyze_hedge_positions(
    exchange_client = Depends(get_exchange_client),  # No user/tenant check!
    api_client = Depends(get_api_client)
):
    # Any anonymous user can access hedge data
```

**Impact:**
- Unauthenticated users can view hedge positions
- Unauthenticated users can execute hedge adjustments
- Unauthenticated users can close all hedges
- Potential for malicious trading activity
- Regulatory compliance violation

**MEDIUM - ensemble.py:**
- Has user authentication via `get_current_user`
- Missing tenant isolation - users might access other tenants' ensembles
- Audit logs incomplete without tenant context

### 1.3 User Pain Points

**Current Issues:**
1. **Security team**: Hedging endpoints flagged in security audit as critical vulnerability
2. **Compliance team**: Missing audit trail for tenant operations
3. **Engineering team**: Inconsistent auth pattern across routers
4. **Operations team**: Cannot trace ensemble/hedging operations by tenant

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

**FR-1: Tenant Context Extraction**
- All 11 endpoints MUST extract `tenant_id` from JWT via middleware
- Pattern consistency: `tenant_id: str = Depends(get_current_tenant_id)`

**FR-2: Authentication Addition (hedging.py)**
- Add `get_current_user` dependency to ALL hedging endpoints
- Add `get_current_tenant_id` dependency to ALL hedging endpoints
- Maintain parameter order: tenant_id → user → other dependencies

**FR-3: Tenant Isolation**
- Ensemble operations scoped to creating tenant
- Hedge operations scoped to tenant's exchange account
- List operations filtered by tenant_id

**FR-4: Audit Logging**
- All operations log tenant context: `[Tenant: {tenant_id}]`
- Include operation type, user, timestamp
- Log both success and failure cases

### 2.2 Non-Functional Requirements

**NFR-1: Performance**
- p99 latency increase < 5ms (JWT parsing overhead)
- No degradation in ensemble prediction throughput
- Hedge execution latency unaffected

**NFR-2: Backward Compatibility**
- Existing ensemble configurations remain functional
- No breaking changes to request/response schemas
- Service layer changes transparent to clients

**NFR-3: Security**
- Zero authentication bypass paths
- Tenant data leakage impossible via API layer
- Rate limiting per tenant (future enhancement)

**NFR-4: Observability**
- Tenant context in all log messages
- Metrics tagged by tenant_id
- Error traces include tenant information

---

## 3. Technical Feasibility

### 3.1 Solution Options Evaluated

**Option A: Reuse Phase 1/2 Pattern (RECOMMENDED)**
- **Pros:**
  - Proven approach (25 endpoints already migrated)
  - 100% pattern consistency across codebase
  - Minimal risk, well-understood
  - Fast implementation
- **Cons:**
  - None identified
- **Score:** 50/50

**Option B: Custom Ensemble-Specific Auth**
- **Pros:**
  - Could optimize for ensemble workload patterns
- **Cons:**
  - Increases code complexity
  - Pattern inconsistency
  - Higher maintenance burden
  - No clear benefit over Option A
- **Score:** 25/50

**Option C: Defer to Service Layer**
- **Pros:**
  - Router layer stays thin
- **Cons:**
  - Inconsistent with Phase 1/2
  - Doesn't fix hedging.py security gap at API boundary
  - Harder to audit API access
- **Score:** 20/50

**Decision: Option A - Reuse Phase 1/2 Pattern**

### 3.2 Implementation Approach

**Pattern to Apply (from risk.py):**
```python
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
from ..dependencies import get_current_user

@router.post("/create")
async def create_ensemble(
    request: CreateEnsembleRequest,
    tenant_id: str = Depends(get_current_tenant_id),  # ADD
    user = Depends(get_current_user),
    service = Depends(get_ensemble_service)
) -> Dict[str, Any]:
    """Create new ensemble configuration."""
    try:
        logger.info(f"[Tenant: {tenant_id}] Creating ensemble: {request.config}")  # ADD
        ensemble_id = await service.create_ensemble(
            config=request.config,
            tenant_id=tenant_id  # PASS TO SERVICE
        )
        logger.info(f"[Tenant: {tenant_id}] Ensemble created: {ensemble_id}")  # ADD
        return {"ensemble_id": ensemble_id}
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Error creating ensemble: {e}")  # ADD
        raise HTTPException(status_code=500, detail=str(e))
```

**Changes per endpoint:**
1. Add import: `from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id`
2. Add parameter: `tenant_id: str = Depends(get_current_tenant_id)`
3. Add logging: `logger.info(f"[Tenant: {tenant_id}] ...")`
4. Pass to service: `service.method(..., tenant_id=tenant_id)`

### 3.3 Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Service layer not ready for tenant_id | Medium | High | Use Task tool to audit service methods first |
| Breaking existing ensemble configs | Low | High | Add backward compat layer in service |
| Hedge execution failures during migration | Medium | Critical | Deploy with feature flag, gradual rollout |
| Performance degradation | Low | Medium | Load test before production |

---

## 4. Scope Definition

### 4.1 In Scope - Phase 3 (P2)

**ensemble.py (8 endpoints):**
1. POST `/create` - Create ensemble configuration
2. POST `/{ensemble_id}/register-agent` - Register agent to ensemble
3. POST `/{ensemble_id}/predict` - Get ensemble prediction
4. GET `/{ensemble_id}/performance` - Get performance metrics
5. GET `/{ensemble_id}/weights` - Get agent weights
6. POST `/{ensemble_id}/optimize-weights` - Optimize weights
7. GET `/` - List ensembles (tenant-filtered)
8. GET `/agent-rankings` - Get agent rankings (tenant-scoped)

**hedging.py (3 endpoints):**
1. GET `/analysis` - Analyze hedge positions (**+AUTH**)
2. POST `/execute` - Execute hedge adjustments (**+AUTH**)
3. POST `/close` - Close all hedges (**+AUTH**)

**Total:** 11 endpoints

### 4.2 Out of Scope

**Deferred to Phase 4 (P3):**
- regime.py (4 endpoints)
- positions.py (3 endpoints)
- liquidity.py (7 endpoints)

**Not in Story 2.4:**
- Service layer refactoring (separate story)
- Database RLS policy updates (covered in EPIC-002 database story)
- Frontend changes (separate epic)

### 4.3 Acceptance Criteria

**AC-1: Code Changes**
- [ ] All 11 endpoints have `get_current_tenant_id` dependency
- [ ] All hedging endpoints have `get_current_user` dependency (security fix)
- [ ] All endpoints log tenant context
- [ ] 100% pattern consistency with Phases 1 & 2

**AC-2: Testing**
- [ ] Integration tests for all 11 endpoints
- [ ] Tenant isolation validated (cannot access other tenant data)
- [ ] Authentication validated (cannot access without valid JWT)
- [ ] Error handling includes tenant context

**AC-3: Documentation**
- [ ] Discovery document (this file)
- [ ] HLD with architecture diagrams
- [ ] Delivery plan with TDD tasks
- [ ] CHANGELOG.md updated

**AC-4: Quality Gates**
- [ ] All tests passing (pytest)
- [ ] No security vulnerabilities (bandit scan)
- [ ] Code coverage maintained (>90%)
- [ ] Pattern consistency check (manual review)

---

## 5. Success Metrics

### 5.1 Leading Indicators

**Implementation Metrics:**
- Endpoints migrated: Target 11/11 (100%)
- Tests written: Target 11 test classes
- Pattern consistency: Target 100% (matches Phase 1/2)
- Code review approval: Required before merge

**Security Metrics:**
- Authentication gaps closed: 3/3 (hedging.py)
- Unauthenticated endpoints: Target 0
- Tenant isolation gaps: Target 0

### 5.2 Lagging Indicators (Post-Deployment)

**Performance Metrics:**
- p50 latency increase: < 2ms
- p99 latency increase: < 5ms
- Error rate change: < 0.1% increase

**Security Metrics:**
- Unauthorized access attempts: 0 (blocked by middleware)
- Audit log completeness: 100% (all operations logged)
- Tenant data leakage incidents: 0

**Operational Metrics:**
- Ensemble creation success rate: > 99%
- Hedge execution success rate: > 99%
- Rollback rate: < 1%

---

## 6. Dependencies & Constraints

### 6.1 Technical Dependencies

**Required Infrastructure:**
- ✅ JWT middleware (`get_current_tenant_id`)
- ✅ Tenant context middleware (from Phase 1)
- ✅ FastAPI dependency injection
- ⚠️ Service layer may need updates (to be verified)

**Blocking Dependencies:**
- None (all prerequisites completed in Phase 1)

### 6.2 Timeline Constraints

**Sprint Capacity:**
- Available: 1 sprint (2 weeks)
- Estimated effort: 6.5 SP
- Buffer: 30% (for service layer adjustments)

**Critical Path:**
1. Service layer audit (0.5 days)
2. ensemble.py implementation (1.5 days)
3. hedging.py implementation + security fix (1 day)
4. Integration tests (1.5 days)
5. Documentation + review (0.5 days)

**Total: 5 days (within 1 sprint)**

### 6.3 Resource Constraints

**Team:**
- 1 backend engineer (Claude Code)
- 1 reviewer (user approval)
- No QA bandwidth (rely on automated tests)

**Tooling:**
- FastAPI dependency injection (available)
- pytest for integration tests (available)
- GitHub Actions CI/CD (available)

---

## 7. Stakeholder Sign-Off

**Business Stakeholders:**
- [ ] Product Owner: Approves scope and priority
- [ ] Security Team: Approves security fix approach
- [ ] Compliance Team: Approves audit logging approach

**Technical Stakeholders:**
- [ ] Tech Lead: Approves architecture decision
- [ ] Backend Team: Approves implementation pattern
- [ ] DevOps Team: Approves deployment approach

**Sign-off Date:** _[To be filled after review]_

---

## 8. Next Steps

1. ✅ **Discovery Complete** - This document
2. ⏭️ **Design Phase** - Create HLD with architecture views
3. ⏭️ **Planning Phase** - Create delivery plan with TDD tasks
4. ⏭️ **Implementation** - Execute TDD cycle (RED → GREEN → REFACTOR)
5. ⏭️ **Review** - Code review and testing
6. ⏭️ **Release** - Merge to main and deploy

**Status:** Discovery phase complete, ready for design phase.

---

## Appendix

### A. Related Documents
- Story 2.4 Phase 1 Discovery: `docs/discovery/story-2.4-phase1-discovery.md`
- Story 2.4 Phase 2 Discovery: `docs/discovery/story-2.4-phase2-discovery.md`
- EPIC-002 Overview: GitHub Issue #141

### B. RICE Score Calculation

**Reach:** 8/10
- Affects all users managing ensembles (primary feature)
- Affects all users executing hedges (risk management feature)
- Higher reach than Phase 2 due to ensemble importance

**Impact:** 10/10
- **CRITICAL security gap** in hedging.py (no auth)
- Core multi-tenant functionality
- Regulatory compliance requirement

**Confidence:** 80%
- Pattern proven in Phases 1 & 2 (25 endpoints)
- Only uncertainty: service layer readiness

**Effort:** 1 sprint
- 11 endpoints × 0.5 days avg = 5.5 days
- 30% buffer = 7 days total
- Fits in 1 sprint (10 working days)

**RICE = (8 × 10 × 0.8) / 1 = 64.0**

**Priority: CRITICAL** (highest of all phases due to security gap)

### C. Glossary
- **Ensemble:** Collection of AI agents with weighted voting
- **Hedge:** Risk mitigation position (offsetting trade)
- **RLS:** Row-Level Security (database-level tenant isolation)
- **P0/P1/P2/P3:** Priority levels (P0=Critical, P3=Low)
