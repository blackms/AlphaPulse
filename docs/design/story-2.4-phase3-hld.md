# Story 2.4 Phase 3: High-Level Design

**Date**: 2025-11-03
**Story**: EPIC-002 Story 2.4 Phase 3
**Phase**: Design the Solution
**Status**: In Review
**Version**: 1.0

---

## Executive Summary

This document details the high-level design for integrating tenant context into 11 P2 priority API endpoints across 2 routers (ensemble, hedging). Phase 3 completes the multi-tenant security architecture initiated in Phases 1 & 2.

**Key Decision**: Reuse Phase 1/2 pattern with CRITICAL security fix for hedging.py
**Complexity**: LOW
**Risk Level**: MEDIUM (due to authentication gap in hedging.py)
**Estimated Effort**: 2-3 days
**RICE Score**: 64.0 (CRITICAL - highest of all phases)

**Critical Security Finding**: hedging.py endpoints currently lack ANY authentication. Phase 3 MUST add both `get_current_user` AND `get_current_tenant_id` dependencies to establish baseline security.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 3 Scope](#phase-3-scope)
3. [Security Fix - Critical](#security-fix---critical)
4. [Architecture Blueprint](#architecture-blueprint)
5. [Decision Log](#decision-log)
6. [Validation Strategy](#validation-strategy)
7. [Quality Gates](#quality-gates)

---

## Phase 3 Scope

### Endpoints to Migrate

**Ensemble Router (8 endpoints)**:
1. `POST /ensemble/create` - Create ensemble configuration
2. `POST /ensemble/{ensemble_id}/register-agent` - Register agent with ensemble
3. `POST /ensemble/{ensemble_id}/predict` - Get ensemble prediction
4. `GET /ensemble/{ensemble_id}/performance` - Get ensemble performance metrics
5. `GET /ensemble/{ensemble_id}/weights` - Get agent weights
6. `POST /ensemble/{ensemble_id}/optimize-weights` - Optimize agent weights
7. `GET /ensemble/` - List all ensembles
8. `GET /ensemble/agent-rankings` - Get agent performance rankings

**Hedging Router (3 endpoints)**:
1. `GET /hedging/analysis` - Analyze hedge positions
2. `POST /hedging/execute` - Execute hedge adjustments
3. `POST /hedging/close` - Close all hedge positions

**Total**: 11 endpoints (8 + 3)

### Pattern Applied

Same as Phases 1 & 2:
- Add `tenant_id: str = Depends(get_current_tenant_id)` parameter
- Add logging: `logger.info(f"[Tenant: {tenant_id}] ...")`
- Pass tenant_id to service layer for data scoping
- No response model changes (tenant_id remains internal)

---

## Security Fix - Critical

### Hedging.py Authentication Gap

**Current State** (Lines 41-46, 99-102, 168-171 in hedging.py):
```python
# CRITICAL: No authentication at all!
@router.get(
    "/analysis",
    response_model=HedgeAnalysisResponse,
    dependencies=[Depends(get_api_client)]  # ← Only get_api_client, no auth
)
async def analyze_hedge_positions(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Analyze current hedge positions and provide recommendations."""
```

**Problem**:
- Zero authentication enforcement
- Any unauthenticated request can access hedge analysis/execution
- Critical trading operations exposed without authorization
- No tenant isolation possible (no tenant_id available)

**Phase 3 Fix**:
```python
# SECURE: Full authentication with tenant context
@router.get(
    "/analysis",
    response_model=HedgeAnalysisResponse,
    dependencies=[Depends(get_api_client)]
)
async def analyze_hedge_positions(
    tenant_id: str = Depends(get_current_tenant_id),  # ← ADD: Tenant context
    current_user = Depends(get_current_user),         # ← ADD: User authentication
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Analyze current hedge positions and provide recommendations."""
    logger.info(f"[Tenant: {tenant_id}] Analyzing hedge positions")
    # ... rest of endpoint
```

### Security Implementation Diagram

```
BEFORE Phase 3 (Hedging.py):
┌──────────────┐
│ Any Request  │ (NO VALIDATION)
└──────┬───────┘
       │
       │ HTTP GET /hedging/analysis
       │ NO Authorization header required
       │ NO tenant_id required
       │
       ▼
┌─────────────────────────────────────┐
│ analyze_hedge_positions() endpoint   │
│  ❌ No get_current_user dependency   │
│  ❌ No get_current_tenant_id         │
│  ❌ No authentication check          │
│  ❌ No tenant isolation              │
│  ✅ Only get_api_client (weak)       │
└─────────────────────────────────────┘
       │
       ▼ (UNRESTRICTED ACCESS)
  Get hedge analysis data
  Execute trades without auth
  Close positions without approval


AFTER Phase 3 (Hedging.py):
┌──────────────────────────────────┐
│ Authenticated Request             │
│ Authorization: Bearer <JWT_TOKEN> │
│ (Contains tenant_id + user claims)│
└──────┬───────────────────────────┘
       │
       │ HTTP GET /hedging/analysis
       │ Authorization header REQUIRED
       │ JWT validation REQUIRED
       │
       ▼
┌──────────────────────────────────────┐
│ Depends(get_current_user)            │
│ - Validates JWT signature             │
│ - Checks user permissions             │
│ - Validates user is active            │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Depends(get_current_tenant_id)       │
│ - Extracts tenant_id from JWT        │
│ - Validates tenant context           │
│ - Sets request.state.tenant_id       │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ analyze_hedge_positions() endpoint    │
│  ✅ User authenticated                │
│  ✅ Tenant isolated                   │
│  ✅ Tenant context logged             │
│  ✅ Data scoped to tenant             │
└──────┬───────────────────────────────┘
       │
       ▼ (RESTRICTED ACCESS)
  Get hedge analysis data (tenant-scoped)
  Execute trades (with user approval)
  Close positions (with tenant authorization)
```

### Imports to Add

In `hedging.py`, add at line 23 (after existing imports):
```python
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
from ..dependencies import get_current_user  # Already available
```

---

## Architecture Blueprint

### Architectural Patterns

Phase 3 follows the exact same pattern as Phases 1 & 2. For detailed architecture (context view, container view, deployment view), **reference** `/Users/a.rocchi/Projects/Personal/AlphaPulse/docs/design/story-2.4-phase2-hld.md` sections:

- **Context View** (Phase 2 HLD, lines 114-167)
- **Container View** (Phase 2 HLD, lines 171-233)
- **Component View** (Phase 2 HLD, lines 235-290)
- **Runtime View** (Phase 2 HLD, lines 292-333)
- **Deployment View** (Phase 2 HLD, lines 335-369)

**No architectural changes in Phase 3** - JWT middleware, tenant context extraction, and data scoping mechanisms remain identical.

### Phase 3-Specific: Ensemble Router Tenant Isolation

**Ensemble Service Integration**:
```
┌─────────────────────────────────┐
│ Client Request                   │
│ POST /ensemble/create            │
│ Authorization: Bearer <JWT>      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Dependencies Injection           │
│ 1. get_current_user()           │
│ 2. get_current_tenant_id()      │
│ 3. get_ensemble_service()       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Endpoint Handler                │
│ async def create_ensemble(      │
│   request: CreateEnsembleRequest│
│   tenant_id: str                │ ← Tenant context
│   user: User                    │ ← User authentication
│   service: EnsembleService      │
│ ):                              │
└────────┬────────────────────────┘
         │
         │ logger.info(f"[Tenant: {tenant_id}] Creating ensemble...")
         │
         ▼
┌─────────────────────────────────┐
│ Service Layer                    │
│ service.create_ensemble(         │
│   config=config,               │
│   tenant_id=tenant_id          │ ← Pass tenant context
│ )                               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Data Layer                       │
│ Query: INSERT INTO ensembles    │
│   WHERE tenant_id = 'tenant-abc'│ ← Tenant scoped
│   AND created_by = user_id      │
└─────────────────────────────────┘
```

### Phase 3-Specific: Hedging Router Security Fix

**Hedging Authentication Flow** (NEW in Phase 3):
```
┌──────────────────────────────────┐
│ Request with JWT Token           │
│ GET /hedging/analysis            │
│ Authorization: Bearer <JWT>      │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ TenantContextMiddleware          │
│ 1. Extract Authorization header  │
│ 2. Decode JWT token              │
│ 3. Extract tenant_id claim       │
│ 4. Extract user_id claim         │
│ 5. Set request.state.tenant_id   │
│ 6. Set request.state.user_id     │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Depends(get_current_user)        │ ← NEW: Auth check
│ - Validates JWT signature        │
│ - Checks user permissions        │
│ - Returns User object            │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Depends(get_current_tenant_id)   │ ← NEW: Tenant check
│ - Retrieves request.state.tenant_id
│ - Validates tenant_id exists     │
│ - Returns tenant_id string       │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ analyze_hedge_positions()        │ ← NOW PROTECTED
│ - User authenticated             │
│ - Tenant isolated                │
│ - Hedge analysis scoped to tenant│
└──────────────────────────────────┘
```

---

## Decision Log

### Decision 1: Reuse Phase 1/2 Pattern vs Custom Approach

**Description**: Apply identical `Depends(get_current_tenant_id)` pattern from Phases 1 & 2 to all 11 Phase 3 endpoints, plus add missing authentication to hedging.py.

**Pros**:
- ✅ Proven pattern (75+ tests passing across Phases 1 & 2)
- ✅ Consistent codebase (same pattern everywhere)
- ✅ Minimal risk (no new abstractions)
- ✅ Fast implementation (pattern reuse)
- ✅ Leverages existing middleware and auth infrastructure
- ✅ Security improvement aligns with risk-first approach

**Cons**:
- ❌ Requires modifying 11 endpoint signatures (manual work)
- ❌ Test duplication (2 new test files)
- ❌ hedging.py requires extra care (authentication gap)

**Alternative Considered**: Custom hedging authentication decorator
- Would abstract security fix, but increases complexity
- Inconsistent with Phase 1/2 (maintenance burden)
- Rejected: Single-point patterns are preferred for security

**Trade-Off Matrix**:

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Complexity** | 9/10 | Lowest complexity option |
| **Cost** | 10/10 | Zero infrastructure cost |
| **Time-to-Market** | 9/10 | 2-3 days (fastest) |
| **Risk** | 8/10 | Low risk (proven) + medium (hedging auth gap) |
| **Security Impact** | 10/10 | Closes critical authentication gap |
| **Opportunity Impact** | 10/10 | Completes multi-tenant architecture |

**Total Score**: 56/60 ⭐ **HIGHEST SCORE**

**Decision**: ✅ **SELECTED** - Reuse Phase 1/2 pattern with security-first approach to hedging.py

### Related ADRs

- **ADR-001**: Multi-Tenant Architecture via JWT Middleware (Phase 1) → **REUSED**
- **ADR-002**: Tenant Context Dependency Injection (Phase 2) → **REUSED**
- **NEW**: Hedging Router Authentication Security Fix (Phase 3)

---

## Risk Assessment & Controls

### Risks Specific to Phase 3

| Risk ID | Failure Mode | Likelihood | Impact | Mitigation | Owner |
|---------|--------------|------------|--------|------------|-------|
| R3P-001 | hedging.py auth implementation incomplete | MEDIUM | CRITICAL | Security code review checklist, pair programming with security lead | Security Lead |
| R3P-002 | Ensemble service integration breaks tenant isolation | LOW | HIGH | Integration tests + unit tests for service layer | Engineer |
| R3P-003 | Cross-tenant hedging operations possible | LOW | CRITICAL | Integration tests verify tenant isolation in hedge execution | Security Lead |
| R3P-004 | Import errors in hedging.py (circular deps) | LOW | MEDIUM | Test imports locally before CI, check existing risk.py imports | Engineer |
| R3P-005 | Ensemble performance metrics expose tenant data | LOW | MEDIUM | Response schema audit, verify no tenant_id in responses | Engineer |
| R3P-006 | Hedge execution without proper user permission checks | MEDIUM | CRITICAL | Permission validation in endpoint, audit logging of all trades | Security Lead |

### Trading Operation Security Controls

**Hedging-Specific Controls**:

1. **Pre-Execution Validation**:
   - ✅ User must be authenticated (get_current_user)
   - ✅ Tenant context must exist (get_current_tenant_id)
   - ✅ User must have "execute_trades" permission (future enhancement)
   - ✅ Positions must be tenant-scoped

2. **Audit Logging**:
   - ✅ Log all hedge operations with tenant_id
   - ✅ Log user_id and timestamp
   - ✅ Log executed trades with quantities and prices
   - ✅ Log any failures or exceptions

3. **Risk Controls**:
   - ✅ Position limits enforced by risk manager
   - ✅ Leverage limits enforced by hedge config
   - ✅ Margin usage limits enforced by hedge manager
   - ✅ Grid bot disabled (safety-first)

**Ensemble-Specific Controls**:

1. **Weight Optimization**:
   - ✅ Optimization scoped to tenant's ensemble
   - ✅ Cannot access or modify other tenants' ensembles
   - ✅ Historical data filtered by tenant_id

2. **Agent Registration**:
   - ✅ Agents registered to specific ensemble
   - ✅ Ensemble belongs to specific tenant
   - ✅ Agent weights are tenant-specific

3. **Performance Metrics**:
   - ✅ Performance data scoped by ensemble_id AND tenant_id
   - ✅ Rankings filtered by tenant's agents
   - ✅ Lookback period enforced per tenant

---

## Validation Strategy

### Testing Coverage

**Unit Tests**: ~15 tests
- Tenant context extraction for each endpoint
- Error handling when tenant_id missing
- Permission validation in hedging endpoints

**Integration Tests**: ~45 tests
- 8 tests per ensemble endpoint (tenant isolation verified)
- 3 tests per hedging endpoint (tenant isolation + auth verified)
- Cross-tenant data leakage verification (critical)

**Security Tests**: ~12 tests
- Hedging auth gap fix verification
- Missing authentication returns 401
- Invalid JWT returns 401
- Missing tenant_id claim returns 401
- Cross-tenant access denied (403)
- SQL injection with tenant_id parameter

### Test Pattern (From Phase 2)

```python
@pytest.mark.asyncio
async def test_create_ensemble_tenant_isolation(client, mock_tenant_token):
    """Test that POST /ensemble/create enforces tenant isolation."""
    # Arrange
    tenant_id = "test-tenant-123"
    request_data = {
        "name": "test-ensemble",
        "ensemble_type": "weighted_voting"
    }

    # Act
    response = await client.post(
        "/ensemble/create",
        json=request_data,
        headers={"Authorization": f"Bearer {mock_tenant_token(tenant_id)}"}
    )

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["ensemble_id"] is not None
    # Verify tenant context was applied (via logs or service behavior)
```

### Quality Gates

**Before Merge**:
- [ ] All 72 new tests passing (15 unit + 45 integration + 12 security)
- [ ] Code coverage ≥ 90% for modified endpoints
- [ ] No critical or high security vulnerabilities (bandit scan)
- [ ] hedging.py authentication gap fixed (code review)
- [ ] Tenant isolation verified for all 11 endpoints
- [ ] Import errors resolved (hedging.py + ensemble.py)

**Before Production Deployment**:
- [ ] CI/CD green for 2 consecutive runs
- [ ] Load tests show no performance degradation
- [ ] Security scan passes (bandit, safety, OWASP)
- [ ] Staging deployment successful
- [ ] Monitoring configured and validated

**Success Criteria**:
- Zero cross-tenant data leakage incidents
- Zero authentication bypass incidents
- p99 latency < 200ms (no degradation)
- All 11 endpoints enforcing tenant isolation
- hedging.py endpoints require authentication

---

## Implementation Checklist

### Ensemble Router (8 endpoints)

**File**: `src/alpha_pulse/api/routers/ensemble.py`

Each endpoint requires:
1. Add `from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id`
2. Add `tenant_id: str = Depends(get_current_tenant_id)` to endpoint signature
3. Add `logger.info(f"[Tenant: {tenant_id}] ...")` to endpoint body
4. Pass `tenant_id` to service methods (if applicable)

**Endpoints**:
- [ ] POST /ensemble/create
- [ ] POST /ensemble/{ensemble_id}/register-agent
- [ ] POST /ensemble/{ensemble_id}/predict
- [ ] GET /ensemble/{ensemble_id}/performance
- [ ] GET /ensemble/{ensemble_id}/weights
- [ ] POST /ensemble/{ensemble_id}/optimize-weights
- [ ] GET /ensemble/
- [ ] GET /ensemble/agent-rankings

### Hedging Router (3 endpoints - CRITICAL SECURITY)

**File**: `src/alpha_pulse/api/routers/hedging.py`

Each endpoint requires:
1. Add imports:
   ```python
   from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
   from ..auth import get_current_user  # Verify correct import path
   ```
2. Add `tenant_id: str = Depends(get_current_tenant_id)` to endpoint signature
3. Add `current_user = Depends(get_current_user)` to endpoint signature
4. Add `logger.info(f"[Tenant: {tenant_id}] ...")` to endpoint body
5. Pass `tenant_id` to service methods (if applicable)
6. **CRITICAL**: Verify no unauthenticated paths remain

**Endpoints**:
- [ ] GET /hedging/analysis (CRITICAL - currently unauthenticated)
- [ ] POST /hedging/execute (CRITICAL - currently unauthenticated)
- [ ] POST /hedging/close (CRITICAL - currently unauthenticated)

### Test Files (2 files to create)

**File 1**: `tests/api/test_ensemble_tenant_context.py`
- Test all 8 endpoints for tenant isolation
- Test that ensemble operations scoped to tenant
- Test weight optimization is tenant-specific
- ~40 integration tests

**File 2**: `tests/api/test_hedging_tenant_context.py`
- Test all 3 endpoints for authentication AND tenant isolation
- Test missing JWT returns 401
- Test missing tenant_id claim returns 401
- Test hedge execution is tenant-scoped
- Test cross-tenant access denied
- ~32 integration tests (more due to auth fixes)

---

## Appendix: Phase Comparison

| Aspect | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|----------|
| **Endpoints** | 14 | 16 | 11 |
| **Routers** | 3 | 5 | 2 |
| **Pattern** | New (Depends) | Reuse Phase 1 | Reuse Phase 1/2 |
| **Auth Gap** | None | None | CRITICAL (hedging) |
| **Complexity** | MEDIUM | LOW | LOW |
| **Risk** | MEDIUM | LOW | MEDIUM |
| **Effort** | 3 days | 2-3 days | 2-3 days |
| **RICE Score** | 52.0 | 58.0 | 64.0 |
| **Critical Blocker** | No | No | Yes (auth fix) |

---

## Key Takeaways

1. **Pattern Continuity**: Phase 3 follows the proven Phases 1 & 2 pattern with zero architectural changes
2. **Security Priority**: hedging.py authentication gap is CRITICAL and must be fixed before production
3. **Tenant Isolation**: All 11 endpoints will enforce tenant isolation via Depends() pattern
4. **Ensemble Integration**: Ensemble service must be tenant-aware (no breaking changes assumed)
5. **Testing**: Comprehensive test coverage required, especially for hedging security fix

---

**Document Version**: 1.0
**Last Updated**: 2025-11-03
**Status**: In Review
**Next Review**: After stakeholder approval

**References**:
- [Story 2.4 Phase 2 HLD](/Users/a.rocchi/Projects/Personal/AlphaPulse/docs/design/story-2.4-phase2-hld.md)
- [Story 2.4 Phase 1 HLD](/Users/a.rocchi/Projects/Personal/AlphaPulse/docs/design/story-2.4-phase1-hld.md)
- [Tenant Context Middleware](/Users/a.rocchi/Projects/Personal/AlphaPulse/src/alpha_pulse/api/middleware/tenant_context.py)
- [Authentication Module](/Users/a.rocchi/Projects/Personal/AlphaPulse/src/alpha_pulse/api/auth.py)
