# ADR-003: API Tenant Context Integration Pattern

## Status

**ACCEPTED** (Pending Implementation - Blocked by #191)

**Date**: 2025-11-02
**Decision Makers**: Tech Lead, Senior Engineers
**Story**: #165 (Story 2.4 - Update API routes with tenant context)
**EPIC**: #141 (EPIC-002 - Application Multi-Tenancy)

## Context

AlphaPulse is transitioning to a multi-tenant SaaS architecture. The database layer (EPIC-001) and core services (AgentManager, RiskManager) already support tenant isolation via `tenant_id` parameters. However, **API endpoints do not yet extract or propagate tenant context** from authenticated requests to underlying services.

### Current State

1. **TenantContextMiddleware exists** (`src/alpha_pulse/api/middleware/tenant_context.py`)
   - Extracts `tenant_id` from JWT tokens
   - Sets `request.state.tenant_id` for route handlers
   - Sets PostgreSQL session variable `app.current_tenant_id` for RLS
   - Already registered in `main.py` (line 167)

2. **JWT tokens include tenant_id** (line 225 in `main.py`)
   ```python
   access_token = create_access_token(
       data={"sub": user.username, "tenant_id": user.tenant_id}
   )
   ```

3. **21 router files with 100+ endpoints** do NOT use tenant context:
   - alerts.py, backtesting.py, correlation.py, data_lake.py
   - data_quality.py, ensemble.py, explainability.py, gpu.py
   - hedging.py, liquidity.py, metrics.py, online_learning.py
   - portfolio.py, positions.py, regime.py, risk.py
   - risk_budget.py, smart_order_router.py, system.py, trades.py

4. **Services expect tenant_id** (AgentManager, RiskManager require it)

### Problem

**API routes do not pass tenant_id to services**, creating three critical gaps:

1. **Broken service calls**: Services like `AgentManager.generate_signals()` will raise `ValueError: tenant_id is mandatory`
2. **Tenant isolation bypass**: Without tenant context propagation, RLS at database level is insufficient for application-level isolation
3. **Audit trail gaps**: Operations are not logged with tenant context

### Alternatives Considered

#### Option 1: Manual tenant_id extraction in each endpoint
```python
@router.get("/portfolio")
async def get_portfolio(request: Request):
    tenant_id = request.state.tenant_id  # Manual extraction
    return await portfolio_service.get(tenant_id=tenant_id)
```

**Pros**:
- Explicit and visible
- No magic dependencies

**Cons**:
- **100+ endpoints** need updates (high error risk)
- Boilerplate code duplication
- Easy to forget in new endpoints
- Violates DRY principle

#### Option 2: Dependency injection with `get_current_tenant_id`
```python
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id

@router.get("/portfolio")
async def get_portfolio(tenant_id: str = Depends(get_current_tenant_id)):
    return await portfolio_service.get(tenant_id=tenant_id)
```

**Pros**:
- **Clean and consistent** across all endpoints
- **Type-safe** (FastAPI validates dependency)
- **Testable** (easy to mock `get_current_tenant_id`)
- **Self-documenting** (tenant_id appears in function signature)
- **Fails fast** (HTTPException if middleware not configured)

**Cons**:
- Requires adding `Depends()` to 100+ endpoints
- Dependency on middleware being correctly configured

#### Option 3: Service-level context propagation (AsyncLocalStorage)
```python
# Using contextvars
tenant_context = ContextVar('tenant_id')

# Middleware sets it
tenant_context.set(tenant_id)

# Services read it
def get_portfolio():
    tenant_id = tenant_context.get()
```

**Pros**:
- No endpoint changes needed
- Fully automatic

**Cons**:
- **Hidden behavior** (hard to debug)
- **Not thread-safe** in all scenarios
- **Testing complexity** (global state)
- **Violates explicit is better than implicit** principle

## Decision

**Use Option 2: Dependency Injection with `get_current_tenant_id`**

### Implementation Pattern

#### 1. Import the dependency
```python
from fastapi import APIRouter, Depends
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
```

#### 2. Add to endpoint signature
```python
@router.get("/portfolio")
async def get_portfolio(
    tenant_id: str = Depends(get_current_tenant_id),
    # ... other parameters
):
    # tenant_id is automatically injected by FastAPI
    return await portfolio_service.get(tenant_id=tenant_id)
```

#### 3. Pass to services
```python
# AgentManager
signals = await agent_manager.generate_signals(
    market_data,
    tenant_id=tenant_id
)

# RiskManager
exposure = await risk_manager.calculate_risk_exposure(
    tenant_id=tenant_id
)
```

### Scope of Changes

**21 router files, 100+ endpoints require updates:**

| Router | Endpoints | Priority | Reason |
|--------|-----------|----------|--------|
| `risk.py` | 5 | P0 | Calls RiskManager (requires tenant_id) |
| `risk_budget.py` | 6 | P0 | Tenant-specific risk limits |
| `portfolio.py` | 3 | P0 | Critical tenant data isolation |
| `trades.py` | 1 | P0 | Tenant-specific trade history |
| `alerts.py` | 2 | P1 | Tenant-specific alerts |
| `metrics.py` | 4 | P1 | Tenant-specific performance |
| `ensemble.py` | 9 | P1 | Calls AgentManager (requires tenant_id) |
| `backtesting.py` | 6 | P2 | Tenant-specific backtests |
| `correlation.py` | 5 | P2 | Tenant-specific analysis |
| `data_lake.py` | 9 | P2 | Tenant data isolation |
| Others | 50+ | P3 | Gradual migration |

### Migration Strategy

**Phase 1: Critical Services (P0)**
- Update 4 routers: risk, risk_budget, portfolio, trades
- ~14 endpoints affected
- **Blocking**: These are broken without tenant_id

**Phase 2: Agent & Analytics (P1)**
- Update 5 routers: alerts, metrics, ensemble, hedging, regime
- ~26 endpoints affected
- **Important**: User-facing features

**Phase 3: Supporting Services (P2-P3)**
- Update 12+ routers: backtesting, data_lake, explainability, etc.
- ~60+ endpoints affected
- **Enhancement**: Gradual rollout

## Consequences

### Positive

1. **Tenant Isolation**: All API operations are tenant-aware from request → service → database
2. **Type Safety**: FastAPI validates tenant_id parameter exists
3. **Audit Trail**: Operations logged with tenant context
4. **Fail-Fast**: Clear errors if middleware misconfigured
5. **Testing**: Easy to mock `get_current_tenant_id` in tests
6. **Documentation**: OpenAPI docs automatically show tenant context
7. **Consistency**: Same pattern across all 100+ endpoints

### Negative

1. **Large Scope**: 100+ endpoints need updates (high effort)
2. **Breaking Change**: Endpoints will fail until updated
3. **Coordination**: Requires careful rollout to avoid downtime
4. **Testing Burden**: All endpoint tests need tenant_id mocking

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Pre-existing circular imports block testing** | P0 | Track in #191, fix before implementation |
| **Missed endpoints cause runtime errors** | HIGH | Comprehensive endpoint audit (completed) |
| **Inconsistent implementation** | MEDIUM | Create implementation guide, code review |
| **Test coverage drops** | MEDIUM | Update all API tests with tenant context |
| **Performance impact** | LOW | Dependency injection is fast (~0.1ms) |

## Implementation Plan

### Prerequisites (BLOCKED by #191)

1. **Fix circular import in `data_lake/lake_manager.py`**
   - Prevents `main.py` from importing
   - Blocks all API tests from running
   - **Status**: Documented in issue #191

### Implementation Steps (Once Unblocked)

1. **Create Implementation Guide** (Documentation)
   - Code examples for each endpoint type
   - Testing patterns
   - Common pitfalls

2. **Update MIGRATION.md** (Documentation)
   - API migration guide
   - Before/after examples for each router type

3. **Phase 1: Critical Services** (14 endpoints)
   - Update risk.py, risk_budget.py, portfolio.py, trades.py
   - Update corresponding tests
   - Integration testing

4. **Phase 2: Agent & Analytics** (26 endpoints)
   - Update alerts.py, metrics.py, ensemble.py, hedging.py, regime.py
   - Update tests
   - Load testing

5. **Phase 3: Supporting Services** (60+ endpoints)
   - Gradual rollout with monitoring
   - Update remaining routers

### Testing Strategy

```python
# Test pattern for tenant-aware endpoints
def test_endpoint_requires_tenant_context(client, create_test_token, tenant_1_id):
    token = create_test_token("admin", tenant_1_id)
    response = client.get(
        "/api/v1/portfolio",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    # Verify tenant isolation
```

### Rollback Plan

If implementation causes issues:
1. **Revert router changes** (Git revert)
2. **Deploy previous version** (tag-based rollback)
3. **Disable TenantContextMiddleware** temporarily
4. **Root cause analysis** before re-attempt

## Related Documents

- **EPIC-002**: #141 (Application Multi-Tenancy)
- **Story 2.4**: #165 (Update API routes with tenant context)
- **Blocking Issue**: #191 (Pre-existing circular import)
- **MIGRATION.md**: Multi-tenant API migration guide
- **TenantContextMiddleware**: `src/alpha_pulse/api/middleware/tenant_context.py`
- **ADR-001**: Database Multi-Tenancy (RLS policies)
- **ADR-002**: Tenant Provisioning Architecture

## References

- FastAPI Dependency Injection: https://fastapi.tiangolo.com/tutorial/dependencies/
- Multi-Tenant SaaS Best Practices: https://docs.aws.amazon.com/wellarchitected/latest/saas-lens/
- PostgreSQL RLS: https://www.postgresql.org/docs/current/ddl-rowsecurity.html

## Stakeholder Sign-Off

- **Tech Lead**: Approved (2025-11-02)
- **Senior Engineers**: Review pending
- **Security Team**: Review required (Sprint 7)

---

**Last Updated**: 2025-11-02
**Status**: ACCEPTED (Implementation blocked by #191)
