# API Tenant Context Implementation Guide

**Story**: #165 (Story 2.4 - Update API routes with tenant context)
**EPIC**: #141 (EPIC-002 - Application Multi-Tenancy)
**ADR**: [ADR-003](../adr/003-api-tenant-context-integration.md)

## Overview

This guide provides step-by-step instructions for updating API endpoints to use tenant context from the `TenantContextMiddleware`.

## Prerequisites

Before implementing tenant-aware endpoints:

1. ✅ **TenantContextMiddleware** is registered in `main.py`
2. ✅ **JWT tokens include tenant_id** claim
3. ✅ **Services support tenant_id** parameter (Ag

entManager, RiskManager, etc.)
4. ⚠️ **Circular imports resolved** (see issue #191)

## Implementation Pattern

### Step 1: Import the Dependency

```python
from fastapi import APIRouter, Depends
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
```

### Step 2: Add to Endpoint Signature

```python
@router.get("/endpoint")
async def my_endpoint(
    tenant_id: str = Depends(get_current_tenant_id),
    # ... other parameters
):
    # tenant_id is automatically injected
    pass
```

### Step 3: Pass to Services

```python
# Pass tenant_id to all service calls
result = await service.method(
    param1,
    param2,
    tenant_id=tenant_id
)
```

## Common Endpoint Patterns

### Pattern 1: Simple GET Endpoint

**Before:**
```python
@router.get("/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    metric_accessor: MetricsDataAccessor = Depends(lambda: metrics_accessor)
) -> List[Dict[str, Any]]:
    return await metric_accessor.get_metrics(metric_type=metric_type)
```

**After:**
```python
@router.get("/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    tenant_id: str = Depends(get_current_tenant_id),  # ← Add this
    metric_accessor: MetricsDataAccessor = Depends(lambda: metrics_accessor)
) -> List[Dict[str, Any]]:
    return await metric_accessor.get_metrics(
        metric_type=metric_type,
        tenant_id=tenant_id  # ← Pass to service
    )
```

### Pattern 2: POST Endpoint with Request Body

**Before:**
```python
@router.post("/trades/execute")
async def execute_trade(
    trade_request: TradeRequest,
    risk_manager: RiskManager = Depends(get_risk_manager)
) -> TradeResponse:
    approved = await risk_manager.evaluate_trade(
        symbol=trade_request.symbol,
        side=trade_request.side,
        quantity=trade_request.quantity
    )
    # ...
```

**After:**
```python
@router.post("/trades/execute")
async def execute_trade(
    trade_request: TradeRequest,
    tenant_id: str = Depends(get_current_tenant_id),  # ← Add this
    risk_manager: RiskManager = Depends(get_risk_manager)
) -> TradeResponse:
    approved = await risk_manager.evaluate_trade(
        symbol=trade_request.symbol,
        side=trade_request.side,
        quantity=trade_request.quantity,
        tenant_id=tenant_id  # ← Pass to service
    )
    # ...
```

### Pattern 3: Endpoint with Permission Check

**Before:**
```python
@router.get("/portfolio")
async def get_portfolio(
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
):
    return await portfolio_accessor.get_portfolio()
```

**After:**
```python
@router.get("/portfolio")
async def get_portfolio(
    tenant_id: str = Depends(get_current_tenant_id),  # ← Add this
    _: Dict[str, Any] = Depends(require_view_portfolio),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
):
    return await portfolio_accessor.get_portfolio(tenant_id=tenant_id)  # ← Pass to service
```

### Pattern 4: Endpoint Accessing Request State

**Before:**
```python
@router.get("/system")
async def get_system_metrics(request: Request):
    cache_service = request.app.state.caching_service
    metrics = await cache_service.get_metrics()
    return metrics
```

**After:**
```python
@router.get("/system")
async def get_system_metrics(
    request: Request,
    tenant_id: str = Depends(get_current_tenant_id)  # ← Add this
):
    cache_service = request.app.state.caching_service
    metrics = await cache_service.get_metrics(tenant_id=tenant_id)  # ← Pass to service
    return metrics
```

### Pattern 5: Endpoint with Multiple Service Calls

**Before:**
```python
@router.post("/signals/generate")
async def generate_signals(
    market_data: MarketDataRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    # Generate signals
    signals = await agent_manager.generate_signals(market_data)

    # Evaluate risk
    exposure = await risk_manager.calculate_risk_exposure()

    return {"signals": signals, "exposure": exposure}
```

**After:**
```python
@router.post("/signals/generate")
async def generate_signals(
    market_data: MarketDataRequest,
    tenant_id: str = Depends(get_current_tenant_id),  # ← Add this ONCE
    agent_manager: AgentManager = Depends(get_agent_manager),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    # Pass tenant_id to ALL service calls
    signals = await agent_manager.generate_signals(
        market_data,
        tenant_id=tenant_id  # ← Pass to service
    )

    exposure = await risk_manager.calculate_risk_exposure(
        tenant_id=tenant_id  # ← Pass to service
    )

    return {"signals": signals, "exposure": exposure}
```

## Testing Patterns

### Pattern 1: Unit Test with Mocked Dependency

```python
from unittest.mock import AsyncMock, patch
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id

@pytest.mark.asyncio
async def test_endpoint_uses_tenant_context():
    # Mock the dependency to return a fixed tenant_id
    test_tenant_id = "00000000-0000-0000-0000-000000000001"

    with patch(
        'alpha_pulse.api.routers.metrics.get_current_tenant_id',
        return_value=test_tenant_id
    ):
        # Your test logic here
        response = await get_metrics(...)

        # Verify tenant_id was passed to service
        mock_service.get_metrics.assert_called_with(
            metric_type="performance",
            tenant_id=test_tenant_id
        )
```

### Pattern 2: Integration Test with TestClient

```python
from fastapi.testclient import TestClient
from jose import jwt
from datetime import datetime, timedelta

def test_endpoint_extracts_tenant_from_jwt(client: TestClient):
    # Create JWT with tenant_id
    payload = {
        "sub": "testuser",
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    token = jwt.encode(payload, "secret", algorithm="HS256")

    # Call endpoint
    response = client.get(
        "/api/v1/portfolio",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
```

### Pattern 3: Tenant Isolation Test

```python
@pytest.mark.asyncio
async def test_endpoint_isolates_by_tenant():
    tenant_1_id = "00000000-0000-0000-0000-000000000001"
    tenant_2_id = "00000000-0000-0000-0000-000000000002"

    # Request for tenant 1
    with patch('...get_current_tenant_id', return_value=tenant_1_id):
        result_1 = await get_portfolio()

    # Request for tenant 2
    with patch('...get_current_tenant_id', return_value=tenant_2_id):
        result_2 = await get_portfolio()

    # Results should be different (tenant-isolated)
    assert result_1 != result_2
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Import Dependency

```python
# WRONG - dependency not imported
@router.get("/endpoint")
async def my_endpoint(tenant_id: str = Depends(get_current_tenant_id)):
    #                                          ^^^^^^^^^^^^^^^^^^^^^^
    # NameError: name 'get_current_tenant_id' is not defined
```

**Fix**: Always import `from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id`

### ❌ Pitfall 2: Not Passing tenant_id to Service

```python
# WRONG - tenant_id extracted but not used
@router.get("/endpoint")
async def my_endpoint(tenant_id: str = Depends(get_current_tenant_id)):
    return await service.get_data()  # ← Missing tenant_id parameter
    # Will raise ValueError: tenant_id is mandatory
```

**Fix**: Pass `tenant_id=tenant_id` to all service calls

### ❌ Pitfall 3: Using request.state.tenant_id Directly

```python
# WRONG - bypasses dependency injection
@router.get("/endpoint")
async def my_endpoint(request: Request):
    tenant_id = request.state.tenant_id  # ← Don't do this
    return await service.get_data(tenant_id=tenant_id)
```

**Fix**: Use `tenant_id: str = Depends(get_current_tenant_id)` instead

### ❌ Pitfall 4: Inconsistent Parameter Naming

```python
# WRONG - confusing parameter name
@router.get("/endpoint")
async def my_endpoint(current_tenant: str = Depends(get_current_tenant_id)):
    #                     ^^^^^^^^^^^^^^ - Confusing name
    return await service.get_data(tenant_id=current_tenant)
```

**Fix**: Always use `tenant_id` for consistency

### ❌ Pitfall 5: Missing tenant_id in Test Mocks

```python
# WRONG - test doesn't mock tenant context
def test_endpoint():
    response = client.get("/api/v1/portfolio")
    # Will fail with 401 Unauthorized (no JWT token)
```

**Fix**: Create JWT token with tenant_id or mock the dependency

## Router-Specific Guidance

### alerts.py (2 endpoints)

**Services to update:**
- `alert_manager.get_all_alerts()` → add `tenant_id`
- `alert_manager.acknowledge_alert()` → add `tenant_id`

**Example:**
```python
@router.get("/alerts")
async def get_alerts(
    tenant_id: str = Depends(get_current_tenant_id),
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    return await alert_manager.get_all_alerts(tenant_id=tenant_id)
```

### portfolio.py (3 endpoints)

**Services to update:**
- `portfolio_accessor.get_portfolio()` → add `tenant_id`

**Example:**
```python
@router.get("/portfolio")
async def get_portfolio(
    tenant_id: str = Depends(get_current_tenant_id),
    portfolio_accessor: PortfolioDataAccessor = Depends(get_portfolio_accessor)
):
    return await portfolio_accessor.get_portfolio(tenant_id=tenant_id)
```

### risk.py (5 endpoints)

**Services to update:**
- `risk_manager.calculate_risk_exposure()` → add `tenant_id`
- `risk_manager.get_risk_metrics()` → add `tenant_id`
- `risk_manager.calculate_position_size()` → add `tenant_id`

**Example:**
```python
@router.get("/risk/exposure")
async def get_risk_exposure(
    tenant_id: str = Depends(get_current_tenant_id),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    return await risk_manager.calculate_risk_exposure(tenant_id=tenant_id)
```

## Rollout Checklist

For each router file:

- [ ] Import `get_current_tenant_id` dependency
- [ ] Add `tenant_id: str = Depends(get_current_tenant_id)` to ALL endpoints
- [ ] Pass `tenant_id=tenant_id` to ALL service calls
- [ ] Update unit tests to mock `get_current_tenant_id`
- [ ] Update integration tests with JWT tokens containing `tenant_id`
- [ ] Verify tenant isolation with multi-tenant test cases
- [ ] Update OpenAPI examples (if custom examples exist)
- [ ] Code review for consistency
- [ ] Merge and deploy

## Success Criteria

After implementation, verify:

1. ✅ All endpoints extract `tenant_id` from middleware
2. ✅ All service calls include `tenant_id` parameter
3. ✅ Tests pass with tenant context mocking
4. ✅ Integration tests verify tenant isolation
5. ✅ OpenAPI documentation shows tenant context
6. ✅ No runtime errors from missing `tenant_id`
7. ✅ Logs include tenant context

## Related Documents

- [ADR-003](../adr/003-api-tenant-context-integration.md): API Tenant Context Integration Pattern
- [MIGRATION.md](../../MIGRATION.md): API Migration Guide for v2.0.0
- [TenantContextMiddleware](../../src/alpha_pulse/api/middleware/tenant_context.py): Middleware implementation
- Issue #165: Story 2.4 - Update API routes with tenant context
- Issue #191: Pre-existing circular import (blocking implementation)

---

**Last Updated**: 2025-11-02
**Status**: Ready for Implementation (Blocked by #191)
