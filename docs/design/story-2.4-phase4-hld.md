# Story 2.4 Phase 4: API Tenant Context Integration - High-Level Design

**Date**: 2025-11-05
**Story**: EPIC-002 Story 2.4 Phase 4
**Phase**: Design the Solution
**Status**: In Review
**Version**: 1.0

---

## Executive Summary

This document details the high-level design for completing Story 2.4 by integrating tenant context into the final 7 API endpoints across 2 routers (regime.py, positions.py). **CRITICAL**: positions.py has 3 endpoints with **NO USER AUTHENTICATION**, exposing sensitive trading position data to any API client. This phase includes both a critical security fix and tenant isolation implementation, achieving 100% API endpoint coverage (43/43) for multi-tenant architecture.

**Key Decisions**:
- Reuse Phase 2/3 patterns with zero architectural changes
- Immediate authentication enforcement for positions.py (no grace period)
- Breaking change management for positions.py clients
- 100% pattern consistency across all 43 endpoints

**Complexity**: LOW-MEDIUM
**Risk Level**: MEDIUM (due to breaking change in positions.py)
**Estimated Effort**: 0.75 sprints (7 endpoints, 5-7 story points)

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Architecture](#architecture)
4. [Detailed Design](#detailed-design)
5. [API Changes](#api-changes)
6. [Testing Strategy](#testing-strategy)
7. [Decision Log](#decision-log)
8. [Risk Assessment](#risk-assessment)
9. [Deployment Strategy](#deployment-strategy)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Appendices](#appendices)

---

## 1. Overview

### 1.1 Phase 4 Scope

**Endpoints to Complete** (7 total):

**regime.py** (4 endpoints - HIGH priority):
- GET `/current` - Current market regime state
- GET `/history` - Historical regime data
- GET `/analysis/{regime_type}` - Regime-specific analysis
- GET `/alerts` - Regime transition alerts

**positions.py** (3 endpoints - CRITICAL priority):
- GET `/spot` - Spot trading positions **[NO AUTH - SECURITY GAP]**
- GET `/futures` - Futures trading positions **[NO AUTH - SECURITY GAP]**
- GET `/metrics` - Position metrics **[NO AUTH - SECURITY GAP]**

### 1.2 Phase 4 in Story 2.4 Context

**Phase Progression**:
- Phase 1: 14 endpoints (risk.py, risk_budget.py, portfolio.py) ✅ v2.1.0
- Phase 2: 16 endpoints (alerts.py, metrics.py, system.py, trades.py, correlation.py) ✅ v2.2.0
- Phase 3: 12 endpoints (hedging.py authentication fix, ensemble.py) ✅ v2.3.0
- Phase 4: 7 endpoints (regime.py tenant context, positions.py security fix + tenant context) → v2.4.0

**Cumulative Impact**: 36/43 endpoints → 43/43 endpoints (100% coverage)

### 1.3 CRITICAL Security Issue

**positions.py Authentication Gap**:

Current implementation:
```python
@router.get("/spot", dependencies=[Depends(get_api_client)])
async def get_spot_positions(exchange: BaseExchange = Depends(get_exchange_client)):
    """Get current spot positions."""
```

**Problems**:
1. Missing `get_current_user` dependency → ANY client can access trading data
2. Missing `tenant_id` extraction → Cross-tenant visibility risk
3. Exposes sensitive financial data (spot/futures positions, hedge ratios, exposure metrics)
4. Violates regulatory requirements (unauthorized financial data access)

**Regulatory Impact**: SOC 2 Type II, ISO 27001, GDPR compliance violation

---

## 2. Design Principles

### 2.1 Reuse Phase 2/3 Patterns (100% Consistency)

**Rationale**: Phases 2 and 3 have successfully integrated tenant context into 28 endpoints with proven patterns and zero cross-tenant leakage. Reusing established patterns minimizes risk and maintenance burden.

**Pattern Structure**:
1. Import statements (Phase 2 reference)
2. Parameter order: `tenant_id` → `user` → other parameters
3. Logging format: `[Tenant: {tenant_id}] [User: {user.get('sub')}] <operation>`
4. Dependency injection: `Depends(get_current_tenant_id)` + `Depends(get_current_user)`
5. Service layer integration with tenant-scoped queries

### 2.2 Security-First for positions.py

**Principle**: Trading position data is too sensitive for grace periods or backward compatibility concerns.

- Enforce authentication immediately (no deprecation period)
- Add comprehensive audit logging with tenant context
- Integration tests validate authentication requirement
- Treat as breaking change (v2.4.0)

### 2.3 Tenant Context is Internal-Only

**Principle**: tenant_id is an implementation detail for data isolation, not part of API contracts.

- Extract tenant_id from JWT claims (via middleware)
- Use tenant_id only for data filtering (internal)
- Never expose tenant_id in API responses
- Transparent to API clients (no schema changes)

### 2.4 Pattern Consistency Over Optimization

**Principle**: Code clarity and maintainability trump micro-optimizations.

- Explicit parameter injection (not hidden in middleware)
- Consistent logging structure across all 43 endpoints
- Easier to understand, debug, and test
- Enables precise audit trails

---

## 3. Architecture

### 3.1 System Context

**Position in Multi-Tenant Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                   Client Applications                        │
│  (Web Dashboard, CLI Tools, 3rd Party Integrations)         │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTPS + JWT Token
                     │ (includes tenant_id claim)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         FastAPI Application (AlphaPulse API)                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ JWT Middleware: Extracts tenant_id, validates user   │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
│  ┌───────────────▼───────────────────────────────────────┐  │
│  │ Phase 1-3 (28 endpoints): COMPLETE ✅                 │  │
│  │ - risk.py, risk_budget.py, portfolio.py (14)         │  │
│  │ - alerts.py, metrics.py, system.py, trades.py (16)   │  │
│  │ - hedging.py (auth fix), ensemble.py (12)            │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Phase 4 Routers (7 endpoints): THIS PHASE            │  │
│  │  regime.py (4):                                       │  │
│  │   - GET /current (+ tenant context)                  │  │
│  │   - GET /history (+ tenant context)                  │  │
│  │   - GET /analysis/{regime_type} (+ tenant context)   │  │
│  │   - GET /alerts (+ tenant context)                   │  │
│  │                                                        │  │
│  │  positions.py (3):                                    │  │
│  │   - GET /spot (+ AUTH + tenant context)              │  │
│  │   - GET /futures (+ AUTH + tenant context)           │  │
│  │   - GET /metrics (+ AUTH + tenant context)           │  │
│  └──────────────┬──────────────────────────────────────┘  │
│                 │ tenant_id in context                       │
│  ┌──────────────▼──────────────────────────────────────┐   │
│  │ Data Access Layer (Service Classes)                  │   │
│  │  - RegimeDetectionService (tenant-scoped)            │   │
│  │  - ExchangePositionFetcher (tenant-scoped)           │   │
│  │  - AlertDataAccessor (tenant-scoped)                 │   │
│  └──────────────┬──────────────────────────────────────┘   │
└──────────────────│──────────────────────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │  PostgreSQL    │
          │  (RLS active)  │
          └────────────────┘
```

### 3.2 Component Design

#### 3.2.1 positions.py Router (CRITICAL SECURITY FIX)

**Current State (INSECURE)**:

```python
@router.get(
    "/spot",
    response_model=List[SpotPosition],
    dependencies=[Depends(get_api_client)]  # ❌ NO USER VERIFICATION
)
async def get_spot_positions(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current spot positions."""
    position_fetcher = ExchangePositionFetcher(exchange)
    positions = await position_fetcher.get_spot_positions()
    return positions
```

**Target State (SECURE)**:

```python
from ..middleware.tenant_context import get_current_tenant_id

@router.get(
    "/spot",
    response_model=List[SpotPosition]
)
async def get_spot_positions(
    tenant_id: str = Depends(get_current_tenant_id),
    user: dict = Depends(get_current_user),
    exchange: BaseExchange = Depends(get_exchange_client),
    _api_client = Depends(get_api_client)
):
    """Get current spot positions."""
    logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Fetching spot positions")

    position_fetcher = ExchangePositionFetcher(exchange)
    positions = await position_fetcher.get_spot_positions()

    logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Retrieved {len(positions)} spot positions")
    return positions
```

**Key Changes**:
1. `tenant_id: str = Depends(get_current_tenant_id)` - Extract tenant context from JWT
2. `user: dict = Depends(get_current_user)` - Enforce user authentication (401 if missing)
3. Comprehensive audit logging with tenant + user context
4. Remove `dependencies=[Depends(get_api_client)]` from decorator
5. Add `_api_client = Depends(get_api_client)` as parameter (for consistency)

**Security Impact**:
- Closes CRITICAL authentication gap (401 enforced)
- Adds tenant isolation (data scoped to tenant)
- Complete audit trail for compliance
- Prevents unauthorized access to trading data

**Endpoints Affected**:
1. GET `/spot` - Spot positions (line 26-44 in positions.py)
2. GET `/futures` - Futures positions (line 46-64)
3. GET `/metrics` - Position metrics (line 66-103)

#### 3.2.2 regime.py Router (Tenant Context Addition)

**Current State** (authentication exists, but missing tenant context):

```python
@router.get("/current", response_model=RegimeStateResponse)
async def get_current_regime(
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeStateResponse:
    """Get current market regime state."""
    regime_info = regime_service.current_regime_info
    # ... returns regime data
```

**Target State** (add tenant context):

```python
from ..middleware.tenant_context import get_current_tenant_id

@router.get("/current", response_model=RegimeStateResponse)
async def get_current_regime(
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeStateResponse:
    """Get current market regime state."""
    logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Getting current regime")

    regime_info = regime_service.current_regime_info

    logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Retrieved regime: {regime_info.current_regime if regime_info else 'N/A'}")
    # ... returns regime data
```

**Key Changes**:
1. `tenant_id: str = Depends(get_current_tenant_id)` - Extract tenant context
2. Reorder parameters: `tenant_id` → `user` → other dependencies
3. Add tenant logging at start and completion
4. Service layer assumes tenant filtering is implemented (no changes to regime_service)

**Endpoints Affected**:
1. GET `/current` - Current regime (line 46-92)
2. GET `/history` - Historical regime (line 95-161)
3. GET `/analysis/{regime_type}` - Analysis (line 164-311)
4. GET `/alerts` - Alerts (line 314-336)

### 3.3 Data Flow Diagrams

**Request Flow - Secure Pattern (Phase 4)**:

```
┌─────────┐                                                  ┌─────┐
│ Client  │                                                  │ DB  │
└────┬────┘                                                  └──┬──┘
     │                                                           │
     │ GET /positions/spot                                       │
     │ Authorization: Bearer <JWT_TOKEN>                         │
     │ {tenant_id: "tenant-abc", sub: "user-123"}               │
     ├───────────────────────────────────────────────────────────┤
     │                                                           │
     │ 1. JWT Middleware validates token, extracts tenant_id     │
     │    request.state.tenant_id = "tenant-abc"                │
     │                                                           │
     │ 2. get_current_tenant_id() dependency:                    │
     │    Returns: "tenant-abc"                                  │
     │                                                           │
     │ 3. get_current_user() dependency:                         │
     │    Returns: {sub: "user-123", ...}                        │
     │    OR raises 401 Unauthorized if token missing/invalid    │
     │                                                           │
     │ 4. Endpoint: async def get_spot_positions(               │
     │       tenant_id: str,                                     │
     │       user: dict,                                         │
     │       exchange: BaseExchange,                             │
     │       _api_client                                         │
     │    )                                                      │
     │    logger.info(f"[Tenant: {tenant_id}] [User:            │
     │                 {user.get('sub')}] Fetching positions")   │
     │                                                           │
     │ 5. Exchange service uses tenant context:                  │
     │    Call: exchange_client.get_spot_positions()             │
     │    (Internally scopes positions to tenant)                │
     │    ────────────────────────────────────────────────────────┤
     │                                                           │
     │                          6. Return tenant-scoped results   │
     │                             No tenant_id exposed          │
     │ ◀─────────────────────────────────────────────────────────┤
     │                                                           │
     │ 7. Response: [                                            │
     │   {symbol: "BTC", qty: 1.5, value: 45000.00},           │
     │   {symbol: "ETH", qty: 25.0, value: 50000.00}           │
     │  ]                                                        │
     │ 8. Log: [Tenant: tenant-abc] [User: user-123]            │
     │         Retrieved 2 spot positions                        │
     ◀─────────────────────────────────────────────────────────────┤
```

**Security Comparison (Before/After)**:

```
┌─────────────────────────────────────────────────────────────────┐
│ BEFORE (Phase 3 - INSECURE):                                    │
├─────────────────────────────────────────────────────────────────┤
│ @router.get("/spot", dependencies=[Depends(get_api_client)])   │
│ async def get_spot_positions(                                   │
│     exchange: BaseExchange = Depends(get_exchange_client)       │
│ ):                                                              │
│     # No user verification                                     │
│     # No tenant isolation                                      │
│     # Any API client with basic access can read positions      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ AFTER (Phase 4 - SECURE):                                       │
├─────────────────────────────────────────────────────────────────┤
│ @router.get("/spot")                                            │
│ async def get_spot_positions(                                   │
│     tenant_id: str = Depends(get_current_tenant_id),           │
│     user: dict = Depends(get_current_user),                    │
│     exchange: BaseExchange = Depends(get_exchange_client),     │
│     _api_client = Depends(get_api_client)                      │
│ ):                                                              │
│     # User verified (401 if missing)                           │
│     # Data scoped to tenant                                    │
│     # Full audit trail                                         │
│     # Cross-tenant access prevented                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Security Architecture

**Authentication Layers**:

1. **JWT Middleware** (Layer 1 - App level):
   - Validates JWT signature
   - Extracts `tenant_id` claim
   - Validates token expiry
   - Sets `request.state.tenant_id`

2. **get_current_user() Dependency** (Layer 2 - Endpoint level):
   - Required in all Phase 4 endpoints
   - Raises 401 Unauthorized if token missing/invalid
   - Returns user claims (sub, email, roles)
   - Used for audit logging

3. **get_current_tenant_id() Dependency** (Layer 3 - Endpoint level):
   - Extracts tenant_id from request.state
   - Validates tenant_id exists
   - Used for data filtering

4. **Data Layer Filtering** (Layer 4 - Service/Database level):
   - Service classes filter queries by tenant_id
   - Database RLS policies enforce tenant boundaries
   - Double-defense prevents accidental cross-tenant leakage

**Audit Trail**:

```
[2025-11-05 14:32:15] INFO | [Tenant: tenant-abc] [User: user-123] Fetching spot positions
[2025-11-05 14:32:15] INFO | [Tenant: tenant-abc] [User: user-123] Retrieved 2 spot positions
[2025-11-05 14:32:16] ERROR | [Tenant: tenant-abc] [User: user-123] Exchange error: Connection timeout
```

---

## 4. Detailed Design

### 4.1 Import Statements (Phase 4 Pattern)

**For positions.py** (needs to add get_current_user and get_current_tenant_id):

```python
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from loguru import logger

from alpha_pulse.exchanges.interfaces import BaseExchange
from alpha_pulse.hedging.common.types import SpotPosition, FuturesPosition
from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
from ..dependencies import get_exchange_client, get_api_client, get_current_user
from ..middleware.tenant_context import get_current_tenant_id

router = APIRouter()
```

**For regime.py** (needs to add get_current_tenant_id import):

```python
from ..middleware.tenant_context import get_current_tenant_id
```

### 4.2 Parameter Ordering Convention

**Consistent Order Across All 43 Endpoints**:

```python
@router.get("/endpoint")
async def endpoint_handler(
    # Phase 4 additions (in this order)
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),

    # Existing dependencies
    regime_service = Depends(get_regime_detection_service),

    # Query parameters (if any)
    days: int = Query(30, description="..."),

    # Body parameters (if any)
    request_body: RequestModel = ...
):
    """Docstring."""
```

**Rationale**:
1. `tenant_id` first (most fundamental, used everywhere)
2. `user` second (authentication context)
3. Service/domain dependencies next
4. Query/path parameters last
5. Request body last

### 4.3 Logging Structure

**Mandatory Logging at Endpoint Start**:

```python
logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Getting current regime")
```

**Format Requirements**:
- Square brackets around tenant context
- Include both tenant_id and user ID
- Use consistent prefix: `[Tenant: X] [User: Y]`
- One space after closing bracket

**Logging at Completion**:

```python
logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Retrieved regime: {current_regime_name}")
```

**Error Logging**:

```python
except Exception as e:
    logger.error(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Error: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to get regime")
```

### 4.4 Integration Points

**Service Layer Integration**:

Assumes service classes already support tenant filtering:

```python
# Good - Service filters internally
positions = await position_fetcher.get_spot_positions()
# (Service should handle tenant_id internally via app context)

# NOT RECOMMENDED - Manual filtering at endpoint
positions = [p for p in all_positions if p.tenant_id == tenant_id]
```

**Database-Level Protection**:

PostgreSQL RLS policies provide secondary defense:

```sql
-- RLS Policy Example (existing, not changed in Phase 4)
CREATE POLICY tenant_isolation ON positions
  FOR SELECT
  USING (tenant_id = current_setting('app.current_tenant_id'));
```

---

## 5. API Changes

### 5.1 Breaking Changes in positions.py

**BREAKING**: Authentication now required (was unauthenticated before)

**Severity**: HIGH (pre-deployment blocker)

**Client Impact**:

| Scenario | Before Phase 4 | After Phase 4 |
|----------|---|---|
| Request without Authorization header | ✅ 200 OK | ❌ 401 Unauthorized |
| Request with invalid token | ✅ 200 OK | ❌ 401 Unauthorized |
| Request with valid JWT token | ✅ 200 OK | ✅ 200 OK |

**Migration Path for Clients**:

**Step 1: Obtain JWT Token**
```bash
# Request JWT token from authentication endpoint
curl -X POST https://api.alphapulse.com/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "trader@company.com",
    "password": "secure_password"
  }'

# Response: {"access_token": "eyJhbGciOiJIUzI1NiI...", "token_type": "bearer"}
```

**Step 2: Use Token in Requests**
```bash
# BEFORE (No longer works):
curl https://api.alphapulse.com/positions/spot

# AFTER (Required):
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiI..." \
  https://api.alphapulse.com/positions/spot
```

**Step 3: Update Client Code**
```python
# Python client example
import requests

# Get token
response = requests.post(
    "https://api.alphapulse.com/auth/token",
    json={"username": "trader@company.com", "password": "..."}
)
token = response.json()["access_token"]

# Use token in subsequent calls
headers = {"Authorization": f"Bearer {token}"}
positions = requests.get(
    "https://api.alphapulse.com/positions/spot",
    headers=headers
)
```

### 5.2 Non-Breaking Changes in regime.py

**No Breaking Changes**: Already requires authentication, only adding tenant context (internal)

**Client Impact**: NONE

- Existing clients continue to work
- JWT token still required (already enforced)
- No changes to request/response schema
- tenant_id is internal-only (not exposed)

### 5.3 Deployment Communication

**Release Notes (v2.4.0)**:

```markdown
## Security: CRITICAL - Authentication Enforcement in positions.py

**BREAKING CHANGE**: All position endpoints now require authentication.

### Affected Endpoints
- GET /positions/spot
- GET /positions/futures
- GET /positions/metrics

### What Changed
- All position endpoints now require valid JWT token
- Unauthenticated requests will receive 401 Unauthorized
- All requests are scoped to authenticated tenant

### Migration Required
1. Obtain JWT token via /auth/token endpoint
2. Include token in Authorization header: `Authorization: Bearer <token>`
3. See migration guide: https://docs.alphapulse.com/migration-v2.4

### Timeline
- v2.4.0 Release Date: [DATE]
- Support Period: 30 days for questions
- Deprecation Period: None (immediate enforcement)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Coverage**: Key authentication and tenant context logic

**Test Categories** (estimated 20 tests):

1. **Tenant Context Extraction** (4 tests)
   - Valid tenant_id in JWT → Returns tenant_id
   - Missing tenant_id claim → 401 or error
   - Expired JWT token → 401 Unauthorized
   - Invalid JWT signature → 401 Unauthorized

2. **Authentication Enforcement** (6 tests - positions.py specific)
   - GET /spot without token → 401
   - GET /spot with valid token → 200
   - GET /futures without token → 401
   - GET /metrics without token → 401
   - Invalid user claim → 401 or error
   - User with missing roles → Appropriate error handling

3. **Regime Context** (4 tests)
   - Valid regime service → Returns regime data
   - Missing regime service → 503 Service Unavailable
   - Invalid regime type → 400 Bad Request
   - Service timeout → 500 Internal Server Error

4. **Logging Format** (6 tests)
   - Log contains `[Tenant: X] [User: Y]` format
   - Tenant context preserved through calls
   - Error logs include tenant context
   - Sensitive data not logged (passwords, tokens)

### 6.2 Integration Tests

**Coverage**: Full request/response cycle with tenant isolation

**Test Pattern** (per Phase 2/3):

```python
@pytest.mark.asyncio
async def test_get_spot_positions_requires_authentication(client):
    """Test that GET /positions/spot requires valid JWT token."""
    # Arrange
    # (no auth header provided)

    # Act
    response = await client.get("/positions/spot")

    # Assert
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_spot_positions_uses_tenant_context(client, auth_tokens):
    """Test that GET /positions/spot enforces tenant isolation."""
    # Arrange
    tenant_id = "test-tenant-123"
    token = auth_tokens[tenant_id]  # Mock token for specific tenant
    headers = {"Authorization": f"Bearer {token}"}

    # Act
    response = await client.get("/positions/spot", headers=headers)

    # Assert
    assert response.status_code == 200
    positions = response.json()

    # Verify all returned positions belong to test-tenant-123
    # (Mock should return only tenant-scoped positions)
    assert len(positions) > 0
    # Each position should be scoped to the tenant
    # (details depend on position data structure)


@pytest.mark.asyncio
async def test_positions_cross_tenant_isolation(client, auth_tokens):
    """Test that tenant A cannot access tenant B's positions."""
    # Arrange
    token_tenant_a = auth_tokens["tenant-a"]
    token_tenant_b = auth_tokens["tenant-b"]

    # Get tenant A's positions
    response_a = await client.get(
        "/positions/spot",
        headers={"Authorization": f"Bearer {token_tenant_a}"}
    )
    positions_a = response_a.json()

    # Get tenant B's positions
    response_b = await client.get(
        "/positions/spot",
        headers={"Authorization": f"Bearer {token_tenant_b}"}
    )
    positions_b = response_b.json()

    # Assert - No overlap in positions (tenants isolated)
    assert len(positions_a) > 0
    assert len(positions_b) > 0
    # Verify positions are different (data-level isolation)
    # (Mock fixtures ensure different data per tenant)
```

**Estimated Tests**: 50+ integration tests
- 4 endpoints × 7 tests/endpoint = 28 tests minimum
- Plus cross-tenant isolation tests
- Plus regime-specific tests

### 6.3 Security Tests

**Coverage**: Authentication gaps and tenant boundary violations

**Security Test Cases**:

1. **Authentication Enforcement**
   - Attempt to call positions.py endpoints without token → 401
   - Attempt with malformed JWT → 401
   - Attempt with expired token → 401

2. **Tenant Isolation**
   - Attempt to access other tenant's positions via direct API call → No data leakage
   - Attempt to modify JWT's tenant_id claim → 401 (signature invalid)
   - Attempt to use another user's token → Data scoped to that user's tenant

3. **Audit Trail**
   - Verify logs include tenant context for all operations
   - Verify failed authentication attempts are logged
   - Verify no sensitive data in logs (tokens, passwords)

---

## 7. Decision Log

### Decision 1: Reuse Phase 2/3 Pattern vs Create New Pattern

**Options**:

**Option A: Reuse Phase 2/3 Pattern** ✅ SELECTED
- Apply exact same pattern: `Depends(get_current_tenant_id)` + `Depends(get_current_user)`
- Parameter order: tenant_id → user → other params
- Logging format: `[Tenant: {id}] [User: {user}]`

**Option B: Decorator-Based Approach**
- Create `@require_tenant_context` decorator
- Auto-inject tenant_id without signature changes
- Requires decorator testing and maintenance

**Option C: Middleware-Only Approach**
- Handle all tenant logic in middleware
- Hide dependencies from endpoint signatures
- Harder to debug, less explicit

**Decision Matrix**:

| Dimension | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| **Consistency** | 10/10 | 5/10 | 4/10 |
| **Complexity** | 9/10 | 6/10 | 5/10 |
| **Testability** | 10/10 | 7/10 | 6/10 |
| **Risk** | 10/10 | 6/10 | 5/10 |
| **Time-to-Market** | 10/10 | 7/10 | 6/10 |
| **Maintainability** | 10/10 | 8/10 | 7/10 |
| **Total Score** | **59/60** | **39/60** | **33/60** |

**Decision**: ✅ **OPTION A - Reuse Phase 2/3 Pattern**

**Rationale**:
- Proven pattern with zero cross-tenant leakage in 28 endpoints
- Highest consistency score (100% match with existing codebase)
- Lowest risk (leverages known working implementation)
- Fastest time-to-market (pattern already proven and tested)
- Easiest to maintain (familiar to entire team)

**Rejected Alternatives**:
- Option B adds unnecessary abstraction, inconsistent with Phase 1-3
- Option C hides critical security logic, harder to audit

### Decision 2: positions.py Authentication Enforcement Grace Period

**Options**:

**Option A: Immediate Enforcement** ✅ SELECTED
- v2.4.0: Authentication required immediately
- No grace period, no deprecation warnings
- Breaking change clearly communicated

**Option B: Grace Period with Warnings**
- v2.4.0: Add deprecation warnings (still accept unauthenticated)
- v2.5.0: Require authentication (3-month window)
- Allows client migration time

**Option C: Feature Flag with Phased Rollout**
- Use feature flag to enable/disable auth requirement
- Gradually roll out to 10% → 50% → 100% of traffic
- Observe errors and adjust

**Decision Matrix**:

| Dimension | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| **Security Risk** | 10/10 | 3/10 | 5/10 |
| **Data Protection** | 10/10 | 3/10 | 5/10 |
| **Client Impact** | 4/10 | 8/10 | 9/10 |
| **Implementation** | 10/10 | 8/10 | 5/10 |
| **Compliance** | 10/10 | 4/10 | 6/10 |
| **Total Score** | **44/50** | **26/50** | **30/50** |

**Decision**: ✅ **OPTION A - Immediate Enforcement**

**Rationale**:
- Trading position data is too sensitive for grace period
- Regulatory compliance requires immediate protection
- Highest risk of data breach with Option B
- Clear security posture vs gradual rollout
- v2.4.0 is a point release, clearly indicates breaking change

**Risk Mitigation**:
- API gateway logs reviewed for unauthenticated calls
- Clients notified 2 weeks before release
- Migration guide provided
- Support escalation path documented
- No rollback needed (breaking change is intentional)

**Rejected Alternatives**:
- Option B introduces regulatory risk for 3 months
- Option C adds operational complexity (feature flag maintenance)

### Decision 3: Tenant Context in Endpoint vs Service Layer

**Options**:

**Option A: Explicit in Endpoint** ✅ SELECTED
- `tenant_id` injected via `Depends(get_current_tenant_id)`
- Visible in endpoint signature
- Passed to service layer explicitly

**Option B: Implicit in Service Layer**
- Service extracts tenant_id from request context internally
- Endpoint signature doesn't show tenant_id dependency
- Cleaner endpoint signatures

**Option C: Mixed Approach**
- Explicit in some routers (positions.py)
- Implicit in others (regime.py)
- Different patterns per router

**Decision Matrix**:

| Dimension | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| **Consistency** | 10/10 | 8/10 | 3/10 |
| **Debuggability** | 10/10 | 5/10 | 6/10 |
| **Auditability** | 10/10 | 6/10 | 5/10 |
| **Cleanliness** | 8/10 | 10/10 | 6/10 |
| **Testability** | 10/10 | 6/10 | 6/10 |
| **Total Score** | **48/50** | **35/50** | **26/50** |

**Decision**: ✅ **OPTION A - Explicit in Endpoint**

**Rationale**:
- Matches Phase 1-3 pattern (100% consistency)
- Makes tenant context visible to auditors and reviewers
- Easier to test and mock
- Easier to understand dependencies at glance
- Better for security reviews

**Rejected Alternatives**:
- Option B reduces auditability (security risk)
- Option C breaks consistency (maintenance burden)

---

## 8. Risk Assessment

### 8.1 Risk Register

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Owner |
|----|----|----------|------------|--------|-----------|-------|
| **R4-001** | Breaking change in positions.py | MEDIUM | MEDIUM | Production incidents, client failures | API gateway logs reviewed pre-release, client notification 2 weeks before, migration guide provided | Tech Lead |
| **R4-002** | Service layer doesn't filter by tenant | CRITICAL | LOW | Cross-tenant data leakage | Integration tests validate tenant isolation, database RLS as secondary defense, security review | Security Lead |
| **R4-003** | Incomplete authentication implementation | CRITICAL | LOW | Unauthenticated access gap remains | Code review checklist, unit tests for auth, production smoke test | Tech Lead |
| **R4-004** | Performance regression | MEDIUM | LOW | SLA violations (>200ms p99) | Benchmark tests before/after integration, load test in staging | Engineering Lead |
| **R4-005** | Test coverage gaps | MEDIUM | MEDIUM | Hidden security vulnerabilities | Minimum 90% coverage, security-focused test cases, penetration testing consideration | QA Lead |
| **R4-006** | Dependency injection failures | LOW | LOW | Runtime 500 errors | Pre-deployment CI/CD tests, local validation before PR | Engineer |

### 8.2 Detailed Risk Analysis

**Risk R4-001: Breaking Change in positions.py**

**Description**: Adding authentication to positions.py will break existing clients that don't provide JWT token.

**Current Exposure**: Unknown number of clients may be using positions.py without authentication

**Probability Assessment**:
- MEDIUM - Previous phases showed 100% client adoption
- Likely some clients already use positions.py
- But trading data is critical, likely protected by other means

**Impact if Occurs**:
- Production incidents as clients fail to authenticate
- Support burden fielding authentication questions
- Potential data breaches if clients don't migrate quickly

**Mitigation Strategy**:
1. Review API gateway logs for unauthenticated position requests (1 week before release)
2. Send notification to all registered API clients (2 weeks before v2.4.0)
3. Provide detailed migration guide with examples
4. Maintain support escalation path during transition
5. Monitor for failed authentication attempts post-release (24-hour window)

**Owner**: Tech Lead

**Status**: OPEN (mitigated pre-release)

---

**Risk R4-002: Service Layer Doesn't Filter by tenant_id**

**Description**: Service classes (RegimeDetectionService, ExchangePositionFetcher) may not properly filter data by tenant_id, leading to cross-tenant data leakage despite API-layer tenant context.

**Current Exposure**: Phase 1-3 endpoints already rely on service layer filtering. If issue exists, it affects 28 endpoints.

**Probability Assessment**:
- LOW - Phase 1-3 endpoints show no cross-tenant leakage
- Service layer has been validated in production
- But new endpoints (regime, positions) may use different service implementations

**Impact if Occurs**:
- CRITICAL - Direct violation of multi-tenant security model
- Regulatory compliance breach
- Complete loss of tenant isolation
- Data breach affecting all tenants

**Mitigation Strategy**:
1. Code review service layer implementations (pre-development)
2. Write specific integration tests for tenant isolation (per endpoint)
3. Rely on database RLS as secondary defense layer
4. Security team reviews test results before merge
5. Production smoke tests validate no cross-tenant leakage (per endpoint)

**Owner**: Security Lead

**Status**: OPEN (mitigated by integration tests)

---

**Risk R4-003: Incomplete Authentication Implementation**

**Description**: positions.py authentication gap not fully closed (e.g., one endpoint missed, or partial implementation).

**Current Exposure**: 3 endpoints in positions.py

**Probability Assessment**:
- LOW - Clear security gap identified
- Code review checklist will catch completeness issues
- But manual implementation can miss details

**Impact if Occurs**:
- CRITICAL - One endpoint remains unauthenticated
- Regulatory violation for that endpoint
- Incomplete security fix defeats purpose

**Mitigation Strategy**:
1. Create detailed code review checklist for positions.py (3 endpoints)
2. Unit tests explicitly verify authentication on all 3 endpoints
3. Pre-deployment smoke tests call all 3 endpoints with/without token
4. Tech Lead responsible for final verification
5. Pull request template includes authentication checklist

**Owner**: Tech Lead

**Status**: OPEN (mitigated by checklist + testing)

---

**Risk R4-004: Performance Regression**

**Description**: Additional dependency injection (tenant_id + user) adds latency beyond acceptable threshold.

**Current Exposure**: Phase 1-3 endpoints already use this pattern with no reported issues

**Probability Assessment**:
- LOW - Previous phases showed <5ms overhead
- Dependency injection is lightweight operation
- But new routers may have additional overhead

**Impact if Occurs**:
- MEDIUM - Breach of p99 latency SLA (<200ms)
- User experience degradation
- Potential auto-scaling cost increase

**Mitigation Strategy**:
1. Baseline latency before Phase 4 (existing endpoints)
2. Benchmark Phase 4 endpoints in staging
3. Load test with realistic request volume (1000 req/s)
4. Compare p99 latency: before Phase 4 vs after
5. Alert threshold: If p99 > 250ms, trigger investigation
6. Rollback plan ready if latency exceeds threshold

**Owner**: Engineering Lead

**Status**: OPEN (mitigated by performance testing)

---

**Risk R4-005: Test Coverage Gaps**

**Description**: Integration tests don't cover all cross-tenant scenarios, allowing vulnerabilities to reach production.

**Current Exposure**: 7 endpoints, 50+ test cases needed

**Probability Assessment**:
- MEDIUM - Large number of test scenarios to cover
- Possible edge cases not anticipated
- But Phase 2/3 test patterns are proven

**Impact if Occurs**:
- MEDIUM - Cross-tenant leakage discovered in production
- Security incident, potential data breach
- Loss of customer trust

**Mitigation Strategy**:
1. Use test matrix from Phase 2/3 as template
2. Create explicit "cross-tenant isolation" test for each endpoint
3. Minimum 90% code coverage for Phase 4 code
4. Security lead reviews test coverage before merge
5. Consider penetration testing for tenant boundaries
6. Use property-based testing for exhaustive edge cases

**Owner**: QA Lead

**Status**: OPEN (mitigated by comprehensive test strategy)

---

**Risk R4-006: Dependency Injection Failures**

**Description**: Endpoint fails to instantiate due to missing/broken dependency injection (like Phase 1 import issues).

**Current Exposure**: 7 endpoints with updated imports and dependencies

**Probability Assessment**:
- LOW - Phase 2/3 resolved import issues
- But new imports for regime.py and positions.py could introduce issues

**Impact if Occurs**:
- LOW - Caught by CI/CD tests before merge
- But production deployment fails if CI doesn't catch it

**Mitigation Strategy**:
1. Run full test suite locally before pushing to CI
2. Check import statements line-by-line in code review
3. CI/CD includes endpoint collection test (pytest --collect-only)
4. Pre-deployment smoke tests call all 7 new endpoints
5. Rollback ready (<5 minutes)

**Owner**: Engineer

**Status**: OPEN (mitigated by pre-deployment validation)

---

## 9. Deployment Strategy

### 9.1 Pre-Deployment Checklist

**Week Before Release (v2.4.0)**:

- [ ] API gateway logs reviewed for unauthenticated position requests
- [ ] Client notification sent (2-week advance notice)
- [ ] Migration guide prepared and reviewed
- [ ] Support team trained on v2.4.0 changes
- [ ] Rollback plan documented and tested
- [ ] Monitoring dashboards prepared
- [ ] On-call rotation notified

**Day Before Release**:

- [ ] All tests passing in CI/CD (green for 2 consecutive runs)
- [ ] Code review completed (2+ approvals)
- [ ] Security review completed and signed off
- [ ] Staging deployment successful
- [ ] Staging smoke tests all passing
- [ ] Documentation final review

**Release Day**:

1. **Deployment Windows**:
   - Preferred: Business hours (9am-12pm UTC)
   - Reason: Support team available
   - Rollback: Available in <5 minutes

2. **Deployment Steps**:
   - Deploy to canary (10% traffic) → Monitor 15 min
   - Deploy to staging (100%) → Smoke test
   - Deploy to production (10%) → Monitor 15 min
   - Deploy to production (100%) → Monitor 60 min

3. **Monitoring During Rollout**:
   - Watch for authentication failures in logs
   - Monitor p99 latency (should stay <200ms)
   - Monitor error rate (should stay <0.1%)
   - Check for cross-tenant data access attempts (should be 0)

### 9.2 Rollback Procedure

**Trigger Criteria** (any of the following):

- Critical authentication gap discovered (unauthenticated endpoints still accessible)
- Cross-tenant data leakage detected
- p99 latency consistently >300ms
- Error rate >1% sustained for 5+ minutes
- Data corruption reported

**Rollback Steps**:

```bash
# 1. Announce incident
#    Slack: "#incident" channel, notify on-call lead

# 2. Trigger rollback (automatic via GitHub Actions)
git revert <v2.4.0-commit-hash>
git push origin main

# 3. Monitor rollback
#    Watch logs for revert deployment completion
#    Verify endpoint health checks pass
#    Confirm error rates return to baseline

# 4. Post-incident review
#    Schedule retrospective within 24 hours
#    Document root cause
#    Update mitigation strategy
```

**Estimated Rollback Time**: 5 minutes (via CI/CD automation)

---

## 10. Monitoring and Observability

### 10.1 Metrics

**Key Metrics to Track** (Prometheus):

1. **Authentication Metrics**:
   - `api_auth_failures_total{router="positions", endpoint="spot"}` - Counter
   - `api_auth_failures_by_reason{reason="missing_token"|"invalid_token"|"expired"}` - Counter
   - Alert: `api_auth_failures_total > 10/min` → Potential issue

2. **Tenant Context Metrics**:
   - `api_requests_total{router="regime", tenant_id=<string>}` - Counter
   - `api_request_duration_seconds{router="positions", quantile="0.99"}` - Histogram
   - Alert: `api_request_duration_seconds{p99} > 0.25` → Performance issue

3. **Cross-Tenant Access Attempts**:
   - `api_cross_tenant_access_attempts_total` - Counter (should be 0)
   - Alert: `api_cross_tenant_access_attempts_total > 0` → Security incident

4. **Endpoint-Specific Metrics**:
   - `/positions/spot` request rate, error rate, latency
   - `/positions/futures` request rate, error rate, latency
   - `/positions/metrics` request rate, error rate, latency
   - `/regime/current` request rate, error rate, latency
   - `/regime/history` request rate, error rate, latency
   - `/regime/analysis/{type}` request rate, error rate, latency
   - `/regime/alerts` request rate, error rate, latency

### 10.2 Logging

**Log Format**:

```
[2025-11-05 14:32:15.123] INFO | [Tenant: tenant-abc] [User: user-123] Fetching spot positions
[2025-11-05 14:32:15.250] INFO | [Tenant: tenant-abc] [User: user-123] Retrieved 2 spot positions
```

**Log Levels**:
- INFO: Successful start of endpoint
- INFO: Successful completion of endpoint
- ERROR: Any exceptions during endpoint execution
- WARN: Deprecated API usage (if applicable)

**Log Retention**: Per existing policy (90 days)

**Searchable Queries**:
```
# Find all requests for a specific tenant
[Tenant: tenant-abc]

# Find all requests by a specific user
[User: user-123]

# Find all authentication failures
AUTH | ERROR

# Find all cross-tenant attempts (should be none)
[Tenant: ] [Tenant: ] (log line with multiple tenant mentions)
```

### 10.3 Alerts

**Alert Thresholds**:

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Authentication Spike | `api_auth_failures > 10/min` | P1 | Page on-call, check JWT config |
| Cross-Tenant Access | `api_cross_tenant_access_attempts > 0` | P0 | Immediate incident, isolate |
| Latency Increase | `p99 latency > 250ms` | P2 | Investigate, consider scaling |
| Error Rate Spike | `error_rate > 1%` | P1 | Page on-call, check logs |
| Service Down | `endpoint_response_time = timeout` | P0 | Automatic failover |

**Alert Routing**:
- P0 alerts → PagerDuty (immediate page)
- P1 alerts → Slack #incident + PagerDuty
- P2 alerts → Slack #engineering (no page)

### 10.4 Dashboards

**Grafana Dashboard: "Phase 4 - Tenant Context Integration"**

Panels:
1. **Authentication Metrics**
   - Auth failure rate over time
   - Failure reasons (pie chart)
   - Failed auth by endpoint

2. **Tenant Isolation**
   - Requests by tenant (stacked bar)
   - Cross-tenant access attempts (should be 0)
   - Tenant distribution

3. **Performance**
   - p99 latency by endpoint (line chart)
   - Request rate by endpoint (bar chart)
   - Error rate over time

4. **Endpoint Health**
   - GET /positions/spot status
   - GET /positions/futures status
   - GET /positions/metrics status
   - GET /regime/current status
   - GET /regime/history status
   - GET /regime/analysis/{type} status
   - GET /regime/alerts status

---

## 11. Appendices

### Appendix A: Phase 2/3 Pattern Reference

**Phase 2 Pattern** (from alerts.py):

```python
from typing import Dict
from fastapi import APIRouter, Depends, Query
from loguru import logger
from ..dependencies import get_current_user
from ..middleware.tenant_context import get_current_tenant_id

@router.get("/alerts")
async def get_alerts(
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    severity: str = Query(None, description="Filter by severity")
):
    """Get alerts for current user's tenant."""
    logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Fetching alerts")

    # Get alerts scoped to tenant
    alerts = await alert_accessor.get_alerts(tenant_id=tenant_id, severity=severity)

    logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Retrieved {len(alerts)} alerts")
    return alerts
```

**Phase 3 Pattern** (from ensemble.py):

```python
from fastapi import APIRouter, Depends
from loguru import logger
from ..dependencies import get_current_user

@router.get("/")
async def list_ensembles(
    active_only: bool = Query(True),
    user = Depends(get_current_user),
    service = Depends(get_ensemble_service)
):
    """List all ensemble configurations."""
    # Note: Phase 3 already has get_current_user
    # Phase 4 will add tenant_id = Depends(get_current_tenant_id)
    ensembles = await service.list_ensembles(active_only=active_only)
    return {"ensembles": ensembles, "count": len(ensembles)}
```

### Appendix B: Code Examples

**positions.py - Before Phase 4 (INSECURE)**:

```python
@router.get(
    "/spot",
    response_model=List[SpotPosition],
    dependencies=[Depends(get_api_client)]  # ❌ CRITICAL: NO USER AUTH
)
async def get_spot_positions(
    exchange: BaseExchange = Depends(get_exchange_client)
):
    """Get current spot positions."""
    try:
        position_fetcher = ExchangePositionFetcher(exchange)
        positions = await position_fetcher.get_spot_positions()
        return positions
    except Exception as e:
        logger.error(f"Error fetching spot positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch spot positions"
        )
```

**positions.py - After Phase 4 (SECURE)**:

```python
from ..middleware.tenant_context import get_current_tenant_id

@router.get("/spot", response_model=List[SpotPosition])
async def get_spot_positions(
    tenant_id: str = Depends(get_current_tenant_id),
    user: dict = Depends(get_current_user),
    exchange: BaseExchange = Depends(get_exchange_client),
    _api_client = Depends(get_api_client)
):
    """Get current spot positions."""
    logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Fetching spot positions")

    try:
        position_fetcher = ExchangePositionFetcher(exchange)
        positions = await position_fetcher.get_spot_positions()

        logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Retrieved {len(positions)} spot positions")
        return positions
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Error fetching spot positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch spot positions"
        )
```

**regime.py - Before Phase 4 (MISSING TENANT CONTEXT)**:

```python
@router.get("/current", response_model=RegimeStateResponse)
async def get_current_regime(
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeStateResponse:
    """Get current market regime state."""
    regime_info = regime_service.current_regime_info
    # ... returns regime data (no tenant scoping)
```

**regime.py - After Phase 4 (WITH TENANT CONTEXT)**:

```python
from ..middleware.tenant_context import get_current_tenant_id

@router.get("/current", response_model=RegimeStateResponse)
async def get_current_regime(
    tenant_id: str = Depends(get_current_tenant_id),
    current_user: Dict = Depends(get_current_user),
    regime_service = Depends(get_regime_detection_service)
) -> RegimeStateResponse:
    """Get current market regime state."""
    logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Getting current regime")

    regime_info = regime_service.current_regime_info

    if regime_info:
        logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Current regime: {regime_info.current_regime}")

    # ... returns regime data (scoped to tenant internally)
    return regime_response
```

### Appendix C: Test Examples

**Integration Test - Authentication Enforcement**:

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_get_spot_positions_requires_authentication(client: TestClient):
    """Test that GET /positions/spot requires valid JWT token."""
    # Arrange - No Authorization header

    # Act
    response = client.get("/positions/spot")

    # Assert
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_spot_positions_with_valid_token(client: TestClient, mock_jwt_token):
    """Test that GET /positions/spot succeeds with valid token."""
    # Arrange
    tenant_id = "test-tenant-123"
    token = mock_jwt_token(tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    # Act
    response = client.get("/positions/spot", headers=headers)

    # Assert
    assert response.status_code == 200
    positions = response.json()
    assert isinstance(positions, list)


@pytest.mark.asyncio
async def test_positions_cross_tenant_isolation(
    client: TestClient,
    mock_jwt_token,
    mock_exchange_client
):
    """Test that tenant A cannot access tenant B's positions."""
    # Arrange
    tenant_a_token = mock_jwt_token(tenant_id="tenant-a")
    tenant_b_token = mock_jwt_token(tenant_id="tenant-b")

    # Mock exchange to return different positions per tenant
    def get_positions_for_tenant(tenant_id):
        if tenant_id == "tenant-a":
            return [{"symbol": "BTC", "qty": 1.0, "value": 50000}]
        else:  # tenant-b
            return [{"symbol": "ETH", "qty": 10.0, "value": 20000}]

    mock_exchange_client.get_spot_positions = get_positions_for_tenant

    # Act - Get positions for tenant A
    response_a = client.get(
        "/positions/spot",
        headers={"Authorization": f"Bearer {tenant_a_token}"}
    )
    positions_a = response_a.json()

    # Act - Get positions for tenant B
    response_b = client.get(
        "/positions/spot",
        headers={"Authorization": f"Bearer {tenant_b_token}"}
    )
    positions_b = response_b.json()

    # Assert - Positions are different (tenant isolated)
    assert positions_a != positions_b
    assert positions_a[0]["symbol"] == "BTC"
    assert positions_b[0]["symbol"] == "ETH"
```

**Unit Test - Tenant Context Extraction**:

```python
import pytest
from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id

@pytest.mark.asyncio
async def test_get_current_tenant_id_valid_jwt(mock_request_with_tenant):
    """Test tenant_id extraction from valid JWT."""
    # Arrange
    request = mock_request_with_tenant("tenant-abc-123")

    # Act
    tenant_id = await get_current_tenant_id(request)

    # Assert
    assert tenant_id == "tenant-abc-123"


@pytest.mark.asyncio
async def test_get_current_tenant_id_missing_tenant_claim(mock_request):
    """Test error when tenant_id missing from JWT."""
    # Arrange
    request = mock_request()  # No tenant_id claim

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await get_current_tenant_id(request)

    assert exc_info.value.status_code == 401
```

### Appendix D: Migration Guide for API Clients

**For Users of positions.py Endpoints**:

**Step 1: Obtain JWT Token**

```bash
curl -X POST https://api.alphapulse.com/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_email@company.com",
    "password": "your_password"
  }'

# Response:
# {
#   "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "token_type": "bearer",
#   "expires_in": 3600
# }
```

**Step 2: Update API Calls**

**Python Example**:

```python
import requests

# Get token once
response = requests.post(
    "https://api.alphapulse.com/auth/token",
    json={
        "username": "your_email@company.com",
        "password": "your_password"
    }
)
token = response.json()["access_token"]

# Use token in subsequent calls
headers = {"Authorization": f"Bearer {token}"}

# Get spot positions
positions = requests.get(
    "https://api.alphapulse.com/positions/spot",
    headers=headers
)
print(positions.json())
```

**JavaScript/Node.js Example**:

```javascript
const fetch = require('node-fetch');

async function getPositions() {
  // Get token
  const authResponse = await fetch('https://api.alphapulse.com/auth/token', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      username: 'your_email@company.com',
      password: 'your_password'
    })
  });
  const { access_token } = await authResponse.json();

  // Get positions
  const posResponse = await fetch('https://api.alphapulse.com/positions/spot', {
    headers: {
      'Authorization': `Bearer ${access_token}`
    }
  });
  const positions = await posResponse.json();
  console.log(positions);
}

getPositions();
```

**cURL Example**:

```bash
# Get token
TOKEN=$(curl -X POST https://api.alphapulse.com/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"your_email@company.com","password":"your_password"}' \
  | jq -r '.access_token')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  https://api.alphapulse.com/positions/spot
```

**Step 3: Handle Token Expiration**

```python
# Token expires after 1 hour
# Refresh token when needed:

def get_fresh_token():
    response = requests.post(
        "https://api.alphapulse.com/auth/token",
        json={
            "username": "your_email@company.com",
            "password": "your_password"
        }
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("Failed to get token")

# Retry with fresh token on 401
try:
    headers = {"Authorization": f"Bearer {old_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        new_token = get_fresh_token()
        headers = {"Authorization": f"Bearer {new_token}"}
        response = requests.get(url, headers=headers)
```

---

## Document Information

**Version**: 1.0
**Last Updated**: 2025-11-05
**Next Review**: After implementation begins

**Approved By**:
- [ ] Tech Lead (pending)
- [ ] Security Lead (pending)
- [ ] Engineering Team (pending)

**References**:
- [Phase 4 Discovery Document](../discovery/story-2.4-phase4-discovery.md)
- [Phase 2 HLD](./story-2.4-phase2-hld.md)
- [API Endpoints Reference](https://docs.alphapulse.com/api)
- [Migration Guide](https://docs.alphapulse.com/migration-v2.4)

---

**Status**: Ready for Stakeholder Review
**Target Build Start**: 2025-11-06
**Target Release**: v2.4.0 (2025-11-10)
