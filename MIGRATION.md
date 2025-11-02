# Migration Guide

This document provides guidance for upgrading between major versions of AlphaPulse.

## Upgrading to v2.0.0 (Multi-Tenant Support)

### Overview

Version 2.0.0 introduces **multi-tenant support** to the AlphaPulse trading system, enabling data isolation between different tenants (users/organizations). This is a **breaking change** that requires code updates for all applications using `AgentManager`.

### Breaking Changes

#### AgentManager API Changes

The following `AgentManager` methods now **require** a `tenant_id` parameter:

1. `generate_signals(market_data, tenant_id: str)`
2. `register_agent(agent, tenant_id: str)`
3. `create_and_register_agent(agent_type, tenant_id, config=None)`
4. `_aggregate_signals_with_ensemble(signals, tenant_id: str)` (internal method)

### Migration Steps

#### Before (v1.x.x)

```python
from alpha_pulse.agents.manager import AgentManager
from alpha_pulse.agents.interfaces import MarketData

# Initialize manager
manager = AgentManager(config=config)
await manager.initialize()

# Generate signals (OLD API)
market_data = MarketData(prices=prices_df, volumes=volumes_df)
signals = await manager.generate_signals(market_data)

# Register agent (OLD API)
await manager.register_agent(technical_agent)

# Create and register agent (OLD API)
agent = await manager.create_and_register_agent("technical", config)
```

#### After (v2.0.0)

```python
from alpha_pulse.agents.manager import AgentManager
from alpha_pulse.agents.interfaces import MarketData

# Initialize manager (unchanged)
manager = AgentManager(config=config)
await manager.initialize()

# Get tenant_id from request context (example using FastAPI)
# In production, this comes from JWT token or authenticated session
tenant_id = request.state.tenant_id  # e.g., "00000000-0000-0000-0000-000000000001"

# Generate signals (NEW API - requires tenant_id)
market_data = MarketData(prices=prices_df, volumes=volumes_df)
signals = await manager.generate_signals(market_data, tenant_id=tenant_id)

# Register agent (NEW API - requires tenant_id)
await manager.register_agent(technical_agent, tenant_id=tenant_id)

# Create and register agent (NEW API - requires tenant_id)
agent = await manager.create_and_register_agent("technical", tenant_id=tenant_id, config=config)
```

### Error Handling

If you call these methods without `tenant_id`, you'll receive a clear error:

```python
# This will raise ValueError
signals = await manager.generate_signals(market_data)

# Error message:
# ValueError: AgentManager.generate_signals requires 'tenant_id' parameter.
# Multi-tenant context is mandatory for data isolation.
```

### Where to Get tenant_id

The `tenant_id` should come from your authentication layer:

#### Option 1: FastAPI with JWT (Recommended)

```python
from fastapi import Request, Depends
from alpha_pulse.api.auth import get_current_user

@app.post("/api/signals")
async def generate_signals(
    request: Request,
    current_user = Depends(get_current_user)
):
    # Extract tenant_id from authenticated user
    tenant_id = current_user.tenant_id

    # Use tenant_id in AgentManager calls
    signals = await agent_manager.generate_signals(
        market_data,
        tenant_id=tenant_id
    )
    return signals
```

#### Option 2: Middleware (Automatic)

```python
from fastapi import FastAPI
from alpha_pulse.api.middleware import TenantContextMiddleware

app = FastAPI()
app.add_middleware(TenantContextMiddleware)

@app.post("/api/signals")
async def generate_signals(request: Request):
    # tenant_id automatically extracted by middleware
    tenant_id = request.state.tenant_id

    signals = await agent_manager.generate_signals(
        market_data,
        tenant_id=tenant_id
    )
    return signals
```

#### Option 3: Direct Testing/Scripts

For testing or single-tenant scripts, use a fixed tenant ID:

```python
# Use default tenant ID for single-tenant deployments
DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"

signals = await manager.generate_signals(
    market_data,
    tenant_id=DEFAULT_TENANT_ID
)
```

### Benefits of This Change

1. **Data Isolation**: Each tenant's trading signals are isolated and tagged
2. **Audit Trail**: All operations are logged with tenant context
3. **Scalability**: System is now ready for SaaS multi-tenancy
4. **Security**: Prevents accidental data leakage between tenants
5. **Compliance**: Enables tenant-specific regulatory reporting

### Signal Metadata

All generated signals now include `tenant_id` in metadata:

```python
signal = signals[0]
print(signal.metadata["tenant_id"])  # "00000000-0000-0000-0000-000000000001"
print(signal.metadata["agent_type"])  # "technical"
```

### Logging Changes

All log messages now include tenant context:

```
# Before
[INFO] Agent technical generated 5 signals

# After
[INFO] [Tenant: 00000000-0000-0000-0000-000000000001] Agent technical generated 5 signals
```

This makes it easy to filter logs by tenant in production.

### Testing

Update your test fixtures to include `tenant_id`:

```python
import pytest

@pytest.fixture
def default_tenant_id():
    """Default tenant UUID for testing."""
    return "00000000-0000-0000-0000-000000000001"

@pytest.mark.asyncio
async def test_generate_signals(agent_manager, market_data, default_tenant_id):
    # Pass tenant_id to all AgentManager calls
    signals = await agent_manager.generate_signals(
        market_data,
        tenant_id=default_tenant_id
    )
    assert len(signals) > 0
```

### Compatibility Matrix

| AlphaPulse Version | Database Schema | API Compatibility | Notes |
|-------------------|-----------------|-------------------|-------|
| v1.x.x | Pre-RLS | v1 API only | Single-tenant |
| v2.0.0 | RLS enabled | v2 API only | Multi-tenant (BREAKING) |

### Database Requirements

Multi-tenant support requires the following database features to be enabled:

1. Row-Level Security (RLS) policies on all tables
2. `tenant_id` column on all domain tables
3. PostgreSQL session variable: `app.current_tenant_id`

See [migrations/alembic/versions/008_enable_rls_policies.py](migrations/alembic/versions/008_enable_rls_policies.py) for details.

### Rollback Plan

If you need to rollback to v1.x.x:

1. Database is forward-compatible (has tenant_id columns but doesn't require them)
2. Downgrade application code to v1.x.x
3. System will continue to work in single-tenant mode
4. **Warning**: You'll lose tenant isolation benefits

---

## RiskManager API Changes (Story 2.3)

### Overview

RiskManager now requires `tenant_id` parameter for all risk management operations, enabling tenant-specific risk limits and audit trails.

### Breaking Changes

The following `RiskManager` methods now **require** a `tenant_id` parameter:

1. `calculate_risk_exposure(tenant_id: str)`
2. `evaluate_trade(..., tenant_id: str)`
3. `calculate_position_size(..., tenant_id: str)`

### Migration Steps

#### Before (v1.x.x)

```python
from alpha_pulse.risk_management.manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager(exchange=exchange, config=risk_config)

# Calculate risk exposure (OLD API)
exposure = await risk_manager.calculate_risk_exposure()

# Evaluate trade (OLD API)
approved = await risk_manager.evaluate_trade(
    symbol="BTC/USDT",
    side="buy",
    quantity=0.5,
    current_price=50000.0,
    portfolio_value=100000.0,
    current_positions={}
)

# Calculate position size (OLD API)
position = await risk_manager.calculate_position_size(
    symbol="BTC/USDT",
    current_price=50000.0,
    signal_strength=0.8
)
```

#### After (v2.0.0)

```python
from alpha_pulse.risk_management.manager import RiskManager

# Initialize risk manager (unchanged)
risk_manager = RiskManager(exchange=exchange, config=risk_config)

# Get tenant_id from request context (example using FastAPI)
tenant_id = request.state.tenant_id  # e.g., "00000000-0000-0000-0000-000000000001"

# Calculate risk exposure (NEW API - requires tenant_id)
exposure = await risk_manager.calculate_risk_exposure(tenant_id=tenant_id)

# Evaluate trade (NEW API - requires tenant_id)
approved = await risk_manager.evaluate_trade(
    symbol="BTC/USDT",
    side="buy",
    quantity=0.5,
    current_price=50000.0,
    portfolio_value=100000.0,
    current_positions={},
    tenant_id=tenant_id
)

# Calculate position size (NEW API - requires tenant_id)
position = await risk_manager.calculate_position_size(
    symbol="BTC/USDT",
    current_price=50000.0,
    signal_strength=0.8,
    tenant_id=tenant_id
)
```

### Error Handling

If you call these methods without `tenant_id`, you'll receive a clear error:

```python
# This will raise ValueError
exposure = await risk_manager.calculate_risk_exposure()

# Error message:
# ValueError: RiskManager.calculate_risk_exposure requires 'tenant_id' parameter.
# Multi-tenant context is mandatory for data isolation.
```

### API Integration Example

#### FastAPI Endpoint with RiskManager

```python
from fastapi import APIRouter, Depends, Request
from alpha_pulse.risk_management.manager import RiskManager
from alpha_pulse.api.auth import get_current_user

router = APIRouter()

@router.post("/trades/evaluate")
async def evaluate_trade_endpoint(
    request: Request,
    symbol: str,
    side: str,
    quantity: float,
    current_user = Depends(get_current_user),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    # Extract tenant_id from authenticated user
    tenant_id = current_user.tenant_id

    # Get portfolio data
    portfolio_value = await get_portfolio_value(tenant_id)
    current_positions = await get_positions(tenant_id)
    current_price = await get_current_price(symbol)

    # Evaluate trade with tenant context
    approved = await risk_manager.evaluate_trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        current_price=current_price,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
        tenant_id=tenant_id
    )

    return {"approved": approved, "tenant_id": tenant_id}
```

### Risk Exposure Metadata

Risk exposure results now include `tenant_id` in the response:

```python
exposure = await risk_manager.calculate_risk_exposure(tenant_id=tenant_id)

# Result structure:
{
    "BTC_net_exposure": 75000.0,
    "BTC_exposure_pct": 0.15,
    "ETH_net_exposure": 30000.0,
    "ETH_exposure_pct": 0.06,
    "total_exposure": 105000.0,
    "exposure_ratio": 0.21,
    "tenant_id": "00000000-0000-0000-0000-000000000001"  # ← New field
}
```

### Logging Changes

All RiskManager log messages now include tenant context:

```
# Before
[INFO] Evaluating trade: BTC/USDT buy 0.5 @ 50000
[DEBUG] Position value: 25000, Position size: 25.00%
[DEBUG] Trade passed all risk checks

# After
[INFO] [Tenant: 00000000-0000-0000-0000-000000000001] Evaluating trade: BTC/USDT buy 0.5 @ 50000
[DEBUG] [Tenant: 00000000-0000-0000-0000-000000000001] Position value: 25000, Position size: 25.00%
[DEBUG] [Tenant: 00000000-0000-0000-0000-000000000001] Trade passed all risk checks
```

This makes it easy to filter logs by tenant in production.

### Testing

Update your test fixtures to include `tenant_id`:

```python
import pytest

@pytest.fixture
def default_tenant_id():
    """Default tenant UUID for testing."""
    return "00000000-0000-0000-0000-000000000001"

@pytest.mark.asyncio
async def test_evaluate_trade(risk_manager, default_tenant_id):
    # Pass tenant_id to all RiskManager calls
    approved = await risk_manager.evaluate_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.5,
        current_price=50000.0,
        portfolio_value=100000.0,
        current_positions={},
        tenant_id=default_tenant_id
    )
    assert approved is True
```

### Benefits of RiskManager Multi-Tenant Support

1. **Tenant-Specific Risk Limits**: Each tenant can have independent position size limits and leverage caps
2. **Audit Trail**: All risk decisions are logged with tenant context for compliance
3. **Risk Budget Integration**: Supports tenant-specific dynamic risk budgets
4. **Data Isolation**: Prevents cross-tenant risk calculation leakage
5. **Scalability**: Enables SaaS multi-tenancy for risk management

---

### Need Help?

- **Issues**: https://github.com/blackms/AlphaPulse/issues
- **Documentation**: See [EPIC-002](https://github.com/blackms/AlphaPulse/issues/162)
- **Story 2.2**: [AgentManager Multi-Tenant](https://github.com/blackms/AlphaPulse/issues/163)
- **Story 2.3**: [RiskManager Multi-Tenant](https://github.com/blackms/AlphaPulse/issues/189)

---

## API Endpoints Tenant Context (Story 2.4)

### Overview

API endpoints now use tenant context from `TenantContextMiddleware` via dependency injection. All endpoints must extract `tenant_id` from the middleware and pass it to underlying services.

### Breaking Changes

**All API endpoints** now require:
1. JWT token with `tenant_id` claim in Authorization header
2. `tenant_id` parameter injected via `Depends(get_current_tenant_id)`
3. Passing `tenant_id` to all service calls

### Migration Steps

#### Before (v1.x.x)

```python
from fastapi import APIRouter
from ..dependencies import get_risk_manager

router = APIRouter()

@router.get("/risk/exposure")
async def get_risk_exposure(
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    # OLD API - no tenant context
    exposure = await risk_manager.calculate_risk_exposure()
    return exposure
```

#### After (v2.0.0)

```python
from fastapi import APIRouter, Depends
from ..dependencies import get_risk_manager
from ..middleware.tenant_context import get_current_tenant_id  # ← Import dependency

router = APIRouter()

@router.get("/risk/exposure")
async def get_risk_exposure(
    tenant_id: str = Depends(get_current_tenant_id),  # ← Add tenant context
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    # NEW API - includes tenant context
    exposure = await risk_manager.calculate_risk_exposure(
        tenant_id=tenant_id  # ← Pass to service
    )
    return exposure
```

### Client-Side Changes

#### Authentication Required

All API requests must include a valid JWT token with `tenant_id` claim:

```python
import requests

# Login to get token
response = requests.post(
    "https://api.alphapulse.io/token",
    data={"username": "admin", "password": "admin123!@#"}
)
token_data = response.json()
access_token = token_data["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {access_token}"}

# All API calls now require authentication
portfolio = requests.get(
    "https://api.alphapulse.io/api/v1/portfolio",
    headers=headers
).json()
```

#### Token Structure

JWT tokens now include `tenant_id` claim:

```json
{
  "sub": "admin",
  "tenant_id": "00000000-0000-0000-0000-000000000001",
  "exp": 1730577600
}
```

### Error Handling

#### Missing Authorization Header

```python
# Request without Authorization header
response = requests.get("https://api.alphapulse.io/api/v1/portfolio")

# Response: 401 Unauthorized
{
  "detail": "Missing Authorization header"
}
```

#### Missing tenant_id Claim

```python
# Token without tenant_id claim
response = requests.get(
    "https://api.alphapulse.io/api/v1/portfolio",
    headers={"Authorization": f"Bearer {invalid_token}"}
)

# Response: 401 Unauthorized
{
  "detail": "Missing tenant_id claim in JWT token"
}
```

### Affected Endpoints

All endpoints under `/api/v1/` now require tenant context:

**Critical (P0)**:
- `/api/v1/portfolio` - Portfolio data
- `/api/v1/trades` - Trade history
- `/api/v1/risk/*` - Risk management
- `/api/v1/risk-budget/*` - Risk budgeting

**Important (P1)**:
- `/api/v1/alerts` - Alert management
- `/api/v1/metrics/*` - Performance metrics
- `/api/v1/ensemble/*` - Agent ensembles
- `/api/v1/hedging/*` - Hedging positions
- `/api/v1/regime/*` - Market regimes

**Supporting (P2-P3)**:
- `/api/v1/correlation/*` - Correlation analysis
- `/api/v1/backtesting/*` - Backtesting
- `/api/v1/data-lake/*` - Data lake operations
- `/api/v1/liquidity/*` - Liquidity analysis
- All other endpoints

**Exempt Paths** (No authentication required):
- `/health` - Health check
- `/metrics` - Prometheus metrics
- `/docs` - OpenAPI documentation
- `/openapi.json` - OpenAPI spec
- `/token` - Login endpoint

### OpenAPI Documentation Changes

The OpenAPI schema now reflects tenant context requirements:

```yaml
paths:
  /api/v1/portfolio:
    get:
      summary: Get Portfolio
      security:
        - HTTPBearer: []  # ← JWT required
      responses:
        '200':
          description: Successful Response
        '401':
          description: Unauthorized - Missing or invalid token
```

### Testing Changes

Update API tests to include tenant context:

```python
import pytest
from fastapi.testclient import TestClient
from jose import jwt
from datetime import datetime, timedelta

def create_test_token(username: str, tenant_id: str) -> str:
    """Create JWT token for testing."""
    payload = {
        "sub": username,
        "tenant_id": tenant_id,
        "exp": datetime.utcnow() + timedelta(minutes=30)
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")

def test_get_portfolio():
    client = TestClient(app)

    # Create token with tenant_id
    token = create_test_token(
        "testuser",
        "00000000-0000-0000-0000-000000000001"
    )

    # Make authenticated request
    response = client.get(
        "/api/v1/portfolio",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
```

### Python SDK Example

If using a Python SDK/client library:

```python
from alpha_pulse_client import AlphaPulseClient

# Initialize client with credentials
client = AlphaPulseClient(
    base_url="https://api.alphapulse.io",
    username="admin",
    password="admin123!@#"
)

# Client automatically handles:
# 1. Login to get JWT token
# 2. Extracting tenant_id from token
# 3. Including Authorization header in all requests

# All API calls now automatically include tenant context
portfolio = client.get_portfolio()
trades = client.get_trades()
exposure = client.get_risk_exposure()
```

### JavaScript/TypeScript Example

```typescript
import axios from 'axios';

const API_BASE_URL = 'https://api.alphapulse.io';

// Login to get token
const loginResponse = await axios.post(`${API_BASE_URL}/token`, {
  username: 'admin',
  password: 'admin123!@#'
});

const accessToken = loginResponse.data.access_token;

// Create authenticated axios instance
const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});

// All requests now include tenant context
const portfolio = await apiClient.get('/portfolio');
const trades = await apiClient.get('/trades');
const exposure = await apiClient.get('/risk/exposure');
```

### Benefits of API Tenant Context

1. **Automatic Tenant Isolation**: Middleware ensures all requests are tenant-scoped
2. **Security**: JWT validation prevents cross-tenant access
3. **Audit Trail**: All API operations logged with tenant context
4. **Type Safety**: FastAPI validates tenant_id parameter
5. **Developer Experience**: Clear errors if authentication missing
6. **PostgreSQL RLS Integration**: Session variable set automatically

### Implementation Status

**Current Status**: ⚠️ **BLOCKED**

- ✅ TenantContextMiddleware implemented and registered
- ✅ JWT tokens include tenant_id claim
- ✅ Dependency injection pattern defined
- ✅ Test suite created (19 tests)
- ✅ Documentation complete (ADR-003, Implementation Guide)
- ❌ **BLOCKED by pre-existing circular import** (Issue #191)
- ⏳ Endpoint updates pending (100+ endpoints)

**Blocking Issue**: Circular import in `data_lake/lake_manager.py` prevents API from starting and tests from running.

**Next Steps** (After #191 resolved):
1. Phase 1: Update P0 endpoints (risk, portfolio, trades) - 14 endpoints
2. Phase 2: Update P1 endpoints (alerts, metrics, ensemble) - 26 endpoints
3. Phase 3: Update P2-P3 endpoints (backtesting, data_lake, etc.) - 60+ endpoints

---

### Need Help?

- **Issues**: https://github.com/blackms/AlphaPulse/issues
- **Documentation**:
  - [ADR-003](docs/adr/003-api-tenant-context-integration.md): Design decision
  - [API Implementation Guide](docs/guides/API_TENANT_IMPLEMENTATION_GUIDE.md): Step-by-step guide
- **Story 2.4**: [Update API routes with tenant context](https://github.com/blackms/AlphaPulse/issues/165)
- **Blocking Issue**: [Pre-existing circular import](https://github.com/blackms/AlphaPulse/issues/191)

---

**Last Updated**: 2025-11-02
**Applies to**: v2.0.0+
