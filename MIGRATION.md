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

### Need Help?

- **Issues**: https://github.com/blackms/AlphaPulse/issues
- **Documentation**: See [EPIC-002](https://github.com/blackms/AlphaPulse/issues/162)
- **Story**: [Story 2.2 - Refactor Services](https://github.com/blackms/AlphaPulse/issues/163)

---

**Last Updated**: 2025-10-31
**Applies to**: v2.0.0+
