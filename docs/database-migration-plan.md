# Database Migration Plan: Multi-Tenant Schema Transformation

**Date**: 2025-10-22
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead
**Related**: [HLD-MULTI-TENANT-SAAS.md](HLD-MULTI-TENANT-SAAS.md), [ADR-001](adr/001-multi-tenant-data-isolation-strategy.md), Issue #180
**Status**: Draft (Pending DBA Approval)

---

## Executive Summary

This document defines the migration strategy to transform AlphaPulse from a single-tenant system to a multi-tenant SaaS platform. The migration introduces `tenant_id` foreign keys across all domain tables, enables PostgreSQL Row-Level Security (RLS) for data isolation, and maintains zero downtime during the transition.

**Migration Approach**: Online migration with dual-write strategy
**Estimated Duration**: 4-6 hours (excluding backfill)
**Downtime Target**: Zero downtime (rolling deployment)
**Rollback Window**: 24 hours (reversible migrations)
**Risk Level**: Medium (mitigated by extensive testing and rollback procedures)

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Target State Architecture](#target-state-architecture)
3. [Migration Strategy](#migration-strategy)
4. [Alembic Migration Scripts](#alembic-migration-scripts)
5. [Rollback Procedures](#rollback-procedures)
6. [Performance Impact Analysis](#performance-impact-analysis)
7. [Testing Strategy](#testing-strategy)
8. [Risk Mitigation](#risk-mitigation)
9. [Runbook](#runbook)
10. [Approval](#approval)

---

## 1. Current State Assessment

### 1.1 Database Schema (Single-Tenant)

**Current Schema** (simplified):
```sql
-- Users table (no tenant_id)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolios table (no tenant_id)
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    total_value NUMERIC(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trades table (no tenant_id)
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signals table (no tenant_id)
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    agent_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) CHECK (signal_type IN ('buy', 'sell', 'hold')),
    confidence NUMERIC(3, 2) CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions table (no tenant_id)
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    symbol VARCHAR(20) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    average_price NUMERIC(20, 8) NOT NULL,
    current_price NUMERIC(20, 8),
    unrealized_pnl NUMERIC(20, 8),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, symbol)
);
```

**Total Tables**: 28 (core domain tables)
**Estimated Row Counts** (production projection):
- `users`: 1,000
- `portfolios`: 1,500
- `trades`: 500,000
- `signals`: 2,000,000
- `positions`: 5,000
- Total: ~2.5M rows across all tables

### 1.2 Current Limitations

1. **No Data Isolation**: All users share the same table space (single tenant)
2. **No RLS Policies**: Application-level filtering only (vulnerable to bugs)
3. **No Tenant Context**: No concept of organizational boundaries
4. **Credential Sharing**: Exchange credentials stored per-user, not per-tenant
5. **Billing Complexity**: No way to aggregate usage across organization

---

## 2. Target State Architecture

### 2.1 Multi-Tenant Schema

**New Tenants Table**:
```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,  -- e.g., "acme-corp"
    subscription_tier VARCHAR(20) CHECK (subscription_tier IN ('starter', 'pro', 'enterprise')),
    stripe_customer_id VARCHAR(100) UNIQUE,
    status VARCHAR(20) CHECK (status IN ('active', 'suspended', 'cancelled')) DEFAULT 'active',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tenants_slug ON tenants(slug);
CREATE INDEX idx_tenants_stripe ON tenants(stripe_customer_id);
```

**Updated Users Table**:
```sql
ALTER TABLE users ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE users ADD COLUMN role VARCHAR(20) CHECK (role IN ('admin', 'trader', 'viewer')) DEFAULT 'trader';

-- Drop old unique constraint, add tenant-scoped constraint
ALTER TABLE users DROP CONSTRAINT users_email_key;
CREATE UNIQUE INDEX idx_users_email_tenant ON users(email, tenant_id);

-- Add tenant_id to existing index
CREATE INDEX idx_users_tenant ON users(tenant_id);
```

**Updated Domain Tables** (example: trades):
```sql
ALTER TABLE trades ADD COLUMN tenant_id UUID REFERENCES tenants(id);

-- Add tenant_id to all foreign key lookups
CREATE INDEX idx_trades_tenant ON trades(tenant_id);
CREATE INDEX idx_trades_tenant_portfolio ON trades(tenant_id, portfolio_id);
CREATE INDEX idx_trades_tenant_symbol ON trades(tenant_id, symbol);
```

### 2.2 Row-Level Security Policies

**Enable RLS on all domain tables**:
```sql
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;

-- Create RLS policy for tenant isolation
CREATE POLICY tenant_isolation_policy ON users
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_policy ON portfolios
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_policy ON trades
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_policy ON signals
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_policy ON positions
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
```

**Admin Bypass Policy** (for platform administrators):
```sql
-- Create admin role
CREATE ROLE alphapulse_admin;

-- Admin can see all tenants (bypass RLS)
CREATE POLICY admin_bypass_policy ON users
    TO alphapulse_admin
    USING (true);

-- Replicate for all tables
CREATE POLICY admin_bypass_policy ON portfolios TO alphapulse_admin USING (true);
CREATE POLICY admin_bypass_policy ON trades TO alphapulse_admin USING (true);
-- ... (repeat for all tables)
```

---

## 3. Migration Strategy

### 3.1 Migration Phases

**Phase 1: Schema Preparation** (Sprints 5-6)
- Add `tenants` table
- Add `tenant_id` columns to all domain tables (nullable)
- Add tenant-scoped indexes
- Deploy application code with dual-write support

**Phase 2: Data Backfill** (Sprint 7)
- Create default tenant for existing users
- Backfill `tenant_id` for all existing rows
- Validate data integrity (no orphaned records)

**Phase 3: RLS Enablement** (Sprint 8)
- Make `tenant_id` NOT NULL
- Enable RLS on all tables
- Create tenant isolation policies
- Deploy application code with RLS enforcement

**Phase 4: Cleanup** (Sprint 9)
- Remove old unique constraints
- Drop legacy indexes
- Optimize query plans
- Performance validation

### 3.2 Dual-Write Strategy

During Phase 1-2, the application will support both single-tenant and multi-tenant modes:

```python
# src/alpha_pulse/models/base.py

class TenantScopedModel:
    """Base model for tenant-scoped tables."""

    tenant_id: Optional[UUID] = None  # Nullable during migration

    @classmethod
    async def create(cls, db: AsyncSession, tenant_id: Optional[UUID] = None, **kwargs):
        """Create record with optional tenant_id."""
        if tenant_id:
            kwargs['tenant_id'] = tenant_id

        obj = cls(**kwargs)
        db.add(obj)
        await db.flush()
        return obj

    @classmethod
    async def get_by_id(cls, db: AsyncSession, id: UUID, tenant_id: Optional[UUID] = None):
        """Get record by ID with optional tenant filtering."""
        query = select(cls).where(cls.id == id)

        if tenant_id:
            query = query.where(cls.tenant_id == tenant_id)

        result = await db.execute(query)
        return result.scalar_one_or_none()
```

**Feature Flag for RLS Enforcement**:
```python
# src/alpha_pulse/middleware/tenant_context.py

RLS_ENABLED = os.getenv("RLS_ENABLED", "false") == "true"

@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")

    if not token:
        return await call_next(request)

    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    tenant_id = payload.get("tenant_id")

    request.state.tenant_id = tenant_id

    if RLS_ENABLED and tenant_id:
        # Set PostgreSQL session variable for RLS
        await db.execute(f"SET LOCAL app.current_tenant_id = '{tenant_id}'")

    response = await call_next(request)
    return response
```

### 3.3 Zero-Downtime Deployment

**Rolling Deployment Strategy**:
1. Deploy Phase 1 code (dual-write, RLS_ENABLED=false)
2. Gradually roll out to 10% → 50% → 100% of API pods
3. Monitor error rates (rollback if >1% error rate spike)
4. Once stable, run backfill script (Sprint 7)
5. Deploy Phase 3 code (RLS_ENABLED=true)
6. Gradually roll out with same strategy

**Health Check Validation**:
```python
# src/alpha_pulse/api/health.py

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check with RLS validation."""

    # Check database connectivity
    await db.execute(text("SELECT 1"))

    # Check RLS enforcement (if enabled)
    if RLS_ENABLED:
        # Verify RLS policies exist
        result = await db.execute(text("""
            SELECT COUNT(*) FROM pg_policies
            WHERE schemaname = 'public' AND policyname = 'tenant_isolation_policy'
        """))
        policy_count = result.scalar()

        if policy_count < 5:  # Expect policies on 5+ tables
            raise HTTPException(status_code=503, detail="RLS policies not configured")

    return {"status": "healthy", "rls_enabled": RLS_ENABLED}
```

---

## 4. Alembic Migration Scripts

### 4.1 Migration: Add Tenants Table

**File**: `alembic/versions/001_add_tenants_table.py`

```python
"""Add tenants table

Revision ID: 001_add_tenants
Revises:
Create Date: 2025-10-22 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_add_tenants'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('subscription_tier', sa.String(20), nullable=False, server_default='starter'),
        sa.Column('stripe_customer_id', sa.String(100), unique=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('settings', postgresql.JSONB, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.CheckConstraint("subscription_tier IN ('starter', 'pro', 'enterprise')", name='ck_subscription_tier'),
        sa.CheckConstraint("status IN ('active', 'suspended', 'cancelled')", name='ck_status')
    )

    # Create indexes
    op.create_index('idx_tenants_slug', 'tenants', ['slug'])
    op.create_index('idx_tenants_stripe', 'tenants', ['stripe_customer_id'])

    # Create default tenant for existing data
    op.execute("""
        INSERT INTO tenants (id, name, slug, subscription_tier, status)
        VALUES ('00000000-0000-0000-0000-000000000001', 'Default Tenant', 'default', 'pro', 'active')
    """)


def downgrade():
    op.drop_table('tenants')
```

### 4.2 Migration: Add tenant_id to Users

**File**: `alembic/versions/002_add_tenant_id_to_users.py`

```python
"""Add tenant_id to users table

Revision ID: 002_add_tenant_id_users
Revises: 001_add_tenants
Create Date: 2025-10-22 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '002_add_tenant_id_users'
down_revision = '001_add_tenants'
branch_labels = None
depends_on = None


def upgrade():
    # Add tenant_id column (nullable for now)
    op.add_column('users', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))

    # Add foreign key constraint
    op.create_foreign_key('fk_users_tenant', 'users', 'tenants', ['tenant_id'], ['id'])

    # Add role column
    op.add_column('users', sa.Column('role', sa.String(20), server_default='trader', nullable=False))
    op.create_check_constraint('ck_user_role', 'users', "role IN ('admin', 'trader', 'viewer')")

    # Backfill tenant_id for existing users
    op.execute("""
        UPDATE users SET tenant_id = '00000000-0000-0000-0000-000000000001'
        WHERE tenant_id IS NULL
    """)

    # Create tenant-scoped index
    op.create_index('idx_users_tenant', 'users', ['tenant_id'])

    # Drop old unique constraint on email
    op.drop_constraint('users_email_key', 'users', type_='unique')

    # Create tenant-scoped unique constraint
    op.create_index('idx_users_email_tenant', 'users', ['email', 'tenant_id'], unique=True)


def downgrade():
    op.drop_index('idx_users_email_tenant', 'users')
    op.create_unique_constraint('users_email_key', 'users', ['email'])
    op.drop_index('idx_users_tenant', 'users')
    op.drop_constraint('fk_users_tenant', 'users', type_='foreignkey')
    op.drop_column('users', 'tenant_id')
    op.drop_column('users', 'role')
```

### 4.3 Migration: Add tenant_id to Domain Tables

**File**: `alembic/versions/003_add_tenant_id_to_domain_tables.py`

```python
"""Add tenant_id to all domain tables

Revision ID: 003_add_tenant_id_domain
Revises: 002_add_tenant_id_users
Create Date: 2025-10-22 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '003_add_tenant_id_domain'
down_revision = '002_add_tenant_id_users'
branch_labels = None
depends_on = None

DOMAIN_TABLES = [
    'portfolios',
    'trades',
    'signals',
    'positions',
    'risk_metrics',
    'agent_configs',
    'exchange_credentials',
    'audit_logs',
    'usage_metrics'
]


def upgrade():
    for table in DOMAIN_TABLES:
        # Add tenant_id column (nullable)
        op.add_column(table, sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))

        # Add foreign key constraint
        op.create_foreign_key(f'fk_{table}_tenant', table, 'tenants', ['tenant_id'], ['id'])

        # Create tenant index
        op.create_index(f'idx_{table}_tenant', table, ['tenant_id'])

        # Backfill tenant_id from parent relationship
        if table == 'portfolios':
            # portfolios.tenant_id = users.tenant_id (via user_id FK)
            op.execute(f"""
                UPDATE {table} t
                SET tenant_id = u.tenant_id
                FROM users u
                WHERE t.user_id = u.id AND t.tenant_id IS NULL
            """)
        elif table in ['trades', 'signals', 'positions', 'risk_metrics']:
            # Get tenant_id from portfolios
            op.execute(f"""
                UPDATE {table} t
                SET tenant_id = p.tenant_id
                FROM portfolios p
                WHERE t.portfolio_id = p.id AND t.tenant_id IS NULL
            """)
        elif table == 'agent_configs':
            # agent_configs.tenant_id = users.tenant_id (via user_id FK)
            op.execute(f"""
                UPDATE {table} t
                SET tenant_id = u.tenant_id
                FROM users u
                WHERE t.user_id = u.id AND t.tenant_id IS NULL
            """)
        elif table == 'exchange_credentials':
            # exchange_credentials.tenant_id = users.tenant_id (via user_id FK)
            op.execute(f"""
                UPDATE {table} t
                SET tenant_id = u.tenant_id
                FROM users u
                WHERE t.user_id = u.id AND t.tenant_id IS NULL
            """)
        elif table in ['audit_logs', 'usage_metrics']:
            # Get tenant_id from users (via user_id)
            op.execute(f"""
                UPDATE {table} t
                SET tenant_id = u.tenant_id
                FROM users u
                WHERE t.user_id = u.id AND t.tenant_id IS NULL
            """)


def downgrade():
    for table in DOMAIN_TABLES:
        op.drop_index(f'idx_{table}_tenant', table)
        op.drop_constraint(f'fk_{table}_tenant', table, type_='foreignkey')
        op.drop_column(table, 'tenant_id')
```

### 4.4 Migration: Enable RLS

**File**: `alembic/versions/004_enable_rls.py`

```python
"""Enable Row-Level Security

Revision ID: 004_enable_rls
Revises: 003_add_tenant_id_domain
Create Date: 2025-10-22 13:00:00.000000

"""
from alembic import op

revision = '004_enable_rls'
down_revision = '003_add_tenant_id_domain'
branch_labels = None
depends_on = None

DOMAIN_TABLES = [
    'users',
    'portfolios',
    'trades',
    'signals',
    'positions',
    'risk_metrics',
    'agent_configs',
    'exchange_credentials',
    'audit_logs',
    'usage_metrics'
]


def upgrade():
    # Make tenant_id NOT NULL (all rows should be backfilled by now)
    for table in DOMAIN_TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id SET NOT NULL")

    # Enable RLS on all domain tables
    for table in DOMAIN_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

    # Create tenant isolation policy
    for table in DOMAIN_TABLES:
        op.execute(f"""
            CREATE POLICY tenant_isolation_policy ON {table}
                USING (tenant_id = current_setting('app.current_tenant_id')::uuid)
        """)

    # Create admin role and bypass policy
    op.execute("CREATE ROLE alphapulse_admin")

    for table in DOMAIN_TABLES:
        op.execute(f"""
            CREATE POLICY admin_bypass_policy ON {table}
                TO alphapulse_admin
                USING (true)
        """)


def downgrade():
    # Drop policies
    for table in DOMAIN_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_policy ON {table}")
        op.execute(f"DROP POLICY IF EXISTS admin_bypass_policy ON {table}")

    # Disable RLS
    for table in DOMAIN_TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Drop admin role
    op.execute("DROP ROLE IF EXISTS alphapulse_admin")

    # Make tenant_id nullable again
    for table in DOMAIN_TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id DROP NOT NULL")
```

### 4.5 Running Migrations

```bash
# Apply migrations
poetry run alembic upgrade head

# Rollback one migration
poetry run alembic downgrade -1

# Rollback to specific revision
poetry run alembic downgrade 002_add_tenant_id_users

# Show migration history
poetry run alembic history

# Check current revision
poetry run alembic current
```

---

## 5. Rollback Procedures

### 5.1 Rollback Triggers

Rollback if any of the following occur within 24 hours of migration:

1. **Error Rate Spike**: API error rate >1% (baseline: 0.1%)
2. **Query Performance Degradation**: p99 latency >500ms (baseline: 150ms)
3. **Data Integrity Issues**: Orphaned records, NULL tenant_id values
4. **RLS Policy Bypass**: Users can access other tenants' data

### 5.2 Rollback Steps

**Immediate Rollback (Phase 3 failure)**:
```bash
# 1. Disable RLS enforcement in application
kubectl set env deployment/api RLS_ENABLED=false

# 2. Rollback database migration
poetry run alembic downgrade 003_add_tenant_id_domain

# 3. Verify rollback
psql -U alphapulse -d alphapulse_prod -c "SELECT COUNT(*) FROM pg_policies WHERE policyname = 'tenant_isolation_policy'"
# Expected: 0

# 4. Deploy previous application version
kubectl rollout undo deployment/api
```

**Partial Rollback (keep schema, disable RLS)**:
```bash
# Keep tenant_id columns but disable RLS
poetry run alembic downgrade 004_enable_rls

# Application continues with optional tenant filtering (dual-write mode)
kubectl set env deployment/api RLS_ENABLED=false
```

**Full Rollback (remove all multi-tenant changes)**:
```bash
# Rollback all migrations
poetry run alembic downgrade 001_add_tenants

# Remove tenant_id columns from all tables
# (handled by downgrade() functions in migration scripts)

# Deploy single-tenant application version
kubectl rollout undo deployment/api --to-revision=<pre-migration-revision>
```

### 5.3 Rollback Validation

After rollback, validate:
```sql
-- Check no RLS policies exist
SELECT * FROM pg_policies WHERE schemaname = 'public';

-- Check tenant_id columns removed (full rollback)
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'public' AND column_name = 'tenant_id';

-- Check users can access data
SELECT COUNT(*) FROM trades;  -- Should return all trades (no RLS filtering)
```

---

## 6. Performance Impact Analysis

### 6.1 Expected Overhead

**RLS Policy Evaluation**:
- **Overhead**: 2-5ms per query (RLS policy check)
- **Baseline Latency**: 10-20ms (simple SELECT query)
- **New Latency**: 12-25ms (20-25% increase)
- **Mitigation**: Proper indexing on `tenant_id` columns

**Index Size Increase**:
- **Current Index Size**: ~500MB (all indexes)
- **New Indexes**: +150MB (tenant_id indexes on 10 tables)
- **Total**: ~650MB (+30% increase)

**Query Plan Changes**:

Before (single-tenant):
```sql
EXPLAIN ANALYZE SELECT * FROM trades WHERE portfolio_id = '123e4567-e89b-12d3-a456-426614174000';

-- Query Plan:
-- Index Scan using idx_trades_portfolio (cost=0.43..8.45 rows=1 width=100) (actual time=0.020..0.021 rows=1 loops=1)
-- Planning Time: 0.102 ms
-- Execution Time: 0.035 ms
```

After (multi-tenant with RLS):
```sql
SET app.current_tenant_id = '00000000-0000-0000-0000-000000000001';
EXPLAIN ANALYZE SELECT * FROM trades WHERE portfolio_id = '123e4567-e89b-12d3-a456-426614174000';

-- Query Plan:
-- Index Scan using idx_trades_tenant_portfolio (cost=0.43..8.50 rows=1 width=100) (actual time=0.025..0.026 rows=1 loops=1)
--   Index Cond: ((tenant_id = current_setting('app.current_tenant_id')::uuid) AND (portfolio_id = '123e4567-e89b-12d3-a456-426614174000'))
--   Filter: (tenant_id = current_setting('app.current_tenant_id')::uuid)  -- RLS policy
-- Planning Time: 0.125 ms
-- Execution Time: 0.045 ms
```

**Performance Impact**: +10ms (+28% increase) due to RLS policy evaluation

### 6.2 Load Testing Results (Projected)

**Test Setup**:
- 100 concurrent users
- 1,000 requests/min
- Mix: 70% reads, 30% writes

**Before Migration**:
- p50 latency: 50ms
- p99 latency: 150ms
- Error rate: 0.1%

**After Migration (Projected)**:
- p50 latency: 60ms (+20%)
- p99 latency: 200ms (+33%)
- Error rate: 0.1% (no change)

**Mitigation**: Add composite indexes on `(tenant_id, portfolio_id)`, `(tenant_id, symbol)` to reduce RLS overhead.

### 6.3 Index Optimization

**Critical Indexes** (reduce RLS overhead):
```sql
-- Trades table (most queried)
CREATE INDEX idx_trades_tenant_portfolio ON trades(tenant_id, portfolio_id);
CREATE INDEX idx_trades_tenant_symbol_date ON trades(tenant_id, symbol, executed_at DESC);

-- Signals table (high volume)
CREATE INDEX idx_signals_tenant_portfolio_date ON signals(tenant_id, portfolio_id, created_at DESC);

-- Positions table (frequent updates)
CREATE INDEX idx_positions_tenant_portfolio_symbol ON positions(tenant_id, portfolio_id, symbol);
```

**Index Size vs. Query Performance Trade-off**:
- Additional indexes: +200MB storage
- Query performance improvement: 40-60% faster for tenant-scoped queries

---

## 7. Testing Strategy

### 7.1 Pre-Migration Testing

**Step 1: Schema Validation (Local)**
```bash
# Create test database
createdb alphapulse_test

# Apply migrations
poetry run alembic -c alembic.test.ini upgrade head

# Verify schema
psql -U alphapulse -d alphapulse_test -c "\d+ tenants"
psql -U alphapulse -d alphapulse_test -c "\d+ users"
psql -U alphapulse -d alphapulse_test -c "SELECT * FROM pg_policies WHERE schemaname = 'public'"
```

**Step 2: Data Integrity Testing**
```python
# tests/test_migration_integrity.py

import pytest
from sqlalchemy import text

@pytest.mark.asyncio
async def test_tenant_id_backfill(db):
    """Verify all rows have tenant_id after backfill."""

    result = await db.execute(text("SELECT COUNT(*) FROM users WHERE tenant_id IS NULL"))
    assert result.scalar() == 0, "Found users without tenant_id"

    result = await db.execute(text("SELECT COUNT(*) FROM trades WHERE tenant_id IS NULL"))
    assert result.scalar() == 0, "Found trades without tenant_id"

@pytest.mark.asyncio
async def test_rls_isolation(db):
    """Verify RLS prevents cross-tenant access."""

    tenant_a = "00000000-0000-0000-0000-000000000001"
    tenant_b = "00000000-0000-0000-0000-000000000002"

    # Create trades for both tenants
    await db.execute(text(f"INSERT INTO trades (tenant_id, symbol, side, quantity, price) VALUES ('{tenant_a}', 'BTC', 'buy', 1.0, 50000)"))
    await db.execute(text(f"INSERT INTO trades (tenant_id, symbol, side, quantity, price) VALUES ('{tenant_b}', 'ETH', 'buy', 10.0, 3000)"))
    await db.commit()

    # Set tenant context to A
    await db.execute(text(f"SET LOCAL app.current_tenant_id = '{tenant_a}'"))

    # Query trades (should only return tenant A's trades)
    result = await db.execute(text("SELECT COUNT(*) FROM trades"))
    count = result.scalar()
    assert count == 1, f"Expected 1 trade for tenant A, got {count}"

    # Query with explicit tenant B filter (should return 0 due to RLS)
    result = await db.execute(text(f"SELECT COUNT(*) FROM trades WHERE tenant_id = '{tenant_b}'"))
    count = result.scalar()
    assert count == 0, "RLS policy failed: can access other tenant's data"
```

### 7.2 Staging Environment Testing

**Step 1: Deploy to Staging**
```bash
# Apply migrations to staging database
poetry run alembic -c alembic.staging.ini upgrade head

# Deploy Phase 1 application (RLS_ENABLED=false)
kubectl apply -f k8s/staging/deployment.yaml

# Wait for rollout
kubectl rollout status deployment/api -n staging
```

**Step 2: Smoke Tests**
```bash
# Run API smoke tests
poetry run pytest tests/smoke/ --env=staging

# Manual verification
curl -H "Authorization: Bearer $STAGING_TOKEN" https://staging.alphapulse.ai/api/health
curl -H "Authorization: Bearer $STAGING_TOKEN" https://staging.alphapulse.ai/api/portfolio
```

**Step 3: Load Testing**
```bash
# Run load tests (k6 or Locust)
k6 run --vus 100 --duration 10m tests/load/api_load_test.js

# Monitor metrics
# - p99 latency < 500ms
# - Error rate < 1%
# - CPU usage < 70%
```

### 7.3 Production Migration Testing

**Dry Run (Read-Only Validation)**:
```sql
-- Connect to production replica (read-only)
psql -U alphapulse -h prod-replica.example.com -d alphapulse_prod

-- Check current schema
\d+ users
\d+ trades

-- Simulate backfill query (no writes)
EXPLAIN ANALYZE
UPDATE users SET tenant_id = '00000000-0000-0000-0000-000000000001'
WHERE tenant_id IS NULL;

-- Estimated rows affected
SELECT COUNT(*) FROM users WHERE tenant_id IS NULL;
SELECT COUNT(*) FROM trades WHERE tenant_id IS NULL;
```

**Blue/Green Deployment**:
1. Deploy Phase 1 to "green" environment (new pods)
2. Route 10% of traffic to green (canary)
3. Monitor for 30 minutes (error rate, latency)
4. Gradually increase to 50% → 100%
5. Decommission "blue" environment (old pods)

---

## 8. Risk Mitigation

### 8.1 Risk Matrix

| Risk | Likelihood | Impact | Severity | Mitigation |
|------|-----------|--------|----------|------------|
| **RIS-001**: Data loss during backfill | Low | Critical | HIGH | Database backup before migration, transaction-based backfill, validation queries |
| **RIS-002**: RLS policy bypass (security) | Medium | Critical | HIGH | Extensive testing, security review, penetration testing |
| **RIS-003**: Query performance degradation | High | Medium | MEDIUM | Composite indexes, load testing, query optimization |
| **RIS-004**: Downtime during migration | Low | Medium | MEDIUM | Rolling deployment, health checks, rollback procedures |
| **RIS-005**: Orphaned records (data integrity) | Medium | Medium | MEDIUM | Foreign key constraints, validation scripts, audit logs |
| **RIS-006**: Application bugs (code errors) | Medium | Medium | MEDIUM | Feature flags, staged rollout, comprehensive testing |
| **RIS-007**: Rollback complexity | Low | Low | LOW | Detailed rollback procedures, automated scripts, practice runs |

### 8.2 Mitigation Strategies

**RIS-001: Data Loss During Backfill**
- **Before Migration**: Full database backup via `pg_dump`
  ```bash
  pg_dump -U alphapulse -h prod.example.com alphapulse_prod > backup_pre_migration_$(date +%Y%m%d).sql
  ```
- **During Migration**: Use transactions with SAVEPOINT
  ```sql
  BEGIN;
  SAVEPOINT before_backfill;

  UPDATE users SET tenant_id = '00000000-0000-0000-0000-000000000001' WHERE tenant_id IS NULL;

  -- Validate
  SELECT COUNT(*) FROM users WHERE tenant_id IS NULL;
  -- Expected: 0

  -- If validation fails
  ROLLBACK TO SAVEPOINT before_backfill;

  -- Otherwise commit
  COMMIT;
  ```
- **After Migration**: Validation queries (see Testing Strategy)

**RIS-002: RLS Policy Bypass**
- **Prevention**: Security design review (completed)
- **Testing**: Penetration testing (Sprint 15)
- **Validation**: Automated RLS tests in CI/CD
  ```python
  @pytest.mark.security
  async def test_rls_prevents_cross_tenant_access(db):
      # Attempt to access other tenant's data
      # Assert empty result set
  ```
- **Monitoring**: Audit logs for suspicious queries

**RIS-003: Query Performance Degradation**
- **Prevention**: Composite indexes on `(tenant_id, ...)` columns
- **Testing**: Load testing with 100-500 concurrent users
- **Monitoring**: Prometheus alerts for p99 latency >500ms
  ```yaml
  - alert: HighLatency
    expr: histogram_quantile(0.99, api_request_duration_seconds) > 0.5
    for: 5m
    annotations:
      summary: "API p99 latency >500ms"
  ```
- **Mitigation**: Query optimization, connection pooling (pgbouncer)

**RIS-004: Downtime During Migration**
- **Prevention**: Rolling deployment (10% → 50% → 100%)
- **Testing**: Health checks (`/health`, `/ready`)
- **Monitoring**: Kubernetes readiness probes
  ```yaml
  readinessProbe:
    httpGet:
      path: /ready
      port: 8000
    initialDelaySeconds: 10
    periodSeconds: 5
  ```
- **Mitigation**: Automatic rollback if health checks fail

---

## 9. Runbook

### 9.1 Pre-Migration Checklist

- [ ] **Database Backup**: Full `pg_dump` backup created and verified
- [ ] **Staging Testing**: All tests pass in staging environment
- [ ] **Load Testing**: p99 latency <500ms, error rate <1%
- [ ] **Feature Flag**: `RLS_ENABLED=false` configured for Phase 1
- [ ] **Rollback Plan**: Rollback procedures documented and tested
- [ ] **Team Availability**: On-call engineer available during migration
- [ ] **Communication**: Stakeholders notified of migration window
- [ ] **Monitoring**: Dashboards and alerts configured

### 9.2 Migration Execution

**Phase 1: Schema Preparation** (Sprint 5-6, Week 1)
```bash
# Day 1: Deploy migrations (add tenants table, tenant_id columns)
kubectl exec -it api-pod -- poetry run alembic upgrade 003_add_tenant_id_domain

# Day 2: Deploy Phase 1 application code (dual-write, RLS_ENABLED=false)
kubectl apply -f k8s/prod/deployment-phase1.yaml

# Day 3-5: Monitor for errors, performance issues
# - Check Grafana dashboard (latency, error rate, CPU)
# - Review logs for tenant_id errors
```

**Phase 2: Data Backfill** (Sprint 7, Week 1)
```bash
# Day 1: Backfill tenant_id for all tables (run during low-traffic window)
psql -U alphapulse -h prod.example.com -d alphapulse_prod <<EOF
BEGIN;

-- Backfill users (already done in migration 002)
UPDATE users SET tenant_id = '00000000-0000-0000-0000-000000000001' WHERE tenant_id IS NULL;

-- Backfill portfolios
UPDATE portfolios p SET tenant_id = u.tenant_id FROM users u WHERE p.user_id = u.id AND p.tenant_id IS NULL;

-- Backfill trades (may take 10-20 minutes for 500k rows)
UPDATE trades t SET tenant_id = p.tenant_id FROM portfolios p WHERE t.portfolio_id = p.id AND t.tenant_id IS NULL;

-- Validate (expect 0)
SELECT COUNT(*) FROM users WHERE tenant_id IS NULL;
SELECT COUNT(*) FROM portfolios WHERE tenant_id IS NULL;
SELECT COUNT(*) FROM trades WHERE tenant_id IS NULL;

COMMIT;
EOF

# Day 2-3: Validate data integrity
poetry run pytest tests/test_migration_integrity.py -v
```

**Phase 3: RLS Enablement** (Sprint 8, Week 1)
```bash
# Day 1: Apply RLS migration (enable RLS, create policies)
kubectl exec -it api-pod -- poetry run alembic upgrade 004_enable_rls

# Day 2: Deploy Phase 3 application code (RLS_ENABLED=true)
kubectl set env deployment/api RLS_ENABLED=true

# Day 3-5: Monitor for RLS-related errors
# - Check logs for "permission denied" errors
# - Verify tenant isolation (run RLS tests)
```

### 9.3 Post-Migration Validation

**Validation Queries**:
```sql
-- Check RLS policies exist
SELECT schemaname, tablename, policyname
FROM pg_policies
WHERE schemaname = 'public' AND policyname = 'tenant_isolation_policy';
-- Expected: 10 rows (one per table)

-- Check all rows have tenant_id
SELECT
    'users' AS table_name, COUNT(*) AS null_count FROM users WHERE tenant_id IS NULL
UNION ALL
SELECT 'portfolios', COUNT(*) FROM portfolios WHERE tenant_id IS NULL
UNION ALL
SELECT 'trades', COUNT(*) FROM trades WHERE tenant_id IS NULL;
-- Expected: 0 for all tables

-- Test RLS isolation
SET app.current_tenant_id = '00000000-0000-0000-0000-000000000001';
SELECT COUNT(*) FROM trades;  -- Should return only tenant's trades

-- Attempt to bypass RLS (should return 0)
SELECT COUNT(*) FROM trades WHERE tenant_id = '00000000-0000-0000-0000-000000000002';
```

**API Validation**:
```bash
# Health check
curl https://api.alphapulse.ai/health
# Expected: {"status":"healthy","rls_enabled":true}

# Get portfolio (should return only tenant's data)
curl -H "Authorization: Bearer $TOKEN" https://api.alphapulse.ai/api/portfolio
```

### 9.4 Rollback Procedures

See [Section 5.2: Rollback Steps](#52-rollback-steps)

---

## 10. Approval

### 10.1 Sign-Off

This migration plan requires approval from the following stakeholders:

**Database Administrator (DBA)**:
- [ ] Schema design reviewed
- [ ] Migration scripts validated
- [ ] Performance impact acceptable
- [ ] Rollback procedures tested

**Signature**: ___________________________  Date: ___________

---

**Tech Lead**:
- [ ] Application code changes ready
- [ ] Feature flags configured
- [ ] Testing strategy executed
- [ ] Runbook reviewed

**Signature**: ___________________________  Date: ___________

---

**Security Lead**:
- [ ] RLS policies reviewed
- [ ] Tenant isolation validated
- [ ] Credential management strategy approved

**Signature**: ___________________________  Date: ___________

---

**CTO**:
- [ ] Migration strategy approved
- [ ] Risk mitigation acceptable
- [ ] Budget and timeline approved

**Signature**: ___________________________  Date: ___________

---

### 10.2 Post-Migration Review

**Review Meeting**: Scheduled for Sprint 9, Week 2

**Agenda**:
1. Migration execution review (what went well, what went wrong)
2. Performance metrics validation (latency, error rate)
3. Security validation (RLS testing, penetration test results)
4. Lessons learned and action items

**Participants**:
- Tech Lead
- DBA Lead
- Security Lead
- Product Manager
- On-call Engineer (who executed migration)

---

## Appendix A: SQL Schema Comparison

### Before Migration (Single-Tenant)

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE portfolios (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL
);

CREATE TABLE trades (
    id UUID PRIMARY KEY,
    portfolio_id UUID REFERENCES portfolios(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8) NOT NULL
);
```

### After Migration (Multi-Tenant with RLS)

```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    subscription_tier VARCHAR(20) CHECK (subscription_tier IN ('starter', 'pro', 'enterprise'))
);

CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('admin', 'trader', 'viewer')) DEFAULT 'trader',
    UNIQUE(email, tenant_id)
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_policy ON users
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE TABLE portfolios (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id) NOT NULL,
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL
);

ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_policy ON portfolios
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE TABLE trades (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id) NOT NULL,
    portfolio_id UUID REFERENCES portfolios(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8) NOT NULL
);

ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_policy ON trades
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
```

---

## Appendix B: References

- [HLD Section 3.2: Database Design](HLD-MULTI-TENANT-SAAS.md#32-database-design)
- [ADR-001: Multi-Tenant Data Isolation Strategy](adr/001-multi-tenant-data-isolation-strategy.md)
- [Security Design Review](security-design-review.md)
- [PostgreSQL RLS Documentation](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

---

**Document Status**: Draft (Pending DBA Approval)
**Review Date**: Sprint 3, Week 2
**Reviewers**: DBA Lead, Tech Lead, Security Lead, CTO

---

**END OF DOCUMENT**
