# ADR 001: Multi-tenant Data Isolation Strategy

Date: 2025-10-20
Status: Proposed

## Context

AlphaPulse is transitioning from a single-instance trading system to a multi-tenant SaaS platform. We need to choose a data isolation strategy that ensures:

- **Security**: Complete data isolation between tenants (no data leakage)
- **Performance**: Minimal query overhead for tenant filtering
- **Scalability**: Support for 100+ tenants in first year, 1000+ long-term
- **Operational Simplicity**: Easy backup/restore, migration, monitoring
- **Cost Efficiency**: Optimize infrastructure costs while maintaining quality
- **Compliance**: Support for data residency requirements (future EU/US data separation)

### Current Database Architecture

- PostgreSQL for persistent data (trades, positions, portfolio history)
- Redis for caching (market data, agent signals, session state)
- Time-series data (OHLCV) stored in PostgreSQL with TimescaleDB extension
- Complex queries joining across trades, positions, risk metrics, and portfolio allocations

### Critical Requirements

1. **Zero data leakage**: Row-level security must be bulletproof
2. **Query performance**: Adding tenant filtering must not degrade performance >10%
3. **Tenant provisioning**: New tenant setup < 5 minutes automated
4. **Backup/restore**: Per-tenant backup and point-in-time recovery
5. **Data sovereignty**: Support future geographic data separation

## Decision

We will implement a **Hybrid Shared Database + Schema-per-tenant** approach:

### Primary Strategy: Shared Database with Row-Level Security (RLS)

**For most tables (95% of data):**
- Single PostgreSQL database with all tenant data
- Add `tenant_id UUID NOT NULL` column to all tenant-specific tables
- Implement PostgreSQL Row-Level Security (RLS) policies
- Use composite indexes: `(tenant_id, primary_key)` for optimal performance
- Set `app.current_tenant_id` session variable for automatic filtering

**Example Implementation:**

```sql
-- Enable RLS on tenant tables
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

-- Create policy for tenant isolation
CREATE POLICY tenant_isolation_policy ON trades
  USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

-- Grant tenant access
GRANT SELECT, INSERT, UPDATE, DELETE ON trades TO app_user;
```

### Secondary Strategy: Schema-per-tenant for High-Value Enterprise Clients

**For Enterprise tier (5% of tenants, potential 50% of revenue):**
- Dedicated PostgreSQL schema per tenant: `tenant_<uuid>`
- Isolated backup/restore capabilities
- Custom performance tuning per tenant
- Support for data residency requirements (future multi-region)
- Migration path: Start in shared schema, graduate to dedicated schema when needed

**Hybrid Approach Workflow:**

```
Starter/Pro Tier → Shared DB + RLS (tenant_id column)
Enterprise Tier  → Dedicated Schema (tenant_<uuid>)
```

### Redis Caching Strategy

- Namespace all cache keys with tenant ID: `tenant:{tenant_id}:{cache_key}`
- Use Redis multi-tenancy features (keyspace notifications per tenant)
- Implement cache quota per tenant to prevent noisy neighbor issues

### Connection Pooling

- Use PgBouncer with session-level pooling (not transaction-level due to RLS session variables)
- Set `app.current_tenant_id` at connection acquisition
- Pool size: 20 connections × 10 app instances = 200 concurrent connections

## Consequences

### Positive

✅ **Performance**: Shared database with RLS adds minimal overhead (~2-5% query time)
✅ **Cost**: Single database reduces infrastructure costs by 60% vs database-per-tenant
✅ **Operational Simplicity**: Single database to monitor, backup, upgrade
✅ **Scalability**: Supports 100+ tenants without infrastructure changes
✅ **Flexibility**: Enterprise clients can graduate to dedicated schemas
✅ **Compliance**: Schema-per-tenant enables data residency requirements
✅ **Query Optimization**: Composite indexes on (tenant_id, id) maintain single-tenant performance

### Negative

⚠️ **RLS Complexity**: Developers must remember to set session variable for every connection
⚠️ **Testing Burden**: Must test tenant isolation exhaustively (integration tests + penetration testing)
⚠️ **Migration Overhead**: All existing tables need tenant_id column + RLS policies + index updates
⚠️ **Session Pooling**: Cannot use transaction-level pooling (must use session pooling)
⚠️ **Noisy Neighbor**: One tenant's heavy queries can impact others (mitigated by query timeouts + connection limits)

### Mitigation Strategies

1. **RLS Enforcement**: Create database middleware that automatically sets `app.current_tenant_id` from JWT claims
2. **Testing**: Implement comprehensive tenant isolation tests in CI/CD
3. **Monitoring**: Per-tenant query performance dashboards
4. **Rate Limiting**: API-level rate limits per tenant + database query timeouts
5. **Graduation Path**: Automated migration from shared to dedicated schema for Enterprise clients

## Alternatives Considered

### Option A: Database-per-tenant

**Pros:**
- Maximum isolation (impossible to leak data across databases)
- Per-tenant performance tuning
- Easy backup/restore per tenant
- Perfect for compliance and data residency

**Cons:**
- ❌ **High operational overhead**: Managing 100+ databases (backups, monitoring, upgrades)
- ❌ **Cost**: 100 tenants × $50/month = $5,000/month vs $500/month for shared
- ❌ **Slow provisioning**: 2-5 minutes to provision new database
- ❌ **Connection limits**: 100 databases × 20 connections = 2,000 connections (PostgreSQL limit ~500)
- ❌ **Query complexity**: Cross-tenant analytics require federation

**Why Rejected:** Operational complexity and cost too high for Starter/Pro tiers. Viable only for Enterprise.

### Option B: Shared Database with Application-Level Filtering

**Pros:**
- Simple implementation (just add WHERE tenant_id = ?)
- Works with any database
- Transaction-level connection pooling

**Cons:**
- ❌ **Security risk**: Forgetting WHERE clause = data leak
- ❌ **No database-level enforcement**: Relies on application code (not defense-in-depth)
- ❌ **SQL injection risk**: Tenant ID must be parameterized everywhere

**Why Rejected:** Too risky. One developer mistake = data breach. RLS provides defense-in-depth.

### Option C: Shared Schema with Discriminator Column (No RLS)

**Pros:**
- Simplest approach
- Best query performance (no RLS overhead)

**Cons:**
- ❌ **No enforcement**: Application must filter everywhere
- ❌ **Audit nightmare**: Hard to prove compliance
- ❌ **High risk**: Single bug = data leak

**Why Rejected:** Unacceptable security posture for financial trading platform.

## Implementation Plan

### Phase 1: Database Schema Migration (Sprint 1-2)

1. Add `tenant_id UUID NOT NULL` to all tenant-specific tables
2. Create composite indexes: `(tenant_id, id)` and `(tenant_id, created_at)`
3. Enable RLS policies on all tables
4. Create database middleware for session variable injection

### Phase 2: Application Code Updates (Sprint 2-3)

1. Update FastAPI dependency injection to extract tenant from JWT
2. Create database session factory that sets `app.current_tenant_id`
3. Update all queries to work with RLS (no code changes needed)
4. Add tenant context to logging and tracing

### Phase 3: Testing & Validation (Sprint 3)

1. Integration tests for tenant isolation (attempt cross-tenant access)
2. Performance benchmarks (compare single-tenant vs multi-tenant query times)
3. Load testing with 50 concurrent tenants
4. Security audit and penetration testing

### Phase 4: Enterprise Schema Support (Sprint 4)

1. Implement schema-per-tenant provisioning for Enterprise tier
2. Create migration scripts to move tenant from shared to dedicated schema
3. Update connection routing logic to support both strategies

## Links

- Issue: [To be created - Multi-tenant Architecture Epic]
- Related: ADR-002 (Tenant Provisioning), ADR-003 (Credential Management), ADR-004 (Caching Strategy)
- Reference: [PostgreSQL RLS Documentation](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- Reference: [Multi-tenancy Patterns](https://docs.microsoft.com/en-us/azure/architecture/guide/multitenant/considerations/tenancy-models)
