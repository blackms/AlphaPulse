# Database Migration Checklist - EPIC-001 (Sprint 5-6)

**EPIC**: EPIC-001 Database Multi-Tenancy
**Sprints**: Sprint 5-6 (Weeks 11-14)
**Story Points**: 21 SP
**Owner**: Backend Engineer + DBA Lead

---

## Overview

This checklist guides the implementation of database multi-tenancy with PostgreSQL Row-Level Security (RLS) following the zero-downtime migration plan documented in `database-migration-plan.md`.

**Success Criteria**:
- Zero downtime during migration
- All data migrated successfully
- RLS policies enforce tenant isolation
- Performance degradation <25%
- Rollback capability at every phase

---

## Pre-Migration Checklist

### Week Before Sprint 5

- [ ] **Review Migration Plan**: Read `docs/database-migration-plan.md` thoroughly
- [ ] **Database Backup**: Create full backup before starting
  ```bash
  pg_dump -h $DB_HOST -U $DB_USER -Fc alphapulse > backup_pre_migration_$(date +%Y%m%d).dump
  ```
- [ ] **Test Environment Ready**: Local PostgreSQL 14+ with RLS enabled
- [ ] **Feature Flag Created**: `RLS_ENABLED=false` in config
- [ ] **Monitoring Baseline**: Capture current query performance
  ```sql
  SELECT query, mean_exec_time, calls
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 20;
  ```
- [ ] **Team Training**: All engineers understand RLS concepts
- [ ] **Rollback Plan Reviewed**: Team knows rollback procedures

---

## Sprint 5: Phase 1 & 2 (Schema + Data Backfill)

### User Story: US-001 - Add tenants table (3 SP)

**File**: `alembic/versions/001_add_tenants_table.py`

- [ ] **Migration Created**:
  ```sql
  CREATE TABLE tenants (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      name VARCHAR(255) NOT NULL,
      slug VARCHAR(100) UNIQUE NOT NULL,
      subscription_tier VARCHAR(50) NOT NULL,
      status VARCHAR(50) NOT NULL DEFAULT 'active',
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
  );

  CREATE INDEX idx_tenants_slug ON tenants(slug);
  CREATE INDEX idx_tenants_status ON tenants(status);
  ```

- [ ] **Migration Applied** (development):
  ```bash
  poetry run alembic upgrade head
  ```

- [ ] **Verification**:
  ```sql
  \d tenants
  SELECT * FROM tenants;
  ```

- [ ] **Default Tenant Inserted**:
  ```sql
  INSERT INTO tenants (id, name, slug, subscription_tier, status)
  VALUES ('00000000-0000-0000-0000-000000000001', 'Default Tenant', 'default', 'pro', 'active');
  ```

- [ ] **Unit Tests Written**: `tests/test_models_tenant.py`
- [ ] **Code Review Completed**
- [ ] **Merged to main**

---

### User Story: US-002 - Add tenant_id to users table (3 SP)

**File**: `alembic/versions/002_add_tenant_id_to_users.py`

- [ ] **Migration Created**:
  ```sql
  -- Add column (nullable initially)
  ALTER TABLE users ADD COLUMN tenant_id UUID;

  -- Add foreign key
  ALTER TABLE users ADD CONSTRAINT fk_users_tenant
      FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE;

  -- Create index
  CREATE INDEX idx_users_tenant_id ON users(tenant_id);
  ```

- [ ] **Data Backfill**:
  ```sql
  -- Backfill existing users with default tenant
  UPDATE users
  SET tenant_id = '00000000-0000-0000-0000-000000000001'
  WHERE tenant_id IS NULL;
  ```

- [ ] **Make Column NOT NULL**:
  ```sql
  ALTER TABLE users ALTER COLUMN tenant_id SET NOT NULL;
  ```

- [ ] **Migration Applied** (development)
- [ ] **Verification**:
  ```sql
  SELECT COUNT(*), tenant_id FROM users GROUP BY tenant_id;
  ```

- [ ] **Unit Tests Written**
- [ ] **Integration Tests Written** (multi-tenant user creation)
- [ ] **Code Review Completed**
- [ ] **Merged to main**

---

### User Story: US-003 - Add tenant_id to domain tables (5 SP)

**File**: `alembic/versions/003_add_tenant_id_to_domain_tables.py`

**Tables to Update**:
- trades
- positions
- orders
- portfolio_snapshots
- risk_metrics
- agent_signals

**For Each Table**:

- [ ] **trades table**:
  ```sql
  ALTER TABLE trades ADD COLUMN tenant_id UUID;
  ALTER TABLE trades ADD CONSTRAINT fk_trades_tenant
      FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE;
  CREATE INDEX idx_trades_tenant_id ON trades(tenant_id);
  CREATE INDEX idx_trades_tenant_created ON trades(tenant_id, created_at DESC);

  UPDATE trades t
  SET tenant_id = u.tenant_id
  FROM users u
  WHERE t.user_id = u.id AND t.tenant_id IS NULL;

  ALTER TABLE trades ALTER COLUMN tenant_id SET NOT NULL;
  ```

- [ ] **positions table** (repeat above pattern)
- [ ] **orders table** (repeat above pattern)
- [ ] **portfolio_snapshots table** (repeat above pattern)
- [ ] **risk_metrics table** (repeat above pattern)
- [ ] **agent_signals table** (repeat above pattern)

- [ ] **Migration Applied** (development)
- [ ] **Verification** (all tables):
  ```sql
  SELECT table_name, column_name, is_nullable
  FROM information_schema.columns
  WHERE table_schema = 'public' AND column_name = 'tenant_id';
  ```

- [ ] **Performance Test**: Query latency before/after indexes
- [ ] **Unit Tests Written** (each model)
- [ ] **Code Review Completed**
- [ ] **Merged to main**

---

## Sprint 6: Phase 3 (RLS Enablement)

### User Story: US-004 - Create RLS policies (5 SP)

**File**: `alembic/versions/004_enable_rls_policies.py`

- [ ] **Enable RLS on All Tables**:
  ```sql
  ALTER TABLE users ENABLE ROW LEVEL SECURITY;
  ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
  ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
  ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
  ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
  ALTER TABLE risk_metrics ENABLE ROW LEVEL SECURITY;
  ALTER TABLE agent_signals ENABLE ROW LEVEL SECURITY;
  ```

- [ ] **Create RLS Policies** (users table example):
  ```sql
  CREATE POLICY tenant_isolation_policy ON users
      USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

  CREATE POLICY tenant_insert_policy ON users
      FOR INSERT
      WITH CHECK (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);
  ```

- [ ] **Repeat for All Tables**: trades, positions, orders, portfolio_snapshots, risk_metrics, agent_signals

- [ ] **Create Bypass Role** (for admin/migrations):
  ```sql
  CREATE ROLE rls_bypass_role;
  ALTER TABLE users FORCE ROW LEVEL SECURITY;  -- Apply RLS to owner too
  GRANT rls_bypass_role TO alphapulse;  -- Grant to app user when needed
  ```

- [ ] **Migration Applied** (development)
- [ ] **Verification**:
  ```sql
  SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';

  SELECT schemaname, tablename, policyname
  FROM pg_policies
  WHERE schemaname = 'public';
  ```

- [ ] **RLS Testing** (critical):
  ```sql
  -- Set tenant context
  SET LOCAL app.current_tenant_id = '00000000-0000-0000-0000-000000000001';

  -- Should return only tenant 1 data
  SELECT COUNT(*) FROM users;
  SELECT COUNT(*) FROM trades;

  -- Try to access tenant 2 data (should return 0)
  SET LOCAL app.current_tenant_id = '00000000-0000-0000-0000-000000000002';
  SELECT COUNT(*) FROM users WHERE tenant_id = '00000000-0000-0000-0000-000000000001';  -- Should be 0
  ```

- [ ] **Unit Tests Written**: `tests/test_rls_policies.py`
- [ ] **Integration Tests Written**: Cross-tenant isolation tests
- [ ] **Code Review Completed**
- [ ] **Merged to main**

---

### User Story: US-005 - Implement tenant context middleware (5 SP)

**File**: `src/alpha_pulse/api/middleware/tenant_context.py`

- [ ] **Middleware Implemented**:
  ```python
  @app.middleware("http")
  async def tenant_context_middleware(request: Request, call_next):
      # 1. Extract JWT from Authorization header
      token = request.headers.get("Authorization", "").replace("Bearer ", "")

      if not token:
          return JSONResponse(status_code=401, content={"detail": "Missing token"})

      try:
          # 2. Validate JWT and extract tenant_id
          payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
          tenant_id = payload.get("tenant_id")

          if not tenant_id:
              return JSONResponse(status_code=401, content={"detail": "Missing tenant_id in token"})

          # 3. Set tenant context (request state)
          request.state.tenant_id = tenant_id

          # 4. Set PostgreSQL RLS session variable (if RLS enabled)
          if settings.RLS_ENABLED:
              async with request.app.state.db_pool.acquire() as conn:
                  await conn.execute(f"SET LOCAL app.current_tenant_id = '{tenant_id}'")

          # 5. Continue request processing
          response = await call_next(request)
          return response

      except JWTError as e:
          return JSONResponse(status_code=401, content={"detail": f"Invalid token: {str(e)}"})
  ```

- [ ] **Feature Flag Integration**:
  ```python
  # config/settings.py
  RLS_ENABLED: bool = os.getenv("RLS_ENABLED", "false").lower() == "true"
  ```

- [ ] **Unit Tests Written**: `tests/test_tenant_middleware.py`
- [ ] **Integration Tests Written**: End-to-end API tests with multi-tenancy
- [ ] **Load Test**: Performance with RLS enabled
- [ ] **Code Review Completed**
- [ ] **Merged to main**

---

## Post-Migration Checklist

### Immediately After Phase 3 (RLS Enabled)

- [ ] **Enable Feature Flag** (staging):
  ```bash
  kubectl set env deployment/alphapulse-api -n alphapulse-staging RLS_ENABLED=true
  ```

- [ ] **Monitor Query Performance** (first 24 hours):
  ```sql
  -- Check slow queries
  SELECT query, mean_exec_time, calls
  FROM pg_stat_statements
  WHERE mean_exec_time > 100  -- queries > 100ms
  ORDER BY mean_exec_time DESC;
  ```

- [ ] **Verify Tenant Isolation** (staging):
  ```bash
  # Test with different tenant tokens
  curl -H "Authorization: Bearer $TENANT1_TOKEN" http://staging.alphapulse.ai/api/users
  curl -H "Authorization: Bearer $TENANT2_TOKEN" http://staging.alphapulse.ai/api/users

  # Verify no cross-tenant data leakage
  ```

- [ ] **Load Testing** (staging):
  ```bash
  k6 run --env BASE_URL=https://staging.alphapulse.ai target-capacity-test.js

  # Verify p99 latency increase <25%
  ```

- [ ] **Rollback Plan Tested**: Verify rollback procedures work

---

### Week After Migration

- [ ] **Performance Monitoring**: No queries >500ms p99
- [ ] **Error Rate Monitoring**: No RLS-related errors
- [ ] **Data Integrity Check**:
  ```sql
  -- Verify no orphaned records
  SELECT COUNT(*) FROM trades WHERE tenant_id NOT IN (SELECT id FROM tenants);
  SELECT COUNT(*) FROM users WHERE tenant_id NOT IN (SELECT id FROM tenants);
  ```

- [ ] **Production Deployment** (if staging validation passed):
  ```bash
  # Enable RLS in production
  kubectl set env deployment/alphapulse-api -n alphapulse RLS_ENABLED=true

  # Monitor closely for 48 hours
  ```

---

## Rollback Procedures

### Rollback Scenario 1: Immediate (within 5 minutes of Phase 3)

**Trigger**: Critical errors, data corruption, complete service outage

**Steps**:
```bash
# 1. Disable RLS feature flag
kubectl set env deployment/alphapulse-api RLS_ENABLED=false

# 2. Restart pods
kubectl rollout restart deployment/alphapulse-api

# 3. Verify service restored
curl http://api.alphapulse.ai/health
```

**Expected Recovery Time**: <2 minutes

---

### Rollback Scenario 2: Partial (within 24 hours)

**Trigger**: Performance degradation >25%, RLS policy bugs

**Steps**:
```bash
# 1. Disable RLS feature flag
kubectl set env deployment/alphapulse-api RLS_ENABLED=false

# 2. Disable RLS policies (keep schema changes)
poetry run alembic downgrade -1  # Downgrade only RLS policies migration
```

**Expected Recovery Time**: <10 minutes

---

### Rollback Scenario 3: Full (within 1 week)

**Trigger**: Fundamental design issues, need complete rollback

**Steps**:
```bash
# 1. Restore from backup
pg_restore -h $DB_HOST -U $DB_USER -d alphapulse backup_pre_migration_YYYYMMDD.dump

# 2. Downgrade all migrations
poetry run alembic downgrade base

# 3. Verify data integrity
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM trades;
```

**Expected Recovery Time**: 1-2 hours (depending on database size)

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| **Zero Downtime** | 0 minutes downtime | ⏳ TBD |
| **Data Migration** | 100% migrated | ⏳ TBD |
| **Tenant Isolation** | 0 cross-tenant leaks | ⏳ TBD |
| **Performance** | Latency increase <25% | ⏳ TBD |
| **Error Rate** | <0.1% RLS errors | ⏳ TBD |

---

## References

- [Database Migration Plan](./database-migration-plan.md)
- [PostgreSQL RLS Documentation](https://www.postgresql.org/docs/14/ddl-rowsecurity.html)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
