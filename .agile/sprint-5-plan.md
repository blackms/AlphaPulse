# Sprint 5 Plan: EPIC-001 Database Multi-Tenancy (Part 1)

**Sprint Duration**: 2 weeks (10 working days)
**Dates**: 2025-11-11 to 2025-11-22
**Phase**: Phase 3 - Build & Validate
**EPIC**: EPIC-001 Database Multi-Tenancy
**Story Points**: 21 SP

---

## Sprint Goals

### Primary Goals
1. 🎯 Implement tenant isolation at database level (PostgreSQL RLS)
2. 🎯 Add tenant context propagation through API middleware
3. 🎯 Ensure zero downtime migration
4. 🎯 Validate tenant isolation with comprehensive tests
5. 🎯 Achieve <25% performance impact

### Success Criteria
- [ ] All 5 user stories completed (21 SP)
- [ ] Zero data leakage between tenants (100% isolation)
- [ ] Performance degradation <25% (p99 latency)
- [ ] 100% test coverage for RLS policies
- [ ] Zero downtime during migration
- [ ] Rollback capability verified

---

## User Stories Breakdown

### US-001: Add tenants table (3 SP)
**Priority**: P0 (Foundation)
**Owner**: Backend Engineer
**Duration**: Days 1-2

**Acceptance Criteria**:
- [ ] Tenants table created with all required fields
- [ ] Indexes created (slug, status)
- [ ] Default tenant inserted
- [ ] Migration applied successfully
- [ ] Unit tests passing (>90% coverage)

**Deliverables**:
1. Alembic migration: `001_add_tenants_table.py`
2. SQLAlchemy model: `src/alpha_pulse/models/tenant.py`
3. Unit tests: `tests/test_tenant_model.py`
4. API endpoints: GET/POST/PATCH/DELETE `/api/tenants`

---

### US-002: Add tenant_id to users table (3 SP)
**Priority**: P0 (Foundation)
**Owner**: Backend Engineer
**Duration**: Days 2-3

**Acceptance Criteria**:
- [ ] tenant_id column added to users table
- [ ] Foreign key constraint created
- [ ] Existing users backfilled with default tenant
- [ ] Column made NOT NULL after backfill
- [ ] Migration reversible (rollback tested)
- [ ] Unit tests passing

**Deliverables**:
1. Alembic migration: `002_add_tenant_id_to_users.py`
2. Updated model: `src/alpha_pulse/models/user.py`
3. Unit tests: `tests/test_user_tenant_association.py`

---

### US-003: Add tenant_id to domain tables (5 SP)
**Priority**: P0 (Foundation)
**Owner**: Backend Engineer
**Duration**: Days 3-5

**Tables to Update**:
- trades
- positions
- orders
- portfolio_snapshots
- risk_metrics
- agent_signals

**Acceptance Criteria**:
- [ ] tenant_id added to all 6 domain tables
- [ ] Foreign keys created
- [ ] Indexes created (tenant_id, tenant_id + created_at)
- [ ] Data backfilled from users.tenant_id
- [ ] All columns NOT NULL after backfill
- [ ] Performance impact measured (<25%)

**Deliverables**:
1. Alembic migration: `003_add_tenant_id_to_domain_tables.py`
2. Updated models (6 files)
3. Unit tests for each model
4. Performance benchmark report

---

### US-004: Create RLS policies (5 SP)
**Priority**: P0 (Security)
**Owner**: Backend Engineer + DBA Lead
**Duration**: Days 6-7

**Acceptance Criteria**:
- [ ] RLS enabled on all tables (7 tables)
- [ ] Policies created (SELECT, INSERT, UPDATE, DELETE)
- [ ] Bypass role created for admin operations
- [ ] Policies tested with multiple tenants
- [ ] Zero cross-tenant data leakage
- [ ] Feature flag created: `RLS_ENABLED=false`

**Deliverables**:
1. Alembic migration: `004_enable_rls_policies.py`
2. RLS test suite: `tests/test_rls_policies.py`
3. Documentation: RLS policy design
4. Feature flag configuration

---

### US-005: Implement tenant context middleware (5 SP)
**Priority**: P0 (Integration)
**Owner**: Backend Engineer
**Duration**: Days 8-9

**Acceptance Criteria**:
- [ ] JWT validation extracts tenant_id
- [ ] PostgreSQL session variable set per request
- [ ] Feature flag controls RLS enablement
- [ ] Error handling for missing tenant_id
- [ ] Middleware tested end-to-end
- [ ] API endpoints respect tenant isolation

**Deliverables**:
1. Middleware: `src/alpha_pulse/api/middleware/tenant_context.py`
2. JWT service: Updated to include tenant_id claim
3. Integration tests: `tests/integration/test_tenant_isolation.py`
4. API documentation updated

---

## Daily Breakdown

### Week 1: Schema Changes (Days 1-5)

#### Day 1 (Monday): US-001 Start
- [ ] Morning: Create tenants table migration
- [ ] Afternoon: Create Tenant model and API endpoints
- [ ] Evening: Write unit tests
- [ ] **Deliverable**: Migration + Model + Tests

#### Day 2 (Tuesday): US-001 Complete + US-002 Start
- [ ] Morning: Complete US-001 (code review, merge)
- [ ] Afternoon: Create users.tenant_id migration
- [ ] Evening: Backfill data, test rollback
- [ ] **Deliverable**: Migration applied, users updated

#### Day 3 (Wednesday): US-002 Complete + US-003 Start
- [ ] Morning: Complete US-002 (tests, merge)
- [ ] Afternoon: Start domain tables migration (trades, positions)
- [ ] Evening: Create indexes
- [ ] **Deliverable**: 2/6 tables updated

#### Day 4 (Thursday): US-003 Continue
- [ ] Morning: Update remaining tables (orders, portfolio_snapshots)
- [ ] Afternoon: Update risk_metrics, agent_signals
- [ ] Evening: Backfill all data
- [ ] **Deliverable**: 6/6 tables updated

#### Day 5 (Friday): US-003 Complete + Mid-Sprint Review
- [ ] Morning: Performance testing (before/after indexes)
- [ ] Afternoon: Complete US-003 (tests, merge)
- [ ] Evening: Mid-sprint review, retrospective
- [ ] **Deliverable**: All schema changes complete

**Mid-Sprint Progress**: 11 SP / 21 SP = 52%

---

### Week 2: RLS & Integration (Days 6-10)

#### Day 6 (Monday): US-004 Start
- [ ] Morning: Create RLS policies migration
- [ ] Afternoon: Enable RLS on all tables
- [ ] Evening: Create bypass role
- [ ] **Deliverable**: RLS enabled

#### Day 7 (Tuesday): US-004 Complete
- [ ] Morning: Test RLS policies (multi-tenant scenarios)
- [ ] Afternoon: Write comprehensive test suite
- [ ] Evening: Complete US-004 (code review, merge)
- [ ] **Deliverable**: RLS tested and validated

#### Day 8 (Wednesday): US-005 Start
- [ ] Morning: Implement tenant context middleware
- [ ] Afternoon: Update JWT service
- [ ] Evening: Test feature flag integration
- [ ] **Deliverable**: Middleware implemented

#### Day 9 (Thursday): US-005 Complete
- [ ] Morning: Write integration tests
- [ ] Afternoon: End-to-end testing (API → DB)
- [ ] Evening: Complete US-005 (code review, merge)
- [ ] **Deliverable**: Full tenant isolation working

#### Day 10 (Friday): Sprint Review & Demo
- [ ] Morning: Prepare demo (multi-tenant scenarios)
- [ ] Afternoon: Sprint 5 review and demo
- [ ] Evening: Sprint 5 retrospective + Sprint 6 planning
- [ ] **Deliverable**: Sprint complete, Sprint 6 planned

---

## Technical Implementation Details

### Database Schema Changes

**Tenants Table**:
```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    subscription_tier VARCHAR(50) NOT NULL CHECK (subscription_tier IN ('starter', 'pro', 'enterprise')),
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'cancelled')),
    max_users INT DEFAULT 5,
    max_api_calls_per_day INT DEFAULT 10000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_tenants_slug ON tenants(slug);
CREATE INDEX idx_tenants_status ON tenants(status);
CREATE INDEX idx_tenants_tier ON tenants(subscription_tier);
```

**RLS Policy Example** (trades table):
```sql
-- Enable RLS
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

-- Policy for SELECT
CREATE POLICY tenant_isolation_select ON trades
    FOR SELECT
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- Policy for INSERT
CREATE POLICY tenant_isolation_insert ON trades
    FOR INSERT
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- Policy for UPDATE
CREATE POLICY tenant_isolation_update ON trades
    FOR UPDATE
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- Policy for DELETE
CREATE POLICY tenant_isolation_delete ON trades
    FOR DELETE
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);
```

---

### Middleware Implementation

```python
# src/alpha_pulse/api/middleware/tenant_context.py
from fastapi import Request, HTTPException
from jose import jwt, JWTError
import logging

logger = logging.getLogger(__name__)

async def tenant_context_middleware(request: Request, call_next):
    """
    Extract tenant_id from JWT and set PostgreSQL session variable.
    """
    # Skip for health/metrics endpoints
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)

    # Extract JWT from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.replace("Bearer ", "")

    try:
        # Decode JWT
        payload = jwt.decode(
            token,
            request.app.state.settings.JWT_SECRET,
            algorithms=[request.app.state.settings.JWT_ALGORITHM]
        )

        tenant_id = payload.get("tenant_id")
        user_id = payload.get("sub")

        if not tenant_id:
            raise HTTPException(status_code=401, detail="Missing tenant_id in token")

        # Store in request state
        request.state.tenant_id = tenant_id
        request.state.user_id = user_id

        # Set PostgreSQL session variable (if RLS enabled)
        if request.app.state.settings.RLS_ENABLED:
            async with request.app.state.db_pool.acquire() as conn:
                await conn.execute(
                    "SET LOCAL app.current_tenant_id = $1",
                    tenant_id
                )
                logger.debug(f"Set tenant context: {tenant_id}")

        # Continue processing request
        response = await call_next(request)
        return response

    except JWTError as e:
        logger.error(f"JWT validation failed: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Tenant context middleware error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## Testing Strategy

### Unit Tests (>90% coverage)
- Model validation (tenants, users with tenant_id)
- RLS policy enforcement (per-table)
- Middleware JWT extraction
- Feature flag behavior

### Integration Tests
- Multi-tenant user creation
- Cross-tenant isolation (user A cannot see user B's data)
- API endpoints respect tenant boundaries
- Error handling (missing tenant_id)

### Performance Tests
- Baseline query performance (without RLS)
- RLS query performance (with RLS enabled)
- Impact measurement (<25% degradation)
- Index effectiveness

### Security Tests
- Attempt to bypass RLS (should fail)
- SQL injection attempts
- JWT tampering detection
- Session variable manipulation

---

## Rollback Plan

### Phase 1: Immediate Rollback (within 5 min)
**Trigger**: Critical errors, service outage

```bash
# Disable RLS feature flag
kubectl set env deployment/alphapulse-api RLS_ENABLED=false
kubectl rollout restart deployment/alphapulse-api
```

**Expected Recovery**: <2 minutes

---

### Phase 2: Partial Rollback (within 24 hours)
**Trigger**: Performance issues, RLS bugs

```bash
# Downgrade RLS policies migration only
poetry run alembic downgrade -1
```

**Expected Recovery**: <10 minutes

---

### Phase 3: Full Rollback (within 1 week)
**Trigger**: Fundamental design issues

```bash
# Restore from backup
pg_restore backup_pre_sprint5.dump

# Downgrade all migrations
poetry run alembic downgrade <previous_version>
```

**Expected Recovery**: 1-2 hours

---

## Monitoring & Metrics

### Key Metrics to Track

| Metric | Baseline | Target | Alert Threshold |
|--------|----------|--------|-----------------|
| **Query Latency (p99)** | 150ms | <188ms (+25%) | >200ms |
| **Error Rate** | 0.1% | <0.2% | >0.5% |
| **RLS Policy Hits** | N/A | 100% | <95% |
| **Cross-Tenant Leaks** | N/A | 0 | >0 |
| **API Response Time** | 120ms | <150ms | >180ms |

### Monitoring Queries

```sql
-- Check RLS is enabled
SELECT schemaname, tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- Check policy enforcement
SELECT COUNT(*) as policy_count
FROM pg_policies
WHERE schemaname = 'public';

-- Monitor query performance
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE query LIKE '%tenant_id%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Verify tenant isolation
SELECT tenant_id, COUNT(*) as user_count
FROM users
GROUP BY tenant_id;
```

---

## Dependencies & Blockers

### Pre-Requisites
- [ ] Sprint 4 load testing complete ✅
- [ ] Stakeholder approvals received (5/6 minimum)
- [ ] Database backup created
- [ ] Feature flag infrastructure ready

### External Dependencies
- GCP access (not needed until production deployment)
- DBA Lead availability (RLS policy review)
- Security Lead review (tenant isolation validation)

### Known Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance degradation >25% | Medium | High | Comprehensive indexing strategy |
| RLS bypass vulnerability | Low | Critical | Security review + penetration testing |
| Data backfill errors | Low | High | Dry-run on staging first |
| Rollback complexity | Medium | Medium | Test rollback procedures before migration |

---

## Definition of Done

Sprint 5 is complete when:
- [ ] All 5 user stories implemented (21 SP)
- [ ] All migrations applied successfully
- [ ] RLS policies enforce tenant isolation (0 leaks)
- [ ] Performance impact <25%
- [ ] Test coverage >90%
- [ ] Code reviewed and merged
- [ ] Documentation updated
- [ ] Rollback tested
- [ ] Sprint demo completed
- [ ] Sprint 6 planned

---

## Sprint 6 Preview

**EPIC-001: Database Multi-Tenancy (Part 2)** (21 SP)

Focus areas:
- Tenant management API endpoints
- Subscription tier enforcement
- Usage tracking and limits
- Tenant onboarding flow
- Admin panel for tenant management

---

## References

- [Database Migration Plan](../docs/database-migration-plan.md)
- [Database Migration Checklist](../docs/database-migration-checklist-sprint5.md)
- [PostgreSQL RLS Documentation](https://www.postgresql.org/docs/14/ddl-rowsecurity.html)
- [Sprint 4 Retrospective](./sprint-4-retrospective.md)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

**Last Updated**: 2025-10-28
**Status**: READY FOR EXECUTION
