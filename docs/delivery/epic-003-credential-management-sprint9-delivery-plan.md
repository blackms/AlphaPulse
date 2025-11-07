# EPIC-003: Credential Management - Delivery Plan

**Epic**: EPIC-003 (#142)
**Sprint**: 9-10
**Story Points**: 36
**Date**: 2025-11-07
**Phase**: Build & Validate
**Author**: Tech Lead (via Claude Code)
**Status**: READY FOR EXECUTION

---

## Executive Summary

**Duration**: 2 sprints (Weeks 17-20)
**Team**: 1 backend engineer (parallel with EPIC-004)
**Velocity**: 18 SP per sprint
**Delivery**: Week of 2025-12-09

---

## Sprint 9 (Weeks 17-18): Infrastructure + Core Service

### Week 17 (Sprint 9, Part 1)

#### Story 3.1: Deploy Vault HA Cluster (8 SP)

**Owner**: Backend Engineer A + DevOps
**Dependencies**: DevOps availability, AWS KMS access

**Tasks**:
1. **RED** (Day 1):
   - Write Vault health check tests (`tests/integration/test_vault_ha.py`)
   - Test: Single node failure recovery
   - Test: Auto-unseal on restart
   - Test: Metrics endpoint accessibility

2. **GREEN** (Days 2-3):
   - Create Vault Kubernetes manifests (`k8s/vault/statefulset.yaml`)
   - Deploy 3-node Raft cluster (vault-0, vault-1, vault-2)
   - Configure auto-unseal with AWS KMS
   - Initialize cluster and distribute unseal keys

3. **REFACTOR** (Day 4):
   - Configure Prometheus integration
   - Set up Grafana dashboard for Vault metrics
   - Document runbook (`docs/runbooks/vault-operations.md`)

4. **QUALITY** (Day 5):
   - Kill vault-0, verify vault-1 takes over
   - Restart cluster, verify auto-unseal
   - Load test: 1000 req/sec for 5 minutes

**Acceptance Criteria**:
- [x] Vault cluster survives killing 1 of 3 nodes
- [x] Auto-unseal works on restart
- [x] Metrics endpoint accessible at vault.internal:9091/metrics

**Commit Message**: `feat(vault): deploy HA cluster with Raft + auto-unseal (Story 3.1)`

---

### Week 18 (Sprint 9, Part 2)

#### Story 3.2: Implement TenantCredentialService (8 SP)

**Owner**: Backend Engineer A
**Dependencies**: Vault HA cluster (Story 3.1), SecretsManager existing code

**Tasks**:
1. **RED** (Day 1):
   - Write service interface tests (`tests/services/test_tenant_credential_service.py`)
   - Test: `get_credentials()` with cache hit/miss
   - Test: `validate_and_store()` success/failure
   - Test: Cache invalidation on update

2. **GREEN** (Days 2-3):
   - Implement `TenantCredentialService` class (`src/alpha_pulse/services/tenant_credential_service.py`)
   - Methods: `get_credentials()`, `validate_and_store()`, `delete_credentials()`
   - Integrate with existing `SecretsManager` (Vault I/O)
   - Integrate with existing `CCXTAdapter` (validation)

3. **REFACTOR** (Day 4):
   - Add caching layer (5-min TTL in-memory LRU)
   - Optimize: Batch metadata queries
   - Add detailed logging and metrics

4. **QUALITY** (Day 5):
   - Integration test with real Vault instance
   - Test: 1000 credential retrievals (<5ms P99)
   - Test: Invalid credentials rejected before storage

**Acceptance Criteria**:
- [x] Credentials stored with tenant-scoped Vault paths
- [x] Invalid credentials rejected via CCXT validation
- [x] <5ms P99 latency (from cache)

**Commit Message**: `feat(credentials): implement TenantCredentialService with validation (Story 3.2)`

---

#### Story 3.3: Create Database Schema & Metadata (5 SP)

**Owner**: Backend Engineer A
**Dependencies**: PostgreSQL multi-tenancy (EPIC-001)

**Tasks**:
1. **RED** (Day 1):
   - Write migration tests (`tests/migrations/test_credential_schema.py`)
   - Test: Table creation with all columns
   - Test: Indexes created
   - Test: RLS policy enforced

2. **GREEN** (Day 2):
   - Create Alembic migration (`alembic/versions/xxx_add_tenant_credentials.py`)
   - Create `tenant_credentials` table with columns:
     - id, tenant_id, exchange, credential_type, vault_path, status, permissions, last_validated_at, created_at, updated_at
   - Create indexes: tenant_id, status, exchange

3. **REFACTOR** (Day 3):
   - Create SQLAlchemy ORM model (`src/alpha_pulse/models/tenant_credential.py`)
   - Add helper methods: `find_by_tenant_and_exchange()`
   - Add validation: Unique constraint on (tenant_id, exchange, credential_type)

4. **QUALITY** (Day 3):
   - Test RLS policy (tenant A cannot see tenant B's credentials)
   - Test unique constraint (cannot insert duplicate)
   - Test cascade delete (tenant delete removes credentials)

**Acceptance Criteria**:
- [x] Table supports all metadata fields
- [x] RLS prevents cross-tenant access
- [x] Indexes on (tenant_id, exchange)

**Commit Message**: `feat(db): add tenant_credentials table with RLS (Story 3.3)`

---

## Sprint 10 (Weeks 19-20): API + Automation

### Week 19 (Sprint 10, Part 1)

#### Story 3.4: Build API Endpoints (5 SP)

**Owner**: Backend Engineer A
**Dependencies**: TenantCredentialService (Story 3.2), Database schema (Story 3.3)

**Tasks**:
1. **RED** (Day 1):
   - Write API integration tests (`tests/api/test_credentials_router.py`)
   - Test: POST /credentials (create)
   - Test: GET /credentials (list)
   - Test: DELETE /credentials/{id}
   - Test: POST /credentials/{id}/validate

2. **GREEN** (Days 2-3):
   - Create router (`src/alpha_pulse/api/routers/credentials.py`)
   - Endpoints:
     - `POST /api/v1/credentials` → `create_or_update_credentials()`
     - `GET /api/v1/credentials` → `list_credentials()`
     - `GET /api/v1/credentials/{id}` → `get_credential()`
     - `DELETE /api/v1/credentials/{id}` → `delete_credential()`
     - `POST /api/v1/credentials/{id}/validate` → `validate_credential()`
     - `GET /api/v1/credentials/{id}/health` → `get_health_status()`

3. **REFACTOR** (Day 4):
   - Add OpenAPI documentation (description, examples)
   - Add request/response models (Pydantic)
   - Add error handling (400, 401, 404, 422)

4. **QUALITY** (Day 5):
   - Security scan (Bandit, Semgrep)
   - Test tenant isolation (tenant A cannot delete tenant B's credentials)
   - Test JWT authentication required

**Acceptance Criteria**:
- [x] All endpoints require JWT authentication
- [x] Swagger docs complete at /docs
- [x] Tenant isolation validated (0% cross-tenant access)

**Commit Message**: `feat(api): add credentials CRUD endpoints with tenant isolation (Story 3.4)`

---

#### Story 3.5: Implement Health Check Job (5 SP)

**Owner**: Backend Engineer A
**Dependencies**: TenantCredentialService (Story 3.2), APScheduler setup

**Tasks**:
1. **RED** (Day 1):
   - Write scheduler tests (`tests/jobs/test_credential_health_check.py`)
   - Test: Job runs on schedule (6h interval)
   - Test: Failed credentials marked as 'invalid'
   - Test: Webhook sent on failure

2. **GREEN** (Days 2-3):
   - Create job (`src/alpha_pulse/jobs/credential_health_check.py`)
   - Implement `CredentialHealthCheckJob` class
   - Method: `run_health_check()` → calls `service.health_check_all_active()`
   - Schedule with APScheduler (interval: 6 hours)

3. **REFACTOR** (Day 4):
   - Add webhook notification integration
   - Add retry logic (3 attempts with exponential backoff)
   - Add detailed logging (success/failure summary)

4. **QUALITY** (Day 5):
   - Test failure detection (mock invalid credentials)
   - Test webhook delivery
   - Verify job doesn't block main thread

**Acceptance Criteria**:
- [x] Job runs every 6 hours
- [x] Failed credentials marked as 'invalid'
- [x] Webhook sent with tenant_id + error details

**Commit Message**: `feat(jobs): add background health check for credentials (Story 3.5)`

---

### Week 20 (Sprint 10, Part 2)

#### Story 3.6: Credential Rotation Flow (5 SP)

**Owner**: Backend Engineer A
**Dependencies**: TenantCredentialService (Story 3.2), Vault versioning

**Tasks**:
1. **RED** (Day 1):
   - Write rotation tests (`tests/services/test_credential_rotation.py`)
   - Test: Successful rotation
   - Test: Old + new credentials valid during transition
   - Test: Automatic cleanup after 1 hour

2. **GREEN** (Days 2-3):
   - Implement `rotate_credentials()` method in `TenantCredentialService`
   - Validate new credentials before storing
   - Store new version in Vault (Vault auto-versions)
   - Update metadata with new `updated_at`

3. **REFACTOR** (Day 4):
   - Add graceful transition period (1 hour)
   - Implement cleanup job to delete old versions after grace period
   - Add audit logging for rotation events

4. **QUALITY** (Day 5):
   - Test zero-downtime rotation (trading continues during rotation)
   - Test rollback (if new credentials fail, revert to old)
   - Test audit log completeness

**Acceptance Criteria**:
- [x] Old + new credentials valid during transition
- [x] Automatic cleanup after 1 hour
- [x] Audit log records rotation

**Commit Message**: `feat(credentials): implement rotation with graceful transition (Story 3.6)`

---

## Integration & Testing (Final Days of Sprint 10)

### End-to-End Testing (Day 1-2)

**Scenarios**:
1. **Full Credential Lifecycle**:
   - Tenant creates credentials → Validation passes → Stored in Vault
   - Trading agent retrieves credentials → Executes trade
   - Tenant rotates credentials → Old + new both work
   - After 1 hour → Old credentials deleted

2. **Health Check Failure**:
   - Tenant's API key expires on exchange
   - Health check job detects failure (6h interval)
   - Credential marked as 'invalid' in database
   - Webhook sent to tenant

3. **Vault Failover**:
   - Kill vault-0 (primary)
   - Vault-1 becomes leader
   - Credential retrieval continues without errors

**Test Locations**:
- `tests/e2e/test_credential_lifecycle.py`
- `tests/e2e/test_health_check_failure.py`
- `tests/e2e/test_vault_failover.py`

### Security Audit (Day 3)

**Checklist**:
- [ ] Vault TLS encryption verified
- [ ] Audit logging enabled and tested
- [ ] Tenant isolation validated (penetration test)
- [ ] No secrets in logs (masked API keys)
- [ ] JWT authentication enforced on all endpoints
- [ ] RLS policies prevent cross-tenant access

**Tools**:
- Bandit (Python security scanner)
- Semgrep (static analysis)
- OWASP ZAP (penetration testing)

### Performance Testing (Day 4)

**Load Tests**:
1. **Credential Retrieval**: 1000 req/sec for 5 minutes
   - Target: P99 <5ms (cached)
   - Target: P99 <50ms (uncached)

2. **Credential Validation**: 100 concurrent validations
   - Target: All complete within 30 seconds

3. **Health Check**: 10,000 active credentials
   - Target: Complete within 1 hour

**Tools**: Locust, JMeter

### Documentation (Day 5)

**User Documentation**:
- `docs/user-guides/adding-exchange-credentials.md`
- `docs/user-guides/rotating-credentials.md`

**Operational Runbooks**:
- `docs/runbooks/vault-operations.md`
- `docs/runbooks/credential-health-check-failure.md`

**API Documentation**:
- OpenAPI/Swagger at `/docs` (auto-generated)

---

## Deployment Plan

### Pre-Deployment Checklist

**Week before Sprint 9 Kickoff**:
- [ ] DevOps: AWS KMS key created for Vault auto-unseal
- [ ] DevOps: Kubernetes namespace ready (`production`)
- [ ] DevOps: Persistent volumes provisioned (3 × 10GB for Vault)
- [ ] Security Team: Vault policies reviewed
- [ ] Security Team: Audit logging destination configured (CloudWatch/ELK)

### Sprint 9 Deployment (Week 18, Friday)

**Steps**:
1. Deploy Vault HA cluster (Story 3.1) → Production
2. Run smoke tests (health checks, failover test)
3. Deploy database migration (Story 3.3) → Production
4. Deploy TenantCredentialService (Story 3.2) → Production (behind feature flag)
5. Monitor for 24 hours (no rollback needed)

### Sprint 10 Deployment (Week 20, Friday)

**Steps**:
1. Deploy API endpoints (Story 3.4) → Production
2. Deploy health check job (Story 3.5) → Production
3. Deploy rotation flow (Story 3.6) → Production
4. Enable feature flag for all tenants
5. Announce to users via email/dashboard notification

---

## Rollback Plan

### Rollback Triggers

- Critical bug discovered (P0 severity)
- Vault downtime >5 minutes
- Credential validation failure rate >10%
- Security vulnerability identified

### Rollback Steps

1. **Disable Feature Flag**: Revert to old credential system
2. **Stop Health Check Job**: Prevent modifications to database
3. **Drain Vault Traffic**: Redirect to backup credential store
4. **Database Rollback**: Alembic downgrade to previous version
5. **Kubernetes Rollback**: `kubectl rollout undo deployment alphapulse-api`

**Recovery Time Objective (RTO)**: <15 minutes

---

## Risk Management

### High-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Vault HA setup complexity | 40% | HIGH | DevOps pairing, use Vault Raft (simpler than Consul) |
| CCXT validation false negatives | 20% | MEDIUM | Comprehensive test suite with real exchange APIs |
| Health check job overload (10K credentials) | 30% | MEDIUM | Parallelize with asyncio.gather(), add rate limiting |

### Contingency Plans

**Plan A** (Vault HA too complex):
- Use single Vault node + daily backups
- Accept higher downtime risk
- Document manual recovery procedure

**Plan B** (CCXT validation unreliable):
- Skip validation, allow tenants to store invalid credentials
- Detect failures during actual trading
- Notify tenant via webhook

---

## Success Metrics

### Sprint 9 Goals

- ✅ Vault HA deployed and operational
- ✅ TenantCredentialService implemented (90% test coverage)
- ✅ Database schema migrated (RLS policies active)

### Sprint 10 Goals

- ✅ All 6 API endpoints operational
- ✅ Health check job running every 6 hours
- ✅ Rotation flow tested with zero downtime

### Overall EPIC Success

- ✅ <5ms P99 credential retrieval latency (from cache)
- ✅ 100% invalid credentials rejected before storage
- ✅ Vault survives single node failure (HA verified)
- ✅ SOC2-compliant audit trail (all access logged)

---

## Team Communication

### Daily Standups

**Time**: 9:00 AM daily (15 minutes)

**Format**:
- Yesterday: What I completed
- Today: What I'm working on
- Blockers: Any issues

### Sprint Review (End of Each Sprint)

**Attendees**: Product Owner, Tech Lead, Backend Engineer A, DevOps

**Agenda**:
- Demo completed stories
- Review metrics (velocity, quality)
- Discuss feedback

### Retrospective (End of Sprint 10)

**Questions**:
- What went well?
- What could be improved?
- Action items for next sprint

---

## Dependencies

### Upstream (Completed)

- ✅ EPIC-001: Database Multi-Tenancy (tenant_id available)
- ✅ EPIC-002: Application Multi-Tenancy (JWT + tenant context)

### External (To Be Confirmed)

- ⏳ DevOps: Vault HA deployment planning (confirm by Week 16)
- ⏳ Security Team: Review Vault policies (confirm by Week 17)
- ⏳ Cloud Provider: KMS for Vault auto-unseal (provision by Week 17)

### Parallel Execution

- EPIC-004 (Caching Layer): Independent, Backend Engineer B

---

## Sign-Off

**Tech Lead**: ✅ Approved (2025-11-07)
**Product Owner**: ⏳ Pending review
**Backend Engineer A**: ⏳ Acknowledged
**DevOps Lead**: ⏳ Acknowledged (Vault deployment)

**Next Milestone**: Sprint 9 Kickoff (Week 17, Monday)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
