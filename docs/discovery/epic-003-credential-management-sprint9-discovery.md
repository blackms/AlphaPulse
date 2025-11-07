# EPIC-003: Credential Management - Sprint 9-10 Discovery

**Epic**: EPIC-003 (#142)
**Sprint**: 9-10 (Parallel with EPIC-004)
**Story Points**: 36
**Date**: 2025-11-07
**Phase**: Discover & Frame
**Author**: Tech Lead (via Claude Code)

---

## Executive Summary

**Objective**: Deploy multi-tenant credential management system with HashiCorp Vault for secure exchange API credentials.

**Current State**:
- ✅ Strong foundation with `SecretsManager` and `HashiCorpVaultProvider`
- ✅ ADR-003 fully documented (369 lines)
- ⚠️ Single-tenant design, no Vault deployment
- ❌ Missing: Multi-tenant service layer, database schema, API endpoints, validation

**Proposed Solution**: Build multi-tenant credential service on top of existing `SecretsManager` infrastructure.

**Confidence**: HIGH (90%) - Strong foundation exists, clear requirements

**RICE Score**: 72.0 (Reach: 10, Impact: 9, Confidence: 80%, Effort: 1 sprint)

---

## Problem Statement

### Current Pain Points

1. **Single-Tenant Design**: `CredentialsManager` only supports one tenant
2. **No Credential Validation**: Invalid credentials stored without testing
3. **No Health Monitoring**: Expired/revoked credentials not detected automatically
4. **Insecure Storage**: JSON files in `config/credentials/` (security risk)
5. **No Audit Trail**: No tracking of who accessed which credentials

### Business Impact

**Without Fix**:
- Cannot support multi-tenant SaaS (blocks 100+ tenant goal)
- Security vulnerabilities (stored credentials in Git risk)
- Operational toil (manual credential rotation)
- Compliance failure (no audit trail for SOC2)

**With Fix**:
- ✅ Secure, multi-tenant credential storage
- ✅ Automatic validation (catch errors early)
- ✅ Health monitoring (6-hour checks)
- ✅ SOC2-compliant audit trail
- ✅ Zero-touch credential rotation

---

## Existing Infrastructure Analysis

### What We Have ✅

#### 1. SecretsManager (`src/alpha_pulse/utils/secrets_manager.py` - 616 lines)

**Providers Implemented**:
```python
- HashiCorpVaultProvider: Full Vault KV v2 integration with hvac client
- AWSSecretsManagerProvider: AWS with 5-min caching
- LocalEncryptedFileProvider: Fernet-encrypted local storage (dev)
- EnvironmentSecretProvider: Environment variable fallback
```

**Key Features**:
- LRU caching with TTL (5 min)
- Audit logging (`_audit_log` method)
- Factory pattern for environment-based selection
- Graceful fallback chain

**Gap**: Not tenant-scoped, lacks validation logic

#### 2. ADR-003 Documentation (369 lines)

**Path**: `docs/adr/003-credential-management-multi-tenant.md`

**Defined Architecture**:
- Vault path: `secret/tenants/{tenant_id}/exchanges/{exchange_id}/credentials`
- Database: `tenant_credentials` table with status tracking
- Cache TTL: 5 minutes
- Health checks: Every 6 hours
- Validation: CCXT test API calls before storage

**Status**: APPROVED, ready for implementation

#### 3. CCXT Exchange Integrations (22 files)

**Key Files**:
- `src/alpha_pulse/exchanges/adapters/ccxt_adapter.py`
- `src/alpha_pulse/exchanges/implementations/binance.py`
- `src/alpha_pulse/exchanges/implementations/bybit.py`

**Credential Flow**:
```
CredentialsManager → Secrets (API key/secret) → CCXT → Exchange
```

**Gap**: No validation wrapper, no multi-tenant support

### What We Need ❌

#### 1. Database Schema

**Table**: `tenant_credentials`
```sql
CREATE TABLE tenant_credentials (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    exchange VARCHAR(50) NOT NULL,
    credential_type VARCHAR(20) NOT NULL, -- 'api_key' | 'api_secret' | 'passphrase'
    vault_path VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'active', -- 'active' | 'expired' | 'invalid' | 'revoked'
    permissions JSONB, -- {trading: true, withdraw: false}
    last_validated_at TIMESTAMP,
    health_check_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, exchange, credential_type)
);

CREATE INDEX idx_tenant_credentials_tenant_id ON tenant_credentials(tenant_id);
CREATE INDEX idx_tenant_credentials_status ON tenant_credentials(status);
```

#### 2. TenantCredentialService Class

**Path**: `src/alpha_pulse/services/tenant_credential_service.py` (NEW)

**Interface**:
```python
class TenantCredentialService:
    async def get_credentials(tenant_id: UUID, exchange: str) -> Dict
    async def validate_and_store(tenant_id: UUID, exchange: str, api_key: str, secret: str) -> bool
    async def delete_credentials(tenant_id: UUID, exchange: str) -> bool
    async def health_check_all_active() -> List[HealthCheckResult]
    async def rotate_credentials(tenant_id: UUID, exchange: str, new_key: str, new_secret: str) -> bool
```

**Dependencies**:
- `SecretsManager` (existing - for Vault integration)
- `CCXTAdapter` (existing - for validation calls)
- `tenant_credentials` table (new - for metadata)

#### 3. API Endpoints

**Router**: `src/alpha_pulse/api/routers/credentials.py` (NEW)

**Endpoints**:
```
POST   /api/v1/credentials          - Create/update credentials
GET    /api/v1/credentials          - List all tenant credentials
GET    /api/v1/credentials/{id}     - Get specific credential metadata
DELETE /api/v1/credentials/{id}     - Delete credentials
POST   /api/v1/credentials/{id}/validate - Manual validation trigger
GET    /api/v1/credentials/{id}/health - Get health check status
```

**Security**: All endpoints require JWT + tenant context

#### 4. Background Health Check Job

**Path**: `src/alpha_pulse/jobs/credential_health_check.py` (NEW)

**Scheduler**:
```python
# APScheduler job - runs every 6 hours
@scheduler.scheduled_job('interval', hours=6)
async def check_all_credentials():
    service = TenantCredentialService()
    results = await service.health_check_all_active()

    for result in results:
        if result.failed:
            await send_webhook(result.tenant_id, result.error)
            await update_status(result.id, 'invalid')
```

---

## Technical Design Summary

### Architecture Layers

```
API Layer
  └─→ credentials.py (FastAPI router)
       └─→ TenantCredentialService
            ├─→ SecretsManager (Vault integration)
            ├─→ CCXTAdapter (validation)
            ├─→ Database (metadata persistence)
            └─→ CachingService (5-min TTL)

Background Jobs
  └─→ CredentialHealthCheck (every 6h)
       └─→ TenantCredentialService.health_check_all_active()
```

### Vault Path Structure (from ADR-003)

```
secret/tenants/{tenant_id}/exchanges/{exchange}/credentials
  ├─ api_key
  ├─ api_secret
  └─ passphrase (optional)

Example:
secret/tenants/00000000-0000-0000-0000-000000000001/exchanges/binance/credentials
  ├─ api_key: "abc123..."
  ├─ api_secret: "def456..."
```

### Validation Flow

```
1. User submits credentials via API
2. TenantCredentialService validates format
3. CCXTAdapter creates test exchange instance
4. Test API call: exchange.fetch_balance()
5. If success:
   - Store in Vault (SecretsManager)
   - Save metadata in database
   - Return success
6. If failure:
   - Return error (do not store)
```

### Caching Strategy

```
L1: In-memory LRU (5-min TTL)
  └─→ L2: Vault (persistent)
       └─→ L3: Database metadata (status tracking)

Cache Key: tenant:{tenant_id}:credentials:{exchange}
TTL: 5 minutes (matches SecretsManager default)
Eviction: LRU per tenant
```

---

## RICE Prioritization

**Reach**: 10/10 (All tenants need secure credentials)
**Impact**: 9/10 (Security + compliance requirement)
**Confidence**: 80% (Strong foundation, clear requirements)
**Effort**: 1 sprint (36 SP / 2 sprints with EPIC-004 parallel)

**RICE Score**: (10 × 9 × 0.80) / 1 = **72.0** (VERY HIGH)

---

## Success Metrics

### Technical Metrics
- **Latency**: <5ms P99 for credential retrieval (from cache)
- **Availability**: Vault survives single node failure (HA with 3 replicas)
- **Security**: 100% credentials stored in Vault (0% in files/Git)
- **Validation**: 100% invalid credentials rejected before storage

### Business Metrics
- **Compliance**: SOC2-compliant audit trail (all access logged)
- **Reliability**: <1% credential health check failures
- **Efficiency**: Zero manual credential rotation (automated)
- **Security**: 0 credential leaks (Vault encryption + audit)

---

## Dependencies

### Upstream
- ✅ EPIC-001: Database Multi-Tenancy (tenant_id available)
- ✅ EPIC-002: Application Multi-Tenancy (JWT + tenant context)

### External
- ⏳ **DevOps**: Vault HA deployment (3 replicas, Raft consensus)
- ⏳ **Security Team**: Review Vault policies and audit logging
- ⏳ **Cloud Provider**: KMS for Vault auto-unseal (AWS/GCP)

### Parallel
- **EPIC-004** (Caching Layer): Can run in parallel, shares CachingService

---

## Risks & Mitigation

| Risk | Severity | Probability | Impact | Mitigation |
|------|----------|-------------|--------|------------|
| Vault downtime blocks all credential access | HIGH | 20% | CRITICAL | HA deployment (3 replicas) + cached credentials (1h grace period) |
| Vault HA setup complexity | MEDIUM | 40% | MEDIUM | Use Vault Raft (simpler than Consul), DevOps pairing |
| Invalid credentials stored despite validation | MEDIUM | 10% | MEDIUM | Comprehensive CCXT tests, permission level detection |
| KMS auto-unseal dependency | MEDIUM | 30% | MEDIUM | Manual unseal runbook, alerts on sealed state |
| Credential rotation breaks live trading | LOW | 20% | HIGH | Graceful transition period (old + new both valid for 1h) |

---

## Stories Breakdown (36 SP)

### Story 3.1: Deploy Vault HA Cluster (8 SP)
**Tasks**:
1. **RED**: Write Vault health check tests
2. **GREEN**: Deploy 3-node Raft cluster (Docker/K8s)
3. **REFACTOR**: Configure auto-unseal with KMS
4. **QUALITY**: Test single-node failure recovery

**Acceptance Criteria**:
- Vault cluster survives killing 1 of 3 nodes
- Auto-unseal works on restart
- Metrics endpoint accessible (Prometheus integration)

### Story 3.2: Implement TenantCredentialService (8 SP)
**Tasks**:
1. **RED**: Write service interface tests
2. **GREEN**: Implement `get_credentials()`, `validate_and_store()`
3. **REFACTOR**: Add caching layer (5-min TTL)
4. **QUALITY**: Integration tests with Vault

**Acceptance Criteria**:
- Credentials stored with tenant-scoped Vault paths
- Invalid credentials rejected via CCXT validation
- <5ms P99 latency (from cache)

### Story 3.3: Create Database Schema & Metadata (5 SP)
**Tasks**:
1. **RED**: Write migration tests
2. **GREEN**: Create `tenant_credentials` table
3. **REFACTOR**: Add indexes for performance
4. **QUALITY**: Test RLS policies

**Acceptance Criteria**:
- Table supports all metadata fields
- RLS prevents cross-tenant access
- Indexes on (tenant_id, exchange)

### Story 3.4: Build API Endpoints (5 SP)
**Tasks**:
1. **RED**: Write API integration tests
2. **GREEN**: Implement CRUD endpoints
3. **REFACTOR**: Add OpenAPI documentation
4. **QUALITY**: Security scan + tenant isolation tests

**Acceptance Criteria**:
- All endpoints require JWT authentication
- Swagger docs complete
- Tenant isolation validated (0% cross-tenant access)

### Story 3.5: Implement Health Check Job (5 SP)
**Tasks**:
1. **RED**: Write scheduler tests
2. **GREEN**: Implement APScheduler job (6h interval)
3. **REFACTOR**: Add webhook notifications
4. **QUALITY**: Test failure detection

**Acceptance Criteria**:
- Job runs every 6 hours
- Failed credentials marked as 'invalid'
- Webhook sent with tenant_id + error details

### Story 3.6: Credential Rotation Flow (5 SP)
**Tasks**:
1. **RED**: Write rotation tests
2. **GREEN**: Implement `rotate_credentials()`
3. **REFACTOR**: Add graceful transition period
4. **QUALITY**: Test zero-downtime rotation

**Acceptance Criteria**:
- Old + new credentials valid during transition
- Automatic cleanup after 1 hour
- Audit log records rotation

---

## Delivery Timeline

### Sprint 9 (Weeks 17-18)
**Focus**: Infrastructure + Core Service

- Week 1: Vault deployment (Story 3.1)
- Week 2: TenantCredentialService (Story 3.2) + Database (Story 3.3)

**Deliverable**: Vault HA operational, core service implemented

### Sprint 10 (Weeks 19-20)
**Focus**: API + Automation

- Week 1: API endpoints (Story 3.4) + Health checks (Story 3.5)
- Week 2: Rotation flow (Story 3.6) + Integration testing

**Deliverable**: Full credential management system operational

---

## Testing Strategy

### Unit Tests (90% coverage target)
- TenantCredentialService methods
- Validation logic (valid/invalid credentials)
- Vault path construction

### Integration Tests
- Vault read/write operations
- CCXT validation calls (mock exchanges)
- Database persistence

### End-to-End Tests
- Full credential lifecycle (create → validate → use → rotate → delete)
- Health check failure scenario
- Vault failover (kill 1 node, verify recovery)

### Security Tests
- Tenant isolation (cannot access other tenant credentials)
- Audit logging (all access recorded)
- Vault encryption (secrets never in plaintext logs)

---

## Monitoring & Observability

### Prometheus Metrics
```
alphapulse_credentials_total{tenant_id, exchange, status}
alphapulse_credentials_validation_duration_seconds{exchange}
alphapulse_credentials_health_check_failures_total{tenant_id, exchange}
alphapulse_vault_operations_total{operation, status}
```

### Grafana Dashboards
- Credential health by tenant
- Validation success/failure rates
- Vault latency (P50, P95, P99)
- Health check job execution history

### Alerts
- Vault sealed state (CRITICAL)
- Vault node down (WARNING)
- Health check failure rate > 5% (WARNING)
- Credential validation failure spike (INFO)

---

## Security Considerations

### Vault Security
- ✅ TLS encryption for all Vault communication
- ✅ Auto-unseal with KMS (no manual unseal keys)
- ✅ Audit logging enabled (all secret access tracked)
- ✅ Least-privilege policies per tenant

### API Security
- ✅ JWT authentication required
- ✅ Tenant context validated
- ✅ Rate limiting (100 req/min per tenant)
- ✅ No credentials in logs (masked API keys)

### Database Security
- ✅ RLS policies (tenant isolation)
- ✅ Encrypted connections (SSL/TLS)
- ✅ No plaintext secrets (only Vault paths stored)
- ✅ Audit trail (created_at, updated_at, last_validated_at)

---

## ADR Reference

**Decision Record**: ADR-003 (Credential Management Multi-Tenant)
**Path**: `docs/adr/003-credential-management-multi-tenant.md`
**Status**: APPROVED
**Date**: 2025-10-20

**Key Decisions**:
1. Use HashiCorp Vault for secret storage (vs AWS Secrets Manager)
2. Tenant-scoped Vault paths: `secret/tenants/{tenant_id}/...`
3. 5-minute cache TTL (balance security + performance)
4. CCXT validation before storage (catch errors early)
5. Background health checks every 6 hours (automatic detection)

---

## Stakeholder Sign-Off

**Tech Lead**: ✅ Approved (2025-11-07)
**Product Owner**: ⏳ Pending review
**Security Team**: ⏳ Pending review (Sprint 9)
**DevOps Team**: ⏳ Pending Vault deployment planning

**Next Review**: Sprint 9 kickoff (after Sprint 8 retrospective)

---

## Appendix: Existing Code Locations

### Core Infrastructure
- `src/alpha_pulse/utils/secrets_manager.py` (616 lines) - SecretsManager with providers
- `src/alpha_pulse/exchanges/credentials/manager.py` - Single-tenant CredentialsManager
- `src/alpha_pulse/exchanges/adapters/ccxt_adapter.py` - CCXT integration
- `docs/adr/003-credential-management-multi-tenant.md` (369 lines) - Architecture

### Exchange Implementations
- `src/alpha_pulse/exchanges/implementations/binance.py`
- `src/alpha_pulse/exchanges/implementations/bybit.py`
- `src/alpha_pulse/exchanges/implementations/coinbase.py`

### Tests
- `tests/exchanges/test_credentials_manager.py` - Existing credential tests
- `tests/utils/test_secrets_manager.py` - SecretsManager unit tests

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
