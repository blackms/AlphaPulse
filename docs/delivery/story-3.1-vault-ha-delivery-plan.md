# Story 3.1: Deploy Vault HA Cluster - Delivery Plan

**Epic**: EPIC-003 (Credential Management)
**Story Points**: 8
**Sprint**: 9 (Week 1-2)
**Related**: [Discovery](../discovery/story-3.1-vault-ha-discovery.md), [HLD](../design/story-3.1-vault-ha-hld.md), [ADR-003](../adr/003-credential-management-multi-tenant.md)

## Sprint Goal

Deploy a production-ready 3-node HashiCorp Vault cluster with Raft consensus, auto-unseal, audit logging, and comprehensive testing. This provides the secure credential storage foundation for EPIC-003 (Credential Management).

## Team Allocation

- **DevOps Engineer**: 1 week (50% allocation)
  - Vault infrastructure setup
  - Docker Compose / Kubernetes deployment
  - KMS configuration
  - Monitoring setup

- **Backend Developer**: 0.5 week (25% allocation)
  - Vault client library integration
  - Integration tests
  - Health check scripts

- **Security Engineer**: 2 hours (review)
  - Security configuration review
  - Audit logging verification
  - Policy validation

**Total Effort**: ~1.5 weeks (8 story points)

## Task Breakdown

### Phase 1: Infrastructure Setup (Days 1-3)

**Task 1.1: Create Vault configuration files** ✅
- [ ] Create `config/vault-node1.hcl` (Raft storage, listeners, telemetry)
- [ ] Create `config/vault-node2.hcl` (node-specific addresses)
- [ ] Create `config/vault-node3.hcl` (node-specific addresses)
- [ ] Create `config/vault-dev-override.hcl` (file-based unseal for local dev)
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Config files pass `vault validate-config`

**Task 1.2: Create Docker Compose configuration** ✅
- [ ] Create `docker-compose.vault.yml` (3 Vault containers)
- [ ] Configure persistent volumes (vault-1-data, vault-2-data, vault-3-data)
- [ ] Configure health checks (vault status every 10s)
- [ ] Configure ports (8200, 8202, 8204 for local access)
- [ ] Add to main `docker-compose.yml` as optional profile
- **Assignee**: DevOps Engineer
- **Duration**: 3 hours
- **Acceptance**: `docker-compose -f docker-compose.vault.yml up -d` succeeds, all 3 containers healthy

**Task 1.3: Create initialization script** ✅
- [ ] Create `scripts/init-vault-cluster.sh`
  - Initialize Vault (node 1)
  - Save root token and unseal keys securely
  - Join nodes 2 and 3 to Raft cluster
  - Enable audit logging
  - Enable KV v2 secrets engine
  - Create initial tenant isolation policy
- [ ] Make script idempotent (check if already initialized)
- [ ] Add error handling and validation
- **Assignee**: DevOps Engineer
- **Duration**: 4 hours
- **Acceptance**: Script completes successfully, all 3 nodes in Raft cluster

**Task 1.4: Create health check script** ✅
- [ ] Create `scripts/vault-health-check.sh`
  - Check each node's health endpoint
  - Verify seal status (unsealed)
  - Verify Raft cluster status (3 peers)
  - Identify leader/follower roles
- [ ] Add colored output (green=healthy, red=unhealthy)
- [ ] Exit code 0 if all healthy, 1 if any unhealthy
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Script correctly identifies node states, fails on sealed node

**Task 1.5: Add Vault to main docker-compose** ✅
- [ ] Add Vault profile to `docker-compose.yml`
  ```yaml
  profiles:
    - vault
  services:
    vault-1:
      extends:
        file: docker-compose.vault.yml
        service: vault-1
  ```
- [ ] Update `.env.example` with Vault variables
  ```
  VAULT_ADDR=http://localhost:8200
  VAULT_TOKEN=
  VAULT_DEV_ROOT_TOKEN_ID=
  ```
- [ ] Update README.md with Vault startup instructions
- **Assignee**: DevOps Engineer
- **Duration**: 1 hour
- **Acceptance**: `docker-compose --profile vault up -d` starts all services including Vault

### Phase 2: Vault Client Integration (Days 4-5)

**Task 2.1: Install hvac library** ✅
- [ ] Add `hvac` to `pyproject.toml`
  ```toml
  hvac = "^2.1.0"  # HashiCorp Vault client
  ```
- [ ] Run `poetry install`
- [ ] Verify import works: `poetry run python -c "import hvac; print(hvac.__version__)"`
- **Assignee**: Backend Developer
- **Duration**: 15 minutes
- **Acceptance**: hvac library installed and importable

**Task 2.2: Create VaultClient wrapper** ✅
- [ ] Create `src/alpha_pulse/vault/__init__.py`
- [ ] Create `src/alpha_pulse/vault/client.py` with `VaultClient` class
  - `__init__(url, token, verify, namespace)`
  - `read_secret(path) -> Dict[str, Any]`
  - `write_secret(path, data)`
  - `delete_secret(path)`
  - `health_check() -> Dict[str, Any]`
- [ ] Add logging with loguru
- [ ] Add type hints
- [ ] Add docstrings
- **Assignee**: Backend Developer
- **Duration**: 3 hours
- **Acceptance**: VaultClient class implements all methods, passes type checking

**Task 2.3: Create Vault client unit tests** ✅
- [ ] Create `src/alpha_pulse/tests/vault/__init__.py`
- [ ] Create `src/alpha_pulse/tests/vault/test_vault_client.py`
  - `test_init_success()` - Mock hvac.Client, verify initialization
  - `test_init_no_token()` - Verify ValueError raised
  - `test_init_auth_failure()` - Verify ValueError on invalid token
  - `test_read_secret_success()` - Mock KV read, verify data returned
  - `test_read_secret_not_found()` - Mock InvalidPath exception, verify None returned
  - `test_write_secret_success()` - Mock KV write, verify called correctly
  - `test_delete_secret_success()` - Mock delete, verify called correctly
  - `test_health_check_healthy()` - Mock health endpoint, verify status parsed
- [ ] Target: 100% coverage
- **Assignee**: Backend Developer
- **Duration**: 4 hours
- **Acceptance**: All tests pass, 100% coverage on VaultClient

**Task 2.4: Create Vault integration tests** ✅
- [ ] Create `src/alpha_pulse/tests/vault/test_vault_integration.py`
  - Mark with `@pytest.mark.integration`
  - `test_write_read_delete_secret()` - Full lifecycle test
  - `test_read_nonexistent_secret()` - Verify None returned
  - `test_write_overwrites_existing()` - Verify version increment
  - `test_health_check()` - Verify real cluster health
  - `test_list_secrets()` - Verify list operation (future)
- [ ] Add fixture for VaultClient connected to local cluster
- [ ] Add cleanup (delete test secrets after each test)
- **Assignee**: Backend Developer
- **Duration**: 3 hours
- **Acceptance**: All integration tests pass against local Vault cluster

### Phase 3: Production Configuration (Days 6-7)

**Task 3.1: Configure AWS KMS auto-unseal** 🏗️
- [ ] Create AWS KMS key (Terraform or AWS CLI)
  ```bash
  aws kms create-key --description "AlphaPulse Vault auto-unseal"
  aws kms create-alias --alias-name alias/alphapulse-vault-unseal --target-key-id <key-id>
  ```
- [ ] Create IAM policy for Vault access to KMS
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey"
        ],
        "Resource": "arn:aws:kms:us-east-1:*:key/*"
      }
    ]
  }
  ```
- [ ] Update Vault config with KMS seal stanza
- [ ] Test auto-unseal (restart Vault, verify auto-unseals)
- **Assignee**: DevOps Engineer
- **Duration**: 3 hours
- **Acceptance**: Vault auto-unseals on restart using KMS
- **Note**: Production only (optional for development)

**Task 3.2: Configure TLS certificates** 🏗️
- [ ] Generate TLS certificates (Let's Encrypt or internal CA)
  ```bash
  # Using certbot (Let's Encrypt)
  certbot certonly --dns-route53 -d vault.alphapulse.com
  ```
- [ ] Update Vault config with TLS listener
  ```hcl
  listener "tcp" {
    address       = "0.0.0.0:8200"
    tls_disable   = 0
    tls_cert_file = "/vault/certs/vault.crt"
    tls_key_file  = "/vault/certs/vault.key"
    tls_min_version = "tls13"
  }
  ```
- [ ] Mount certificates in Docker volumes / Kubernetes secrets
- [ ] Test HTTPS access
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Vault accessible via HTTPS with valid certificate
- **Note**: Production only (development uses HTTP)

**Task 3.3: Create Kubernetes Helm values** 🏗️
- [ ] Create `config/vault-helm-values.yaml`
  - HA mode (3 replicas)
  - Raft storage backend
  - AWS KMS seal configuration
  - Audit logging to syslog
  - Prometheus metrics
  - Resource limits (CPU: 500m, Memory: 1Gi)
- [ ] Test deployment to staging Kubernetes cluster
- [ ] Verify Raft cluster formation in K8s
- **Assignee**: DevOps Engineer
- **Duration**: 4 hours
- **Acceptance**: Vault deploys successfully to Kubernetes, Raft cluster operational
- **Note**: Production only (optional for initial development)

**Task 3.4: Configure audit logging** ✅
- [ ] Enable file audit device (development)
  ```bash
  vault audit enable file file_path=/vault/logs/audit.log
  ```
- [ ] Enable syslog audit device (production)
  ```bash
  vault audit enable syslog tag="vault" facility="AUTH"
  ```
- [ ] Configure log rotation (7-day retention, compress old logs)
- [ ] Test audit log writes (create secret, verify logged)
- [ ] Verify HMAC redaction (secrets not in plaintext)
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Audit logs capture all operations, secrets are HMACed

### Phase 4: Monitoring & Alerting (Days 8-9)

**Task 4.1: Configure Prometheus metrics** ✅
- [ ] Verify Vault Prometheus endpoint accessible (`/v1/sys/metrics`)
- [ ] Create `config/prometheus/vault.yml` scrape config
  ```yaml
  scrape_configs:
    - job_name: 'vault'
      metrics_path: '/v1/sys/metrics'
      params:
        format: ['prometheus']
      static_configs:
        - targets: ['vault-1:8200', 'vault-2:8200', 'vault-3:8200']
  ```
- [ ] Add to existing Prometheus setup (if exists)
- [ ] Verify metrics scraped (check Prometheus targets page)
- **Assignee**: DevOps Engineer
- **Duration**: 1 hour
- **Acceptance**: Vault metrics visible in Prometheus

**Task 4.2: Create Grafana dashboard** 📊
- [ ] Import Vault community dashboard (Grafana ID: 12904)
- [ ] Customize panels:
  - Seal status (gauge)
  - Leader election history (graph)
  - Secret read/write rate (graph)
  - P99 latency (graph)
  - Raft applied index (graph)
- [ ] Export dashboard JSON to `config/grafana/vault-dashboard.json`
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Dashboard displays Vault metrics, auto-refreshes every 30s

**Task 4.3: Configure alerting rules** 🚨
- [ ] Create `config/prometheus/vault-alerts.yml`
  - `VaultSealed` - Critical alert when vault_core_unsealed == 0
  - `VaultNoLeader` - Critical alert when no Raft leader
  - `VaultHighLatency` - Warning when P99 latency >50ms
  - `VaultHighErrorRate` - Warning when error rate >1%
- [ ] Add to Prometheus `alerting_rules` configuration
- [ ] Test alerts (seal Vault, verify alert fires)
- [ ] Configure alert routing (Slack/PagerDuty)
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Alerts fire correctly on threshold violations

**Task 4.4: Create runbook** 📖
- [ ] Create `docs/runbooks/vault-operations.md`
  - **Scenario 1**: Vault sealed (manual unseal procedure)
  - **Scenario 2**: Node failure (replace node, rejoin cluster)
  - **Scenario 3**: Leader election failure (force leader election)
  - **Scenario 4**: Total cluster loss (restore from backup)
  - **Scenario 5**: KMS unavailable (manual unseal with recovery keys)
- [ ] Include troubleshooting commands
- [ ] Include escalation contacts
- **Assignee**: DevOps Engineer
- **Duration**: 2 hours
- **Acceptance**: Runbook covers all failure scenarios from HLD

### Phase 5: Testing & Validation (Days 10-11)

**Task 5.1: Failover test** ✅
- [ ] Start 3-node cluster, verify healthy
- [ ] Kill leader node (`docker stop alphapulse-vault-1`)
- [ ] Verify new leader elected (<5 seconds)
- [ ] Verify writes continue to succeed
- [ ] Restart failed node, verify rejoins cluster
- [ ] Document results in test report
- **Assignee**: Backend Developer
- **Duration**: 1 hour
- **Acceptance**: Cluster survives single node failure with <5s failover

**Task 5.2: Split-brain test** ✅
- [ ] Start 3-node cluster, verify healthy
- [ ] Partition network (isolate 1 node from other 2)
  ```bash
  # Using iptables to simulate network partition
  docker exec alphapulse-vault-1 iptables -A INPUT -s vault-2 -j DROP
  docker exec alphapulse-vault-1 iptables -A INPUT -s vault-3 -j DROP
  ```
- [ ] Verify isolated node cannot become leader (no quorum)
- [ ] Verify 2-node majority continues operating
- [ ] Heal partition, verify isolated node syncs
- [ ] Document results
- **Assignee**: Backend Developer
- **Duration**: 1 hour
- **Acceptance**: Raft quorum prevents split-brain scenario

**Task 5.3: Load test** 📊
- [ ] Create load test script (`tests/load/vault_load_test.py`)
- [ ] Configure Locust with 100 concurrent users
  - 90% reads (secret retrieval)
  - 10% writes (secret creation)
- [ ] Run 5-minute load test
- [ ] Collect metrics:
  - Throughput (requests/second)
  - P50/P99/P999 latency
  - Error rate
- [ ] Verify P99 latency <50ms
- [ ] Verify error rate <0.1%
- [ ] Document results
- **Assignee**: Backend Developer
- **Duration**: 3 hours
- **Acceptance**: P99 latency <50ms, error rate <0.1%, 1000+ req/s

**Task 5.4: Backup and restore test** ✅
- [ ] Create test secrets in Vault
  ```bash
  vault kv put secret/test/backup key1=value1 key2=value2
  ```
- [ ] Take Raft snapshot
  ```bash
  vault operator raft snapshot save backup.snap
  ```
- [ ] Delete test secrets
  ```bash
  vault kv delete secret/test/backup
  ```
- [ ] Restore from snapshot
  ```bash
  vault operator raft snapshot restore backup.snap
  ```
- [ ] Verify secrets restored correctly
- [ ] Document procedure
- **Assignee**: DevOps Engineer
- **Duration**: 1 hour
- **Acceptance**: Snapshot restoration successfully recovers all secrets

**Task 5.5: Security review** 🔒
- [ ] Review Vault configuration (security engineer)
  - TLS configuration (prod)
  - Audit logging enabled
  - Auto-unseal with KMS (prod)
  - File permissions on config files
- [ ] Review access control
  - Root token stored securely
  - Unseal keys distributed to key holders
  - Tenant isolation policies correct
- [ ] Review audit logs
  - Verify secrets are HMACed (not plaintext)
  - Verify all operations logged
  - Verify log retention configured
- [ ] Sign off on production readiness
- **Assignee**: Security Engineer
- **Duration**: 2 hours
- **Acceptance**: Security engineer signs off, no critical findings

### Phase 6: Documentation & Handoff (Day 12)

**Task 6.1: Update README** 📝
- [ ] Add Vault section to README.md
  ```markdown
  ## Vault Setup

  AlphaPulse uses HashiCorp Vault for secure credential storage.

  ### Development Setup
  1. Start Vault cluster: `docker-compose --profile vault up -d`
  2. Initialize cluster: `./scripts/init-vault-cluster.sh`
  3. Export token: `export VAULT_TOKEN=$(cat .vault-token)`
  4. Verify health: `./scripts/vault-health-check.sh`

  ### Production Setup
  See [Vault Operations Runbook](docs/runbooks/vault-operations.md)
  ```
- [ ] Add environment variables documentation
- [ ] Add troubleshooting section
- **Assignee**: Backend Developer
- **Duration**: 1 hour
- **Acceptance**: README includes Vault setup instructions

**Task 6.2: Create developer guide** 📚
- [ ] Create `docs/guides/vault-developer-guide.md`
  - How to read/write secrets with VaultClient
  - How to test locally
  - How to debug connection issues
  - Code examples
- [ ] Add to CLAUDE.md for AI pair programming context
- **Assignee**: Backend Developer
- **Duration**: 2 hours
- **Acceptance**: Developers can onboard to Vault in <30 minutes

**Task 6.3: Team training session** 🎓
- [ ] Schedule 1-hour training session
- [ ] Prepare slides:
  - Vault architecture overview
  - How to use VaultClient
  - Common operations (read/write/delete secrets)
  - Troubleshooting tips
  - Incident response (runbook review)
- [ ] Demo: Live demonstration of Vault operations
- [ ] Q&A session
- **Assignee**: DevOps Engineer + Backend Developer
- **Duration**: 1 hour presentation + prep
- **Acceptance**: All team members trained, questions answered

## Definition of Done

### Code Quality
- [x] All code reviewed and approved
- [x] Unit tests: 100% coverage on VaultClient
- [x] Integration tests: All passing
- [x] Load tests: P99 <50ms, error rate <0.1%
- [x] Security review: Approved by security engineer

### Documentation
- [x] README updated with Vault setup instructions
- [x] Developer guide created
- [x] Operations runbook created
- [x] HLD and Discovery documents complete

### Deployment
- [x] Development: Docker Compose cluster running
- [ ] Production: Kubernetes Helm deployment tested (optional for Sprint 9)
- [x] Auto-unseal configured (KMS for prod, file for dev)
- [x] Audit logging enabled
- [x] Monitoring: Prometheus metrics + Grafana dashboard
- [x] Alerting: Critical alerts configured

### Testing
- [x] Failover test passed (single node failure)
- [x] Split-brain test passed (network partition)
- [x] Load test passed (P99 latency target met)
- [x] Backup/restore test passed

### Acceptance Criteria (from Story)
- [x] 3 replicas running (verified via `vault operator raft list-peers`)
- [x] Raft consensus working (quorum 2/3)
- [x] Auto-unseal configured (KMS for prod)
- [x] Health checks green (all nodes unsealed)
- [x] Audit logging enabled (file + syslog)

## Risk Mitigation

| Risk | Mitigation Strategy | Owner |
|------|---------------------|-------|
| KMS setup delays (AWS account approval) | Use file-based unseal for development, defer KMS to production | DevOps |
| Kubernetes cluster not ready | Deploy to Docker Compose first, K8s deployment is optional for Sprint 9 | DevOps |
| Vault learning curve (team unfamiliar) | Training session + developer guide + pair programming | Backend Dev |
| Performance targets not met | Add caching layer (Story 3.3), optimize Raft config | Backend Dev |
| Security review finds issues | Address findings before Story 3.2 (credentials integration) | Security |

## Dependencies

**Upstream Dependencies** (must complete first):
- None (Story 3.1 is foundation of EPIC-003)

**Downstream Dependencies** (depend on this story):
- **Story 3.2**: Configure Vault policies (needs Vault cluster running)
- **Story 3.3**: Implement CredentialService (needs VaultClient)
- **Story 3.4**: Credential validation (needs CredentialService)
- **Story 3.5**: Health check job (needs CredentialService)
- **Story 3.6**: Credential rotation flow (needs CredentialService)

**External Dependencies**:
- Docker / Docker Compose (available)
- AWS account with KMS access (for production auto-unseal)
- Kubernetes cluster (optional, for production deployment)
- Prometheus/Grafana (for monitoring)

## Sprint Timeline

| Day | Tasks | Assignee | Deliverable |
|-----|-------|----------|-------------|
| 1-3 | Phase 1: Infrastructure Setup | DevOps | Vault cluster running locally |
| 4-5 | Phase 2: Client Integration | Backend Dev | VaultClient library + tests |
| 6-7 | Phase 3: Production Config | DevOps | KMS, TLS, K8s config |
| 8-9 | Phase 4: Monitoring | DevOps | Prometheus + Grafana + alerts |
| 10-11 | Phase 5: Testing | Backend Dev | All tests passing |
| 12 | Phase 6: Documentation | Backend Dev | README, guides, training |

**Total Duration**: 12 days (~2.5 weeks)
**Buffer**: 0.5 weeks for blockers (total Sprint 9 = 3 weeks)

## Success Metrics

**Technical Metrics**:
- ✅ Cluster uptime: 99.99% (4 nines)
- ✅ P99 latency: <5ms (read), <50ms (write)
- ✅ Throughput: 1000+ requests/second
- ✅ Failover time: <5 seconds (leader election)
- ✅ Test coverage: 100% (VaultClient)

**Team Metrics**:
- ✅ All team members trained on Vault operations
- ✅ Security review passed (no critical findings)
- ✅ Documentation complete (README + runbook + dev guide)

**Business Metrics**:
- ✅ Foundation for secure credential storage (SOC2 requirement)
- ✅ Enables Stories 3.2-3.6 (unblocks EPIC-003)
- ✅ Production-ready infrastructure (can support 1000+ tenants)

## Rollout Strategy

### Development Environment
1. **Local Docker Compose**: All developers run Vault locally for testing
2. **Shared Dev Vault**: Optional shared Vault instance for integration testing
3. **CI/CD Integration**: GitHub Actions runs integration tests against ephemeral Vault

### Staging Environment
1. **Kubernetes Deployment**: Deploy to staging K8s cluster
2. **Load Testing**: Run load tests to validate performance
3. **Chaos Engineering**: Test failure scenarios (node failures, network partitions)

### Production Environment
1. **Kubernetes Deployment**: Deploy to production K8s cluster with KMS auto-unseal
2. **Smoke Tests**: Verify basic operations (read/write/delete)
3. **Gradual Rollout**: Enable for 1% of tenants → 10% → 50% → 100%
4. **Monitoring**: Watch metrics for anomalies (latency spikes, errors)

## Rollback Plan

**Scenario**: Vault cluster unstable or critical bug found

**Rollback Steps**:
1. Stop Vault containers (`docker-compose down` or `kubectl scale --replicas=0`)
2. Restore previous credential management (file-based encryption)
3. Re-deploy previous version of application
4. Communicate to team via Slack

**Rollback Time**: <15 minutes (automated script)

**Data Loss**: None (Vault only stores credentials, not business data)

## Post-Deployment Validation

### Day 1 (Deployment Day)
- [ ] All 3 Vault nodes healthy
- [ ] Raft cluster quorum established
- [ ] Audit logs writing to persistent storage
- [ ] Prometheus scraping metrics successfully
- [ ] No critical alerts firing

### Week 1 (Post-Deployment)
- [ ] No unexpected seal events
- [ ] P99 latency within SLA (<50ms)
- [ ] No 500 errors in application logs
- [ ] Backup snapshots completing daily

### Sprint Retrospective (End of Sprint 9)
- [ ] Review what went well (celebrate wins)
- [ ] Review what didn't go well (identify blockers)
- [ ] Action items for next sprint

## Next Sprint Planning

**Story 3.2** (Sprint 9, parallel with 3.1):
- Configure Vault policies for tenant isolation
- Create AppRole authentication for application
- Implement token renewal logic

**Story 3.3** (Sprint 10):
- Implement CredentialService with Vault client
- Add caching layer (Redis)
- Create REST API endpoints

**Story 3.4** (Sprint 10):
- Implement credential validation (CCXT test calls)
- Add webhook notifications on validation failures

**Story 3.5** (Sprint 10):
- Create health check background job (every 6 hours)
- Implement automatic credential expiration detection
