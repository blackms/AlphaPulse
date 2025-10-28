# HashiCorp Vault Load Testing Report

**Date**: 2025-10-20
**Sprint**: 1 (Inception)
**Related**: [SPIKE: Vault Load Testing](#151), [EPIC-003](#142), [ADR-003](../adr/003-credential-management-multi-tenant.md)

---

## Executive Summary

**Objective**: Validate that HashiCorp Vault can handle 10k+ req/sec with P99 latency <10ms for credential retrieval.

**Hypothesis**: Vault HA deployment (3 replicas, Raft consensus) can handle multi-tenant credential workload at scale.

**Decision**: [To be determined after load test execution]

---

## Test Environment

### Vault Configuration

- **Version**: Vault 1.15.x (latest stable)
- **Deployment**: High Availability (HA)
  - 3 Vault replicas
  - Raft consensus (integrated storage)
  - Auto-unseal with AWS KMS
- **Instance Type**: t3.medium (2 vCPU, 4GB RAM) per replica
- **Storage**: GP3 SSD (3000 IOPS baseline)

### Test Data

- **Tenants**: 1,000 simulated tenants
- **Exchanges per Tenant**: 3 (Binance, Coinbase, Kraken)
- **Total Secrets**: 3,000 secrets in Vault
- **Secret Size**: ~200 bytes per secret

### Load Test Configuration

**Tool**: k6 (https://k6.io)

**Test Stages**:
1. Ramp up: 0 → 50 VUs (30s)
2. Ramp up: 50 → 100 VUs (1min)
3. Steady state: 100 VUs (2min)
4. Spike: 100 → 200 VUs (1min)
5. Spike sustain: 200 VUs (1min)
6. Ramp down: 200 → 0 VUs (30s)

**Workload**: 90% read, 10% write (simulate credential rotation)

---

## Test Scenarios

### Scenario 1: Credential Read (90% of traffic)

**Operation**: Get secret from Vault

```bash
GET /v1/secret/data/tenants/{tenant_id}/exchanges/{exchange}
X-Vault-Token: {token}
```

**Expected Performance**:
- P50: <5ms
- P95: <10ms
- P99: <20ms

### Scenario 2: Credential Write (10% of traffic)

**Operation**: Update secret in Vault (credential rotation)

```bash
POST /v1/secret/data/tenants/{tenant_id}/exchanges/{exchange}
X-Vault-Token: {token}
Content-Type: application/json

{
  "data": {
    "api_key": "new_key",
    "secret": "new_secret",
    "permissions": "trading"
  }
}
```

**Expected Performance**:
- P50: <10ms
- P95: <20ms
- P99: <50ms

---

## Results

### [To be filled after load test execution]

#### Overall Performance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Throughput (req/sec) | >10,000 | TBD | TBD |
| Total Requests | N/A | TBD | N/A |
| Duration | 6.5 min | TBD | N/A |
| Error Rate | <1% | TBD | TBD |

#### Latency Statistics

| Metric | Read Operations (ms) | Write Operations (ms) | Combined (ms) |
|--------|----------------------|-----------------------|---------------|
| P50    | TBD | TBD | TBD |
| P95    | TBD | TBD | TBD |
| P99    | TBD | TBD | TBD |
| Max    | TBD | TBD | TBD |

#### Resource Utilization

| Resource | Replica 1 | Replica 2 | Replica 3 |
|----------|-----------|-----------|-----------|
| CPU (avg) | TBD% | TBD% | TBD% |
| Memory (avg) | TBD MB | TBD MB | TBD MB |
| Disk I/O (IOPS) | TBD | TBD | TBD |

---

## Analysis

### [To be filled after load test execution]

**Throughput Analysis**:
- Observed throughput: TBD req/sec
- Peak throughput: TBD req/sec
- Requests/VU: TBD req/sec

**Latency Analysis**:
- P99 read latency: TBD ms
- P99 write latency: TBD ms
- Latency under spike (200 VUs): TBD ms

**Error Analysis**:
- Total errors: TBD
- Error types: [To be added]
- Error rate during spike: TBD%

**Bottlenecks Identified**:
- [To be added after execution]

---

## Decision

### If Throughput >10k req/sec AND P99 <10ms (PASS)

✅ **PROCEED with Vault OSS**

- Vault OSS provides sufficient performance for multi-tenant workload
- HA deployment ensures 99.9% availability
- No upgrade to Enterprise required
- Move forward with EPIC-003 (Credential Management) as planned

**Caching Strategy**:
- 5-minute TTL for credential caching (as designed)
- Expected cache hit rate: >95% (credentials rarely rotate)
- Actual Vault load in production: ~500 req/sec (after caching)

### If Throughput 5k-10k req/sec OR P99 10-20ms (WARNING)

⚠️ **PROCEED with caching mitigation**

- Performance acceptable but below target
- Extend cache TTL to reduce Vault load
- Monitor Vault performance in production

**Mitigation**:
- Extend credential cache TTL: 5 min → 1 hour
- Expected cache hit rate: >99%
- Actual Vault load: ~50 req/sec (after extended caching)
- Re-evaluate in Sprint 8 with production traffic patterns

**Impact**: No timeline change, adjust caching configuration only

### If Throughput <5k req/sec OR P99 >20ms (FAIL)

✗ **ADJUST STRATEGY**

**Option A: Upgrade to Vault Enterprise**
- Vault Enterprise: Better performance, read replicas, HSM integration
- Cost: ~$500/month for 3 replicas
- **Impact**: +$6k/year operational cost

**Option B: Alternative Secret Management**
- AWS Secrets Manager (managed service, $0.40/secret/month = $1,200/month)
- Database-encrypted credentials (less secure, no audit trail)
- **Impact**: Re-design credential management architecture (+2 sprints)

**Option C: Optimize Vault Deployment**
- Increase instance size: t3.medium → t3.large (4 vCPU, 8GB RAM)
- Add read replicas (Vault Enterprise required)
- Tune Raft storage backend
- **Impact**: +1 sprint for optimization, +$200/month infrastructure

**Recommended**: Option C (Optimize) first, then Option A (Enterprise) if needed

---

## Recommendations

### [To be filled after load test execution]

1. [Recommendation based on results]
2. [Caching strategy adjustments]
3. [Infrastructure tuning suggestions]
4. [Production monitoring requirements]

---

## Next Steps

### [To be filled after load test execution]

- [ ] Update EPIC-003 with decision
- [ ] Update risk register (Vault performance risk)
- [ ] Adjust Sprint 9 scope if needed (Vault optimization)
- [ ] Present findings in Sprint 1 Review

---

## Appendix: Running the Load Test

### Prerequisites

```bash
# Install k6
# macOS
brew install k6

# Linux
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Install Vault
brew install vault  # macOS
# or follow: https://www.vaultproject.io/downloads
```

### Setup

```bash
# 1. Start Vault in HA mode (3 replicas)
# See: https://learn.hashicorp.com/tutorials/vault/ha-with-consul

# 2. Set environment variables
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=root

# 3. Populate test data (3,000 secrets)
./scripts/setup_vault_test_data.sh

# Expected output:
#   ✓ Created 3,000 secrets in 120s
```

### Execution

```bash
# Run load test (6.5 minutes)
k6 run --vus 100 --duration 60s scripts/load_test_vault.js

# Higher load (stress test)
k6 run --vus 200 --duration 120s scripts/load_test_vault.js

# Custom configuration
export NUM_TENANTS=2000
k6 run scripts/load_test_vault.js
```

### Expected Output

```
          /\      |‾‾| /‾‾/   /‾‾/
     /\  /  \     |  |/  /   /  /
    /  \/    \    |     (   /   ‾‾\
   /          \   |  |\  \ |  (‾)  |
  / __________ \  |__| \__\ \_____/ .io

  execution: local
     script: scripts/load_test_vault.js
     output: -

  scenarios: (100.00%) 1 scenario, 200 max VUs, 6m30s max duration

     ✓ read status is 200
     ✓ read has data
     ✓ read latency <50ms
     ✓ write status is 200 or 204

     checks.........................: 100.00% ✓ 45230      ✗ 0
     data_received..................: 12 MB   31 kB/s
     data_sent......................: 8.2 MB  21 kB/s
     errors.........................: 0.00%   ✓ 0          ✗ 45230
     http_req_duration..............: avg=8.23ms   p(95)=15.32ms p(99)=18.45ms
     http_reqs......................: 45230   116/s
     iteration_duration.............: avg=18.5ms   p(95)=32.1ms  p(99)=45.2ms
     iterations.....................: 45230   116/s
     read_operations................: 40707   104/s
     vault_latency..................: avg=7.12ms   p(95)=14.2ms  p(99)=17.8ms
     vus............................: 200     min=0        max=200
     vus_max........................: 200     min=200      max=200
     write_operations...............: 4523    12/s

=== Summary Statistics ===
Test Duration: 390s
Total Requests: 45230
Requests/sec: 116.00
Read Operations: 40707
Write Operations: 4523
Error Rate: 0.00%
P50 Latency: 6.45ms
P95 Latency: 14.20ms
P99 Latency: 17.80ms
Max Latency: 45.30ms

=== Test Results ===
✗ FAIL: Throughput <5k req/sec (unacceptable)
✓ PASS: P99 latency <10ms (target met)
✓ PASS: Error rate <1% (target met)

=== Decision ===
Decision: PROCEED with caching mitigation (extend TTL to 1 hour)
```

**Note**: The example above shows a scenario where throughput is low (116 req/sec with only 200 VUs). In a real test with proper tuning, we expect 10k+ req/sec.

---

## References

- [ADR-003: Credential Management for Multi-Tenant](../adr/003-credential-management-multi-tenant.md)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [Vault Performance Tuning](https://learn.hashicorp.com/tutorials/vault/performance-tuning)
- [k6 Load Testing Guide](https://k6.io/docs/)
- [HLD Section 2.1: Component View](../HLD-MULTI-TENANT-SAAS.md#21-architecture-views)
