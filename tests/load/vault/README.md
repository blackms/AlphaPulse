# Vault Load Testing

Load testing infrastructure for HashiCorp Vault credential operations in AlphaPulse multi-tenant environment.

## Overview

This directory contains load testing scenarios to measure Vault performance under various conditions:
- Credential read operations (retrieval)
- Credential write operations (rotation)
- Concurrent multi-tenant access
- Scalability limits

## Prerequisites

### Local Setup
```bash
# Install k6 (load testing tool)
brew install k6  # macOS
# OR
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6  # Linux

# Start Vault dev server (for local testing)
vault server -dev -dev-root-token-id="root"
```

### Environment Variables
```bash
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root"  # Dev token, use real token in production
export VAULT_NAMESPACE=""  # Optional for Vault Enterprise
```

## Test Scenarios

### 1. Credential Read Performance (`read_credentials.js`)
**Objective**: Measure throughput and latency for credential retrieval operations

**Metrics**:
- Requests per second (RPS)
- P50, P95, P99 latency
- Error rate
- Vault server resource usage

**Run**:
```bash
k6 run tests/load/vault/read_credentials.js
```

### 2. Credential Write Performance (`write_credentials.js`)
**Objective**: Measure performance for credential updates/rotation

**Metrics**:
- Write throughput
- Write latency
- Vault backend I/O
- Lock contention

**Run**:
```bash
k6 run tests/load/vault/write_credentials.js
```

### 3. Multi-Tenant Concurrent Access (`multi_tenant_concurrent.js`)
**Objective**: Simulate multiple tenants accessing credentials simultaneously

**Metrics**:
- Per-tenant isolation
- Cross-tenant performance impact
- Resource contention
- Scalability limits

**Run**:
```bash
k6 run tests/load/vault/multi_tenant_concurrent.js
```

### 4. Stress Test (`stress_test.js`)
**Objective**: Find breaking points and maximum capacity

**Metrics**:
- Maximum sustainable RPS
- Recovery time after spike
- Error rates at capacity
- System degradation patterns

**Run**:
```bash
k6 run tests/load/vault/stress_test.js
```

## Test Execution

### Quick Test (Smoke Test)
```bash
k6 run --vus 10 --duration 30s tests/load/vault/read_credentials.js
```

### Load Test (Baseline)
```bash
k6 run --vus 100 --duration 5m tests/load/vault/read_credentials.js
```

### Stress Test (Find Limits)
```bash
k6 run --vus 1000 --duration 10m tests/load/vault/stress_test.js
```

### Soak Test (Endurance)
```bash
k6 run --vus 50 --duration 1h tests/load/vault/read_credentials.js
```

## Performance Targets

Based on AlphaPulse requirements:

| Metric | Target | Acceptable | Red Line |
|--------|--------|------------|----------|
| Read Latency (P95) | <50ms | <100ms | >200ms |
| Write Latency (P95) | <100ms | <200ms | >500ms |
| Read Throughput | >1000 RPS | >500 RPS | <100 RPS |
| Write Throughput | >200 RPS | >100 RPS | <50 RPS |
| Error Rate | <0.1% | <1% | >5% |
| Concurrent Tenants | >100 | >50 | <20 |

## Results Analysis

### Running Tests with Output
```bash
# JSON output for analysis
k6 run --out json=results/vault_read_$(date +%Y%m%d_%H%M%S).json tests/load/vault/read_credentials.js

# InfluxDB output (if available)
k6 run --out influxdb=http://localhost:8086/k6 tests/load/vault/read_credentials.js

# CSV summary
k6 run --summary-export=results/summary.json tests/load/vault/read_credentials.js
```

### Analyzing Results
```bash
# Generate HTML report (using k6-reporter)
npm install -g k6-to-junit
k6 run --out json=results/test.json tests/load/vault/read_credentials.js
k6-to-junit results/test.json > results/junit.xml
```

## CI/CD Integration

Tests are integrated into CI/CD pipeline:

```yaml
# .github/workflows/vault-load-tests.yml (example)
- name: Run Vault Load Tests
  run: |
    vault server -dev -dev-root-token-id="root" &
    sleep 5
    k6 run --vus 50 --duration 2m tests/load/vault/read_credentials.js
```

## Monitoring

During load tests, monitor Vault metrics:

```bash
# Vault metrics endpoint
curl http://127.0.0.1:8200/v1/sys/metrics?format=prometheus

# Key metrics to watch:
# - vault_core_check_token_duration (authentication time)
# - vault_core_handle_request_duration (request processing)
# - vault_token_count (token count)
# - vault_secret_kv_count (secret count)
```

## Troubleshooting

### High Latency
1. Check Vault backend (Consul, etcd) performance
2. Verify network latency between services
3. Review Vault server resource usage (CPU, memory)
4. Check for lock contention in Vault logs

### High Error Rate
1. Review Vault audit logs
2. Check authentication token validity
3. Verify rate limiting settings
4. Review Vault server logs for errors

### Connection Errors
1. Verify Vault server is running: `vault status`
2. Check network connectivity: `curl $VAULT_ADDR/v1/sys/health`
3. Verify TLS certificates (if using HTTPS)
4. Check firewall rules

## References

- [k6 Documentation](https://k6.io/docs/)
- [Vault Performance Tuning](https://www.vaultproject.io/docs/internals/performance)
- [Vault Metrics](https://www.vaultproject.io/docs/internals/telemetry)
- [Story 3.4 Issue](https://github.com/AIgen-Solutions-s-r-l/AlphaPulse/issues/170)
