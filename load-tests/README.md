# AlphaPulse Load Testing

This directory contains load testing scripts for validating the performance of the AlphaPulse multi-tenant SaaS platform.

## Overview

Load testing is a critical approval condition for Phase 3 implementation. We validate that the system can handle target capacity with acceptable latency and error rates.

**Success Criteria**:
- p99 latency <500ms (acceptance)
- p99 latency <200ms (ideal)
- Error rate <1%
- CPU usage <70%
- Memory usage <80%

## Test Scenarios

### 1. Baseline Test (100 concurrent users)

**File**: `baseline-test.js`

**Purpose**: Establish baseline performance metrics with 100 concurrent users.

**Test Mix**:
- 70% reads (GET /portfolio, /trades, /positions)
- 30% writes (POST /trades)

**Duration**: 10 minutes
- 2 minutes: ramp-up (0 → 100 users)
- 6 minutes: sustained load (100 users)
- 2 minutes: ramp-down (100 → 0 users)

**Run**:
```bash
k6 run --env BASE_URL=https://staging.alphapulse.ai baseline-test.js
```

---

### 2. Target Capacity Test (500 concurrent users)

**File**: `target-capacity-test.js`

**Purpose**: Validate target capacity with 500 concurrent users (100 tenants × 5 users average).

**Test Mix**:
- 25% GET /portfolio
- 25% GET /trades
- 20% GET /positions
- 20% POST /trades
- 10% GET /signals

**Duration**: 10 minutes
- 2 minutes: ramp-up (0 → 100 users)
- 2 minutes: ramp-up (100 → 300 users)
- 2 minutes: ramp-up (300 → 500 users)
- 2 minutes: sustained load (500 users)
- 2 minutes: ramp-down (500 → 0 users)

**Run**:
```bash
k6 run --env BASE_URL=https://staging.alphapulse.ai target-capacity-test.js
```

---

## Prerequisites

### 1. Install k6

**macOS**:
```bash
brew install k6
```

**Linux**:
```bash
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**Windows**:
```powershell
choco install k6
```

**Verify installation**:
```bash
k6 version
```

---

### 2. Set Up Staging Environment

**Staging environment must have**:
- Kubernetes cluster (2 nodes: t3.xlarge or equivalent)
- PostgreSQL RDS (db.t3.large or equivalent)
- Redis Cluster (3 pods: 1 master + 2 replicas)
- API deployment (10 replicas with HPA enabled)
- Monitoring stack (Prometheus, Grafana)

**Deploy to staging**:
```bash
# Deploy with Helm
helm upgrade --install alphapulse ./helm/alphapulse \
  --namespace alphapulse-staging \
  --create-namespace \
  --values ./helm/alphapulse/values-staging.yaml

# Verify deployment
kubectl get pods -n alphapulse-staging
kubectl get svc -n alphapulse-staging
```

---

### 3. Create Test Users

**Run seed script** to create test users for load testing:
```bash
poetry run python scripts/seed_load_test_users.py --staging
```

This creates:
- 5 tenants (tenant1-tenant5)
- 2 users per tenant (10 users total)
- Sample portfolio data

---

## Running Tests

### Local Test (Development)

Test against local API (http://localhost:8000):
```bash
k6 run baseline-test.js
```

---

### Staging Test (Recommended)

Test against staging environment:
```bash
# Baseline test (100 users)
k6 run --env BASE_URL=https://staging.alphapulse.ai baseline-test.js

# Target capacity test (500 users)
k6 run --env BASE_URL=https://staging.alphapulse.ai target-capacity-test.js
```

---

### Advanced Options

**Output results to file**:
```bash
k6 run --out json=results.json target-capacity-test.js
```

**Output results to InfluxDB** (for Grafana dashboards):
```bash
k6 run --out influxdb=http://localhost:8086/k6 target-capacity-test.js
```

**Custom test duration**:
```bash
k6 run --duration 30m --vus 500 target-capacity-test.js
```

**Smoke test** (quick validation):
```bash
k6 run --vus 10 --duration 1m baseline-test.js
```

---

## Analyzing Results

### 1. k6 CLI Output

k6 automatically displays:
- Request metrics (http_req_duration: min, avg, max, p95, p99)
- Error rates (http_req_failed)
- Custom metrics (portfolio_latency, trades_latency, etc.)

**Example output**:
```
✓ portfolio status 200
✓ portfolio latency <500ms

checks.........................: 98.50%  ✓ 9850       ✗ 150
data_received..................: 12 MB   120 kB/s
data_sent......................: 5.2 MB  52 kB/s
errors.........................: 0.85%   ✓ 85         ✗ 9915
http_req_duration..............: avg=145ms min=23ms med=120ms max=890ms p(95)=320ms p(99)=450ms
http_reqs......................: 10000   100/s
portfolio_latency..............: avg=125ms min=20ms med=110ms max=450ms p(95)=280ms p(99)=380ms
trades_latency.................: avg=135ms min=25ms med=115ms max=520ms p(95)=310ms p(99)=420ms
vus............................: 100     min=0        max=100
```

**Pass/Fail Evaluation**:
- ✅ **PASS**: p99 <500ms, error rate <1%
- ⚠️ **MARGINAL**: p99 500-700ms, error rate 1-2%
- ❌ **FAIL**: p99 >700ms, error rate >2%

---

### 2. Grafana Dashboards

Monitor real-time metrics during load test:

**Access Grafana**:
```bash
# Port-forward to Grafana
kubectl port-forward -n alphapulse-monitoring svc/grafana 3000:80

# Open browser: http://localhost:3000
# Login: admin / <password from secret>
```

**Key Dashboards**:
- **AlphaPulse API Dashboard**: Request rate, latency (p50, p95, p99), error rate
- **Kubernetes Dashboard**: CPU usage, memory usage, pod count
- **PostgreSQL Dashboard**: Query latency, connection count, cache hit rate
- **Redis Dashboard**: Commands/sec, memory usage, hit rate

**Collect screenshots**:
- API latency (p50, p95, p99) over 10 minutes
- CPU/Memory usage over 10 minutes
- Database query latency over 10 minutes
- Redis hit rate over 10 minutes

---

### 3. Resource Utilization

**Check Kubernetes pod metrics**:
```bash
# CPU usage
kubectl top pods -n alphapulse-staging

# Memory usage
kubectl describe hpa -n alphapulse-staging

# Pod scaling events
kubectl get events -n alphapulse-staging --sort-by='.lastTimestamp' | grep HorizontalPodAutoscaler
```

**Check database metrics**:
```bash
# PostgreSQL connections
psql -U alphapulse -h staging.rds.amazonaws.com -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# PostgreSQL slow queries
psql -U alphapulse -h staging.rds.amazonaws.com -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

**Check Redis metrics**:
```bash
# Redis memory usage
kubectl exec -n alphapulse-staging redis-master-0 -- redis-cli info memory

# Redis hit rate
kubectl exec -n alphapulse-staging redis-master-0 -- redis-cli info stats | grep keyspace
```

---

## Bottleneck Identification

### Common Bottlenecks

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| p99 >500ms, CPU <50% | Database slow queries | Add indexes, optimize queries |
| p99 >500ms, CPU >80% | API pods under-provisioned | Increase HPA maxReplicas, add resources |
| High error rate (>1%) | Database connection exhaustion | Increase connection pool size |
| p99 200-400ms (good), but p99.9 >1s | Outlier queries (N+1, large result sets) | Add caching, pagination |
| High memory usage (>80%) | Memory leak or large objects | Profile memory, optimize serialization |

---

### Optimization Workflow

1. **Identify bottleneck** from k6 output and Grafana dashboards
2. **Reproduce locally** with single-user test
3. **Profile** with `py-spy` or `cProfile`
4. **Optimize** code (add indexes, caching, pagination)
5. **Retest** with baseline-test.js
6. **Validate** with target-capacity-test.js

---

## Deliverable: Load Testing Report

After completing tests, create `docs/load-testing-report.md` with:

**Required Sections**:
1. **Executive Summary**: Pass/fail verdict, key metrics
2. **Test Scenarios**: Baseline (100 users) and target capacity (500 users)
3. **Results**: p50, p95, p99 latency, error rate, resource utilization
4. **Bottleneck Analysis**: Identified bottlenecks and root causes
5. **Optimization Recommendations**: Actionable improvements
6. **Approval Decision**: ✅ APPROVED or ⏳ REQUIRES OPTIMIZATION

**Template**: See `docs/load-testing-report-template.md`

---

## Troubleshooting

### Issue: Connection Refused

**Symptom**: k6 returns "dial tcp: connection refused"

**Solution**:
```bash
# Check API is running
kubectl get pods -n alphapulse-staging

# Check service endpoint
kubectl get svc -n alphapulse-staging

# Port-forward for local testing
kubectl port-forward -n alphapulse-staging svc/api 8000:80
```

---

### Issue: Authentication Failures

**Symptom**: k6 shows "login unsuccessful" or "token received" check fails

**Solution**:
- Verify test users exist: `kubectl exec -n alphapulse-staging <api-pod> -- poetry run python scripts/verify_test_users.py`
- Check JWT secret is set: `kubectl get secret -n alphapulse-staging alphapulse-secrets -o yaml`
- Test login manually: `curl -X POST https://staging.alphapulse.ai/api/auth/login -d '{"email":"user1@tenant1.com","password":"test123"}'`

---

### Issue: High Latency (p99 >1s)

**Symptom**: k6 shows p99 >1000ms

**Investigation**:
1. Check database query latency: `SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;`
2. Check Redis hit rate: `redis-cli info stats | grep keyspace`
3. Check API logs: `kubectl logs -n alphapulse-staging <api-pod> | grep ERROR`
4. Profile API: `py-spy record -o profile.svg -- poetry run python -m alpha_pulse.main`

---

## References

- [k6 Documentation](https://k6.io/docs/)
- [k6 Test Types](https://k6.io/docs/test-types/introduction/)
- [k6 Metrics](https://k6.io/docs/using-k6/metrics/)
- [AlphaPulse Architecture Review](../docs/architecture-review.md)
- [AlphaPulse Database Migration Plan](../docs/database-migration-plan.md)

---

## Next Steps

After completing load testing:

1. ✅ Create `docs/load-testing-report.md` with results
2. ✅ Update Architecture Review approval status (Approval Condition 2)
3. ✅ Present results to stakeholders for sign-off
4. ✅ Proceed to Sprint 5 (EPIC-001: Database Multi-Tenancy implementation)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
