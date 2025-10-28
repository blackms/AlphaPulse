# Operational Runbook: AlphaPulse Multi-Tenant SaaS

**Date**: 2025-10-22
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead + DevOps Engineer
**Related**: [Architecture Review](architecture-review.md), [Security Design Review](security-design-review.md)
**Status**: Draft

---

## Purpose

This operational runbook provides step-by-step procedures for operating and troubleshooting the AlphaPulse multi-tenant SaaS platform in production. It covers:

- Incident response procedures (P0-P3)
- Service-specific troubleshooting (PostgreSQL, Redis, Vault, API)
- Common operational tasks (deployments, rollbacks, scaling)
- Monitoring and alerting
- Disaster recovery

**Target Audience**: On-call engineers, DevOps engineers, SREs

---

## Table of Contents

1. [Incident Response Procedures](#1-incident-response-procedures)
2. [Service Troubleshooting](#2-service-troubleshooting)
3. [Common Operational Tasks](#3-common-operational-tasks)
4. [Monitoring and Alerting](#4-monitoring-and-alerting)
5. [Disaster Recovery](#5-disaster-recovery)
6. [Escalation Procedures](#6-escalation-procedures)
7. [On-Call Rotation](#7-on-call-rotation)

---

## 1. Incident Response Procedures

### 1.1 Incident Severity Levels

Per [Security Design Review](security-design-review.md), we use 4 severity levels:

| Severity | Definition | Response Time | Escalation | Examples |
|----------|-----------|--------------|-----------|----------|
| **P0 (Critical)** | Complete service outage, data loss risk, security breach | <15 minutes | Immediate (CTO, Tech Lead, Security Lead) | API down, database unavailable, data breach |
| **P1 (High)** | Major feature unavailable, significant user impact | <1 hour | Tech Lead | Trading execution fails, authentication broken |
| **P2 (Medium)** | Minor feature degraded, limited user impact | <4 hours | Team | Cache miss rate high, slow queries |
| **P3 (Low)** | No user impact, cosmetic issues | <24 hours | Team | Dashboard chart formatting, log warnings |

### 1.2 P0: Critical Incident Response

**Immediate Actions** (0-15 minutes):

1. **Acknowledge Alert**
   ```bash
   # Acknowledge PagerDuty alert
   # Check #incidents Slack channel
   ```

2. **Assess Impact**
   - Check status page: https://status.alphapulse.ai
   - Check Grafana dashboards: https://grafana.alphapulse.ai
   - Check error tracking: https://sentry.io/alphapulse

   ```bash
   # Check service health
   kubectl get pods -n alphapulse
   kubectl get nodes

   # Check API health
   curl https://api.alphapulse.ai/health
   ```

3. **Create Incident Channel**
   ```
   /incident create title:"API Down" severity:P0
   ```

4. **Notify Stakeholders**
   - Update status page: "Investigating incident"
   - Post in #incidents Slack channel
   - Escalate to CTO and Tech Lead immediately

**Investigation** (15-30 minutes):

5. **Identify Root Cause**
   ```bash
   # Check recent deployments
   kubectl rollout history deployment/api -n alphapulse

   # Check pod logs
   kubectl logs -f deployment/api -n alphapulse --tail=100

   # Check database
   psql -U alphapulse -h prod.example.com -c "SELECT 1"

   # Check Redis
   redis-cli -h redis-cluster.example.com ping

   # Check Vault
   vault status

   # Check Prometheus alerts
   # Visit https://prometheus.alphapulse.ai/alerts
   ```

6. **Gather Evidence**
   - Screenshot Grafana dashboards
   - Export logs from Loki
   - Save error messages from Sentry

**Resolution** (30-60 minutes):

7. **Implement Fix**

   **Common P0 scenarios**:

   - **API pods crashing**: See [2.4 API Troubleshooting](#24-api-troubleshooting)
   - **Database unavailable**: See [2.1 PostgreSQL Troubleshooting](#21-postgresql-troubleshooting)
   - **Redis cluster down**: See [2.2 Redis Troubleshooting](#22-redis-troubleshooting)
   - **Vault sealed**: See [2.3 Vault Troubleshooting](#23-vault-troubleshooting)

8. **Verify Resolution**
   ```bash
   # Check service health
   kubectl get pods -n alphapulse

   # Run smoke tests
   ./scripts/smoke_tests.sh production

   # Monitor error rate
   # Check Grafana: Error rate should drop to <0.1%
   ```

9. **Communicate Resolution**
   - Update status page: "Issue resolved"
   - Post in #incidents channel: "Incident resolved, monitoring for stability"

**Post-Incident** (within 24 hours):

10. **Create Postmortem**
    - Use template: `.bug/postmortem-template.md`
    - Include: Timeline, root cause, impact, action items
    - Schedule blameless postmortem meeting (within 48 hours)

---

### 1.3 P1: High Severity Incident Response

**Response Time**: <1 hour

**Procedure**:

1. **Acknowledge and Assess** (0-15 min)
   - Same as P0, but no immediate CTO escalation
   - Create incident channel: `/incident create title:"..." severity:P1`

2. **Investigate** (15-45 min)
   - Check logs, metrics, traces
   - Identify affected features/tenants

3. **Fix or Workaround** (45-60 min)
   - Implement fix if root cause known
   - Apply workaround if fix requires more time (disable feature flag, scale up pods)

4. **Monitor** (ongoing)
   - Watch metrics for 1 hour after resolution
   - Update stakeholders every 30 minutes

5. **Postmortem** (within 48 hours)
   - Optional for P1 (required if recurring issue)

---

### 1.4 P2/P3: Medium/Low Severity Response

**Response Time**: P2 <4 hours, P3 <24 hours

**Procedure**:

1. **Acknowledge** (within response time)
2. **Create Ticket** (GitHub issue or Jira)
3. **Investigate** (during business hours)
4. **Fix** (in next sprint or hotfix if critical)
5. **No Postmortem** (unless systemic issue)

---

## 2. Service Troubleshooting

### 2.1 PostgreSQL Troubleshooting

#### High Connection Count

**Symptoms**: `FATAL: too many connections` errors

**Diagnosis**:
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Check connections by user
SELECT usename, count(*) FROM pg_stat_activity GROUP BY usename;

-- Check idle connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';
```

**Resolution**:
```sql
-- Kill idle connections (caution!)
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '10 minutes';

-- Increase max_connections (requires restart)
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```

**Prevention**:
- Configure connection pooling (PgBouncer)
- Set connection timeout in application: `POOL_TIMEOUT=30s`

---

#### Slow Queries

**Symptoms**: API latency >500ms, high database CPU

**Diagnosis**:
```sql
-- Find slow queries (running >1 second)
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - pg_stat_activity.query_start > interval '1 second';

-- Check query stats
SELECT query, mean_exec_time, calls FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;

-- Check missing indexes
SELECT schemaname, tablename, indexname FROM pg_indexes WHERE schemaname = 'public';
```

**Resolution**:
```sql
-- Kill long-running query (if needed)
SELECT pg_cancel_backend(pid);  -- Graceful
SELECT pg_terminate_backend(pid);  -- Force

-- Add missing index (example)
CREATE INDEX CONCURRENTLY idx_trades_tenant_symbol ON trades(tenant_id, symbol);

-- Update statistics
ANALYZE;
```

**Prevention**:
- Run `EXPLAIN ANALYZE` on all new queries
- Add composite indexes for tenant-scoped queries

---

#### Replication Lag (if read replicas enabled)

**Symptoms**: Stale data on read queries

**Diagnosis**:
```sql
-- On master: Check replication status
SELECT * FROM pg_stat_replication;

-- On replica: Check lag
SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;
```

**Resolution**:
```bash
# Restart replica (if lag >60 seconds)
kubectl rollout restart statefulset/postgres-replica -n alphapulse

# Check replication after restart
psql -h replica.example.com -U alphapulse -c "SELECT pg_last_xact_replay_timestamp()"
```

---

### 2.2 Redis Troubleshooting

#### Redis Cluster Split-Brain

**Symptoms**: Some keys not found, inconsistent data

**Diagnosis**:
```bash
# Check cluster status
redis-cli -c -h redis-cluster.example.com cluster info

# Check cluster nodes
redis-cli -c -h redis-cluster.example.com cluster nodes

# Check for partitions
redis-cli -c -h redis-cluster.example.com cluster check
```

**Resolution**:
```bash
# Force cluster reconfiguration (CAUTION: data loss possible)
redis-cli -c -h redis-cluster.example.com cluster failover

# Or manually fix slots
redis-cli -c -h redis-cluster.example.com cluster fix

# Verify cluster health
redis-cli -c -h redis-cluster.example.com cluster info | grep cluster_state
# Expected: cluster_state:ok
```

**Prevention**:
- Use Redis Sentinel for automatic failover
- Monitor cluster health: `cluster_state`, `cluster_slots_ok`

---

#### High Memory Usage

**Symptoms**: Redis memory >80%, slow responses

**Diagnosis**:
```bash
# Check memory usage
redis-cli -h redis-cluster.example.com info memory

# Check key eviction stats
redis-cli -h redis-cluster.example.com info stats | grep evicted

# Find large keys
redis-cli -h redis-cluster.example.com --bigkeys
```

**Resolution**:
```bash
# Check eviction policy
redis-cli -h redis-cluster.example.com config get maxmemory-policy
# Should be: allkeys-lru

# Increase maxmemory (if needed)
redis-cli -h redis-cluster.example.com config set maxmemory 2gb

# Or scale up Redis pods
kubectl scale statefulset/redis-cluster -n alphapulse --replicas=9
```

**Prevention**:
- Set appropriate TTLs on all keys
- Monitor cache hit rate (target: >85%)

---

### 2.3 Vault Troubleshooting

#### Vault Sealed

**Symptoms**: `vault status` shows `Sealed: true`, API cannot read credentials

**Diagnosis**:
```bash
# Check Vault status
vault status

# Expected output if sealed:
# Sealed: true
# Seal Type: shamir
```

**Resolution**:
```bash
# Unseal Vault (requires unseal keys)
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# Verify unsealed
vault status
# Expected: Sealed: false
```

**Auto-Unseal** (AWS KMS):
```bash
# Configure auto-unseal (one-time setup)
vault operator init -recovery-shares=5 -recovery-threshold=3

# Vault will auto-unseal on restart using AWS KMS
```

**Prevention**:
- Enable auto-unseal via AWS KMS or Cloud KMS
- Monitor seal status: `vault_core_unsealed` metric

---

#### Vault High Latency

**Symptoms**: Credential lookups >500ms, API slow

**Diagnosis**:
```bash
# Check Vault performance
vault read sys/metrics

# Check Vault logs
kubectl logs -f deployment/vault -n alphapulse

# Check storage backend (Raft)
vault operator raft list-peers
```

**Resolution**:
```bash
# Scale Vault pods (if needed)
kubectl scale deployment/vault -n alphapulse --replicas=5

# Check Raft leader
vault operator raft list-peers
# Leader should be healthy

# Restart Vault (if needed)
kubectl rollout restart deployment/vault -n alphapulse
```

**Prevention**:
- Cache credentials in application (5-minute TTL)
- Monitor Vault latency: `vault_request_duration_seconds`

---

### 2.4 API Troubleshooting

#### API Pods Crashing (CrashLoopBackOff)

**Symptoms**: `kubectl get pods` shows CrashLoopBackOff

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -n alphapulse | grep api

# Check pod logs
kubectl logs -f <pod-name> -n alphapulse --tail=100

# Check pod events
kubectl describe pod <pod-name> -n alphapulse
```

**Common Causes**:

1. **Database connection failure**
   ```bash
   # Check database URL in pod
   kubectl exec <pod-name> -n alphapulse -- env | grep DATABASE_URL

   # Test database connection from pod
   kubectl exec <pod-name> -n alphapulse -- psql $DATABASE_URL -c "SELECT 1"
   ```

2. **Redis connection failure**
   ```bash
   # Test Redis from pod
   kubectl exec <pod-name> -n alphapulse -- redis-cli -h redis-cluster ping
   ```

3. **Vault connection failure**
   ```bash
   # Test Vault from pod
   kubectl exec <pod-name> -n alphapulse -- vault status
   ```

4. **OOM (Out of Memory)**
   ```bash
   # Check memory limits
   kubectl describe pod <pod-name> -n alphapulse | grep -A5 "Limits:"

   # Increase memory limit (if needed)
   kubectl set resources deployment/api -n alphapulse --limits=memory=8Gi
   ```

**Resolution**:
```bash
# Rollback deployment (if recent deployment caused issue)
kubectl rollout undo deployment/api -n alphapulse

# Or fix and redeploy
kubectl apply -f k8s/api-deployment.yaml

# Watch rollout
kubectl rollout status deployment/api -n alphapulse
```

---

#### High API Latency (p99 >500ms)

**Symptoms**: Grafana shows p99 latency >500ms

**Diagnosis**:
```bash
# Check slow endpoints in Grafana
# Query: histogram_quantile(0.99, rate(api_request_duration_seconds_bucket[5m])) by (endpoint)

# Check database slow queries (see 2.1)
# Check Redis cache hit rate
redis-cli -h redis-cluster.example.com info stats | grep keyspace_hits

# Check distributed traces in Jaeger
# https://jaeger.alphapulse.ai
```

**Resolution**:

1. **Scale API pods**
   ```bash
   kubectl scale deployment/api -n alphapulse --replicas=30
   ```

2. **Optimize slow queries** (see 2.1)

3. **Increase cache TTL** (if safe)
   ```python
   # In caching_service.py
   CACHE_TTL = 120  # Increase from 60 to 120 seconds
   ```

4. **Add missing indexes** (see 2.1)

---

## 3. Common Operational Tasks

### 3.1 Deployments

#### Standard Deployment (Rolling Update)

```bash
# 1. Build new Docker image
docker build -t alphapulse/api:v1.2.3 .
docker push alphapulse/api:v1.2.3

# 2. Update Kubernetes deployment
kubectl set image deployment/api api=alphapulse/api:v1.2.3 -n alphapulse

# 3. Watch rollout
kubectl rollout status deployment/api -n alphapulse

# 4. Verify health
kubectl get pods -n alphapulse | grep api
curl https://api.alphapulse.ai/health
```

#### Canary Deployment (5% → 25% → 50% → 100%)

```bash
# 1. Deploy canary (5%)
kubectl apply -f k8s/api-canary.yaml

# 2. Route 5% traffic to canary (via Istio or Traefik)
kubectl apply -f k8s/api-virtualservice-5pct.yaml

# 3. Monitor canary metrics (5-10 minutes)
# Check Grafana: Error rate, latency

# 4. If metrics OK, increase to 25%
kubectl apply -f k8s/api-virtualservice-25pct.yaml

# 5. Repeat for 50%, 100%

# 6. If metrics BAD, rollback canary
kubectl delete -f k8s/api-canary.yaml
```

---

### 3.2 Rollbacks

#### Rollback Deployment

```bash
# 1. Check rollout history
kubectl rollout history deployment/api -n alphapulse

# 2. Rollback to previous version
kubectl rollout undo deployment/api -n alphapulse

# 3. Or rollback to specific revision
kubectl rollout undo deployment/api -n alphapulse --to-revision=5

# 4. Watch rollback
kubectl rollout status deployment/api -n alphapulse

# 5. Verify health
curl https://api.alphapulse.ai/health
```

#### Rollback Database Migration

```bash
# 1. SSH to database server or use pgAdmin
psql -U alphapulse -h prod.example.com -d alphapulse_prod

# 2. Check current migration version
SELECT * FROM alembic_version;

# 3. Rollback one migration
poetry run alembic downgrade -1

# 4. Or rollback to specific version
poetry run alembic downgrade <revision_id>

# 5. Verify rollback
SELECT * FROM alembic_version;
\dt  # Check tables
```

---

### 3.3 Scaling

#### Manual Scaling

```bash
# Scale API pods
kubectl scale deployment/api -n alphapulse --replicas=50

# Scale Agent Workers
kubectl scale deployment/agent-workers -n alphapulse --replicas=120

# Scale Redis Cluster
kubectl scale statefulset/redis-cluster -n alphapulse --replicas=9
```

#### Auto-Scaling (HPA)

```bash
# Check HPA status
kubectl get hpa -n alphapulse

# Edit HPA (increase max replicas)
kubectl edit hpa api-hpa -n alphapulse

# Or apply new HPA config
kubectl apply -f k8s/api-hpa.yaml
```

---

### 3.4 Database Maintenance

#### Backup Database

```bash
# Full backup (daily)
pg_dump -U alphapulse -h prod.example.com alphapulse_prod > backup_$(date +%Y%m%d).sql

# Upload to S3
aws s3 cp backup_$(date +%Y%m%d).sql s3://alphapulse-backups/

# Verify backup
pg_restore --list backup_$(date +%Y%m%d).sql
```

#### Restore Database

```bash
# Download backup from S3
aws s3 cp s3://alphapulse-backups/backup_20251022.sql .

# Restore (CAUTION: drops existing data)
psql -U alphapulse -h prod.example.com -d alphapulse_prod < backup_20251022.sql

# Or selective restore
pg_restore -U alphapulse -h prod.example.com -d alphapulse_prod -t trades backup_20251022.sql
```

#### Vacuum Database

```bash
# Run VACUUM ANALYZE (monthly)
psql -U alphapulse -h prod.example.com -d alphapulse_prod -c "VACUUM ANALYZE"

# Check bloat
psql -U alphapulse -h prod.example.com -d alphapulse_prod -c "SELECT schemaname, tablename, n_dead_tup FROM pg_stat_user_tables WHERE n_dead_tup > 1000 ORDER BY n_dead_tup DESC"
```

---

## 4. Monitoring and Alerting

### 4.1 Key Metrics

**API Metrics**:
- `api_request_duration_seconds` (Histogram): p50, p95, p99 latency
- `api_requests_total` (Counter): Total requests by endpoint, status
- `api_errors_total` (Counter): Error count by type

**Database Metrics**:
- `pg_stat_activity_count` (Gauge): Active connections
- `pg_stat_database_tup_inserted` (Counter): Insert rate
- `pg_stat_database_tup_updated` (Counter): Update rate

**Redis Metrics**:
- `redis_memory_used_bytes` (Gauge): Memory usage
- `redis_keyspace_hits_total` (Counter): Cache hits
- `redis_keyspace_misses_total` (Counter): Cache misses

**Vault Metrics**:
- `vault_core_unsealed` (Gauge): Seal status (1=unsealed, 0=sealed)
- `vault_request_duration_seconds` (Histogram): Vault latency

---

### 4.2 Critical Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| **API Down** | No healthy pods for 2 min | P0 | See [P0 Response](#12-p0-critical-incident-response) |
| **High Latency** | p99 >500ms for 5 min | P1 | See [API Troubleshooting](#24-api-troubleshooting) |
| **High Error Rate** | Error rate >0.5% for 2 min | P1 | Check logs, rollback if recent deployment |
| **DB Connection Pool** | >80% connections used | P1 | Kill idle connections, scale pgbouncer |
| **Redis Memory** | >90% memory used | P2 | Scale Redis, check eviction policy |
| **Vault Sealed** | Vault sealed status | P0 | Unseal Vault (see [2.3](#23-vault-troubleshooting)) |

---

### 4.3 Dashboards

**Grafana Dashboards**:
- **Overview**: https://grafana.alphapulse.ai/d/overview
- **API Metrics**: https://grafana.alphapulse.ai/d/api
- **Database**: https://grafana.alphapulse.ai/d/postgres
- **Redis**: https://grafana.alphapulse.ai/d/redis
- **Vault**: https://grafana.alphapulse.ai/d/vault

**Jaeger Traces**:
- https://jaeger.alphapulse.ai

**Sentry (Error Tracking)**:
- https://sentry.io/alphapulse

---

## 5. Disaster Recovery

### 5.1 Recovery Time Objective (RTO) & Recovery Point Objective (RPO)

| Service | RTO | RPO | Backup Frequency |
|---------|-----|-----|-----------------|
| **PostgreSQL** | 1 hour | 15 minutes | Every 15 min (WAL), Daily (full) |
| **Redis** | 30 minutes | 1 hour | Daily (RDB snapshot) |
| **Vault** | 1 hour | 24 hours | Daily (Raft snapshot) |
| **Application** | 15 minutes | N/A (stateless) | N/A |

---

### 5.2 Database Disaster Recovery

**Scenario: Database Corruption**

```bash
# 1. Stop application (prevent writes)
kubectl scale deployment/api -n alphapulse --replicas=0

# 2. Download latest backup from S3
aws s3 cp s3://alphapulse-backups/backup_latest.sql .

# 3. Download WAL archives (for point-in-time recovery)
aws s3 sync s3://alphapulse-backups/wal/ ./wal/

# 4. Restore base backup
psql -U alphapulse -h prod.example.com -d alphapulse_prod < backup_latest.sql

# 5. Replay WAL archives (point-in-time recovery to 2025-10-22 14:30)
pg_waldump ./wal/000000010000000000000042
# Apply WAL manually or via recovery.conf

# 6. Verify data integrity
psql -U alphapulse -h prod.example.com -d alphapulse_prod -c "SELECT COUNT(*) FROM trades"

# 7. Restart application
kubectl scale deployment/api -n alphapulse --replicas=10
```

---

### 5.3 Redis Disaster Recovery

**Scenario: Redis Cluster Failure**

```bash
# 1. Check Redis cluster status
redis-cli -c -h redis-cluster.example.com cluster info

# 2. Download latest RDB snapshot from S3
aws s3 cp s3://alphapulse-backups/redis/dump.rdb .

# 3. Stop Redis pods
kubectl scale statefulset/redis-cluster -n alphapulse --replicas=0

# 4. Restore RDB snapshot (copy to Redis pods)
kubectl cp dump.rdb redis-cluster-0:/data/dump.rdb -n alphapulse

# 5. Start Redis pods
kubectl scale statefulset/redis-cluster -n alphapulse --replicas=6

# 6. Verify Redis cluster
redis-cli -c -h redis-cluster.example.com cluster info
redis-cli -c -h redis-cluster.example.com cluster nodes
```

**Note**: Redis is cache, not source of truth. Can rebuild from database if needed.

---

### 5.4 Multi-Region Failover (Future)

**Not implemented in Phase 1. Planned for Phase 3.**

---

## 6. Escalation Procedures

### 6.1 Escalation Path

| Level | Contact | When to Escalate |
|-------|---------|-----------------|
| **L1: On-Call Engineer** | PagerDuty | All incidents |
| **L2: Tech Lead** | Slack DM | P0/P1 incidents, L1 cannot resolve within response time |
| **L3: Engineering Leadership** | Phone | P0 incidents lasting >1 hour, data breach |
| **L4: CTO** | Phone | P0 incidents lasting >2 hours, major data breach |

### 6.2 Contact Information

**Stored in PagerDuty and internal wiki (not in public docs).**

---

## 7. On-Call Rotation

### 7.1 Schedule

- **Primary On-Call**: 1 week rotation
- **Secondary On-Call**: Backup (responds if primary doesn't ack within 10 min)
- **On-Call Hours**: 24/7
- **Handoff**: Monday 9:00 AM

### 7.2 On-Call Responsibilities

**Before Shift**:
- Review this runbook
- Review recent incidents (last week)
- Test PagerDuty alerts
- Verify access: Grafana, Sentry, Kubernetes, AWS

**During Shift**:
- Respond to alerts within response time
- Update #incidents channel for all P0/P1 incidents
- Create postmortems for P0 incidents

**After Shift**:
- Handoff to next on-call (document open incidents)
- Complete any pending postmortems

---

## Appendix A: Quick Reference Commands

### Service Health Checks

```bash
# Kubernetes
kubectl get pods -n alphapulse
kubectl get nodes

# API
curl https://api.alphapulse.ai/health

# Database
psql -U alphapulse -h prod.example.com -c "SELECT 1"

# Redis
redis-cli -h redis-cluster.example.com ping

# Vault
vault status
```

### Logs

```bash
# API logs
kubectl logs -f deployment/api -n alphapulse --tail=100

# Database logs (RDS)
aws rds download-db-log-file-portion --db-instance-identifier alphapulse-prod --log-file-name postgresql.log

# Redis logs
kubectl logs -f statefulset/redis-cluster -n alphapulse

# Vault logs
kubectl logs -f deployment/vault -n alphapulse
```

### Metrics

```bash
# Prometheus queries
# API p99 latency
histogram_quantile(0.99, rate(api_request_duration_seconds_bucket[5m]))

# Error rate
rate(api_errors_total[5m])

# Cache hit rate
redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)
```

---

## Appendix B: Incident Postmortem Template

**File**: `.bug/postmortem-YYYYMMDD-title.md`

```markdown
# Postmortem: [Incident Title]

**Date**: YYYY-MM-DD
**Severity**: P0/P1/P2/P3
**Duration**: X hours
**Incident Commander**: [Name]

---

## Summary

[1-2 sentence summary of what happened]

## Timeline (UTC)

| Time | Event |
|------|-------|
| 14:00 | Alert triggered: API p99 latency >500ms |
| 14:05 | On-call engineer acknowledged |
| 14:10 | Root cause identified: Database slow queries |
| 14:30 | Fix deployed: Added missing index |
| 14:45 | Incident resolved |

## Root Cause

[Detailed explanation of what caused the incident]

## Impact

- **Users Affected**: 500 users (50% of active users)
- **Duration**: 45 minutes
- **Failed Requests**: 1,200 (2% error rate)
- **Revenue Impact**: $0 (no SLA breach)

## What Went Well

- Alert triggered within 2 minutes
- Root cause identified quickly
- Fix deployed within 30 minutes

## What Went Wrong

- Missing index should have been caught in load testing
- No automated slow query detection

## Action Items

| Item | Owner | Due Date | Status |
|------|-------|----------|--------|
| Add slow query monitoring | Backend Team | 2025-10-25 | ✅ Done |
| Run load testing before all deployments | DevOps | 2025-10-30 | 🔄 In Progress |
| Update index creation checklist | Tech Lead | 2025-10-23 | ⏳ Pending |

## Lessons Learned

[Key takeaways to prevent recurrence]
```

---

**Document Status**: Draft
**Review Date**: Sprint 3, Week 2
**Owners**: Tech Lead, DevOps Engineer

---

**END OF DOCUMENT**
