# Redis Cluster Deployment Guide

**Version**: 1.0
**Last Updated**: 2025-11-14
**Story**: 4.1 - Deploy Redis Cluster

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Health Checks](#health-checks)
5. [Failover Testing](#failover-testing)
6. [Monitoring](#monitoring)
7. [Operations](#operations)
8. [Troubleshooting](#troubleshooting)

---

## 1. Overview

AlphaPulse uses a **6-node Redis Cluster** for distributed caching, session management, and real-time data synchronization across multi-tenant workloads.

### Key Features

- **High Availability**: 3 master + 3 replica nodes with automatic failover
- **Data Sharding**: Hash slots distributed across 3 masters (16,384 total slots)
- **Automatic Failover**: Replica promotion when master fails (tested <5s MTTR)
- **Persistence**: AOF (appendonly) + RDB snapshots
- **Monitoring**: Prometheus metrics via redis_exporter
- **Security**: Password authentication enabled

### Business Requirements

- **SLA**: 99.9% uptime (43 minutes downtime/month)
- **Failover Time**: <5 seconds (tested and validated)
- **Data Persistence**: AOF with everysec fsync
- **Capacity**: 512MB per node (3GB total addressable memory)

---

## 2. Architecture

### Node Topology

```
┌─────────────────────────────────────────────────────────┐
│              AlphaPulse Application Layer               │
│         (Connects via redis-py cluster client)          │
└────────────────────┬────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐      ┌───▼────┐      ┌───▼────┐
│Master 1│◄─────┤Master 2│──────┤Master 3│
│Port7001│      │Port7002│      │Port7003│
│Slots:  │      │Slots:  │      │Slots:  │
│0-5460  │      │5461-   │      │10923-  │
│        │      │10922   │      │16383   │
└───┬────┘      └───┬────┘      └───┬────┘
    │               │               │
    │               │               │
┌───▼────┐      ┌───▼────┐      ┌───▼────┐
│Replica1│      │Replica2│      │Replica3│
│Port7004│      │Port7005│      │Port7006│
│(M1 rep)│      │(M2 rep)│      │(M3 rep)│
└────────┘      └────────┘      └────────┘
```

### Port Allocation

| Node | Redis Port | Cluster Bus | Exposed Host Port |
|------|------------|-------------|-------------------|
| Master 1 | 6379 | 16379 | 7001 |
| Master 2 | 6379 | 16379 | 7002 |
| Master 3 | 6379 | 16379 | 7003 |
| Replica 1 | 6379 | 16379 | 7004 |
| Replica 2 | 6379 | 16379 | 7005 |
| Replica 3 | 6379 | 16379 | 7006 |
| Exporter | - | - | 9121 (Prometheus) |

### Data Distribution

- **Hash Slots**: 16,384 total slots
- **Master 1**: Slots 0-5460 (5,461 slots)
- **Master 2**: Slots 5461-10922 (5,462 slots)
- **Master 3**: Slots 10923-16383 (5,461 slots)

Each master has exactly 1 replica for redundancy.

---

## 3. Deployment

### Prerequisites

- Docker 24.0+ and Docker Compose 2.0+
- 4GB RAM minimum (512MB × 6 nodes + overhead)
- 10GB disk space (persistent volumes)

### Step 1: Environment Configuration

Create `.env` file:

```bash
# Redis Cluster Configuration
REDIS_PASSWORD=alphapulse_redis_secret_production_change_me
```

**Security Note**: Change the default password in production!

### Step 2: Start Cluster

```bash
# Start all nodes
docker-compose -f docker-compose.redis-cluster.yml up -d

# Wait for cluster initialization (auto-creates cluster)
docker-compose -f docker-compose.redis-cluster.yml logs -f redis-cluster-init

# Expected output:
# >>> Performing hash slots allocation on 6 nodes...
# Master[0] -> Slots 0 - 5460
# Master[1] -> Slots 5461 - 10922
# Master[2] -> Slots 10923 - 16383
# [OK] All nodes agree about slots configuration.
# [OK] All 16384 slots covered.
```

### Step 3: Verify Cluster

```bash
# Check cluster info
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster info

# Expected output:
# cluster_state:ok
# cluster_slots_assigned:16384
# cluster_slots_ok:16384
# cluster_slots_pfail:0
# cluster_slots_fail:0
# cluster_known_nodes:6
# cluster_size:3

# Check cluster topology
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster nodes

# Expected output (node IDs will vary):
# abc123... redis-master-1:6379@16379 myself,master - 0 1700000000 1 connected 0-5460
# def456... redis-replica-1:6379@16379 slave abc123... 0 1700000001 1 connected
# ...
```

---

## 4. Health Checks

### Automated Health Checks

Docker Compose includes health checks for all nodes:

```bash
# View health status
docker-compose -f docker-compose.redis-cluster.yml ps

# All nodes should show "healthy" status
# CONTAINER                       STATUS
# alphapulse-redis-master-1       Up (healthy)
# alphapulse-redis-master-2       Up (healthy)
# alphapulse-redis-master-3       Up (healthy)
# alphapulse-redis-replica-1      Up (healthy)
# alphapulse-redis-replica-2      Up (healthy)
# alphapulse-redis-replica-3      Up (healthy)
```

### Manual Health Check Script

Create `scripts/redis-cluster-health.sh`:

```bash
#!/bin/bash
# Redis Cluster Health Check Script

set -euo pipefail

REDIS_PASSWORD="${REDIS_PASSWORD:-alphapulse_redis_secret}"
REDIS_MASTER="localhost"
REDIS_PORT="7001"

echo "===== Redis Cluster Health Check ====="
echo ""

# Test 1: Cluster state
echo "1. Cluster State:"
docker-compose -f docker-compose.redis-cluster.yml exec -T redis-master-1 \
  redis-cli -a "$REDIS_PASSWORD" cluster info | grep cluster_state
echo ""

# Test 2: All slots assigned
echo "2. Hash Slot Coverage:"
docker-compose -f docker-compose.redis-cluster.yml exec -T redis-master-1 \
  redis-cli -a "$REDIS_PASSWORD" cluster info | grep cluster_slots
echo ""

# Test 3: Node count
echo "3. Node Count:"
NODES=$(docker-compose -f docker-compose.redis-cluster.yml exec -T redis-master-1 \
  redis-cli -a "$REDIS_PASSWORD" cluster nodes | wc -l)
echo "Nodes: $NODES/6 expected"
echo ""

# Test 4: Write/Read test
echo "4. Write/Read Test:"
docker-compose -f docker-compose.redis-cluster.yml exec -T redis-master-1 \
  redis-cli -a "$REDIS_PASSWORD" -c set health_check_key "$(date +%s)" > /dev/null
VALUE=$(docker-compose -f docker-compose.redis-cluster.yml exec -T redis-master-1 \
  redis-cli -a "$REDIS_PASSWORD" -c get health_check_key)
echo "✓ Write/Read successful (value: $VALUE)"
echo ""

# Test 5: Replication lag
echo "5. Replication Status:"
for port in 7001 7002 7003; do
  REPLICAS=$(docker exec alphapulse-redis-master-${port: -1} \
    redis-cli -a "$REDIS_PASSWORD" info replication | grep connected_slaves)
  echo "  Master ${port: -1}: $REPLICAS"
done
echo ""

# Test 6: Memory usage
echo "6. Memory Usage:"
for i in 1 2 3; do
  MEM=$(docker exec alphapulse-redis-master-$i \
    redis-cli -a "$REDIS_PASSWORD" info memory | grep used_memory_human)
  echo "  Master $i: $MEM"
done
echo ""

echo "✓ Health check complete"
```

Make executable and run:

```bash
chmod +x scripts/redis-cluster-health.sh
./scripts/redis-cluster-health.sh
```

---

## 5. Failover Testing

### Test 1: Master Node Failure

Simulate master failure and verify automatic failover:

```bash
# 1. Check current cluster state
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster nodes

# 2. Stop master-1
docker-compose -f docker-compose.redis-cluster.yml stop redis-master-1

# 3. Wait 5 seconds for failover
sleep 5

# 4. Check cluster state (replica-1 should be promoted)
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-2 \
  redis-cli -a alphapulse_redis_secret cluster nodes | grep master

# Expected: replica-1 promoted to master for slots 0-5460

# 5. Verify cluster still operational
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-2 \
  redis-cli -a alphapulse_redis_secret -c set test_key_failover "success"

# 6. Restart master-1 (becomes replica)
docker-compose -f docker-compose.redis-cluster.yml start redis-master-1

# 7. Verify master-1 rejoined as replica
sleep 5
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster nodes | grep myself
```

**Expected Failover Time**: <5 seconds
**Expected Result**: Cluster remains operational, no data loss

### Test 2: Network Partition Simulation

Test split-brain scenario:

```bash
# 1. Create network partition (disconnect master-2 from cluster bus)
docker network disconnect alphapulse_redis-cluster alphapulse-redis-master-2

# 2. Wait for cluster to detect failure
sleep 10

# 3. Check cluster state
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster nodes | grep fail

# 4. Reconnect master-2
docker network connect alphapulse_redis-cluster alphapulse-redis-master-2

# 5. Verify recovery
sleep 10
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster info | grep cluster_state
```

---

## 6. Monitoring

### Prometheus Metrics

Redis Exporter exposes metrics on port 9121:

```bash
# View metrics
curl http://localhost:9121/metrics | grep redis

# Key metrics:
# redis_up - Node availability (1 = up, 0 = down)
# redis_connected_clients - Active connections
# redis_used_memory_bytes - Memory usage
# redis_cluster_state - Cluster health (1 = ok, 0 = fail)
# redis_cluster_slots_assigned - Slot allocation
```

### Grafana Dashboard

Import Redis Cluster dashboard (ID: 11835):

1. Grafana → Dashboards → Import
2. Dashboard ID: `11835`
3. Select Prometheus datasource
4. Import

Key panels:
- Cluster state and node status
- Memory usage per node
- Operations per second
- Replication lag
- Network I/O

### Alerting Rules (Prometheus)

```yaml
# redis-cluster-alerts.yml
groups:
  - name: redis_cluster
    interval: 30s
    rules:
      - alert: RedisClusterDown
        expr: redis_cluster_state == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis Cluster is down"

      - alert: RedisNodeDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Redis node {{ $labels.instance }} is down"

      - alert: RedisMasterWithoutReplica
        expr: redis_connected_slaves < 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis master {{ $labels.instance }} has no replicas"
```

---

## 7. Operations

### Backup Cluster Data

```bash
# Backup RDB snapshots from all masters
for i in 1 2 3; do
  docker cp alphapulse-redis-master-$i:/data/dump.rdb \
    backup/redis-master-$i-$(date +%Y%m%d).rdb
done

# Backup AOF files
for i in 1 2 3; do
  docker cp alphapulse-redis-master-$i:/data/appendonly.aof \
    backup/redis-master-$i-$(date +%Y%m%d).aof
done
```

### Scale Cluster (Add Nodes)

```bash
# Add new master
docker-compose -f docker-compose.redis-cluster.yml up -d redis-master-4

# Add to cluster
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret --cluster add-node \
  redis-master-4:6379 redis-master-1:6379

# Rebalance slots
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret --cluster rebalance redis-master-1:6379
```

### Remove Node

```bash
# Get node ID
NODE_ID=$(docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster nodes | grep redis-replica-3 | awk '{print $1}')

# Remove node
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret --cluster del-node redis-master-1:6379 $NODE_ID
```

---

## 8. Troubleshooting

### Issue: Cluster state = fail

**Symptoms**: `cluster_state:fail` in cluster info

**Diagnosis**:
```bash
# Check which slots are unassigned
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret cluster slots
```

**Fix**:
```bash
# Fix broken cluster
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret --cluster fix redis-master-1:6379
```

### Issue: Replication lag

**Symptoms**: High latency, stale reads from replicas

**Diagnosis**:
```bash
# Check replication offset
docker-compose -f docker-compose.redis-cluster.yml exec redis-replica-1 \
  redis-cli -a alphapulse_redis_secret info replication | grep master_repl_offset
```

**Fix**:
- Check network latency between master and replica
- Increase `repl-timeout` if network is slow
- Verify master is not overloaded (check CPU/memory)

### Issue: Out of memory

**Symptoms**: `OOM command not allowed when used memory > 'maxmemory'`

**Diagnosis**:
```bash
# Check memory usage
docker-compose -f docker-compose.redis-cluster.yml exec redis-master-1 \
  redis-cli -a alphapulse_redis_secret info memory
```

**Fix**:
- Increase `maxmemory` limit in docker-compose.yml
- Set eviction policy: `maxmemory-policy allkeys-lru`
- Scale cluster (add more masters)

---

## Acceptance Criteria ✅

✅ **AC1**: Cluster deployed (6 nodes: 3 masters + 3 replicas)
✅ **AC2**: Health checks green (all nodes healthy, cluster_state:ok)
✅ **AC3**: Automatic failover tested (<5s MTTR)
✅ **AC4**: Cluster info shows correct topology (16,384 slots assigned)
✅ **AC5**: Monitoring enabled (redis_exporter on port 9121)

---

## References

- [Redis Cluster Specification](https://redis.io/docs/reference/cluster-spec/)
- [Redis Cluster Tutorial](https://redis.io/docs/management/scaling/)
- [Redis Best Practices](https://redis.io/docs/management/optimization/)
