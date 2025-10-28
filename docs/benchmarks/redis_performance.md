# Redis Namespace Isolation Performance Benchmark

**Date**: 2025-10-20
**Sprint**: 1 (Inception)
**Related**: [SPIKE: Redis Namespace Isolation](#152), [EPIC-004](#143), [ADR-004](../adr/004-caching-strategy-multi-tenant.md)

---

## Executive Summary

**Objective**: Validate that Redis namespace isolation with rolling counters adds <1ms overhead and LRU eviction completes in <100ms.

**Hypothesis**: Redis namespace prefixes (`tenant:{id}:*`) combined with rolling counters (O(1) operations) provide tenant isolation with minimal performance impact.

**Decision**: [To be determined after benchmark execution]

---

## Test Environment

### Redis Configuration

- **Version**: Redis 7.x (latest stable)
- **Deployment**: Redis Cluster (6 nodes: 3 masters + 3 replicas)
- **Instance Type**: cache.t3.medium (2 vCPU, 3.09GB RAM) per node
- **Eviction Policy**: `allkeys-lru` (automatic eviction when maxmemory reached)
- **Max Memory**: 2GB per master node
- **Persistence**: AOF (Append-Only File) for durability

### Test Data

- **Tenants**: 10 simulated tenants
- **Cache Entries per Tenant**: 100 entries (signals, positions, portfolio data)
- **Entry Size**: ~150 bytes per entry
- **Shared Market Data**: 5 symbols (BTC_USDT, ETH_USDT, XRP_USDT, SOL_USDT, ADA_USDT)
- **TTL**: 5 minutes for tenant data, 1 minute for market data

### Namespace Structure

```
tenant:{tenant_id}:signal:{signal_id}       # Tenant-specific cache
tenant:{tenant_id}:position:{symbol}        # Tenant positions
meta:tenant:{tenant_id}:usage_bytes         # Rolling counter (quota tracking)
meta:tenant:{tenant_id}:lru                 # Sorted set (LRU tracking)
shared:market:{symbol}:price                # Shared market data (all tenants)
```

---

## Benchmark Scenarios

### Scenario 1: Rolling Counter (Quota Enforcement)

**Operation**: Increment usage counter + quota check

```python
# Check current usage
current = await redis.get(f"meta:tenant:{tenant_id}:usage_bytes")

# Enforce quota (100MB limit)
if current is None or int(current) < 100_000_000:
    await redis.incrby(f"meta:tenant:{tenant_id}:usage_bytes", 1000)
```

**Expected Performance**:
- P50: <0.5ms
- P95: <1ms
- P99: <1ms (target)

---

### Scenario 2: LRU Eviction (Sorted Set)

**Operation**: Evict oldest 10 entries when cache exceeds 100 entries

```python
# Get LRU size
size = await redis.zcard(f"meta:tenant:{tenant_id}:lru")

if size > 100:
    # Get oldest 10 keys
    oldest = await redis.zrange(f"meta:tenant:{tenant_id}:lru", 0, 9)

    # Delete cache entries
    await redis.delete(*oldest)

    # Remove from LRU tracking
    await redis.zrem(f"meta:tenant:{tenant_id}:lru", *oldest)

    # Update usage counter
    await redis.decrby(f"meta:tenant:{tenant_id}:usage_bytes", len(oldest) * 150)
```

**Expected Performance**:
- P50: <50ms
- P95: <80ms
- P99: <100ms (target)

---

### Scenario 3: Cache Hit Rate (Shared Market Data)

**Operation**: Read shared market data (accessed by all tenants)

```python
value = await redis.get(f"shared:market:{symbol}:price")
```

**Expected Performance**:
- Hit Rate: >80% (target)
- P50: <0.5ms
- P95: <1ms
- P99: <2ms

---

### Scenario 4: Namespace Isolation (Tenant-Scoped Reads)

**Operation**: Read tenant-specific data

```python
value = await redis.get(f"tenant:{tenant_id}:signal:{signal_id}")
```

**Expected Performance**:
- P50: <0.5ms
- P95: <1ms
- P99: <1ms (target)

---

### Scenario 5: Cross-Tenant Isolation Verification

**Operation**: Verify tenants cannot access each other's data

**Test**:
1. Set data for Tenant A: `tenant:{tenant_a}:secret_data`
2. Attempt to read from Tenant B: `tenant:{tenant_b}:secret_data` → Should be `None`
3. Verify key pattern matching is application-enforced (no Redis-level ACLs in this design)

**Expected**: Application-level enforcement via namespace prefixes prevents accidental cross-tenant access.

---

## Results

### [To be filled after benchmark execution]

#### Scenario 1: Rolling Counter

| Metric | Value (ms) | Target | Status |
|--------|-----------|--------|--------|
| Min    | TBD | N/A | N/A |
| Max    | TBD | N/A | N/A |
| Mean   | TBD | <1ms | TBD |
| P50    | TBD | <1ms | TBD |
| P95    | TBD | <1ms | TBD |
| P99    | TBD | <1ms | TBD |

#### Scenario 2: LRU Eviction

| Metric | Value (ms) | Target | Status |
|--------|-----------|--------|--------|
| Min    | TBD | N/A | N/A |
| Max    | TBD | N/A | N/A |
| Mean   | TBD | <100ms | TBD |
| P50    | TBD | <100ms | TBD |
| P95    | TBD | <100ms | TBD |
| P99    | TBD | <100ms | TBD |

#### Scenario 3: Cache Hit Rate

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Hit Rate | TBD% | >80% | TBD |
| Hits     | TBD | N/A | N/A |
| Misses   | TBD | N/A | N/A |

**Latency Statistics**:

| Metric | Value (ms) | Target | Status |
|--------|-----------|--------|--------|
| Mean   | TBD | <2ms | TBD |
| P50    | TBD | <1ms | TBD |
| P95    | TBD | <2ms | TBD |
| P99    | TBD | <2ms | TBD |

#### Scenario 4: Namespace Isolation

| Metric | Value (ms) | Target | Status |
|--------|-----------|--------|--------|
| Min    | TBD | N/A | N/A |
| Max    | TBD | N/A | N/A |
| Mean   | TBD | <1ms | TBD |
| P50    | TBD | <1ms | TBD |
| P95    | TBD | <1ms | TBD |
| P99    | TBD | <1ms | TBD |

#### Scenario 5: Cross-Tenant Isolation

| Check | Result | Status |
|-------|--------|--------|
| Tenant A keys created | TBD | TBD |
| Cross-tenant access blocked | TBD | TBD |
| Namespace isolation enforced | TBD | TBD |

---

## Analysis

### [To be filled after benchmark execution]

**Rolling Counter Performance**:
- P99 latency: TBD ms
- Overhead vs direct GET: TBD%

**LRU Eviction Performance**:
- P99 latency: TBD ms
- Batch eviction (10 keys): TBD operations

**Cache Hit Rate**:
- Shared market data hit rate: TBD%
- Tenant-specific data hit rate: TBD%

**Observations**:
- [To be added after execution]

---

## Decision

### If Rolling Counter P99 <1ms AND LRU P99 <100ms AND Hit Rate >80% (PASS)

✅ **PROCEED with Redis namespace isolation approach**

- Rolling counters provide O(1) quota enforcement
- Sorted sets enable efficient LRU eviction
- Namespace prefixes provide tenant isolation
- No Redis Cluster ACLs required (application-level enforcement)
- Move forward with EPIC-004 (Caching Layer) as planned

**Cache Quotas (per tier)**:
- **Starter**: 10MB per tenant
- **Pro**: 50MB per tenant
- **Enterprise**: 500MB per tenant (dedicated cache nodes if needed)

### If Rolling Counter P99 1-2ms OR LRU P99 100-200ms (WARNING)

⚠️ **PROCEED with monitoring**

- Performance acceptable but below target
- Add observability for cache performance in production
- Monitor eviction frequency and hit rates

**Mitigation**:
- Tune Redis `maxmemory-policy` settings
- Add Prometheus metrics for cache hit rates
- Budget 1 sprint (Sprint 9) for optimization if needed

**Impact**: No timeline change, adjust monitoring configuration only

### If Rolling Counter P99 >2ms OR LRU P99 >200ms OR Hit Rate <60% (FAIL)

✗ **ADJUST STRATEGY**

**Option A: Redis Enterprise**
- Redis Enterprise: Better performance, Active-Active replication, built-in multi-tenancy
- Cost: ~$500/month for 3 nodes
- **Impact**: +$6k/year operational cost

**Option B: Dedicated Redis Instances per Tenant (Enterprise Tier Only)**
- Starter/Pro remain in shared Redis Cluster
- Enterprise tenants get dedicated Redis instance
- **Impact**: +1 sprint for provisioning logic (EPIC-006)

**Option C: Alternative Caching Strategy**
- In-memory caching (e.g., Python `cachetools`) with Redis as L2 cache
- Reduce Redis load by 70-80%
- **Impact**: +1 sprint for implementation

**Recommended**: Option C (In-memory L1 cache) provides best performance/cost balance.

---

## Recommendations

### [To be filled after benchmark execution]

1. [Recommendation based on results]
2. [Cache configuration tuning suggestions]
3. [Quota enforcement adjustments]
4. [Production monitoring requirements]

---

## Next Steps

### [To be filled after benchmark execution]

- [ ] Update EPIC-004 with decision
- [ ] Update risk register (cache performance risk)
- [ ] Adjust Sprint 7 scope if needed (Redis optimization)
- [ ] Present findings in Sprint 1 Review

---

## Appendix: Running the Benchmark

### Prerequisites

```bash
# Install Redis
# macOS
brew install redis

# Ubuntu/Debian
sudo apt-get install redis-server

# Start Redis
redis-server
```

```bash
# Install Python dependencies
pip install redis asyncio
```

### Execution

```bash
# Run benchmark with default settings (10 tenants)
python scripts/benchmark_redis.py

# Custom settings
python scripts/benchmark_redis.py \
    --redis redis://localhost:6379/0 \
    --tenants 20
```

### Expected Runtime

- Setup: ~5 seconds (create 1,000 test keys)
- Benchmarks: ~3 minutes (5 scenarios with 1k-10k iterations each)
- **Total**: ~3.5 minutes

### Expected Output

```
=== SETUP PHASE ===
Cleaning up test keys...
✓ Test keys cleaned up
Populating test data for 10 tenants...
✓ Populated 1005 keys in 4.23s

=== BENCHMARK PHASE ===

Benchmarking: Rolling Counter (Quota Enforcement)
Iterations: 10,000
  Running benchmark...

Benchmarking: LRU Eviction (Sorted Set)
Iterations: 1,000
  Running benchmark...

Benchmarking: Cache Hit Rate (Shared Market Data)
Iterations: 10,000
  Running benchmark...

Benchmarking: Namespace Isolation
Iterations: 5,000
  Running benchmark...

Verifying: Cross-Tenant Isolation
  Tenant A keys: 101
  Cross-tenant access attempt result: None
  ✓ Namespaces are isolated (application enforces prefix)

=== BENCHMARK RESULTS ===

Rolling Counter (Quota Enforcement)
--------------------------------------------------------------------------------
Iterations: 10,000
Target: <1ms P99

Metric     Value (ms)
--------------------------------------------------------------------------------
MIN             0.145
MAX             2.456
MEAN            0.328
P50             0.312
P95             0.523
P99             0.789

✓ Status: PASS

LRU Eviction (Sorted Set)
--------------------------------------------------------------------------------
Iterations: 1,000
Target: <100ms P99

Metric     Value (ms)
--------------------------------------------------------------------------------
MIN             0.234
MAX            45.123
MEAN            2.145
P50             1.876
P95             5.234
P99            12.456

✓ Status: PASS

Cache Hit Rate (Shared Market Data)
--------------------------------------------------------------------------------
Iterations: 10,000
Target: >80% hit rate

Hit Rate: 95.23% (9,523 hits / 477 misses)

Latency Statistics:
  MEAN: 0.245ms
  P50: 0.234ms
  P95: 0.456ms
  P99: 0.678ms

✓ Status: PASS

Namespace Isolation (Tenant-Scoped Reads)
--------------------------------------------------------------------------------
Iterations: 5,000
Target: <1ms P99

Metric     Value (ms)
--------------------------------------------------------------------------------
MIN             0.123
MAX             1.234
MEAN            0.289
P50             0.276
P95             0.456
P99             0.723

✓ Status: PASS

Cross-Tenant Isolation
--------------------------------------------------------------------------------
Tenant A keys: 101
Cross-tenant access blocked: True

✓ Status: PASS

=== SUMMARY ===
Tests Passed: 5/5

✓ PASS: All benchmarks met targets
  Decision: PROCEED with Redis namespace isolation approach

Key Findings:
  - Rolling counter P99: 0.789ms (target: <1ms)
  - LRU eviction P99: 12.456ms (target: <100ms)
  - Cache hit rate: 95.2% (target: >80%)

=== CLEANUP PHASE ===
Cleaning up test keys...
✓ Test keys cleaned up
✓ Redis connection closed
```

**Note**: The example above shows ideal results. Actual performance depends on Redis configuration, hardware, and network latency.

---

## References

- [ADR-004: Caching Strategy for Multi-Tenant](../adr/004-caching-strategy-multi-tenant.md)
- [Redis Documentation](https://redis.io/docs/)
- [Redis Sorted Sets](https://redis.io/docs/data-types/sorted-sets/)
- [Redis Cluster Tutorial](https://redis.io/docs/manual/scaling/)
- [HLD Section 2.5: Caching Strategy](../HLD-MULTI-TENANT-SAAS.md#25-caching-strategy)
