# PostgreSQL Row-Level Security (RLS) Performance Benchmark

**Date**: 2025-10-20
**Sprint**: 1 (Inception)
**Related**: [SPIKE: PostgreSQL RLS Performance](#150), [EPIC-001](#140), [ADR-001](../adr/001-multi-tenant-data-isolation-strategy.md)

---

## Executive Summary

**Objective**: Validate that PostgreSQL Row-Level Security (RLS) adds <10% query overhead for multi-tenant workloads.

**Hypothesis**: Composite indexes on `(tenant_id, id)` and `(tenant_id, created_at)` will keep RLS overhead below 10%.

**Decision**: [To be determined after benchmark execution]

---

## Test Environment

### Database Configuration

- **PostgreSQL Version**: 14.x (or current production version)
- **Instance**: db.t3.medium (2 vCPU, 4GB RAM) or equivalent
- **Storage**: GP3 SSD (3000 IOPS baseline)
- **Connection Pool**: 20 connections

### Test Data

- **Tenants**: 10 simulated tenants
- **Rows**: 100,000 trades distributed evenly across tenants
- **Symbols**: 5 cryptocurrencies (BTC_USDT, ETH_USDT, XRP_USDT, SOL_USDT, ADA_USDT)
- **Time Range**: 365 days of historical data

### Schema

```sql
CREATE TABLE test_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    side VARCHAR(10) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Critical indexes for RLS performance
CREATE INDEX idx_trades_tenant_id ON test_trades(tenant_id, id);
CREATE INDEX idx_trades_tenant_created ON test_trades(tenant_id, created_at DESC);

-- RLS policy
ALTER TABLE test_trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON test_trades
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);
```

---

## Benchmark Scenarios

### Scenario 1: Simple SELECT (LIMIT 100)

**Query (without RLS)**:
```sql
SELECT * FROM test_trades WHERE tenant_id = $1 LIMIT 100;
```

**Query (with RLS)**:
```sql
SET LOCAL app.current_tenant_id = $1;
SELECT * FROM test_trades LIMIT 100;
```

**Expected Performance**:
- Baseline: ~1-2ms
- RLS: ~1-2ms
- Expected overhead: <5%

---

### Scenario 2: Aggregation (GROUP BY)

**Query (without RLS)**:
```sql
SELECT symbol, COUNT(*), AVG(price), SUM(quantity)
FROM test_trades
WHERE tenant_id = $1
GROUP BY symbol;
```

**Query (with RLS)**:
```sql
SET LOCAL app.current_tenant_id = $1;
SELECT symbol, COUNT(*), AVG(price), SUM(quantity)
FROM test_trades
GROUP BY symbol;
```

**Expected Performance**:
- Baseline: ~5-10ms
- RLS: ~5-11ms
- Expected overhead: <10%

---

### Scenario 3: JOIN (Multi-Table)

**Query (without RLS)**:
```sql
SELECT t.id, t.symbol, t.quantity, p.current_value
FROM test_trades t
JOIN test_positions p ON t.symbol = p.symbol AND t.tenant_id = p.tenant_id
WHERE t.tenant_id = $1
LIMIT 100;
```

**Query (with RLS)**:
```sql
SET LOCAL app.current_tenant_id = $1;
SELECT t.id, t.symbol, t.quantity, p.current_value
FROM test_trades t
JOIN test_positions p ON t.symbol = p.symbol
LIMIT 100;
```

**Expected Performance**:
- Baseline: ~3-5ms
- RLS: ~3-6ms
- Expected overhead: <10%

---

### Scenario 4: Time-Range Query (7 days)

**Query (without RLS)**:
```sql
SELECT * FROM test_trades
WHERE tenant_id = $1 AND created_at >= $2
ORDER BY created_at DESC
LIMIT 1000;
```

**Query (with RLS)**:
```sql
SET LOCAL app.current_tenant_id = $1;
SELECT * FROM test_trades
WHERE created_at >= $1
ORDER BY created_at DESC
LIMIT 1000;
```

**Expected Performance**:
- Baseline: ~2-4ms
- RLS: ~2-5ms
- Expected overhead: <10%

---

## Results

### [To be filled after benchmark execution]

#### Scenario 1: Simple SELECT (LIMIT 100)

| Metric | Baseline (ms) | RLS (ms) | Overhead (%) | Status |
|--------|---------------|----------|--------------|--------|
| Mean   | TBD | TBD | TBD | TBD |
| P50    | TBD | TBD | TBD | TBD |
| P95    | TBD | TBD | TBD | TBD |
| P99    | TBD | TBD | TBD | TBD |

#### Scenario 2: Aggregation (GROUP BY)

| Metric | Baseline (ms) | RLS (ms) | Overhead (%) | Status |
|--------|---------------|----------|--------------|--------|
| Mean   | TBD | TBD | TBD | TBD |
| P50    | TBD | TBD | TBD | TBD |
| P95    | TBD | TBD | TBD | TBD |
| P99    | TBD | TBD | TBD | TBD |

#### Scenario 3: JOIN (Multi-Table)

| Metric | Baseline (ms) | RLS (ms) | Overhead (%) | Status |
|--------|---------------|----------|--------------|--------|
| Mean   | TBD | TBD | TBD | TBD |
| P50    | TBD | TBD | TBD | TBD |
| P95    | TBD | TBD | TBD | TBD |
| P99    | TBD | TBD | TBD | TBD |

#### Scenario 4: Time-Range Query

| Metric | Baseline (ms) | RLS (ms) | Overhead (%) | Status |
|--------|---------------|----------|--------------|--------|
| Mean   | TBD | TBD | TBD | TBD |
| P50    | TBD | TBD | TBD | TBD |
| P95    | TBD | TBD | TBD | TBD |
| P99    | TBD | TBD | TBD | TBD |

---

## Analysis

### [To be filled after benchmark execution]

**Average P99 Overhead**: TBD%
**Maximum P99 Overhead**: TBD%

**EXPLAIN ANALYZE Output** (sample queries):

```
[To be added: Query planner output showing index usage]
```

**Index Usage**:
- [ ] Composite index `(tenant_id, id)` used for simple SELECTs
- [ ] Composite index `(tenant_id, created_at)` used for time-range queries
- [ ] No sequential scans detected

**Observations**:
- [To be added after execution]

---

## Decision

### If Overhead <10% (PASS)

✅ **PROCEED with RLS approach**

- RLS provides sufficient performance for multi-tenant workloads
- Composite indexes effectively optimize tenant filtering
- No additional mitigation required
- Move forward with EPIC-001 (Database Multi-Tenancy) as planned

### If Overhead 10-20% (WARNING)

⚠️ **PROCEED with RLS, monitor in production**

- Overhead acceptable but warrants monitoring
- Implement query performance tracking in production
- Consider query optimization (e.g., materialized views for aggregations)
- Re-evaluate if production metrics show degradation

**Mitigation**:
- Add query performance monitoring dashboard
- Set alerts for queries >100ms
- Budget 1 sprint (Sprint 7) for optimization if needed

### If Overhead >20% (FAIL)

✗ **ADJUST STRATEGY**

**Option A: Table Partitioning**
- Partition tables by `tenant_id` (e.g., HASH partitioning with 10 partitions)
- RLS still enabled per partition for defense-in-depth
- **Impact**: +1 sprint (Sprint 7) for partitioning implementation

**Option B: Dedicated Schemas (Enterprise Only)**
- Move high-value tenants to dedicated schemas: `tenant_{uuid}`
- Starter/Pro remain in shared schema with RLS
- **Impact**: No change to timeline, adjust provisioning logic (EPIC-006)

**Option C: Application-Level Filtering (Not Recommended)**
- Remove RLS, rely on application WHERE clauses
- **Risk**: HIGH (single developer mistake = data leak)
- **Only consider if**: Options A & B insufficient

**Recommended**: Option A (Partitioning) provides best balance of performance and security.

---

## Recommendations

### [To be filled after benchmark execution]

1. [Recommendation based on results]
2. [Query optimization suggestions]
3. [Index tuning suggestions]
4. [Production monitoring requirements]

---

## Next Steps

### [To be filled after benchmark execution]

- [ ] Update EPIC-001 with decision
- [ ] Update risk register (promote/demote RLS performance risk)
- [ ] Adjust Sprint 5-6 scope if needed (add partitioning stories)
- [ ] Present findings in Sprint 1 Review

---

## Appendix: Running the Benchmark

### Prerequisites

```bash
# Install dependencies
pip install asyncpg

# Create test database
createdb alphapulse_test
```

### Execution

```bash
# Run benchmark with default settings (10 tenants, 100k rows)
python scripts/benchmark_rls.py

# Custom settings
python scripts/benchmark_rls.py \
    --database postgresql://localhost/alphapulse_test \
    --tenants 20 \
    --rows 500000
```

### Expected Runtime

- Setup: ~30 seconds (schema creation + data insertion)
- Benchmarks: ~10 minutes (4 scenarios × 500-1000 iterations each)
- **Total**: ~11 minutes

---

## References

- [ADR-001: Multi-Tenant Data Isolation Strategy](../adr/001-multi-tenant-data-isolation-strategy.md)
- [PostgreSQL RLS Documentation](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [PostgreSQL Index Performance](https://www.postgresql.org/docs/current/indexes-multicolumn.html)
- [HLD Section 2.2: Data Design](../HLD-MULTI-TENANT-SAAS.md#22-data-design)
