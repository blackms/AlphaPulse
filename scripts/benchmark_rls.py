#!/usr/bin/env python3
"""
PostgreSQL Row-Level Security (RLS) Performance Benchmarking

This script validates that PostgreSQL RLS adds <10% query overhead for
multi-tenant isolation.

SPIKE #150: PostgreSQL RLS Performance Benchmarking
Time Box: 3 days
Success Criteria:
- Baseline benchmarks collected (100k rows per table)
- RLS benchmarks collected (same dataset)
- Performance report documenting overhead percentage
- RLS overhead <10%
"""

import asyncio
import asyncpg
import time
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass
import uuid
import random
import json


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""

    query_type: str
    rls_enabled: bool
    iterations: int
    total_duration: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    rows_processed: int


class RLSBenchmark:
    """PostgreSQL RLS performance benchmarking"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "alphapulse",
        user: str = "alphapulse",
        password: str = "alphapulse",
    ):
        """
        Initialize RLS benchmark.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.tenant_ids: List[uuid.UUID] = []
        self.results: List[BenchmarkResult] = []

    async def connect(self):
        """Establish database connection"""
        self.conn = await asyncpg.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        print(f"✓ Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")

    async def disconnect(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            print("✓ Disconnected from PostgreSQL")

    async def setup_test_table(self, num_tenants: int = 10, rows_per_tenant: int = 10000):
        """
        Create test table and populate with data.

        Args:
            num_tenants: Number of tenants to simulate
            rows_per_tenant: Number of rows per tenant
        """
        print(f"\n{'='*60}")
        print("Setting up test table")
        print(f"{'='*60}")

        # Drop existing table
        await self.conn.execute("DROP TABLE IF EXISTS test_trades CASCADE")
        print("✓ Dropped existing test_trades table")

        # Create table
        await self.conn.execute(
            """
            CREATE TABLE test_trades (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                quantity DECIMAL(18, 8) NOT NULL,
                price DECIMAL(18, 2) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """
        )
        print("✓ Created test_trades table")

        # Create composite index (critical for RLS performance)
        await self.conn.execute(
            """
            CREATE INDEX idx_trades_tenant_id
            ON test_trades(tenant_id, created_at DESC)
        """
        )
        print("✓ Created composite index on (tenant_id, created_at)")

        # Generate tenant IDs
        self.tenant_ids = [uuid.uuid4() for _ in range(num_tenants)]
        print(f"✓ Generated {num_tenants} tenant UUIDs")

        # Insert test data
        total_rows = num_tenants * rows_per_tenant
        print(f"Inserting {total_rows:,} rows...")

        symbols = ["BTC_USDT", "ETH_USDT", "XRP_USDT", "SOL_USDT", "ADA_USDT"]
        batch_size = 1000
        inserted = 0

        start_time = time.time()

        for tenant_id in self.tenant_ids:
            for batch_start in range(0, rows_per_tenant, batch_size):
                batch_end = min(batch_start + batch_size, rows_per_tenant)
                batch_data = []

                for _ in range(batch_end - batch_start):
                    batch_data.append(
                        (
                            tenant_id,
                            random.choice(symbols),
                            round(random.uniform(0.001, 10.0), 8),
                            round(random.uniform(100, 100000), 2),
                        )
                    )

                await self.conn.executemany(
                    """
                    INSERT INTO test_trades (tenant_id, symbol, quantity, price)
                    VALUES ($1, $2, $3, $4)
                """,
                    batch_data,
                )

                inserted += len(batch_data)
                if inserted % 10000 == 0:
                    print(f"  Progress: {inserted:,}/{total_rows:,} rows")

        duration = time.time() - start_time
        print(
            f"✓ Inserted {total_rows:,} rows in {duration:.2f}s "
            f"({total_rows/duration:.0f} rows/sec)"
        )

        # Analyze table for query planner
        await self.conn.execute("ANALYZE test_trades")
        print("✓ Analyzed table statistics")

    async def enable_rls(self):
        """Enable Row-Level Security on test table"""
        print(f"\n{'='*60}")
        print("Enabling Row-Level Security")
        print(f"{'='*60}")

        # Enable RLS
        await self.conn.execute("ALTER TABLE test_trades ENABLE ROW LEVEL SECURITY")
        print("✓ Enabled RLS on test_trades")

        # Drop existing policy if exists
        await self.conn.execute(
            "DROP POLICY IF EXISTS tenant_isolation ON test_trades"
        )

        # Create RLS policy
        await self.conn.execute(
            """
            CREATE POLICY tenant_isolation ON test_trades
            FOR ALL
            USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
        """
        )
        print("✓ Created tenant_isolation policy")

    async def disable_rls(self):
        """Disable Row-Level Security on test table"""
        print(f"\n{'='*60}")
        print("Disabling Row-Level Security")
        print(f"{'='*60}")

        await self.conn.execute("ALTER TABLE test_trades DISABLE ROW LEVEL SECURITY")
        print("✓ Disabled RLS on test_trades")

    async def set_tenant_context(self, tenant_id: uuid.UUID):
        """
        Set tenant context for RLS.

        Args:
            tenant_id: Tenant UUID to set in session
        """
        await self.conn.execute(
            f"SET app.current_tenant_id = '{tenant_id}'"
        )

    async def run_query_benchmark(
        self,
        query_name: str,
        query: str,
        iterations: int = 100,
        rls_enabled: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark for a specific query.

        Args:
            query_name: Name of the query being benchmarked
            query: SQL query to execute
            iterations: Number of iterations to run
            rls_enabled: Whether RLS is enabled

        Returns:
            BenchmarkResult with performance metrics
        """
        latencies = []
        total_rows = 0

        # Set random tenant context if RLS enabled
        if rls_enabled and self.tenant_ids:
            await self.set_tenant_context(random.choice(self.tenant_ids))

        # Warm-up queries (not measured)
        for _ in range(5):
            await self.conn.fetch(query)

        # Measured iterations
        start_time = time.time()

        for i in range(iterations):
            # Rotate tenant context every 10 queries if RLS enabled
            if rls_enabled and i % 10 == 0 and self.tenant_ids:
                await self.set_tenant_context(random.choice(self.tenant_ids))

            query_start = time.time()
            rows = await self.conn.fetch(query)
            query_duration = (time.time() - query_start) * 1000  # Convert to ms

            latencies.append(query_duration)
            total_rows += len(rows)

        total_duration = time.time() - start_time

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        min_latency = min(latencies)
        max_latency = max(latencies)

        return BenchmarkResult(
            query_type=query_name,
            rls_enabled=rls_enabled,
            iterations=iterations,
            total_duration=total_duration,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            rows_processed=total_rows,
        )

    async def benchmark_simple_select(
        self, iterations: int = 100, rls_enabled: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark 1: Simple SELECT query
        Target: P99 <5ms

        Args:
            iterations: Number of iterations
            rls_enabled: Whether RLS is enabled

        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*60}")
        print(f"Benchmark 1: Simple SELECT (RLS: {'ON' if rls_enabled else 'OFF'})")
        print(f"{'='*60}")

        # Use a random tenant for baseline (without RLS policy check)
        tenant_id = random.choice(self.tenant_ids) if self.tenant_ids else uuid.uuid4()

        if rls_enabled:
            # With RLS: policy automatically filters by session tenant_id
            query = """
                SELECT id, symbol, quantity, price, created_at
                FROM test_trades
                LIMIT 100
            """
        else:
            # Baseline: explicit WHERE clause
            query = f"""
                SELECT id, symbol, quantity, price, created_at
                FROM test_trades
                WHERE tenant_id = '{tenant_id}'
                LIMIT 100
            """

        result = await self.run_query_benchmark(
            "Simple SELECT", query, iterations, rls_enabled
        )

        status = "✓ PASS" if result.p99_latency_ms < 5.0 else "✗ FAIL"
        print(f"\n{status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Avg latency: {result.avg_latency_ms:.3f}ms")
        print(f"  P50 latency: {result.p50_latency_ms:.3f}ms")
        print(f"  P95 latency: {result.p95_latency_ms:.3f}ms")
        print(
            f"  P99 latency: {result.p99_latency_ms:.3f}ms "
            f"{'✓' if result.p99_latency_ms < 5.0 else '✗ (target: <5ms)'}"
        )

        return result

    async def benchmark_aggregation(
        self, iterations: int = 100, rls_enabled: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark 2: Aggregation query
        Target: P99 <50ms

        Args:
            iterations: Number of iterations
            rls_enabled: Whether RLS is enabled

        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*60}")
        print(f"Benchmark 2: Aggregation (RLS: {'ON' if rls_enabled else 'OFF'})")
        print(f"{'='*60}")

        tenant_id = random.choice(self.tenant_ids) if self.tenant_ids else uuid.uuid4()

        query = f"""
            SELECT
                symbol,
                COUNT(*) as trade_count,
                SUM(quantity) as total_quantity,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM test_trades
            WHERE tenant_id = '{tenant_id}'
            GROUP BY symbol
        """

        if rls_enabled:
            query = """
                SELECT
                    symbol,
                    COUNT(*) as trade_count,
                    SUM(quantity) as total_quantity,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price
                FROM test_trades
                GROUP BY symbol
            """

        result = await self.run_query_benchmark(
            "Aggregation", query, iterations, rls_enabled
        )

        status = "✓ PASS" if result.p99_latency_ms < 50.0 else "✗ FAIL"
        print(f"\n{status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Avg latency: {result.avg_latency_ms:.3f}ms")
        print(f"  P99 latency: {result.p99_latency_ms:.3f}ms "
              f"{'✓' if result.p99_latency_ms < 50.0 else '✗ (target: <50ms)'}")

        return result

    async def benchmark_time_range(
        self, iterations: int = 100, rls_enabled: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark 3: Time-range query
        Target: P99 <10ms

        Args:
            iterations: Number of iterations
            rls_enabled: Whether RLS is enabled

        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*60}")
        print(f"Benchmark 3: Time-Range Query (RLS: {'ON' if rls_enabled else 'OFF'})")
        print(f"{'='*60}")

        tenant_id = random.choice(self.tenant_ids) if self.tenant_ids else uuid.uuid4()

        query = f"""
            SELECT id, symbol, quantity, price, created_at
            FROM test_trades
            WHERE tenant_id = '{tenant_id}'
              AND created_at >= NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC
            LIMIT 1000
        """

        if rls_enabled:
            query = """
                SELECT id, symbol, quantity, price, created_at
                FROM test_trades
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                ORDER BY created_at DESC
                LIMIT 1000
            """

        result = await self.run_query_benchmark(
            "Time-Range Query", query, iterations, rls_enabled
        )

        status = "✓ PASS" if result.p99_latency_ms < 10.0 else "✗ FAIL"
        print(f"\n{status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Avg latency: {result.avg_latency_ms:.3f}ms")
        print(f"  P99 latency: {result.p99_latency_ms:.3f}ms "
              f"{'✓' if result.p99_latency_ms < 10.0 else '✗ (target: <10ms)'}")

        return result

    async def benchmark_join(
        self, iterations: int = 100, rls_enabled: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark 4: JOIN query
        Target: P99 <20ms

        Args:
            iterations: Number of iterations
            rls_enabled: Whether RLS is enabled

        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*60}")
        print(f"Benchmark 4: JOIN Query (RLS: {'ON' if rls_enabled else 'OFF'})")
        print(f"{'='*60}")

        # Create a second table for join test
        await self.conn.execute("DROP TABLE IF EXISTS test_positions CASCADE")
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_positions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                position_size DECIMAL(18, 8) NOT NULL
            )
        """
        )

        # Insert position data
        for tenant_id in self.tenant_ids[:5]:  # Only 5 tenants for join test
            await self.conn.executemany(
                """
                INSERT INTO test_positions (tenant_id, symbol, position_size)
                VALUES ($1, $2, $3)
            """,
                [
                    (tenant_id, "BTC_USDT", 1.5),
                    (tenant_id, "ETH_USDT", 10.0),
                    (tenant_id, "XRP_USDT", 1000.0),
                ],
            )

        # Enable RLS on positions table if needed
        if rls_enabled:
            await self.conn.execute(
                "ALTER TABLE test_positions ENABLE ROW LEVEL SECURITY"
            )
            await self.conn.execute(
                "DROP POLICY IF EXISTS tenant_isolation_positions ON test_positions"
            )
            await self.conn.execute(
                """
                CREATE POLICY tenant_isolation_positions ON test_positions
                FOR ALL
                USING (tenant_id = current_setting('app.current_tenant_id', true)::uuid)
            """
            )

        tenant_id = random.choice(self.tenant_ids[:5])

        query = f"""
            SELECT
                t.symbol,
                COUNT(*) as trade_count,
                SUM(t.quantity) as total_quantity,
                p.position_size
            FROM test_trades t
            INNER JOIN test_positions p ON t.symbol = p.symbol AND t.tenant_id = p.tenant_id
            WHERE t.tenant_id = '{tenant_id}'
            GROUP BY t.symbol, p.position_size
        """

        if rls_enabled:
            query = """
                SELECT
                    t.symbol,
                    COUNT(*) as trade_count,
                    SUM(t.quantity) as total_quantity,
                    p.position_size
                FROM test_trades t
                INNER JOIN test_positions p ON t.symbol = p.symbol
                GROUP BY t.symbol, p.position_size
            """

        result = await self.run_query_benchmark(
            "JOIN Query", query, iterations, rls_enabled
        )

        status = "✓ PASS" if result.p99_latency_ms < 20.0 else "✗ FAIL"
        print(f"\n{status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Avg latency: {result.avg_latency_ms:.3f}ms")
        print(f"  P99 latency: {result.p99_latency_ms:.3f}ms "
              f"{'✓' if result.p99_latency_ms < 20.0 else '✗ (target: <20ms)'}")

        return result

    async def run_all_benchmarks(self):
        """Run all benchmarks with and without RLS"""
        print("\n" + "=" * 60)
        print("SPIKE #150: PostgreSQL RLS Performance Benchmarking")
        print("=" * 60)

        await self.connect()

        try:
            # Setup test data
            await self.setup_test_table(num_tenants=10, rows_per_tenant=10000)

            # Baseline benchmarks (RLS disabled)
            print("\n" + "=" * 60)
            print("PHASE 1: Baseline Benchmarks (RLS DISABLED)")
            print("=" * 60)

            baseline_results = []
            baseline_results.append(
                await self.benchmark_simple_select(iterations=100, rls_enabled=False)
            )
            baseline_results.append(
                await self.benchmark_aggregation(iterations=100, rls_enabled=False)
            )
            baseline_results.append(
                await self.benchmark_time_range(iterations=100, rls_enabled=False)
            )
            baseline_results.append(
                await self.benchmark_join(iterations=100, rls_enabled=False)
            )

            # RLS benchmarks
            print("\n" + "=" * 60)
            print("PHASE 2: RLS Benchmarks (RLS ENABLED)")
            print("=" * 60)

            await self.enable_rls()

            rls_results = []
            rls_results.append(
                await self.benchmark_simple_select(iterations=100, rls_enabled=True)
            )
            rls_results.append(
                await self.benchmark_aggregation(iterations=100, rls_enabled=True)
            )
            rls_results.append(
                await self.benchmark_time_range(iterations=100, rls_enabled=True)
            )
            rls_results.append(
                await self.benchmark_join(iterations=100, rls_enabled=True)
            )

            # Store all results
            self.results = baseline_results + rls_results

            # Print summary
            self.print_summary(baseline_results, rls_results)

        finally:
            await self.disconnect()

    def print_summary(
        self, baseline_results: List[BenchmarkResult], rls_results: List[BenchmarkResult]
    ):
        """Print comparison summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)

        total_overhead = []

        for baseline, rls in zip(baseline_results, rls_results):
            overhead = (
                (rls.p99_latency_ms - baseline.p99_latency_ms)
                / baseline.p99_latency_ms
                * 100
            )
            total_overhead.append(overhead)

            print(f"\n{baseline.query_type}:")
            print(f"  Baseline P99: {baseline.p99_latency_ms:.3f}ms")
            print(f"  RLS P99:      {rls.p99_latency_ms:.3f}ms")
            print(f"  Overhead:     {overhead:+.1f}% {'✓' if overhead < 10 else '✗ (target: <10%)'}")

        avg_overhead = statistics.mean(total_overhead)

        print("\n" + "=" * 60)
        print(f"AVERAGE RLS OVERHEAD: {avg_overhead:+.1f}%")
        print("=" * 60)

        if avg_overhead < 10:
            print("✓ SUCCESS: RLS overhead is within acceptable range (<10%)")
            print("\nDECISION: ✅ PROCEED with PostgreSQL RLS for multi-tenant isolation")
        else:
            print(f"✗ FAIL: RLS overhead ({avg_overhead:.1f}%) exceeds 10% target")
            print("\nDECISION: ⚠️ REVIEW RLS strategy or optimize queries")

    def export_results(self, filename: str = "rls_benchmark_results.json"):
        """Export results to JSON"""
        results_dict = {
            "timestamp": time.time(),
            "database": self.database,
            "host": self.host,
            "benchmarks": [
                {
                    "query_type": r.query_type,
                    "rls_enabled": r.rls_enabled,
                    "iterations": r.iterations,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                }
                for r in self.results
            ],
        }

        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results exported to {filename}")


async def main():
    """Main entry point"""
    # Configure database connection
    benchmark = RLSBenchmark(
        host="localhost",
        port=5432,
        database="alphapulse",
        user="alphapulse",
        password="alphapulse",
    )

    try:
        await benchmark.run_all_benchmarks()
        benchmark.export_results()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n✗ Benchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
