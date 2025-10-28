#!/usr/bin/env python3
"""
PostgreSQL RLS Performance Benchmarking

Validates that Row-Level Security adds <10% query overhead.

Usage:
    python scripts/benchmark_rls.py --database alphapulse_test --tenants 10 --rows 100000
"""

import argparse
import asyncio
import statistics
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4

import asyncpg


class RLSBenchmark:
    """Benchmark PostgreSQL RLS performance."""

    def __init__(self, database_url: str, num_tenants: int = 10, num_rows: int = 100000):
        self.database_url = database_url
        self.num_tenants = num_tenants
        self.num_rows = num_rows
        self.tenant_ids = [uuid4() for _ in range(num_tenants)]
        self.pool = None

    async def setup(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=5, max_size=20)
        print(f"✓ Connected to database: {self.database_url}")

    async def teardown(self):
        """Close database connection pool."""
        await self.pool.close()
        print("✓ Database connection closed")

    async def create_test_schema(self):
        """Create test tables for benchmarking."""
        async with self.pool.acquire() as conn:
            # Drop existing test tables
            await conn.execute("DROP TABLE IF EXISTS test_positions CASCADE")
            await conn.execute("DROP TABLE IF EXISTS test_trades CASCADE")

            # Create test_trades table
            await conn.execute("""
                CREATE TABLE test_trades (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    price DECIMAL(18, 8) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)

            # Create test_positions table (for JOIN tests)
            await conn.execute("""
                CREATE TABLE test_positions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    avg_price DECIMAL(18, 8) NOT NULL,
                    current_value DECIMAL(18, 8) NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)

            # Create composite indexes (critical for RLS performance)
            await conn.execute("""
                CREATE INDEX idx_trades_tenant_id
                ON test_trades(tenant_id, id)
            """)
            await conn.execute("""
                CREATE INDEX idx_trades_tenant_created
                ON test_trades(tenant_id, created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX idx_positions_tenant_id
                ON test_positions(tenant_id, id)
            """)

            print(f"✓ Created test schema with indexes")

    async def populate_test_data(self):
        """Insert test data across multiple tenants."""
        symbols = ['BTC_USDT', 'ETH_USDT', 'XRP_USDT', 'SOL_USDT', 'ADA_USDT']
        sides = ['BUY', 'SELL']

        print(f"Inserting {self.num_rows:,} test rows...")
        start = time.time()

        async with self.pool.acquire() as conn:
            # Batch insert trades
            trades_data = []
            for i in range(self.num_rows):
                tenant_id = self.tenant_ids[i % self.num_tenants]
                symbol = symbols[i % len(symbols)]
                trades_data.append((
                    tenant_id,
                    symbol,
                    float(1 + (i % 100)),  # quantity
                    float(1000 + (i % 50000)),  # price
                    sides[i % 2],
                    datetime.now() - timedelta(days=i % 365)
                ))

            await conn.executemany("""
                INSERT INTO test_trades
                (tenant_id, symbol, quantity, price, side, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, trades_data)

            # Insert positions (for JOIN tests)
            positions_data = []
            for tenant_id in self.tenant_ids:
                for symbol in symbols:
                    positions_data.append((
                        tenant_id,
                        symbol,
                        float(100),  # quantity
                        float(30000),  # avg_price
                        float(3000000)  # current_value
                    ))

            await conn.executemany("""
                INSERT INTO test_positions
                (tenant_id, symbol, quantity, avg_price, current_value)
                VALUES ($1, $2, $3, $4, $5)
            """, positions_data)

        elapsed = time.time() - start
        print(f"✓ Inserted {self.num_rows:,} trades + {len(positions_data)} positions in {elapsed:.2f}s")

    async def enable_rls(self):
        """Enable Row-Level Security on test tables."""
        async with self.pool.acquire() as conn:
            # Enable RLS
            await conn.execute("ALTER TABLE test_trades ENABLE ROW LEVEL SECURITY")
            await conn.execute("ALTER TABLE test_positions ENABLE ROW LEVEL SECURITY")

            # Create RLS policies
            await conn.execute("""
                CREATE POLICY tenant_isolation_trades ON test_trades
                USING (tenant_id = current_setting('app.current_tenant_id')::uuid)
            """)
            await conn.execute("""
                CREATE POLICY tenant_isolation_positions ON test_positions
                USING (tenant_id = current_setting('app.current_tenant_id')::uuid)
            """)

            # Grant permissions
            await conn.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON test_trades TO CURRENT_USER")
            await conn.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON test_positions TO CURRENT_USER")

        print("✓ Enabled RLS policies")

    async def run_query_without_rls(self, query: str, params: tuple) -> float:
        """Execute query without RLS (baseline)."""
        async with self.pool.acquire() as conn:
            start = time.time()
            await conn.fetch(query, *params)
            return time.time() - start

    async def run_query_with_rls(self, query: str, tenant_id: Any, params: tuple) -> float:
        """Execute query with RLS (set session variable)."""
        async with self.pool.acquire() as conn:
            # Set tenant context
            await conn.execute(f"SET LOCAL app.current_tenant_id = '{tenant_id}'")

            start = time.time()
            await conn.fetch(query, *params)
            return time.time() - start

    async def benchmark_query(
        self,
        name: str,
        query_without_rls: str,
        query_with_rls: str,
        params: tuple,
        iterations: int = 1000
    ) -> Dict[str, Any]:
        """Benchmark a query with and without RLS."""
        print(f"\nBenchmarking: {name}")
        print(f"Iterations: {iterations}")

        # Warm up
        tenant_id = self.tenant_ids[0]
        for _ in range(10):
            await self.run_query_without_rls(query_without_rls, params)
            await self.run_query_with_rls(query_with_rls, tenant_id, ())

        # Benchmark without RLS
        print("  Running baseline (without RLS)...")
        times_without_rls = []
        for i in range(iterations):
            tenant_id = self.tenant_ids[i % self.num_tenants]
            elapsed = await self.run_query_without_rls(
                query_without_rls,
                (tenant_id,) + params
            )
            times_without_rls.append(elapsed * 1000)  # Convert to ms

        # Benchmark with RLS
        print("  Running with RLS...")
        times_with_rls = []
        for i in range(iterations):
            tenant_id = self.tenant_ids[i % self.num_tenants]
            elapsed = await self.run_query_with_rls(
                query_with_rls,
                tenant_id,
                params
            )
            times_with_rls.append(elapsed * 1000)  # Convert to ms

        # Calculate statistics
        baseline = {
            'min': min(times_without_rls),
            'max': max(times_without_rls),
            'mean': statistics.mean(times_without_rls),
            'p50': statistics.median(times_without_rls),
            'p95': statistics.quantiles(times_without_rls, n=20)[18],  # 95th percentile
            'p99': statistics.quantiles(times_without_rls, n=100)[98],  # 99th percentile
        }

        rls = {
            'min': min(times_with_rls),
            'max': max(times_with_rls),
            'mean': statistics.mean(times_with_rls),
            'p50': statistics.median(times_with_rls),
            'p95': statistics.quantiles(times_with_rls, n=20)[18],
            'p99': statistics.quantiles(times_with_rls, n=100)[98],
        }

        overhead = {
            'mean': ((rls['mean'] - baseline['mean']) / baseline['mean']) * 100,
            'p50': ((rls['p50'] - baseline['p50']) / baseline['p50']) * 100,
            'p95': ((rls['p95'] - baseline['p95']) / baseline['p95']) * 100,
            'p99': ((rls['p99'] - baseline['p99']) / baseline['p99']) * 100,
        }

        return {
            'name': name,
            'iterations': iterations,
            'baseline': baseline,
            'rls': rls,
            'overhead': overhead
        }

    async def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run all benchmark scenarios."""
        results = []

        # 1. Simple SELECT (single tenant, LIMIT 100)
        results.append(await self.benchmark_query(
            name="Simple SELECT (LIMIT 100)",
            query_without_rls="""
                SELECT * FROM test_trades
                WHERE tenant_id = $1
                LIMIT 100
            """,
            query_with_rls="""
                SELECT * FROM test_trades
                LIMIT 100
            """,
            params=(),
            iterations=1000
        ))

        # 2. Aggregation (single tenant, GROUP BY)
        results.append(await self.benchmark_query(
            name="Aggregation (GROUP BY symbol)",
            query_without_rls="""
                SELECT symbol, COUNT(*), AVG(price), SUM(quantity)
                FROM test_trades
                WHERE tenant_id = $1
                GROUP BY symbol
            """,
            query_with_rls="""
                SELECT symbol, COUNT(*), AVG(price), SUM(quantity)
                FROM test_trades
                GROUP BY symbol
            """,
            params=(),
            iterations=500
        ))

        # 3. JOIN (multi-table)
        results.append(await self.benchmark_query(
            name="JOIN (trades + positions)",
            query_without_rls="""
                SELECT t.id, t.symbol, t.quantity, p.current_value
                FROM test_trades t
                JOIN test_positions p ON t.symbol = p.symbol AND t.tenant_id = p.tenant_id
                WHERE t.tenant_id = $1
                LIMIT 100
            """,
            query_with_rls="""
                SELECT t.id, t.symbol, t.quantity, p.current_value
                FROM test_trades t
                JOIN test_positions p ON t.symbol = p.symbol
                LIMIT 100
            """,
            params=(),
            iterations=500
        ))

        # 4. Time-range query (7 days, ORDER BY)
        results.append(await self.benchmark_query(
            name="Time-range query (7 days)",
            query_without_rls="""
                SELECT * FROM test_trades
                WHERE tenant_id = $1
                  AND created_at >= $2
                ORDER BY created_at DESC
                LIMIT 1000
            """,
            query_with_rls="""
                SELECT * FROM test_trades
                WHERE created_at >= $1
                ORDER BY created_at DESC
                LIMIT 1000
            """,
            params=(datetime.now() - timedelta(days=7),),
            iterations=500
        ))

        return results

    def print_results(self, results: List[Dict[str, Any]]):
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)

        for result in results:
            print(f"\n{result['name']}")
            print("-" * 80)
            print(f"Iterations: {result['iterations']:,}")
            print()
            print(f"{'Metric':<10} {'Baseline (ms)':>15} {'RLS (ms)':>15} {'Overhead (%)':>15}")
            print("-" * 80)

            for metric in ['mean', 'p50', 'p95', 'p99']:
                baseline_val = result['baseline'][metric]
                rls_val = result['rls'][metric]
                overhead_val = result['overhead'][metric]

                # Color code overhead (green if <10%, yellow if 10-20%, red if >20%)
                overhead_color = "✓" if overhead_val < 10 else "⚠" if overhead_val < 20 else "✗"

                print(f"{metric.upper():<10} {baseline_val:>14.2f}  {rls_val:>14.2f}  {overhead_color} {overhead_val:>12.2f}%")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        avg_overhead = statistics.mean([r['overhead']['p99'] for r in results])
        max_overhead = max([r['overhead']['p99'] for r in results])

        print(f"Average P99 overhead: {avg_overhead:.2f}%")
        print(f"Maximum P99 overhead: {max_overhead:.2f}%")
        print()

        if max_overhead < 10:
            print("✓ PASS: RLS overhead <10% (target met)")
            print("  Decision: PROCEED with RLS approach")
        elif max_overhead < 20:
            print("⚠ WARNING: RLS overhead 10-20% (above target but acceptable)")
            print("  Decision: PROCEED with RLS, monitor in production")
        else:
            print("✗ FAIL: RLS overhead >20% (unacceptable)")
            print("  Decision: Consider partitioning or dedicated schemas")

    async def run(self):
        """Execute full benchmark suite."""
        try:
            await self.setup()

            # Setup phase
            print("\n" + "="*80)
            print("SETUP PHASE")
            print("="*80)
            await self.create_test_schema()
            await self.populate_test_data()
            await self.enable_rls()

            # Benchmark phase
            print("\n" + "="*80)
            print("BENCHMARK PHASE")
            print("="*80)
            results = await self.run_all_benchmarks()

            # Results
            self.print_results(results)

        finally:
            await self.teardown()


async def main():
    parser = argparse.ArgumentParser(description="Benchmark PostgreSQL RLS performance")
    parser.add_argument(
        "--database",
        default="postgresql://localhost/alphapulse_test",
        help="Database URL (default: postgresql://localhost/alphapulse_test)"
    )
    parser.add_argument(
        "--tenants",
        type=int,
        default=10,
        help="Number of tenants to simulate (default: 10)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100000,
        help="Number of rows to insert (default: 100,000)"
    )

    args = parser.parse_args()

    benchmark = RLSBenchmark(
        database_url=args.database,
        num_tenants=args.tenants,
        num_rows=args.rows
    )

    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
