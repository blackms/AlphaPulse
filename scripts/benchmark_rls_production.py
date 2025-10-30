#!/usr/bin/env python3
"""
PostgreSQL RLS Performance Benchmarking (Production Tables)

Validates that Row-Level Security adds <10% query overhead using actual production tables.

Usage:
    python scripts/benchmark_rls_production.py
"""

import asyncio
import statistics
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID

import asyncpg


class ProductionRLSBenchmark:
    """Benchmark PostgreSQL RLS performance on production tables."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        # Use existing tenant ID
        self.tenant_id = "00000000-0000-0000-0000-000000000001"

    async def setup(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=5, max_size=20)
        print(f"✓ Connected to database")

    async def teardown(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
        print("✓ Database connection closed")

    async def verify_rls_status(self):
        """Verify RLS is enabled on production tables."""
        async with self.pool.acquire() as conn:
            result = await conn.fetch("""
                SELECT tablename, rowsecurity
                FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename IN ('trades', 'positions', 'users', 'trading_accounts')
                ORDER BY tablename
            """)

            print("\n" + "="*80)
            print("RLS STATUS CHECK")
            print("="*80)
            for row in result:
                status = "✓ ENABLED" if row['rowsecurity'] else "✗ DISABLED"
                print(f"{row['tablename']:<20} {status}")
            print()

            all_enabled = all(row['rowsecurity'] for row in result)
            if not all_enabled:
                raise RuntimeError("RLS not enabled on all tables!")

    async def get_table_stats(self):
        """Get row counts for production tables."""
        async with self.pool.acquire() as conn:
            trades_count = await conn.fetchval("SELECT COUNT(*) FROM trades WHERE tenant_id = $1", UUID(self.tenant_id))
            positions_count = await conn.fetchval("SELECT COUNT(*) FROM positions WHERE tenant_id = $1", UUID(self.tenant_id))
            users_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE tenant_id = $1", UUID(self.tenant_id))

            print("="*80)
            print("PRODUCTION TABLE STATS (Tenant 1)")
            print("="*80)
            print(f"Trades:     {trades_count:,} rows")
            print(f"Positions:  {positions_count:,} rows")
            print(f"Users:      {users_count:,} rows")
            print()

            if trades_count < 100:
                print("⚠️  WARNING: Low row count. Results may not be representative.")
                print("   Consider running system to generate more trades first.")
            print()

    async def run_query_baseline(self, query: str, params: tuple, iterations: int = 100) -> List[float]:
        """Execute query with explicit tenant_id filter (baseline)."""
        times = []
        async with self.pool.acquire() as conn:
            # Warm up
            for _ in range(10):
                await conn.fetch(query, *params)

            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                await conn.fetch(query, *params)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

        return times

    async def run_query_with_rls(self, query: str, params: tuple, iterations: int = 100) -> List[float]:
        """Execute query with RLS (set session variable)."""
        times = []
        async with self.pool.acquire() as conn:
            # Set tenant context
            await conn.execute(f"SET LOCAL app.current_tenant_id = '{self.tenant_id}'")

            # Warm up
            for _ in range(10):
                await conn.fetch(query, *params)

            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                await conn.fetch(query, *params)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

        return times

    def calculate_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate statistics from timing data."""
        return {
            'min': min(times),
            'max': max(times),
            'mean': statistics.mean(times),
            'p50': statistics.median(times),
            'p95': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            'p99': statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
        }

    async def benchmark_scenario(
        self,
        name: str,
        query_baseline: str,
        query_rls: str,
        params_baseline: tuple,
        params_rls: tuple,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark a specific query scenario."""
        print(f"\nBenchmarking: {name}")
        print(f"Iterations: {iterations}")

        # Run baseline
        print("  Running baseline (explicit tenant_id filter)...")
        times_baseline = await self.run_query_baseline(query_baseline, params_baseline, iterations)

        # Run with RLS
        print("  Running with RLS (session variable)...")
        times_rls = await self.run_query_with_rls(query_rls, params_rls, iterations)

        # Calculate stats
        baseline_stats = self.calculate_stats(times_baseline)
        rls_stats = self.calculate_stats(times_rls)

        # Calculate overhead
        overhead = {}
        for metric in ['mean', 'p50', 'p95', 'p99']:
            baseline_val = baseline_stats[metric]
            rls_val = rls_stats[metric]
            if baseline_val > 0:
                overhead[metric] = ((rls_val - baseline_val) / baseline_val) * 100
            else:
                overhead[metric] = 0.0

        return {
            'name': name,
            'iterations': iterations,
            'baseline': baseline_stats,
            'rls': rls_stats,
            'overhead': overhead
        }

    async def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run all benchmark scenarios on production tables."""
        results = []
        tenant_uuid = UUID(self.tenant_id)

        # 1. Simple SELECT on trades (most common query)
        results.append(await self.benchmark_scenario(
            name="Trades: SELECT with LIMIT 100",
            query_baseline="SELECT * FROM trades WHERE tenant_id = $1 LIMIT 100",
            query_rls="SELECT * FROM trades LIMIT 100",
            params_baseline=(tenant_uuid,),
            params_rls=(),
            iterations=200
        ))

        # 2. Aggregation on trades
        results.append(await self.benchmark_scenario(
            name="Trades: GROUP BY symbol",
            query_baseline="SELECT symbol, COUNT(*), AVG(price) FROM trades WHERE tenant_id = $1 GROUP BY symbol",
            query_rls="SELECT symbol, COUNT(*), AVG(price) FROM trades GROUP BY symbol",
            params_baseline=(tenant_uuid,),
            params_rls=(),
            iterations=100
        ))

        # 3. Time-range query on trades
        cutoff_date = datetime.now() - timedelta(days=7)
        results.append(await self.benchmark_scenario(
            name="Trades: Time-range (7 days)",
            query_baseline="SELECT * FROM trades WHERE tenant_id = $1 AND executed_at >= $2 ORDER BY executed_at DESC LIMIT 100",
            query_rls="SELECT * FROM trades WHERE executed_at >= $1 ORDER BY executed_at DESC LIMIT 100",
            params_baseline=(tenant_uuid, cutoff_date),
            params_rls=(cutoff_date,),
            iterations=100
        ))

        # 4. Positions query
        results.append(await self.benchmark_scenario(
            name="Positions: SELECT ALL",
            query_baseline="SELECT * FROM positions WHERE tenant_id = $1",
            query_rls="SELECT * FROM positions",
            params_baseline=(tenant_uuid,),
            params_rls=(),
            iterations=200
        ))

        # 5. Users query
        results.append(await self.benchmark_scenario(
            name="Users: SELECT ALL",
            query_baseline="SELECT * FROM users WHERE tenant_id = $1",
            query_rls="SELECT * FROM users",
            params_baseline=(tenant_uuid,),
            params_rls=(),
            iterations=200
        ))

        return results

    def print_results(self, results: List[Dict[str, Any]]):
        """Print benchmark results in a readable format."""
        print("\n" + "="*90)
        print("BENCHMARK RESULTS")
        print("="*90)

        for result in results:
            print(f"\n{result['name']}")
            print("-" * 90)
            print(f"Iterations: {result['iterations']:,}")
            print()
            print(f"{'Metric':<10} {'Baseline (ms)':>16} {'RLS (ms)':>16} {'Overhead':>20}")
            print("-" * 90)

            for metric in ['mean', 'p50', 'p95', 'p99']:
                baseline_val = result['baseline'][metric]
                rls_val = result['rls'][metric]
                overhead_val = result['overhead'][metric]

                # Color code overhead
                if overhead_val < 5:
                    status = "✓ Excellent"
                elif overhead_val < 10:
                    status = "✓ Good"
                elif overhead_val < 15:
                    status = "⚠ Acceptable"
                else:
                    status = "✗ High"

                print(f"{metric.upper():<10} {baseline_val:>15.3f}  {rls_val:>15.3f}  {status} {overhead_val:>11.2f}%")

        # Summary
        print("\n" + "="*90)
        print("DECISION ANALYSIS")
        print("="*90)

        avg_p99_overhead = statistics.mean([r['overhead']['p99'] for r in results])
        max_p99_overhead = max([r['overhead']['p99'] for r in results])
        worst_scenario = max(results, key=lambda r: r['overhead']['p99'])

        print(f"\nAverage P99 overhead:  {avg_p99_overhead:>6.2f}%")
        print(f"Maximum P99 overhead:  {max_p99_overhead:>6.2f}% ({worst_scenario['name']})")
        print()

        # Go/No-Go decision
        if max_p99_overhead < 10:
            decision = "GO"
            status = "✓ PASS"
            recommendation = "PROCEED with RLS approach as planned"
            confidence = "HIGH"
        elif max_p99_overhead < 15:
            decision = "GO (with monitoring)"
            status = "⚠ CONDITIONAL PASS"
            recommendation = "PROCEED with RLS, add performance monitoring"
            confidence = "MEDIUM"
        else:
            decision = "NO-GO"
            status = "✗ FAIL"
            recommendation = "CONSIDER alternative strategies (partitioning, dedicated schemas)"
            confidence = "LOW"

        print("="*90)
        print(f"DECISION: {decision}")
        print("="*90)
        print(f"Status:         {status}")
        print(f"Recommendation: {recommendation}")
        print(f"Confidence:     {confidence}")
        print()

        if decision.startswith("GO"):
            print("✓ RLS overhead meets <10% target (ADR-001 requirement)")
            print("  Safe to proceed with multi-tenant RLS strategy")
        else:
            print("✗ RLS overhead exceeds 10% target")
            print("  Recommend revisiting data isolation strategy")

    async def run(self):
        """Execute full benchmark suite."""
        try:
            await self.setup()
            await self.verify_rls_status()
            await self.get_table_stats()

            print("="*90)
            print("BENCHMARK PHASE")
            print("="*90)
            results = await self.run_all_benchmarks()

            self.print_results(results)

            return results

        finally:
            await self.teardown()


async def main():
    database_url = "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse"

    benchmark = ProductionRLSBenchmark(database_url)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
