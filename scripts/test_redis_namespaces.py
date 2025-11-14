#!/usr/bin/env python3
"""
Redis Cluster Namespace Isolation and Performance Testing

This script validates the namespace isolation approach for multi-tenant caching
and measures performance impact of rolling counters.

SPIKE #152: Redis Cluster Namespace Isolation
Time Box: 2 days
Success Criteria:
- Namespace isolation verified (100 simulated tenants)
- Rolling counter overhead <1ms per write
- LRU eviction completes in <100ms
- Cache hit rate >80% for shared market data
"""

import asyncio
import random
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import redis.asyncio as redis
from redis.asyncio import Redis, RedisCluster
import statistics
import json


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""

    test_name: str
    success: bool
    metrics: Dict[str, float]
    notes: str


class RedisNamespaceTester:
    """Comprehensive Redis namespace isolation and performance testing"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the tester with Redis connection.

        Args:
            redis_url: Redis connection URL (cluster or standalone)
        """
        self.redis_url = redis_url
        self.client: Redis = None
        self.results: List[BenchmarkResult] = []

    async def connect(self):
        """Establish connection to Redis"""
        # Try cluster connection first, fall back to standalone
        try:
            self.client = await redis.from_url(
                self.redis_url, decode_responses=True, encoding="utf-8"
            )
            await self.client.ping()
            print(f"✓ Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"✗ Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            print("✓ Disconnected from Redis")

    async def test_namespace_isolation(self, num_tenants: int = 100) -> BenchmarkResult:
        """
        Test 1: Namespace Isolation

        Verify that 100 tenants can write to their namespaces simultaneously
        without any key collisions.

        Args:
            num_tenants: Number of tenants to simulate

        Returns:
            BenchmarkResult with isolation verification
        """
        print(f"\n{'='*60}")
        print("Test 1: Namespace Isolation")
        print(f"{'='*60}")

        test_key = "test_key"
        start_time = time.time()

        # Write to all tenants simultaneously
        print(f"Writing to {num_tenants} tenant namespaces concurrently...")
        tasks = []
        for tenant_id in range(1, num_tenants + 1):
            tasks.append(
                self.client.set(
                    f"tenant:{tenant_id}:{test_key}", f"value_{tenant_id}"
                )
            )

        await asyncio.gather(*tasks)
        write_duration = time.time() - start_time

        # Verify no collisions
        print(f"Verifying data integrity across {num_tenants} tenants...")
        collisions = 0
        missing = 0

        for tenant_id in range(1, num_tenants + 1):
            value = await self.client.get(f"tenant:{tenant_id}:{test_key}")
            expected = f"value_{tenant_id}"

            if value is None:
                missing += 1
            elif value != expected:
                collisions += 1
                print(
                    f"  ✗ Collision detected! Tenant {tenant_id}: "
                    f"expected '{expected}', got '{value}'"
                )

        # Cleanup
        cleanup_tasks = [
            self.client.delete(f"tenant:{tid}:{test_key}")
            for tid in range(1, num_tenants + 1)
        ]
        await asyncio.gather(*cleanup_tasks)

        # Results
        success = collisions == 0 and missing == 0
        status = "✓ PASS" if success else "✗ FAIL"

        print(f"\n{status}")
        print(f"  Tenants tested: {num_tenants}")
        print(f"  Write duration: {write_duration:.3f}s")
        print(f"  Collisions: {collisions}")
        print(f"  Missing keys: {missing}")
        print(f"  Isolation: {'100%' if success else f'{(1-(collisions+missing)/num_tenants)*100:.1f}%'}")

        return BenchmarkResult(
            test_name="Namespace Isolation",
            success=success,
            metrics={
                "tenants_tested": num_tenants,
                "write_duration_seconds": write_duration,
                "collisions": collisions,
                "missing_keys": missing,
                "isolation_percentage": (
                    100.0 if success else (1 - (collisions + missing) / num_tenants) * 100
                ),
            },
            notes=f"Tested {num_tenants} tenants with concurrent writes",
        )

    async def track_usage(
        self, tenant_id: str, key: str, payload_size: int, timestamp: float
    ):
        """
        Track usage with rolling counter (mimics production usage tracking).

        Args:
            tenant_id: Tenant identifier
            key: Cache key
            payload_size: Size of data in bytes
            timestamp: Timestamp for LRU tracking
        """
        pipeline = self.client.pipeline()
        pipeline.incrby(f"meta:tenant:{tenant_id}:usage_bytes", payload_size)
        pipeline.zadd(
            f"meta:tenant:{tenant_id}:lru", {f"tenant:{tenant_id}:{key}": timestamp}
        )
        await pipeline.execute()

    async def test_rolling_counter_performance(
        self, num_writes: int = 10000, num_tenants: int = 100
    ) -> BenchmarkResult:
        """
        Test 2: Rolling Counter Performance

        Measure overhead of tracking cache usage with rolling counters.
        Target: <1ms per write

        Args:
            num_writes: Number of write operations to benchmark
            num_tenants: Number of tenants to distribute writes across

        Returns:
            BenchmarkResult with latency metrics
        """
        print(f"\n{'='*60}")
        print("Test 2: Rolling Counter Performance")
        print(f"{'='*60}")

        latencies = []
        start_time = time.time()

        print(f"Executing {num_writes:,} writes across {num_tenants} tenants...")

        for i in range(num_writes):
            tenant_id = f"tenant_{random.randint(1, num_tenants)}"
            payload_size = random.randint(100, 10000)  # 100B - 10KB
            timestamp = time.time()

            op_start = time.time()
            await self.track_usage(tenant_id, f"key_{i}", payload_size, timestamp)
            latency_ms = (time.time() - op_start) * 1000
            latencies.append(latency_ms)

            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1:,}/{num_writes:,} writes")

        total_duration = time.time() - start_time
        throughput = num_writes / total_duration

        # Calculate statistics
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        avg_latency = statistics.mean(latencies)

        # Cleanup
        print("Cleaning up test data...")
        cleanup_keys = []
        for tenant_id in range(1, num_tenants + 1):
            cleanup_keys.extend(
                [
                    f"meta:tenant_tenant_{tenant_id}:usage_bytes",
                    f"meta:tenant_tenant_{tenant_id}:lru",
                ]
            )

        if cleanup_keys:
            await self.client.delete(*cleanup_keys)

        # Results
        success = p99 < 1.0  # Target: <1ms at P99
        status = "✓ PASS" if success else "✗ FAIL"

        print(f"\n{status}")
        print(f"  Total writes: {num_writes:,}")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} writes/sec")
        print(f"  Latency (avg): {avg_latency:.3f}ms")
        print(f"  Latency (P50): {p50:.3f}ms")
        print(f"  Latency (P95): {p95:.3f}ms")
        print(f"  Latency (P99): {p99:.3f}ms {'✓' if p99 < 1.0 else '✗ (target: <1ms)'}")

        return BenchmarkResult(
            test_name="Rolling Counter Performance",
            success=success,
            metrics={
                "total_writes": num_writes,
                "duration_seconds": total_duration,
                "throughput_writes_per_sec": throughput,
                "latency_avg_ms": avg_latency,
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
            },
            notes=f"Tested {num_writes:,} writes across {num_tenants} tenants",
        )

    async def evict_tenant_keys(
        self, tenant_id: str, target_size: int
    ) -> Tuple[int, float]:
        """
        Evict keys for a tenant using LRU sorted set.

        Args:
            tenant_id: Tenant to evict keys for
            target_size: Target size in bytes

        Returns:
            Tuple of (keys_evicted, duration_seconds)
        """
        lru_key = f"meta:tenant:{tenant_id}:lru"
        usage_key = f"meta:tenant:{tenant_id}:usage_bytes"

        start_time = time.time()

        # Get current usage
        current_usage = int(await self.client.get(usage_key) or 0)

        if current_usage <= target_size:
            return 0, 0.0

        # Get oldest keys from sorted set
        keys_to_evict = await self.client.zrange(lru_key, 0, -1)

        keys_evicted = 0
        bytes_freed = 0

        for key in keys_to_evict:
            if current_usage - bytes_freed <= target_size:
                break

            # Get key size and delete
            key_size = await self.client.memory_usage(key) or 1000  # Estimate 1KB
            if await self.client.delete(key):
                keys_evicted += 1
                bytes_freed += key_size

                # Remove from LRU tracking
                await self.client.zrem(lru_key, key)

        # Update usage counter
        if bytes_freed > 0:
            await self.client.decrby(usage_key, bytes_freed)

        duration = time.time() - start_time
        return keys_evicted, duration

    async def test_lru_eviction_performance(
        self, num_keys: int = 1000
    ) -> BenchmarkResult:
        """
        Test 3: LRU Eviction Performance

        Measure time to evict keys using sorted set LRU tracking.
        Target: <100ms

        Args:
            num_keys: Number of keys to create before eviction

        Returns:
            BenchmarkResult with eviction metrics
        """
        print(f"\n{'='*60}")
        print("Test 3: LRU Eviction Performance")
        print(f"{'='*60}")

        tenant_id = "eviction_test_tenant"
        key_size = 10000  # 10KB per key

        # Fill cache
        print(f"Creating {num_keys:,} keys ({key_size:,} bytes each)...")
        fill_start = time.time()

        for i in range(num_keys):
            key = f"tenant:{tenant_id}:key_{i}"
            await self.client.set(key, "x" * key_size)
            await self.track_usage(tenant_id, f"key_{i}", key_size, time.time() - i)

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1:,}/{num_keys:,} keys")

        fill_duration = time.time() - fill_start
        total_bytes = num_keys * key_size

        print(
            f"✓ Created {num_keys:,} keys ({total_bytes/1_000_000:.1f} MB) in {fill_duration:.2f}s"
        )

        # Trigger eviction (evict to 50% of size)
        target_size = total_bytes // 2
        print(f"\nTriggering eviction to {target_size/1_000_000:.1f} MB...")

        eviction_start = time.time()
        keys_evicted, eviction_duration = await self.evict_tenant_keys(
            tenant_id, target_size
        )
        eviction_ms = eviction_duration * 1000

        # Verify remaining keys
        remaining_usage = int(
            await self.client.get(f"meta:tenant:{tenant_id}:usage_bytes") or 0
        )

        # Cleanup
        print("Cleaning up test data...")
        cleanup_keys = [f"tenant:{tenant_id}:key_{i}" for i in range(num_keys)]
        cleanup_keys.extend(
            [
                f"meta:tenant:{tenant_id}:usage_bytes",
                f"meta:tenant:{tenant_id}:lru",
            ]
        )
        await self.client.delete(*cleanup_keys)

        # Results
        success = eviction_ms < 100  # Target: <100ms
        status = "✓ PASS" if success else "✗ FAIL"

        print(f"\n{status}")
        print(f"  Keys created: {num_keys:,}")
        print(f"  Keys evicted: {keys_evicted:,}")
        print(f"  Eviction time: {eviction_ms:.2f}ms {'✓' if success else '✗ (target: <100ms)'}")
        print(f"  Eviction rate: {keys_evicted/(eviction_duration or 0.001):.0f} keys/sec")
        print(f"  Remaining usage: {remaining_usage/1_000_000:.1f} MB")

        return BenchmarkResult(
            test_name="LRU Eviction Performance",
            success=success,
            metrics={
                "keys_created": num_keys,
                "keys_evicted": keys_evicted,
                "eviction_time_ms": eviction_ms,
                "eviction_rate_keys_per_sec": keys_evicted / (eviction_duration or 0.001),
                "remaining_usage_bytes": remaining_usage,
            },
            notes=f"Evicted {keys_evicted:,} keys in {eviction_ms:.2f}ms",
        )

    async def test_cache_hit_rate(
        self, num_requests: int = 10000, num_tenants: int = 100
    ) -> BenchmarkResult:
        """
        Test 4: Cache Hit Rate for Shared Market Data

        Simulate multiple tenants requesting the same market data.
        Target: >80% cache hit rate

        Args:
            num_requests: Number of cache requests to simulate
            num_tenants: Number of tenants to simulate

        Returns:
            BenchmarkResult with hit rate metrics
        """
        print(f"\n{'='*60}")
        print("Test 4: Cache Hit Rate (Shared Market Data)")
        print(f"{'='*60}")

        symbols = ["BTC_USDT", "ETH_USDT", "XRP_USDT", "SOL_USDT", "ADA_USDT"]
        hits = 0
        misses = 0

        print(f"Simulating {num_requests:,} cache requests across {num_tenants} tenants...")

        for i in range(num_requests):
            tenant_id = f"tenant_{random.randint(1, num_tenants)}"
            symbol = random.choice(symbols)

            # Shared cache key (all tenants use same market data)
            cache_key = f"shared:market:binance:{symbol}:1m:ohlcv"

            # Try to get from cache
            data = await self.client.get(cache_key)

            if data:
                hits += 1
            else:
                misses += 1
                # Simulate fetch from exchange
                mock_data = json.dumps(
                    {
                        "symbol": symbol,
                        "timestamp": time.time(),
                        "open": 50000.0,
                        "high": 51000.0,
                        "low": 49000.0,
                        "close": 50500.0,
                        "volume": 1234.56,
                    }
                )
                await self.client.set(cache_key, mock_data, ex=60)  # 60s TTL

            # Progress indicator
            if (i + 1) % 1000 == 0:
                current_hit_rate = hits / (hits + misses) * 100
                print(f"  Progress: {i+1:,}/{num_requests:,} (hit rate: {current_hit_rate:.1f}%)")

        # Cleanup
        print("Cleaning up test data...")
        cleanup_keys = [
            f"shared:market:binance:{symbol}:1m:ohlcv" for symbol in symbols
        ]
        await self.client.delete(*cleanup_keys)

        # Results
        hit_rate = hits / (hits + misses) * 100
        success = hit_rate > 80  # Target: >80%
        status = "✓ PASS" if success else "✗ FAIL"

        print(f"\n{status}")
        print(f"  Total requests: {num_requests:,}")
        print(f"  Cache hits: {hits:,}")
        print(f"  Cache misses: {misses:,}")
        print(f"  Hit rate: {hit_rate:.1f}% {'✓' if success else '✗ (target: >80%)'}")

        return BenchmarkResult(
            test_name="Cache Hit Rate",
            success=success,
            metrics={
                "total_requests": num_requests,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate_percentage": hit_rate,
            },
            notes=f"Simulated {num_requests:,} requests across {num_tenants} tenants",
        )

    async def run_all_tests(self):
        """Run all benchmark tests and generate summary"""
        print("\n" + "=" * 60)
        print("SPIKE #152: Redis Cluster Namespace Isolation")
        print("=" * 60)

        await self.connect()

        try:
            # Run all tests
            self.results.append(await self.test_namespace_isolation(num_tenants=100))
            self.results.append(
                await self.test_rolling_counter_performance(
                    num_writes=10000, num_tenants=100
                )
            )
            self.results.append(await self.test_lru_eviction_performance(num_keys=1000))
            self.results.append(
                await self.test_cache_hit_rate(num_requests=10000, num_tenants=100)
            )

        finally:
            await self.disconnect()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary with pass/fail status"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        all_passed = all(r.success for r in self.results)

        for result in self.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"\n{result.test_name}: {status}")
            print(f"  {result.notes}")

            # Print key metrics
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  - {metric}: {value:.2f}")
                else:
                    print(f"  - {metric}: {value}")

        print("\n" + "=" * 60)
        overall_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
        print(f"OVERALL: {overall_status}")
        print("=" * 60)

        # Decision
        print("\nDECISION:")
        if all_passed:
            print("✓ Proceed with rolling counter approach")
            print("  - Namespace isolation verified (100% success)")
            print("  - Performance targets met across all tests")
            print("  - Ready for production deployment")
        else:
            print("✗ Performance targets not met")
            print("  Recommendations:")
            print("  - Review failed tests and optimize implementation")
            print("  - Consider alternative approaches for failing tests")
            print("  - Re-run tests after optimization")

    def export_results(self, filename: str = "redis_performance_results.json"):
        """Export results to JSON file"""
        results_dict = {
            "timestamp": time.time(),
            "redis_url": self.redis_url,
            "tests": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "metrics": r.metrics,
                    "notes": r.notes,
                }
                for r in self.results
            ],
            "overall_success": all(r.success for r in self.results),
        }

        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results exported to {filename}")


async def main():
    """Main entry point"""
    # Configure Redis connection
    redis_url = "redis://localhost:6379"  # Change for cluster: redis://localhost:7000

    tester = RedisNamespaceTester(redis_url=redis_url)

    try:
        await tester.run_all_tests()
        tester.export_results()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
