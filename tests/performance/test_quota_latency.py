"""
Performance tests for QuotaEnforcementMiddleware - RED Phase (Story 4.3).

Tests performance targets:
- p99 latency < 10ms
- Throughput >= 1000 req/s per tenant
- Cache hit rate > 90%

These tests should FAIL until GREEN phase implementation meets performance targets.
"""

import pytest
import time
import asyncio
import statistics
from uuid import UUID
from httpx import AsyncClient

# These imports will fail until GREEN phase
from alpha_pulse.middleware.quota_enforcement import QuotaEnforcementMiddleware


TEST_TENANT = UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
async def setup_perf_test(setup_test_quota, redis_client):
    """Setup for performance testing."""
    # Create tenant with large quota (avoid rejections during perf test)
    await setup_test_quota(TEST_TENANT, quota_mb=10000)

    # Warm up cache
    quota_key = f"quota:cache:{TEST_TENANT}:quota_mb"
    await redis_client.setex(quota_key, 300, "10000")

    yield

    # Cleanup
    await redis_client.flushdb()


class TestLatencyTarget:
    """Test latency performance targets."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_latency_within_target(
        self,
        test_app,
        setup_perf_test
    ):
        """
        AC-5: Test p99 latency < 10ms target.

        GIVEN: Middleware with warm cache
        WHEN: 1000 sequential requests
        THEN: p99 latency < 10ms
        """
        latencies = []

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 1000 requests with latency measurement
            for _ in range(1000):
                start = time.perf_counter()

                response = await client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT)}
                )

                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

                assert response.status_code == 200

            # Calculate percentiles
            p50 = statistics.quantiles(latencies, n=100)[49]
            p95 = statistics.quantiles(latencies, n=100)[94]
            p99 = statistics.quantiles(latencies, n=100)[98]

            # Print metrics for analysis
            print(f"\nLatency Metrics:")
            print(f"  p50: {p50:.2f}ms")
            print(f"  p95: {p95:.2f}ms")
            print(f"  p99: {p99:.2f}ms")
            print(f"  Mean: {statistics.mean(latencies):.2f}ms")

            # Assert performance targets
            assert p50 < 3, f"p50 latency {p50:.2f}ms exceeds 3ms target"
            assert p95 < 8, f"p95 latency {p95:.2f}ms exceeds 8ms target"
            assert p99 < 10, f"p99 latency {p99:.2f}ms exceeds 10ms target"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_hit_latency(
        self,
        test_app,
        setup_perf_test,
        redis_client
    ):
        """
        Test latency with cache hits (fast path).

        GIVEN: Quota cached in Redis
        WHEN: Quota check is performed
        THEN: Latency < 3ms (cache hit)
        """
        # Warm up cache
        quota_key = f"quota:cache:{TEST_TENANT}:quota_mb"
        await redis_client.setex(quota_key, 300, "10000")

        latencies = []

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 100 requests (all should hit cache)
            for _ in range(100):
                start = time.perf_counter()

                response = await client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT)}
                )

                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            p99 = statistics.quantiles(latencies, n=100)[98]

            print(f"\nCache Hit Latency:")
            print(f"  Mean: {statistics.mean(latencies):.2f}ms")
            print(f"  p99: {p99:.2f}ms")

            # Cache hits should be very fast
            assert p99 < 3, f"Cache hit p99 {p99:.2f}ms exceeds 3ms target"


class TestThroughput:
    """Test throughput performance targets."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_1000_req_per_sec(
        self,
        test_app,
        setup_perf_test
    ):
        """
        Test throughput >= 1000 req/s per tenant.

        GIVEN: Middleware with warm cache
        WHEN: 1000 concurrent requests
        THEN: All complete within 1 second
        """
        start_time = time.perf_counter()

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 1000 concurrent requests
            tasks = [
                client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT)}
                )
                for _ in range(1000)
            ]

            responses = await asyncio.gather(*tasks)

            elapsed_time = time.perf_counter() - start_time

            # Assert all succeeded
            assert all(r.status_code == 200 for r in responses)

            # Calculate throughput
            throughput = 1000 / elapsed_time

            print(f"\nThroughput Metrics:")
            print(f"  Requests: 1000")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Throughput: {throughput:.0f} req/s")

            # Assert throughput target
            assert throughput >= 1000, \
                f"Throughput {throughput:.0f} req/s below 1000 req/s target"


class TestCacheHitRate:
    """Test cache hit rate performance."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_hit_rate(
        self,
        test_app,
        setup_perf_test,
        redis_client
    ):
        """
        Test cache hit rate > 90%.

        GIVEN: Quota cached in Redis
        WHEN: 100 quota checks
        THEN: Cache hit rate > 90%
        """
        # Warm up cache
        quota_key = f"quota:cache:{TEST_TENANT}:quota_mb"
        await redis_client.setex(quota_key, 300, "10000")

        # Track cache hits/misses
        cache_hits = 0
        cache_misses = 0

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 100 requests
            for _ in range(100):
                # Check if quota was in cache before request
                exists = await redis_client.exists(quota_key)

                response = await client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT)}
                )

                if exists:
                    cache_hits += 1
                else:
                    cache_misses += 1

                assert response.status_code == 200

            # Calculate hit rate
            cache_hit_rate = (cache_hits / 100) * 100

            print(f"\nCache Hit Rate:")
            print(f"  Hits: {cache_hits}")
            print(f"  Misses: {cache_misses}")
            print(f"  Hit Rate: {cache_hit_rate:.1f}%")

            # Assert hit rate target
            assert cache_hit_rate > 90, \
                f"Cache hit rate {cache_hit_rate:.1f}% below 90% target"


class TestScalability:
    """Test scalability under load."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_load(
        self,
        test_app,
        setup_perf_test
    ):
        """
        Test sustained load performance.

        GIVEN: Middleware with warm cache
        WHEN: 5000 requests over 5 seconds
        THEN: p99 latency remains < 10ms throughout
        """
        latencies = []

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 5000 requests
            for i in range(5000):
                start = time.perf_counter()

                response = await client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT)}
                )

                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

                assert response.status_code == 200

                # Small delay to sustain ~1000 req/s
                if i % 100 == 0:
                    await asyncio.sleep(0.1)

            # Analyze latency distribution over time
            # Split into 5 buckets (1000 requests each)
            bucket_size = 1000
            for i in range(5):
                bucket = latencies[i*bucket_size:(i+1)*bucket_size]
                p99 = statistics.quantiles(bucket, n=100)[98]

                print(f"\nBucket {i+1} (requests {i*bucket_size}-{(i+1)*bucket_size}):")
                print(f"  p99: {p99:.2f}ms")

                # Assert latency doesn't degrade over time
                assert p99 < 10, \
                    f"Bucket {i+1} p99 latency {p99:.2f}ms exceeds 10ms target"


class TestMemoryUsage:
    """Test memory usage under load."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_redis_memory_efficient(
        self,
        test_app,
        setup_perf_test,
        redis_client
    ):
        """
        Test Redis memory usage is efficient.

        GIVEN: Middleware caching quota data
        WHEN: 1000 requests processed
        THEN: Redis memory usage stable (no leaks)
        """
        # Get initial Redis memory
        info_before = await redis_client.info("memory")
        memory_before = info_before["used_memory"]

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 1000 requests
            tasks = [
                client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT)}
                )
                for _ in range(1000)
            ]

            await asyncio.gather(*tasks)

        # Get final Redis memory
        info_after = await redis_client.info("memory")
        memory_after = info_after["used_memory"]

        memory_increase = memory_after - memory_before

        print(f"\nRedis Memory Usage:")
        print(f"  Before: {memory_before / 1024:.2f} KB")
        print(f"  After: {memory_after / 1024:.2f} KB")
        print(f"  Increase: {memory_increase / 1024:.2f} KB")

        # Assert memory increase is reasonable (< 1MB for quota cache)
        assert memory_increase < 1024 * 1024, \
            f"Redis memory increased by {memory_increase / 1024:.2f} KB (expected < 1MB)"


# Mark all tests as requiring asyncio and performance fixtures
pytest_plugins = ('pytest_asyncio',)
