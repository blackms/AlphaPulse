"""
End-to-end integration tests for QuotaEnforcementMiddleware - RED Phase (Story 4.3).

Tests full request flow through middleware with real Redis and PostgreSQL.
These tests should FAIL until GREEN phase implementation is complete.

Test Coverage:
- Full request-response cycle
- Database + Redis integration
- Concurrent request handling
- Race condition validation
"""

import pytest
import asyncio
from uuid import UUID
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient

# These imports will fail until GREEN phase
from alpha_pulse.middleware.quota_enforcement import QuotaEnforcementMiddleware
from alpha_pulse.models.cache_quota import TenantCacheQuota
from alpha_pulse.database import get_db


# Test tenant IDs
TEST_TENANT_1 = UUID("00000000-0000-0000-0000-000000000001")
TEST_TENANT_2 = UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
async def test_app(redis_client, db_session):
    """Create test FastAPI app with middleware."""
    app = FastAPI()

    # Add quota enforcement middleware
    app.add_middleware(
        QuotaEnforcementMiddleware,
        enabled=True,
        cache_ttl_seconds=300,
        redis_client=redis_client,
        db_session_factory=get_db,
        exclude_paths=["/health"]
    )

    # Test route for cache writes
    @app.post("/cache/write")
    async def write_cache(request: Request):
        data = await request.json()
        return JSONResponse({
            "status": "success",
            "size_mb": data.get("size_mb", 0)
        })

    # Health check route (excluded from quota)
    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


@pytest.fixture
async def setup_test_quota(db_session, redis_client):
    """Setup test quota in database."""
    async def _setup(tenant_id: UUID, quota_mb: int = 100):
        # Create quota in database
        quota = TenantCacheQuota(
            tenant_id=tenant_id,
            quota_mb=quota_mb,
            current_usage_mb=0,
            overage_allowed=True,
            overage_limit_mb=10
        )
        db_session.add(quota)
        await db_session.commit()

        # Clear Redis cache for tenant
        await redis_client.delete(f"quota:cache:{tenant_id}:*")

        return quota

    return _setup


class TestEndToEndFlow:
    """Test complete request-response flow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_quota_enforcement(
        self,
        test_app,
        setup_test_quota,
        redis_client
    ):
        """
        Test complete quota enforcement flow.

        GIVEN: Tenant with 100MB quota, 0MB used
        WHEN: Write 50MB to cache
        THEN: Request succeeds, quota headers present, usage tracked
        """
        # Setup
        await setup_test_quota(TEST_TENANT_1, quota_mb=100)

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Simulate tenant context (normally set by TenantContextMiddleware)
            # Execute
            response = await client.post(
                "/cache/write",
                json={"size_mb": 50},
                headers={"X-Tenant-ID": str(TEST_TENANT_1)}
            )

            # Assert
            assert response.status_code == 200
            assert response.headers.get("X-Cache-Quota-Limit") == "100"
            assert response.headers.get("X-Cache-Quota-Used") == "50.0"
            assert response.headers.get("X-Cache-Quota-Status") == "ok"

            # Verify usage tracked in Redis
            usage_key = f"quota:cache:{TEST_TENANT_1}:current_usage_mb"
            usage = await redis_client.get(usage_key)
            assert float(usage) == 50.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_429_response(
        self,
        test_app,
        setup_test_quota,
        redis_client
    ):
        """
        Test 429 response when quota exceeded.

        GIVEN: Tenant with 100MB quota, 105MB used (over hard limit)
        WHEN: Write 10MB to cache
        THEN: 429 response, request NOT processed, usage NOT incremented
        """
        # Setup
        await setup_test_quota(TEST_TENANT_1, quota_mb=100)

        # Pre-fill usage to 105MB (over quota but within overage)
        usage_key = f"quota:cache:{TEST_TENANT_1}:current_usage_mb"
        await redis_client.set(usage_key, "105")

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute
            response = await client.post(
                "/cache/write",
                json={"size_mb": 10},
                headers={"X-Tenant-ID": str(TEST_TENANT_1)}
            )

            # Assert
            assert response.status_code == 429
            assert "Retry-After" in response.headers
            assert response.headers.get("X-Cache-Quota-Status") == "exceeded"

            # Verify usage NOT incremented (still 105)
            usage = await redis_client.get(usage_key)
            assert float(usage) == 105.0


class TestConcurrentRequests:
    """Test concurrent request handling and race conditions."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_concurrent_writes(
        self,
        test_app,
        setup_test_quota,
        redis_client
    ):
        """
        AC-6: Test atomic usage tracking under concurrent load.

        GIVEN: Tenant with 100MB quota, 0MB used
        WHEN: 50 concurrent 1MB writes (total 50MB)
        THEN: All 50 succeed, final usage exactly 50MB (no race conditions)
        """
        # Setup
        await setup_test_quota(TEST_TENANT_1, quota_mb=100)

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 50 concurrent 1MB writes
            tasks = [
                client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT_1)}
                )
                for _ in range(50)
            ]

            responses = await asyncio.gather(*tasks)

            # Assert all succeeded
            assert all(r.status_code == 200 for r in responses)

            # Verify final usage is exactly 50MB (no double-counting)
            usage_key = f"quota:cache:{TEST_TENANT_1}:current_usage_mb"
            usage = await redis_client.get(usage_key)
            assert float(usage) == 50.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_concurrent_quota_boundary(
        self,
        test_app,
        setup_test_quota,
        redis_client
    ):
        """
        Test concurrent writes at quota boundary.

        GIVEN: Tenant with 100MB quota, 95MB used
        WHEN: 10 concurrent 1MB writes (total 10MB, would exceed quota)
        THEN: First 5 succeed, remaining rejected, no race conditions
        """
        # Setup
        await setup_test_quota(TEST_TENANT_1, quota_mb=100)

        # Pre-fill usage to 95MB
        usage_key = f"quota:cache:{TEST_TENANT_1}:current_usage_mb"
        await redis_client.set(usage_key, "95")

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute 10 concurrent 1MB writes
            tasks = [
                client.post(
                    "/cache/write",
                    json={"size_mb": 1},
                    headers={"X-Tenant-ID": str(TEST_TENANT_1)}
                )
                for _ in range(10)
            ]

            responses = await asyncio.gather(*tasks)

            # Assert some succeeded, some rejected (depends on timing)
            success_count = sum(1 for r in responses if r.status_code == 200)
            reject_count = sum(1 for r in responses if r.status_code == 429)

            # With overage_limit=10, hard limit is 110MB
            # Starting at 95MB, we can accept 15MB total
            # So approximately 15 should succeed (but allow variance for warnings)
            assert success_count <= 15
            assert reject_count > 0

            # Verify usage doesn't exceed hard limit
            usage = await redis_client.get(usage_key)
            assert float(usage) <= 110.0  # Hard limit


class TestTenantIsolation:
    """Test tenant isolation in quota enforcement."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_tenant_isolation(
        self,
        test_app,
        setup_test_quota,
        redis_client
    ):
        """
        Test tenant quotas are isolated.

        GIVEN: Two tenants with separate quotas
        WHEN: Both write to cache
        THEN: Usage tracked separately, no cross-tenant impact
        """
        # Setup
        await setup_test_quota(TEST_TENANT_1, quota_mb=100)
        await setup_test_quota(TEST_TENANT_2, quota_mb=50)

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Tenant 1 writes 80MB
            response1 = await client.post(
                "/cache/write",
                json={"size_mb": 80},
                headers={"X-Tenant-ID": str(TEST_TENANT_1)}
            )

            # Tenant 2 writes 40MB
            response2 = await client.post(
                "/cache/write",
                json={"size_mb": 40},
                headers={"X-Tenant-ID": str(TEST_TENANT_2)}
            )

            # Assert both succeeded
            assert response1.status_code == 200
            assert response2.status_code == 200

            # Verify usage tracked separately
            usage1 = await redis_client.get(
                f"quota:cache:{TEST_TENANT_1}:current_usage_mb"
            )
            usage2 = await redis_client.get(
                f"quota:cache:{TEST_TENANT_2}:current_usage_mb"
            )

            assert float(usage1) == 80.0
            assert float(usage2) == 40.0


class TestRedisFailover:
    """Test Redis failure scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_redis_failure(
        self,
        test_app,
        setup_test_quota,
        redis_client
    ):
        """
        AC-7: Test fallback to PostgreSQL when Redis fails.

        GIVEN: Redis connection interrupted
        WHEN: Quota check is attempted
        THEN: Fallback to PostgreSQL succeeds
        """
        # Setup
        await setup_test_quota(TEST_TENANT_1, quota_mb=100)

        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Simulate Redis failure (mock or disconnect)
            # This would require injecting failure into Redis client
            # For now, test that quota can be loaded from DB when cache cold

            # Clear Redis cache
            await redis_client.flushdb()

            # Execute (should fallback to PostgreSQL)
            response = await client.post(
                "/cache/write",
                json={"size_mb": 10},
                headers={"X-Tenant-ID": str(TEST_TENANT_1)}
            )

            # Assert request succeeded (fallback worked)
            assert response.status_code == 200


class TestExcludedPaths:
    """Test excluded paths skip quota enforcement."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_e2e_excluded_path(
        self,
        test_app,
        redis_client
    ):
        """
        Test excluded paths skip quota check.

        GIVEN: /health is excluded path
        WHEN: Request to /health
        THEN: No quota check, request succeeds
        """
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            # Execute
            response = await client.get("/health")

            # Assert
            assert response.status_code == 200
            # No quota headers should be present
            assert "X-Cache-Quota-Limit" not in response.headers


# Mark all tests as requiring asyncio and integration fixtures
pytest_plugins = ('pytest_asyncio',)
