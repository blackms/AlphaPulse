"""
Unit tests for QuotaEnforcementMiddleware - RED Phase (Story 4.3).

Tests all quota enforcement scenarios following TDD methodology.
These tests should FAIL until GREEN phase implementation is complete.

Test Coverage:
- Quota check decisions (ALLOW, WARN, REJECT)
- Usage tracking (atomic increment/decrement)
- Cache hit/miss scenarios
- Response header injection
- Feature flag control
- Error handling and fallback
"""

import pytest
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from fastapi import Request, Response
from starlette.datastructures import Headers

# These imports will fail until GREEN phase - expected in RED phase
from alpha_pulse.middleware.quota_enforcement import QuotaEnforcementMiddleware
from alpha_pulse.services.quota_checker import QuotaChecker
from alpha_pulse.services.quota_cache_service import QuotaCacheService
from alpha_pulse.services.usage_tracker import UsageTracker
from alpha_pulse.models.quota import QuotaConfig, QuotaDecision, QuotaStatus


@pytest.fixture
def test_tenant_id() -> UUID:
    """Test tenant UUID."""
    return UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.incrbyfloat = AsyncMock(return_value=10.0)
    redis.decrbyfloat = AsyncMock(return_value=5.0)
    redis.pipeline = MagicMock()
    return redis


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    return session


@pytest.fixture
def quota_config(test_tenant_id) -> QuotaConfig:
    """Default quota configuration."""
    return QuotaConfig(
        tenant_id=test_tenant_id,
        quota_mb=100,
        current_usage_mb=50.0,
        overage_allowed=True,
        overage_limit_mb=10
    )


@pytest.fixture
def mock_request(test_tenant_id):
    """Mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/cache/write"
    request.state.tenant_id = test_tenant_id
    request.headers = Headers({"content-type": "application/json"})
    return request


@pytest.fixture
async def middleware(mock_redis, mock_db_session):
    """Create middleware instance."""
    return QuotaEnforcementMiddleware(
        app=MagicMock(),
        enabled=True,
        cache_ttl_seconds=300,
        redis_client=mock_redis,
        db_session_factory=lambda: mock_db_session,
        exclude_paths=["/health", "/metrics"]
    )


class TestQuotaDecisions:
    """Test quota enforcement decision logic."""

    @pytest.mark.asyncio
    async def test_check_quota_allow(
        self,
        mock_redis,
        mock_db_session,
        test_tenant_id,
        quota_config
    ):
        """
        AC-1: Quota check allows write when within quota.

        GIVEN: Tenant with quota_mb=100 and current_usage_mb=50
        WHEN: Tenant writes 10MB
        THEN: Decision is ALLOW (60 < 100)
        """
        # Setup
        checker = QuotaChecker(
            cache_service=QuotaCacheService(mock_redis, mock_db_session),
            usage_tracker=UsageTracker(mock_redis)
        )

        # Mock: Current usage will be 60MB after write
        mock_redis.incrbyfloat.return_value = 60.0

        # Mock: Get quota from cache
        with patch.object(
            checker.cache_service,
            'get_quota',
            return_value=quota_config
        ):
            # Execute
            result = await checker.check_quota(test_tenant_id, write_size_mb=10.0)

            # Assert
            assert result.decision == QuotaDecision.ALLOW
            assert result.allowed
            assert not result.rejected
            assert result.new_usage_mb == 60.0
            # Should be called once for increment only (no rollback)
            assert mock_redis.incrbyfloat.call_count == 1

    @pytest.mark.asyncio
    async def test_check_quota_warn(
        self,
        mock_redis,
        mock_db_session,
        test_tenant_id,
        quota_config
    ):
        """
        AC-2: Quota check warns when over quota but within overage.

        GIVEN: Tenant with quota_mb=100, overage_limit_mb=10, usage_mb=95
        WHEN: Tenant writes 10MB
        THEN: Decision is WARN (105 > 100 but <= 110)
        """
        # Setup
        quota_config.current_usage_mb = 95.0
        checker = QuotaChecker(
            cache_service=QuotaCacheService(mock_redis, mock_db_session),
            usage_tracker=UsageTracker(mock_redis)
        )

        # Mock: Current usage will be 105MB after write
        mock_redis.incrbyfloat.return_value = 105.0

        # Mock: Get quota from cache
        with patch.object(
            checker.cache_service,
            'get_quota',
            return_value=quota_config
        ):
            # Execute
            result = await checker.check_quota(test_tenant_id, write_size_mb=10.0)

            # Assert
            assert result.decision == QuotaDecision.WARN
            assert result.allowed
            assert not result.rejected
            assert result.new_usage_mb == 105.0
            # Should be called once for increment only (no rollback)
            assert mock_redis.incrbyfloat.call_count == 1

    @pytest.mark.asyncio
    async def test_check_quota_reject(
        self,
        mock_redis,
        mock_db_session,
        test_tenant_id,
        quota_config
    ):
        """
        AC-3: Quota check rejects when exceeding hard limit.

        GIVEN: Tenant with quota_mb=100, overage_limit_mb=10, usage_mb=105
        WHEN: Tenant writes 10MB
        THEN: Decision is REJECT (115 > 110 hard limit)
        """
        # Setup
        quota_config.current_usage_mb = 105.0
        checker = QuotaChecker(
            cache_service=QuotaCacheService(mock_redis, mock_db_session),
            usage_tracker=UsageTracker(mock_redis)
        )

        # Mock: Current usage would be 115MB after write
        mock_redis.incrbyfloat.return_value = 115.0

        # Mock: Get quota from cache
        with patch.object(
            checker.cache_service,
            'get_quota',
            return_value=quota_config
        ):
            # Execute
            result = await checker.check_quota(test_tenant_id, write_size_mb=10.0)

            # Assert
            assert result.decision == QuotaDecision.REJECT
            assert not result.allowed
            assert result.rejected
            assert result.new_usage_mb is None  # Not allocated
            # Should be called twice: once for increment, once for rollback (with negative value)
            assert mock_redis.incrbyfloat.call_count == 2


class TestUsageTracking:
    """Test atomic usage counter operations."""

    @pytest.mark.asyncio
    async def test_usage_increment_atomic(self, mock_redis, test_tenant_id):
        """
        Test atomic usage increment.

        GIVEN: UsageTracker with Redis client
        WHEN: increment() is called with 5.5 MB
        THEN: Redis INCRBYFLOAT is called atomically
        """
        # Setup
        tracker = UsageTracker(mock_redis)
        mock_redis.incrbyfloat.return_value = 55.5

        # Execute
        new_usage = await tracker.increment(test_tenant_id, 5.5)

        # Assert
        assert new_usage == 55.5
        mock_redis.incrbyfloat.assert_called_once_with(
            f"quota:cache:{test_tenant_id}:current_usage_mb",
            5.5
        )

    @pytest.mark.asyncio
    async def test_usage_rollback_on_reject(self, mock_redis, test_tenant_id):
        """
        AC-6: Usage is rolled back when write is rejected.

        GIVEN: Write exceeds hard limit
        WHEN: Decision is REJECT
        THEN: Usage counter is decremented (rollback)
        """
        # Setup
        tracker = UsageTracker(mock_redis)
        mock_redis.decrbyfloat.return_value = 110.0

        # Execute
        rolled_back_usage = await tracker.decrement(test_tenant_id, 10.0)

        # Assert
        assert rolled_back_usage == 110.0
        mock_redis.decrbyfloat.assert_called_once_with(
            f"quota:cache:{test_tenant_id}:current_usage_mb",
            10.0
        )


class TestCacheService:
    """Test quota cache service (two-tier caching)."""

    @pytest.mark.asyncio
    async def test_cache_hit_redis(
        self,
        mock_redis,
        mock_db_session,
        test_tenant_id,
        quota_config
    ):
        """
        Test cache hit from Redis (fast path).

        GIVEN: Quota exists in Redis cache
        WHEN: get_quota() is called
        THEN: Quota returned from Redis, no DB query
        """
        # Setup
        cache_service = QuotaCacheService(
            mock_redis,
            mock_db_session,
            cache_ttl=300
        )

        # Mock: Redis has cached quota
        mock_redis.pipeline().execute.return_value = [
            "100",    # quota_mb
            "true",   # overage_allowed
            "10",     # overage_limit_mb
            "50.0"    # current_usage_mb
        ]

        # Execute
        quota = await cache_service.get_quota(test_tenant_id)

        # Assert
        assert quota.quota_mb == 100
        assert quota.overage_allowed is True
        assert quota.overage_limit_mb == 10
        assert quota.current_usage_mb == 50.0
        # DB session should NOT be called
        assert not mock_db_session.execute.called

    @pytest.mark.asyncio
    async def test_cache_miss_fallback_to_db(
        self,
        mock_redis,
        mock_db_session,
        test_tenant_id,
        quota_config
    ):
        """
        Test cache miss fallback to PostgreSQL (slow path).

        GIVEN: Quota NOT in Redis cache
        WHEN: get_quota() is called
        THEN: Quota loaded from PostgreSQL, cached to Redis
        """
        # Setup
        cache_service = QuotaCacheService(
            mock_redis,
            mock_db_session,
            cache_ttl=300
        )

        # Mock: Redis cache miss (returns None values)
        mock_redis.pipeline().execute.return_value = [None, None, None, None]

        # Mock: PostgreSQL returns quota
        with patch.object(
            cache_service,
            '_get_from_db',
            return_value=quota_config
        ):
            # Execute
            quota = await cache_service.get_quota(test_tenant_id)

            # Assert
            assert quota == quota_config
            # Redis should cache the result
            assert mock_redis.pipeline().execute.called


class TestMiddlewareIntegration:
    """Test middleware integration with FastAPI."""

    @pytest.mark.asyncio
    async def test_middleware_allows_write(
        self,
        middleware,
        mock_request,
        test_tenant_id,
        quota_config
    ):
        """
        Test middleware allows write within quota.

        GIVEN: Request to write cache data
        WHEN: Quota check passes (ALLOW)
        THEN: Request proceeds, response has quota headers
        """
        # Setup
        mock_request.json = AsyncMock(return_value={"size_mb": 10})
        call_next = AsyncMock(return_value=Response(status_code=200))

        # Mock quota check returns ALLOW
        with patch.object(
            middleware.quota_checker,
            'check_quota',
            return_value=QuotaDecision.ALLOW
        ):
            # Execute
            response = await middleware.dispatch(mock_request, call_next)

            # Assert
            assert response.status_code == 200
            assert call_next.called
            # Headers should be injected (tested separately)

    @pytest.mark.asyncio
    async def test_middleware_rejects_write(
        self,
        middleware,
        mock_request,
        test_tenant_id
    ):
        """
        AC-3: Middleware returns 429 when quota exceeded.

        GIVEN: Request to write cache data
        WHEN: Quota check fails (REJECT)
        THEN: 429 response returned, write NOT processed
        """
        # Setup
        mock_request.json = AsyncMock(return_value={"size_mb": 10})
        call_next = AsyncMock()

        # Mock quota check returns REJECT
        with patch.object(
            middleware.quota_checker,
            'check_quota',
            return_value=QuotaDecision.REJECT
        ):
            # Execute
            response = await middleware.dispatch(mock_request, call_next)

            # Assert
            assert response.status_code == 429
            assert not call_next.called  # Request not processed
            assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_response_headers_injected(
        self,
        middleware,
        mock_request,
        test_tenant_id,
        quota_config
    ):
        """
        AC-4: Response includes X-Cache-Quota-* headers.

        GIVEN: Request processed successfully
        WHEN: Response is returned
        THEN: Quota status headers are present
        """
        # Setup
        mock_request.json = AsyncMock(return_value={"size_mb": 10})
        mock_response = Response(status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Mock quota status
        with patch.object(
            middleware,
            '_get_quota_status',
            return_value=QuotaStatus(
                limit_mb=100,
                used_mb=60.0,
                remaining_mb=40.0,
                percent=60.0,
                status=QuotaDecision.ALLOW
            )
        ):
            # Execute
            response = await middleware.dispatch(mock_request, call_next)

            # Assert
            assert "X-Cache-Quota-Limit" in response.headers
            assert response.headers["X-Cache-Quota-Limit"] == "100"
            assert "X-Cache-Quota-Used" in response.headers
            assert "X-Cache-Quota-Remaining" in response.headers
            assert "X-Cache-Quota-Percent" in response.headers
            assert "X-Cache-Quota-Status" in response.headers


class TestFeatureFlag:
    """Test feature flag control."""

    @pytest.mark.asyncio
    async def test_middleware_disabled(
        self,
        mock_redis,
        mock_db_session,
        mock_request
    ):
        """
        AC-8: Middleware skips quota check when disabled.

        GIVEN: Feature flag enabled=False
        WHEN: Request is processed
        THEN: Quota check is skipped, request proceeds normally
        """
        # Setup
        middleware = QuotaEnforcementMiddleware(
            app=MagicMock(),
            enabled=False,  # Feature flag OFF
            cache_ttl_seconds=300,
            redis_client=mock_redis,
            db_session_factory=lambda: mock_db_session
        )

        call_next = AsyncMock(return_value=Response(status_code=200))

        # Execute
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.status_code == 200
        assert call_next.called
        # Quota checker should NOT be called
        assert not mock_redis.incrbyfloat.called

    @pytest.mark.asyncio
    async def test_middleware_excludes_path(
        self,
        middleware,
        mock_request
    ):
        """
        Test middleware skips excluded paths.

        GIVEN: Request to excluded path (/health)
        WHEN: Request is processed
        THEN: Quota check is skipped
        """
        # Setup
        mock_request.url.path = "/health"
        call_next = AsyncMock(return_value=Response(status_code=200))

        # Execute
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.status_code == 200
        assert call_next.called


class TestErrorHandling:
    """Test error handling and fallback scenarios."""

    @pytest.mark.asyncio
    async def test_redis_failure_fallback(
        self,
        mock_redis,
        mock_db_session,
        test_tenant_id,
        quota_config
    ):
        """
        AC-7: Fallback to PostgreSQL when Redis fails.

        GIVEN: Redis connection error
        WHEN: get_quota() is called
        THEN: Fallback to PostgreSQL succeeds
        """
        # Setup
        cache_service = QuotaCacheService(
            mock_redis,
            mock_db_session,
            cache_ttl=300
        )

        # Mock: Redis raises exception
        mock_redis.pipeline().execute.side_effect = ConnectionError("Redis down")

        # Mock: PostgreSQL works
        with patch.object(
            cache_service,
            '_get_from_db',
            return_value=quota_config
        ):
            # Execute
            quota = await cache_service.get_quota(test_tenant_id)

            # Assert
            assert quota == quota_config  # Fallback succeeded


class TestQuotaConfig:
    """Test QuotaConfig model properties."""

    def test_hard_limit_calculation(self, test_tenant_id):
        """
        Test hard_limit_mb property calculation.

        GIVEN: QuotaConfig with overage allowed
        WHEN: hard_limit_mb is accessed
        THEN: Returns quota_mb + overage_limit_mb
        """
        # Setup
        quota = QuotaConfig(
            tenant_id=test_tenant_id,
            quota_mb=100,
            current_usage_mb=50.0,
            overage_allowed=True,
            overage_limit_mb=10
        )

        # Assert
        assert quota.hard_limit_mb == 110

    def test_hard_limit_no_overage(self, test_tenant_id):
        """
        Test hard_limit_mb when overage not allowed.

        GIVEN: QuotaConfig with overage_allowed=False
        WHEN: hard_limit_mb is accessed
        THEN: Returns quota_mb (no overage)
        """
        # Setup
        quota = QuotaConfig(
            tenant_id=test_tenant_id,
            quota_mb=100,
            current_usage_mb=50.0,
            overage_allowed=False,
            overage_limit_mb=10
        )

        # Assert
        assert quota.hard_limit_mb == 100

    def test_usage_percent_calculation(self, test_tenant_id):
        """
        Test usage_percent property calculation.

        GIVEN: QuotaConfig with usage
        WHEN: usage_percent is accessed
        THEN: Returns (current_usage_mb / quota_mb) * 100
        """
        # Setup
        quota = QuotaConfig(
            tenant_id=test_tenant_id,
            quota_mb=100,
            current_usage_mb=75.0,
            overage_allowed=True,
            overage_limit_mb=10
        )

        # Assert
        assert quota.usage_percent == 75.0


# Mark all tests as requiring asyncio
pytest_plugins = ('pytest_asyncio',)
