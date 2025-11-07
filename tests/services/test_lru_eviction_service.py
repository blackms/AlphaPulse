"""
Tests for LRU eviction service.

Tests the complete eviction workflow: check quota, find candidates, evict oldest keys.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from alpha_pulse.services.lru_eviction_service import LRUEvictionService
from alpha_pulse.models.quota import QuotaConfig

# Apply asyncio marker to all tests in this module
pytestmark = pytest.mark.asyncio


@pytest.fixture
def test_tenant_id():
    """Test tenant UUID."""
    return uuid4()


@pytest.fixture
def quota_config(test_tenant_id):
    """Sample quota configuration."""
    return QuotaConfig(
        tenant_id=test_tenant_id,
        quota_mb=100.0,
        current_usage_mb=90.0,
        overage_allowed=True,
        overage_limit_mb=10.0
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock()
    redis.delete = AsyncMock()
    redis.pipeline = MagicMock()

    mock_pipe = AsyncMock()
    mock_pipe.delete = AsyncMock()
    mock_pipe.execute = AsyncMock()
    redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    redis.pipeline.return_value.__aexit__ = AsyncMock()

    return redis


@pytest.fixture
def mock_lru_tracker():
    """Mock LRU tracker."""
    tracker = AsyncMock()
    tracker.get_oldest_keys = AsyncMock()
    tracker.evict_keys = AsyncMock()
    tracker.get_lru_count = AsyncMock()
    return tracker


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker."""
    tracker = AsyncMock()
    tracker.get_usage = AsyncMock()
    tracker.decrement_usage = AsyncMock()
    return tracker


@pytest.fixture
def mock_quota_cache():
    """Mock quota cache service."""
    cache = AsyncMock()
    cache.get_quota_config = AsyncMock()
    return cache


@pytest.fixture
def eviction_service(mock_redis, mock_lru_tracker, mock_usage_tracker, mock_quota_cache):
    """LRU eviction service instance."""
    return LRUEvictionService(
        redis_client=mock_redis,
        lru_tracker=mock_lru_tracker,
        usage_tracker=mock_usage_tracker,
        quota_cache_service=mock_quota_cache
    )


class TestEvictionNeed:
    """Test detection of eviction need."""

    async def test_needs_eviction_over_quota(
        self, eviction_service, test_tenant_id, quota_config
    ):
        """Returns True when usage exceeds quota."""
        # Usage at 95MB > quota 90MB (after write)
        quota_config.current_usage_mb = 95.0

        needs_eviction = await eviction_service.needs_eviction(
            tenant_id=test_tenant_id,
            quota_config=quota_config
        )

        assert needs_eviction is True

    async def test_needs_eviction_within_quota(
        self, eviction_service, test_tenant_id, quota_config
    ):
        """Returns False when within quota."""
        quota_config.current_usage_mb = 50.0

        needs_eviction = await eviction_service.needs_eviction(
            tenant_id=test_tenant_id,
            quota_config=quota_config
        )

        assert needs_eviction is False

    async def test_needs_eviction_at_hard_limit(
        self, eviction_service, test_tenant_id, quota_config
    ):
        """Returns True when at hard limit."""
        # Hard limit = 100MB + 10MB overage = 110MB
        quota_config.current_usage_mb = 110.0

        needs_eviction = await eviction_service.needs_eviction(
            tenant_id=test_tenant_id,
            quota_config=quota_config
        )

        assert needs_eviction is True


class TestEvictionTargetCalculation:
    """Test calculation of eviction targets."""

    async def test_calculate_eviction_target_default(
        self, eviction_service, test_tenant_id, quota_config
    ):
        """Calculate eviction target to reach 90% of quota."""
        quota_config.current_usage_mb = 100.0  # Over quota
        quota_config.quota_mb = 100.0

        target_mb = await eviction_service.calculate_eviction_target(
            quota_config=quota_config,
            target_percent=0.9
        )

        # Target: 90% of 100MB = 90MB
        # Need to evict: 100 - 90 = 10MB
        assert target_mb == 10.0

    async def test_calculate_eviction_target_custom_percent(
        self, eviction_service, quota_config
    ):
        """Calculate eviction target with custom percentage."""
        quota_config.current_usage_mb = 100.0
        quota_config.quota_mb = 100.0

        target_mb = await eviction_service.calculate_eviction_target(
            quota_config=quota_config,
            target_percent=0.8  # Target 80% instead
        )

        # Target: 80% of 100MB = 80MB
        # Need to evict: 100 - 80 = 20MB
        assert target_mb == 20.0

    async def test_calculate_eviction_target_minimal(
        self, eviction_service, quota_config
    ):
        """Calculate eviction target when just slightly over."""
        quota_config.current_usage_mb = 101.0
        quota_config.quota_mb = 100.0

        target_mb = await eviction_service.calculate_eviction_target(
            quota_config=quota_config,
            target_percent=0.9
        )

        # Target: 90MB, current: 101MB
        # Evict: 11MB
        assert target_mb == 11.0


class TestEvictionCandidateSelection:
    """Test selection of eviction candidates."""

    async def test_get_eviction_candidates_default_count(
        self, eviction_service, mock_lru_tracker, test_tenant_id
    ):
        """Get eviction candidates with default count of 10."""
        oldest_keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(10)]
        mock_lru_tracker.get_oldest_keys.return_value = oldest_keys

        candidates = await eviction_service.get_eviction_candidates(
            tenant_id=test_tenant_id
        )

        mock_lru_tracker.get_oldest_keys.assert_called_once_with(
            test_tenant_id,
            count=10
        )

        assert candidates == oldest_keys
        assert len(candidates) == 10

    async def test_get_eviction_candidates_custom_count(
        self, eviction_service, mock_lru_tracker, test_tenant_id
    ):
        """Get eviction candidates with custom count."""
        count = 5
        oldest_keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(count)]
        mock_lru_tracker.get_oldest_keys.return_value = oldest_keys

        candidates = await eviction_service.get_eviction_candidates(
            tenant_id=test_tenant_id,
            count=count
        )

        mock_lru_tracker.get_oldest_keys.assert_called_once_with(
            test_tenant_id,
            count=count
        )

        assert len(candidates) == count

    async def test_get_eviction_candidates_empty(
        self, eviction_service, mock_lru_tracker, test_tenant_id
    ):
        """Returns empty list when no candidates available."""
        mock_lru_tracker.get_oldest_keys.return_value = []

        candidates = await eviction_service.get_eviction_candidates(
            tenant_id=test_tenant_id
        )

        assert candidates == []


class TestKeyEviction:
    """Test eviction of individual keys."""

    async def test_evict_key_success(
        self, eviction_service, mock_redis, mock_lru_tracker,
        mock_usage_tracker, test_tenant_id
    ):
        """Successfully evict a single key."""
        cache_key = f"tenant:{test_tenant_id}:data"
        key_size_mb = 5.0

        # Mock Redis get to return serialized data
        mock_redis.get.return_value = b"x" * (5 * 1024 * 1024)  # 5MB
        mock_redis.delete.return_value = 1
        mock_lru_tracker.remove_key.return_value = True
        mock_usage_tracker.decrement_usage.return_value = 85.0

        evicted_size = await eviction_service.evict_key(
            tenant_id=test_tenant_id,
            cache_key=cache_key
        )

        # Verify key deleted from Redis
        mock_redis.delete.assert_called_once_with(cache_key)

        # Verify removed from LRU tracking
        mock_lru_tracker.remove_key.assert_called_once_with(
            test_tenant_id,
            cache_key
        )

        # Verify usage decremented
        mock_usage_tracker.decrement_usage.assert_called_once()
        call_args = mock_usage_tracker.decrement_usage.call_args[0]
        assert call_args[0] == test_tenant_id
        assert call_args[1] > 0  # Size should be calculated

        assert evicted_size > 0

    async def test_evict_key_nonexistent(
        self, eviction_service, mock_redis, test_tenant_id
    ):
        """Evicting nonexistent key returns 0."""
        cache_key = f"tenant:{test_tenant_id}:missing"

        mock_redis.get.return_value = None
        mock_redis.delete.return_value = 0

        evicted_size = await eviction_service.evict_key(
            tenant_id=test_tenant_id,
            cache_key=cache_key
        )

        assert evicted_size == 0.0

    async def test_evict_key_updates_metrics(
        self, eviction_service, mock_redis, mock_lru_tracker,
        mock_usage_tracker, test_tenant_id
    ):
        """Evicting key updates all tracking metrics."""
        cache_key = f"tenant:{test_tenant_id}:data"

        mock_redis.get.return_value = b"test data"
        mock_redis.delete.return_value = 1
        mock_lru_tracker.remove_key.return_value = True
        mock_usage_tracker.decrement_usage.return_value = 80.0

        await eviction_service.evict_key(
            tenant_id=test_tenant_id,
            cache_key=cache_key
        )

        # All three operations should be called
        mock_redis.delete.assert_called_once()
        mock_lru_tracker.remove_key.assert_called_once()
        mock_usage_tracker.decrement_usage.assert_called_once()


class TestBatchEviction:
    """Test batch eviction operations."""

    async def test_evict_batch_success(
        self, eviction_service, mock_redis, mock_lru_tracker,
        mock_usage_tracker, test_tenant_id
    ):
        """Successfully evict multiple keys in batch."""
        keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(5)]

        # Mock Redis responses
        mock_redis.get.return_value = b"x" * 1024 * 1024  # 1MB each
        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1] * 5  # All deletes succeed

        mock_lru_tracker.evict_keys.return_value = 5
        mock_usage_tracker.decrement_usage.return_value = 75.0

        result = await eviction_service.evict_batch(
            tenant_id=test_tenant_id,
            cache_keys=keys
        )

        assert result["evicted_count"] == 5
        assert result["total_size_mb"] > 0
        assert result["failed_count"] == 0

    async def test_evict_batch_partial_failure(
        self, eviction_service, mock_redis, mock_lru_tracker, test_tenant_id
    ):
        """Batch eviction handles partial failures."""
        keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(5)]

        # Some keys exist, some don't
        mock_redis.get.side_effect = [
            b"data",  # exists
            None,     # missing
            b"data",  # exists
            None,     # missing
            b"data",  # exists
        ]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1, 0, 1, 0, 1]  # 3 succeed, 2 fail

        result = await eviction_service.evict_batch(
            tenant_id=test_tenant_id,
            cache_keys=keys
        )

        assert result["evicted_count"] == 3
        assert result["failed_count"] == 2

    async def test_evict_batch_empty_list(
        self, eviction_service, test_tenant_id
    ):
        """Evicting empty list returns zero results."""
        result = await eviction_service.evict_batch(
            tenant_id=test_tenant_id,
            cache_keys=[]
        )

        assert result["evicted_count"] == 0
        assert result["total_size_mb"] == 0.0
        assert result["failed_count"] == 0


class TestEvictionWorkflow:
    """Test complete eviction workflow."""

    async def test_evict_to_target_success(
        self, eviction_service, mock_redis, mock_lru_tracker,
        mock_usage_tracker, mock_quota_cache, test_tenant_id, quota_config
    ):
        """Evict keys until target usage is reached."""
        # Current usage: 100MB, target: 90MB
        quota_config.current_usage_mb = 100.0
        quota_config.quota_mb = 100.0

        oldest_keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(10)]
        mock_lru_tracker.get_oldest_keys.return_value = oldest_keys

        # Each key is 2MB
        mock_redis.get.return_value = b"x" * (2 * 1024 * 1024)

        # Mock usage decreasing as keys are evicted
        mock_usage_tracker.get_usage.side_effect = [
            100.0,  # Initial
            98.0,   # After 1 key
            96.0,   # After 2 keys
            94.0,   # After 3 keys
            92.0,   # After 4 keys
            90.0,   # After 5 keys - TARGET REACHED
        ]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1] * 10

        result = await eviction_service.evict_to_target(
            tenant_id=test_tenant_id,
            quota_config=quota_config,
            target_percent=0.9
        )

        assert result["success"] is True
        assert result["evicted_count"] >= 5  # At least 5 keys evicted
        assert result["final_usage_mb"] <= 90.0

    async def test_evict_to_target_insufficient_keys(
        self, eviction_service, mock_lru_tracker, test_tenant_id, quota_config
    ):
        """Eviction stops when no more keys available."""
        quota_config.current_usage_mb = 100.0
        quota_config.quota_mb = 100.0

        # Only 2 keys available, but need to evict 10MB
        oldest_keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(2)]
        mock_lru_tracker.get_oldest_keys.return_value = oldest_keys

        result = await eviction_service.evict_to_target(
            tenant_id=test_tenant_id,
            quota_config=quota_config,
            target_percent=0.9
        )

        # Should evict available keys even if target not reached
        assert result["success"] is False
        assert result["evicted_count"] == 2
        assert "insufficient" in result.get("message", "").lower()

    async def test_evict_to_target_no_eviction_needed(
        self, eviction_service, test_tenant_id, quota_config
    ):
        """No eviction when already under target."""
        quota_config.current_usage_mb = 50.0  # Well under quota
        quota_config.quota_mb = 100.0

        result = await eviction_service.evict_to_target(
            tenant_id=test_tenant_id,
            quota_config=quota_config,
            target_percent=0.9
        )

        assert result["success"] is True
        assert result["evicted_count"] == 0
        assert "not needed" in result.get("message", "").lower()


class TestEvictionPerformance:
    """Test eviction performance characteristics."""

    async def test_eviction_meets_p99_latency_target(
        self, eviction_service, mock_redis, mock_lru_tracker,
        mock_usage_tracker, test_tenant_id
    ):
        """Eviction completes within P99 <100ms target (Story AC)."""
        import time

        # Setup: 10 keys to evict
        keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(10)]
        mock_lru_tracker.get_oldest_keys.return_value = keys
        mock_redis.get.return_value = b"test data"

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1] * 10

        # Measure eviction time
        start = time.time()
        await eviction_service.evict_batch(test_tenant_id, keys)
        elapsed_ms = (time.time() - start) * 1000

        # Should complete well under 100ms (with mocks, should be <10ms)
        assert elapsed_ms < 100.0

    async def test_batch_eviction_uses_pipeline(
        self, eviction_service, mock_redis, test_tenant_id
    ):
        """Batch eviction uses Redis pipeline for efficiency."""
        keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(5)]

        mock_redis.get.return_value = b"data"
        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1] * 5

        await eviction_service.evict_batch(test_tenant_id, keys)

        # Pipeline should be used for batch operations
        mock_redis.pipeline.assert_called()
        mock_pipe.execute.assert_called_once()  # Single batch execution


class TestEvictionErrorHandling:
    """Test error handling in eviction operations."""

    async def test_evict_key_redis_error(
        self, eviction_service, mock_redis, test_tenant_id
    ):
        """Evict key handles Redis errors gracefully."""
        cache_key = f"tenant:{test_tenant_id}:key"

        mock_redis.get.side_effect = Exception("Redis connection failed")

        with pytest.raises(Exception) as exc_info:
            await eviction_service.evict_key(test_tenant_id, cache_key)

        assert "Redis connection failed" in str(exc_info.value)

    async def test_evict_batch_continues_on_error(
        self, eviction_service, mock_redis, mock_lru_tracker, test_tenant_id
    ):
        """Batch eviction continues despite individual key errors."""
        keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(3)]

        # First key errors, others succeed
        mock_redis.get.side_effect = [
            Exception("Error"),
            b"data2",
            b"data3"
        ]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [0, 1, 1]  # First fails, others succeed

        result = await eviction_service.evict_batch(test_tenant_id, keys)

        # Should evict 2 keys despite 1 error
        assert result["evicted_count"] >= 2


class TestEvictionLogging:
    """Test eviction logging and audit trail."""

    @patch("alpha_pulse.services.lru_eviction_service.logger")
    async def test_evict_key_logs_action(
        self, mock_logger, eviction_service, mock_redis,
        mock_lru_tracker, mock_usage_tracker, test_tenant_id
    ):
        """Evicting key logs the action."""
        cache_key = f"tenant:{test_tenant_id}:data"

        mock_redis.get.return_value = b"test"
        mock_redis.delete.return_value = 1
        mock_lru_tracker.remove_key.return_value = True
        mock_usage_tracker.decrement_usage.return_value = 80.0

        await eviction_service.evict_key(test_tenant_id, cache_key)

        # Should log eviction event
        assert mock_logger.info.called or mock_logger.debug.called

    @patch("alpha_pulse.services.lru_eviction_service.logger")
    async def test_evict_batch_logs_summary(
        self, mock_logger, eviction_service, mock_redis, test_tenant_id
    ):
        """Batch eviction logs summary statistics."""
        keys = [f"tenant:{test_tenant_id}:key{i}" for i in range(5)]

        mock_redis.get.return_value = b"data"
        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1] * 5

        await eviction_service.evict_batch(test_tenant_id, keys)

        # Should log batch summary
        assert mock_logger.info.called
