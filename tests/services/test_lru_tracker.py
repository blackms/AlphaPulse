"""
Tests for LRU tracking service.

Tests the sorted set-based LRU tracking for cache entries per ADR-004.
"""

import pytest
import time
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from alpha_pulse.services.lru_tracker import LRUTracker

# Apply asyncio marker to all tests in this module
pytestmark = pytest.mark.asyncio


@pytest.fixture
def test_tenant_id():
    """Test tenant UUID."""
    return uuid4()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.pipeline = MagicMock()

    # Mock pipeline behavior
    mock_pipe = AsyncMock()
    mock_pipe.zadd = AsyncMock()
    mock_pipe.zremrangebyrank = AsyncMock()
    mock_pipe.zcard = AsyncMock()
    mock_pipe.zrange = AsyncMock()
    mock_pipe.zrem = AsyncMock()
    mock_pipe.execute = AsyncMock()
    redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    redis.pipeline.return_value.__aexit__ = AsyncMock()

    return redis


@pytest.fixture
def lru_tracker(mock_redis):
    """LRU tracker instance."""
    return LRUTracker(mock_redis)


class TestLRUKeyGeneration:
    """Test LRU key generation."""

    async def test_get_lru_key_format(self, lru_tracker, test_tenant_id):
        """LRU key follows meta:tenant:{id}:lru format."""
        key = lru_tracker._get_lru_key(test_tenant_id)
        assert key == f"meta:tenant:{test_tenant_id}:lru"
        assert key.startswith("meta:tenant:")
        assert key.endswith(":lru")


class TestLRUTracking:
    """Test LRU tracking operations."""

    async def test_track_access_adds_to_sorted_set(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Tracking access adds key to sorted set with current timestamp."""
        cache_key = "tenant:test:data"

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1]  # zadd result

        timestamp = await lru_tracker.track_access(test_tenant_id, cache_key)

        # Verify zadd was called
        mock_pipe.zadd.assert_called_once()
        call_args = mock_pipe.zadd.call_args

        # Verify sorted set key
        assert call_args[0][0] == f"meta:tenant:{test_tenant_id}:lru"

        # Verify mapping format {cache_key: timestamp}
        mapping = call_args[1]['mapping']
        assert cache_key in mapping
        assert isinstance(mapping[cache_key], float)
        assert mapping[cache_key] > 0

        # Verify returned timestamp
        assert isinstance(timestamp, float)
        assert timestamp > 0

    async def test_track_access_updates_existing_entry(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Tracking access updates timestamp for existing key."""
        cache_key = "tenant:test:data"

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [0]  # zadd returns 0 for updates

        # Track twice
        ts1 = await lru_tracker.track_access(test_tenant_id, cache_key)
        time.sleep(0.01)  # Small delay
        ts2 = await lru_tracker.track_access(test_tenant_id, cache_key)

        # Second timestamp should be later
        assert ts2 > ts1

        # zadd should be called twice
        assert mock_pipe.zadd.call_count == 2

    async def test_track_multiple_keys(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Can track multiple keys for same tenant."""
        keys = ["key1", "key2", "key3"]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1]

        timestamps = []
        for key in keys:
            ts = await lru_tracker.track_access(test_tenant_id, key)
            timestamps.append(ts)
            time.sleep(0.01)  # Ensure different timestamps

        # All timestamps should be different and increasing
        assert timestamps[0] < timestamps[1] < timestamps[2]

        # zadd called for each key
        assert mock_pipe.zadd.call_count == len(keys)


class TestLRURemoval:
    """Test LRU key removal."""

    async def test_remove_key_from_lru(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Removing key deletes it from sorted set."""
        cache_key = "tenant:test:data"

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1]  # zrem result

        removed = await lru_tracker.remove_key(test_tenant_id, cache_key)

        # Verify zrem called with correct arguments
        mock_pipe.zrem.assert_called_once_with(
            f"meta:tenant:{test_tenant_id}:lru",
            cache_key
        )

        assert removed is True

    async def test_remove_nonexistent_key(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Removing nonexistent key returns False."""
        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [0]  # zrem returns 0

        removed = await lru_tracker.remove_key(test_tenant_id, "nonexistent")

        assert removed is False


class TestLRURetrieval:
    """Test LRU key retrieval operations."""

    async def test_get_lru_count(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get count of tracked keys."""
        mock_redis.zcard = AsyncMock(return_value=42)

        count = await lru_tracker.get_lru_count(test_tenant_id)

        mock_redis.zcard.assert_called_once_with(
            f"meta:tenant:{test_tenant_id}:lru"
        )
        assert count == 42

    async def test_get_oldest_keys(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get N oldest keys in LRU order."""
        oldest_keys = ["key1", "key2", "key3"]
        mock_redis.zrange = AsyncMock(return_value=oldest_keys)

        result = await lru_tracker.get_oldest_keys(test_tenant_id, count=3)

        # Verify zrange called correctly (ascending order, limit 3)
        mock_redis.zrange.assert_called_once_with(
            f"meta:tenant:{test_tenant_id}:lru",
            0,
            2  # Redis range is inclusive: 0-2 = 3 keys
        )

        assert result == oldest_keys
        assert len(result) == 3

    async def test_get_oldest_keys_default_count(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get oldest keys with default count of 10."""
        mock_redis.zrange = AsyncMock(return_value=[])

        await lru_tracker.get_oldest_keys(test_tenant_id)

        # Default should get 10 keys (0-9)
        mock_redis.zrange.assert_called_once_with(
            f"meta:tenant:{test_tenant_id}:lru",
            0,
            9
        )

    async def test_get_newest_keys(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get N newest keys in reverse LRU order."""
        newest_keys = ["key3", "key2", "key1"]
        mock_redis.zrevrange = AsyncMock(return_value=newest_keys)

        result = await lru_tracker.get_newest_keys(test_tenant_id, count=3)

        # Verify zrevrange called correctly (descending order)
        mock_redis.zrevrange.assert_called_once_with(
            f"meta:tenant:{test_tenant_id}:lru",
            0,
            2
        )

        assert result == newest_keys


class TestLRUEviction:
    """Test LRU-based eviction operations."""

    async def test_get_keys_to_evict(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get keys that should be evicted (oldest N)."""
        target_count = 5
        oldest = [f"key{i}" for i in range(target_count)]

        mock_redis.zrange = AsyncMock(return_value=oldest)

        keys = await lru_tracker.get_keys_to_evict(test_tenant_id, target_count)

        assert keys == oldest
        assert len(keys) == target_count

        # Should call get_oldest_keys internally
        mock_redis.zrange.assert_called_once()

    async def test_evict_keys_batch(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Evict a batch of keys from LRU tracking."""
        keys_to_evict = ["key1", "key2", "key3"]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1, 1, 1]  # Each zrem succeeds

        evicted_count = await lru_tracker.evict_keys(test_tenant_id, keys_to_evict)

        # Verify zrem called for each key
        assert mock_pipe.zrem.call_count == len(keys_to_evict)

        # Verify evicted count
        assert evicted_count == 3

    async def test_evict_keys_partial_success(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Eviction handles keys that don't exist."""
        keys_to_evict = ["exists1", "missing", "exists2"]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1, 0, 1]  # Middle key missing

        evicted_count = await lru_tracker.evict_keys(test_tenant_id, keys_to_evict)

        # Only 2 keys actually evicted
        assert evicted_count == 2


class TestLRUClearance:
    """Test clearing LRU tracking data."""

    async def test_clear_lru_for_tenant(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Clear all LRU tracking for a tenant."""
        mock_redis.delete = AsyncMock(return_value=1)

        cleared = await lru_tracker.clear_lru(test_tenant_id)

        mock_redis.delete.assert_called_once_with(
            f"meta:tenant:{test_tenant_id}:lru"
        )

        assert cleared is True

    async def test_clear_lru_nonexistent(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Clearing nonexistent LRU returns False."""
        mock_redis.delete = AsyncMock(return_value=0)

        cleared = await lru_tracker.clear_lru(test_tenant_id)

        assert cleared is False


class TestLRUMetrics:
    """Test LRU metrics and statistics."""

    async def test_get_lru_stats(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get LRU statistics for a tenant."""
        mock_redis.zcard = AsyncMock(return_value=100)
        mock_redis.zrange = AsyncMock(return_value=["oldest_key"])
        mock_redis.zrevrange = AsyncMock(return_value=["newest_key"])
        mock_redis.zscore = AsyncMock(side_effect=[1000.0, 2000.0])  # oldest, newest timestamps

        stats = await lru_tracker.get_lru_stats(test_tenant_id)

        assert stats["total_keys"] == 100
        assert stats["oldest_key"] == "oldest_key"
        assert stats["newest_key"] == "newest_key"
        assert stats["oldest_timestamp"] == 1000.0
        assert stats["newest_timestamp"] == 2000.0
        assert stats["age_range_seconds"] == 1000.0  # 2000 - 1000


class TestLRUErrorHandling:
    """Test error handling in LRU operations."""

    async def test_get_oldest_keys_redis_error(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Get oldest keys handles Redis errors."""
        mock_redis.zrange = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(Exception):
            await lru_tracker.get_oldest_keys(test_tenant_id)


class TestLRUPerformance:
    """Test LRU tracking performance characteristics."""

    async def test_track_access_uses_pipeline(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Track access uses Redis pipeline for efficiency."""
        await lru_tracker.track_access(test_tenant_id, "key")

        # Pipeline should be used
        mock_redis.pipeline.assert_called()

    async def test_evict_keys_batches_operations(
        self, lru_tracker, mock_redis, test_tenant_id
    ):
        """Evict keys batches Redis operations."""
        keys = [f"key{i}" for i in range(20)]

        mock_pipe = mock_redis.pipeline.return_value.__aenter__.return_value
        mock_pipe.execute.return_value = [1] * 20

        await lru_tracker.evict_keys(test_tenant_id, keys)

        # All zrem operations should be in single pipeline
        mock_pipe.execute.assert_called_once()
