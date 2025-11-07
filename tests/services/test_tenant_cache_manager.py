"""
Tests for TenantCacheManager - RED Phase (Story 4.1).

This test suite validates tenant namespace isolation, quota enforcement,
and cache operations with multi-tenant awareness.
"""
import pytest
import asyncio
from uuid import UUID, uuid4
from datetime import datetime
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from decimal import Decimal

from alpha_pulse.services.tenant_cache_manager import (
    TenantCacheManager,
    QuotaExceededException
)


@pytest.fixture
def tenant_a_id():
    """Tenant A UUID fixture."""
    return UUID('00000000-0000-0000-0000-000000000001')


@pytest.fixture
def tenant_b_id():
    """Tenant B UUID fixture."""
    return UUID('00000000-0000-0000-0000-000000000002')


@pytest.fixture
def mock_redis_manager():
    """Mock RedisManager fixture."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.mget = AsyncMock(return_value={})
    mock.incr = AsyncMock(return_value=1)
    mock.incrby = AsyncMock(return_value=100)
    mock.decrby = AsyncMock(return_value=0)

    # Mock pipeline
    pipeline_mock = AsyncMock()
    pipeline_mock.execute = AsyncMock(return_value=[True, True])
    pipeline_mock.incrby = Mock(return_value=None)
    pipeline_mock.zadd = Mock(return_value=None)
    pipeline_mock.decrby = Mock(return_value=None)
    pipeline_mock.zrem = Mock(return_value=None)
    mock.pipeline = Mock(return_value=pipeline_mock)

    return mock


@pytest.fixture
def mock_db_session():
    """Mock database session fixture."""
    mock = AsyncMock()

    # Mock execute result
    result_mock = AsyncMock()
    result_mock.fetchone = Mock(return_value=[100])  # 100MB quota
    mock.execute = AsyncMock(return_value=result_mock)

    return mock


@pytest.fixture
async def cache_manager(mock_redis_manager, mock_db_session):
    """TenantCacheManager fixture."""
    manager = TenantCacheManager(
        redis_manager=mock_redis_manager,
        db_session=mock_db_session
    )
    return manager


class TestNamespaceIsolation:
    """Test tenant namespace isolation."""

    @pytest.mark.asyncio
    async def test_build_key_with_namespace(self, cache_manager, tenant_a_id):
        """Test namespace key building."""
        key = cache_manager._build_key(tenant_a_id, 'signals', 'technical:BTC')

        expected = f"tenant:{tenant_a_id}:signals:technical:BTC"
        assert key == expected

    @pytest.mark.asyncio
    async def test_get_uses_namespaced_key(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test get() uses tenant namespace."""
        # Mock cache hit
        mock_redis_manager.get.return_value = b'{"price": 67000}'

        result = await cache_manager.get(tenant_a_id, 'signals', 'BTC')

        # Verify namespace used
        expected_key = f"tenant:{tenant_a_id}:signals:BTC"
        mock_redis_manager.get.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_set_uses_namespaced_key(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test set() uses tenant namespace."""
        # Mock quota OK
        mock_redis_manager.get.return_value = b'5242880'  # 5MB usage

        await cache_manager.set(
            tenant_a_id,
            'signals',
            'BTC',
            {'price': 67000},
            ttl=300
        )

        # Verify namespace used in set call
        calls = mock_redis_manager.set.call_args_list
        assert len(calls) == 1

        namespaced_key = calls[0][0][0]
        expected_key = f"tenant:{tenant_a_id}:signals:BTC"
        assert namespaced_key == expected_key

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id,
        tenant_b_id
    ):
        """Test tenant A cannot access tenant B's cache."""
        # Tenant A sets value
        mock_redis_manager.get.return_value = b'5242880'  # 5MB usage
        await cache_manager.set(tenant_a_id, 'signals', 'BTC', {'price': 67000})

        # Reset mock
        mock_redis_manager.get.return_value = None

        # Tenant B tries to get Tenant A's value
        result = await cache_manager.get(tenant_b_id, 'signals', 'BTC')

        # Should be None (different namespace)
        assert result is None

        # Verify different keys were used
        tenant_a_key = f"tenant:{tenant_a_id}:signals:BTC"
        tenant_b_key = f"tenant:{tenant_b_id}:signals:BTC"
        assert tenant_a_key != tenant_b_key


class TestCacheOperations:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_get_cache_hit(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test cache get with hit."""
        # Mock cache hit
        mock_redis_manager.get.return_value = b'{"price": 67000, "symbol": "BTC"}'

        result = await cache_manager.get(tenant_a_id, 'signals', 'technical:BTC')

        assert result is not None
        assert result['price'] == 67000
        assert result['symbol'] == 'BTC'

    @pytest.mark.asyncio
    async def test_get_cache_miss(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test cache get with miss."""
        # Mock cache miss
        mock_redis_manager.get.return_value = None

        result = await cache_manager.get(tenant_a_id, 'signals', 'technical:BTC')

        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test cache set with TTL."""
        # Mock quota OK
        mock_redis_manager.get.return_value = b'5242880'  # 5MB usage

        value = {'price': 67000, 'timestamp': datetime.utcnow().isoformat()}
        await cache_manager.set(
            tenant_a_id,
            'signals',
            'technical:BTC',
            value,
            ttl=300
        )

        # Verify set called with TTL
        calls = mock_redis_manager.set.call_args_list
        assert len(calls) == 1
        assert calls[0][1]['ttl'] == 300

    @pytest.mark.asyncio
    async def test_delete_removes_key(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test cache delete."""
        # Mock existing value
        mock_redis_manager.get.return_value = b'{"price": 67000}'

        deleted = await cache_manager.delete(tenant_a_id, 'signals', 'technical:BTC')

        assert deleted is True
        mock_redis_manager.delete.assert_called_once()


class TestQuotaEnforcement:
    """Test quota enforcement logic."""

    @pytest.mark.asyncio
    async def test_set_under_quota_succeeds(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test set succeeds when under quota."""
        # Mock usage: 50MB (under 100MB quota)
        mock_redis_manager.get.return_value = b'52428800'

        result = await cache_manager.set(
            tenant_a_id,
            'signals',
            'BTC',
            {'price': 67000},
            ttl=300
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_set_over_quota_triggers_eviction(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test eviction triggered when quota exceeded."""
        # Mock usage: 98MB (over 100MB quota)
        mock_redis_manager.get.return_value = b'102760448'  # 98MB

        # Mock zpopmin (eviction)
        mock_redis_manager.zpopmin = AsyncMock(return_value=[
            (b'tenant:xxx:signals:ETH', 1234567890.0)
        ])

        # This should trigger eviction
        await cache_manager.set(
            tenant_a_id,
            'signals',
            'BTC',
            {'price': 67000},
            ttl=300
        )

        # Verify eviction was called
        assert mock_redis_manager.zpopmin.called

    @pytest.mark.asyncio
    async def test_quota_exceeded_exception(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test exception raised when quota cannot be freed."""
        # Mock usage: 110MB (over quota, cannot evict)
        mock_redis_manager.get.return_value = b'115343360'  # 110MB

        # Mock zpopmin returns no keys (cannot evict)
        mock_redis_manager.zpopmin = AsyncMock(return_value=[])

        with pytest.raises(QuotaExceededException) as exc_info:
            await cache_manager.set(
                tenant_a_id,
                'signals',
                'BTC',
                {'price': 67000},
                ttl=300
            )

        assert exc_info.value.tenant_id == tenant_a_id

    @pytest.mark.asyncio
    async def test_get_usage(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test get_usage returns current usage."""
        # Mock usage counter
        mock_redis_manager.get.return_value = b'52428800'  # 50MB

        usage = await cache_manager.get_usage(tenant_a_id)

        assert usage == 52428800  # 50MB in bytes

    @pytest.mark.asyncio
    async def test_get_quota(
        self,
        cache_manager,
        mock_db_session,
        tenant_a_id
    ):
        """Test get_quota fetches from database."""
        # Mock database query (100MB quota)
        result_mock = AsyncMock()
        result_mock.fetchone = Mock(return_value=[100])
        mock_db_session.execute = AsyncMock(return_value=result_mock)

        quota = await cache_manager._get_quota(tenant_a_id)

        assert quota == 104857600  # 100MB in bytes
        mock_db_session.execute.assert_called_once()


class TestMetrics:
    """Test metrics collection."""

    @pytest.mark.asyncio
    async def test_get_increments_hit_counter(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test cache hit increments counter."""
        # Mock cache hit
        mock_redis_manager.get.return_value = b'{"price": 67000}'

        await cache_manager.get(tenant_a_id, 'signals', 'BTC')

        # Verify hit counter incremented
        assert mock_redis_manager.incr.called
        hit_key = f"meta:tenant:{tenant_a_id}:hits"
        mock_redis_manager.incr.assert_called_with(hit_key)

    @pytest.mark.asyncio
    async def test_get_increments_miss_counter(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test cache miss increments counter."""
        # Mock cache miss
        mock_redis_manager.get.return_value = None

        await cache_manager.get(tenant_a_id, 'signals', 'BTC')

        # Verify miss counter incremented
        assert mock_redis_manager.incr.called
        miss_key = f"meta:tenant:{tenant_a_id}:misses"
        mock_redis_manager.incr.assert_called_with(miss_key)

    @pytest.mark.asyncio
    async def test_get_metrics_calculates_hit_rate(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test get_metrics calculates hit rate correctly."""
        # Mock counters: 80 hits, 20 misses
        def mock_mget(keys):
            result = {}
            for key in keys:
                if 'hits' in key:
                    result[key] = b'80'
                elif 'misses' in key:
                    result[key] = b'20'
                elif 'usage_bytes' in key:
                    result[key] = b'52428800'  # 50MB
            return result

        mock_redis_manager.mget = AsyncMock(side_effect=mock_mget)

        # Mock database query for quota
        result_mock = AsyncMock()
        result_mock.fetchone = Mock(return_value=[100])
        cache_manager.db.execute = AsyncMock(return_value=result_mock)

        metrics = await cache_manager.get_metrics(tenant_a_id)

        assert metrics['hits'] == 80
        assert metrics['misses'] == 20
        assert metrics['hit_rate'] == 80.0  # 80 / (80 + 20) * 100
        assert metrics['usage_bytes'] == 52428800
        assert metrics['quota_bytes'] == 104857600


class TestUsageTracking:
    """Test usage tracking with Redis counters and sorted sets."""

    @pytest.mark.asyncio
    async def test_track_usage_increments_counter(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test _track_usage increments usage counter."""
        cache_key = f"tenant:{tenant_a_id}:signals:BTC"
        payload_size = 1024  # 1KB

        await cache_manager._track_usage(tenant_a_id, cache_key, payload_size)

        # Verify pipeline was used
        assert mock_redis_manager.pipeline.called
        pipeline = mock_redis_manager.pipeline.return_value
        pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_usage_adds_to_lru(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test _track_usage adds key to LRU sorted set."""
        cache_key = f"tenant:{tenant_a_id}:signals:BTC"
        payload_size = 1024

        await cache_manager._track_usage(tenant_a_id, cache_key, payload_size)

        # Verify zadd was called on pipeline
        pipeline = mock_redis_manager.pipeline.return_value
        pipeline.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_untrack_usage_decrements_counter(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test _untrack_usage decrements usage counter."""
        cache_key = f"tenant:{tenant_a_id}:signals:BTC"
        payload_size = 1024

        await cache_manager._untrack_usage(tenant_a_id, cache_key, payload_size)

        # Verify pipeline was used with decrby
        pipeline = mock_redis_manager.pipeline.return_value
        pipeline.decrby.assert_called_once()
        pipeline.execute.assert_called_once()


class TestEviction:
    """Test LRU eviction logic."""

    @pytest.mark.asyncio
    async def test_evict_tenant_lru_removes_oldest_keys(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test eviction removes oldest keys first."""
        # Mock current usage: 100MB
        mock_redis_manager.get.return_value = b'104857600'

        # Mock zpopmin returns oldest keys
        mock_redis_manager.zpopmin = AsyncMock(return_value=[
            (b'tenant:xxx:signals:ETH', 1234567890.0),
            (b'tenant:xxx:signals:SOL', 1234567891.0),
        ])

        # Mock get for key sizes
        mock_redis_manager.get = AsyncMock(side_effect=[
            b'104857600',  # Initial usage
            b'{"price": 123}' * 100,  # ETH value
            b'{"price": 456}' * 100,  # SOL value
        ])

        target_size = 94371840  # 90MB
        bytes_evicted = await cache_manager._evict_tenant_lru(
            tenant_a_id,
            target_size
        )

        assert bytes_evicted > 0
        assert mock_redis_manager.zpopmin.called

    @pytest.mark.asyncio
    async def test_eviction_stops_when_target_reached(
        self,
        cache_manager,
        mock_redis_manager,
        tenant_a_id
    ):
        """Test eviction stops when target usage reached."""
        # Mock current usage: 95MB
        mock_redis_manager.get.return_value = b'99614720'

        # Mock zpopmin
        mock_redis_manager.zpopmin = AsyncMock(return_value=[
            (b'tenant:xxx:signals:ETH', 1234567890.0),
        ])

        # Mock key size: 10MB
        mock_redis_manager.get = AsyncMock(side_effect=[
            b'99614720',  # Initial usage
            b'x' * 10485760,  # 10MB value
        ])

        target_size = 94371840  # 90MB
        bytes_evicted = await cache_manager._evict_tenant_lru(
            tenant_a_id,
            target_size
        )

        # Should evict ~10MB to reach target
        assert bytes_evicted >= 10485760


class TestSerialization:
    """Test serialization/deserialization."""

    @pytest.mark.asyncio
    async def test_serialize_dict(self, cache_manager):
        """Test serialization of dict."""
        data = {'price': 67000, 'symbol': 'BTC', 'timestamp': '2025-11-07T10:00:00'}

        serialized = cache_manager._serialize(data)

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    @pytest.mark.asyncio
    async def test_deserialize_dict(self, cache_manager):
        """Test deserialization of dict."""
        data = {'price': 67000, 'symbol': 'BTC'}
        serialized = cache_manager._serialize(data)

        deserialized = cache_manager._deserialize(serialized)

        assert deserialized == data

    @pytest.mark.asyncio
    async def test_serialize_decimal(self, cache_manager):
        """Test serialization of Decimal values."""
        data = {'price': Decimal('67123.45'), 'volume': Decimal('123.456')}

        serialized = cache_manager._serialize(data)
        deserialized = cache_manager._deserialize(serialized)

        # Decimals should be preserved as strings
        assert deserialized['price'] == '67123.45'
        assert deserialized['volume'] == '123.456'


# Mark all tests as requiring asyncio
pytest_plugins = ('pytest_asyncio',)
