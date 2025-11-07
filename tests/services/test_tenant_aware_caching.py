"""
Tests for tenant-aware caching service.

Tests namespace isolation per ADR-004:
- tenant:{tenant_id}:* for tenant-specific data
- shared:market:* for shared market data
- meta:tenant:{tenant_id}:* for metadata

Validates that tenants cannot access each other's data and that shared
market data is accessible to all tenants.
"""

import pytest
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

from alpha_pulse.services.tenant_aware_caching import TenantAwareCachingService

# Apply asyncio marker to all tests in this module
pytestmark = pytest.mark.asyncio


@pytest.fixture
def tenant1_id() -> UUID:
    """First tenant UUID."""
    return UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def tenant2_id() -> UUID:
    """Second tenant UUID."""
    return UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock()
    redis.set = AsyncMock()
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    redis.exists = AsyncMock()
    redis.mget = AsyncMock()
    redis.mset = AsyncMock()
    return redis


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker service."""
    tracker = AsyncMock()
    tracker.increment_usage = AsyncMock(return_value=50.0)
    tracker.get_current_usage = AsyncMock(return_value=50.0)
    return tracker


@pytest.fixture
def mock_lru_tracker():
    """Mock LRU tracker service."""
    tracker = AsyncMock()
    tracker.track_access = AsyncMock()
    return tracker


@pytest.fixture
def mock_shared_cache():
    """Mock shared market cache."""
    cache = AsyncMock()
    cache.get_ohlcv = AsyncMock()
    cache.set_ohlcv = AsyncMock()
    cache.get_ticker = AsyncMock()
    cache.set_ticker = AsyncMock()
    return cache


@pytest.fixture
def tenant_cache(
    mock_redis,
    mock_usage_tracker,
    mock_lru_tracker,
    mock_shared_cache
):
    """Tenant-aware caching service instance."""
    return TenantAwareCachingService(
        redis_client=mock_redis,
        usage_tracker=mock_usage_tracker,
        lru_tracker=mock_lru_tracker,
        shared_cache=mock_shared_cache
    )


class TestNamespaceGeneration:
    """Test namespace key generation per ADR-004."""

    def test_tenant_signal_key_format(self, tenant_cache, tenant1_id):
        """Tenant signal keys follow tenant:{id}:signals:* format."""
        key = tenant_cache.get_tenant_key(
            tenant_id=tenant1_id,
            key="signals:technical:BTC_USDT"
        )

        assert key.startswith("tenant:")
        assert str(tenant1_id) in key
        assert "signals:technical:BTC_USDT" in key
        assert key == f"tenant:{tenant1_id}:signals:technical:BTC_USDT"

    def test_tenant_portfolio_key_format(self, tenant_cache, tenant1_id):
        """Tenant portfolio keys follow tenant:{id}:portfolio:* format."""
        key = tenant_cache.get_tenant_key(
            tenant_id=tenant1_id,
            key="portfolio:positions"
        )

        assert key == f"tenant:{tenant1_id}:portfolio:positions"

    def test_tenant_session_key_format(self, tenant_cache, tenant1_id):
        """Tenant session keys follow tenant:{id}:session:* format."""
        session_id = uuid4()
        key = tenant_cache.get_tenant_key(
            tenant_id=tenant1_id,
            key=f"session:{session_id}"
        )

        assert key == f"tenant:{tenant1_id}:session:{session_id}"

    def test_metadata_quota_key_format(self, tenant_cache, tenant1_id):
        """Metadata keys follow meta:tenant:{id}:* format."""
        key = tenant_cache.get_metadata_key(
            tenant_id=tenant1_id,
            key="quota"
        )

        assert key.startswith("meta:tenant:")
        assert str(tenant1_id) in key
        assert key == f"meta:tenant:{tenant1_id}:quota"

    def test_metadata_usage_key_format(self, tenant_cache, tenant1_id):
        """Usage metadata keys follow meta:tenant:{id}:usage format."""
        key = tenant_cache.get_metadata_key(
            tenant_id=tenant1_id,
            key="usage"
        )

        assert key == f"meta:tenant:{tenant1_id}:usage"


class TestTenantIsolation:
    """Test that tenant data is isolated (no cross-tenant access)."""

    async def test_set_tenant_data_uses_tenant_namespace(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Setting tenant data stores in tenant-specific namespace."""
        await tenant_cache.set(
            tenant_id=tenant1_id,
            key="signals:technical:BTC_USDT",
            value={"rsi": 65.5, "macd": 0.05},
            ttl=300
        )

        # Verify Redis set called with tenant-namespaced key
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]

        key = call_args[0]
        assert key.startswith(f"tenant:{tenant1_id}:")
        assert "signals:technical:BTC_USDT" in key

    async def test_get_tenant_data_reads_from_tenant_namespace(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Getting tenant data reads from tenant-specific namespace."""
        mock_redis.get.return_value = b'{"rsi": 65.5}'

        await tenant_cache.get(
            tenant_id=tenant1_id,
            key="signals:technical:BTC_USDT"
        )

        # Verify Redis get called with tenant-namespaced key
        mock_redis.get.assert_called_once()
        call_key = mock_redis.get.call_args[0][0]

        assert call_key.startswith(f"tenant:{tenant1_id}:")

    async def test_different_tenants_different_namespaces(
        self, tenant_cache, mock_redis, tenant1_id, tenant2_id
    ):
        """Different tenants write to different namespaces."""
        data = {"value": 42}

        # Tenant 1 writes
        await tenant_cache.set(tenant1_id, "test_key", data, ttl=60)
        tenant1_key = mock_redis.setex.call_args[0][0]

        # Tenant 2 writes same key
        await tenant_cache.set(tenant2_id, "test_key", data, ttl=60)
        tenant2_key = mock_redis.setex.call_args[0][0]

        # Keys should be different
        assert tenant1_key != tenant2_key
        assert str(tenant1_id) in tenant1_key
        assert str(tenant2_id) in tenant2_key

    async def test_tenant_cannot_access_other_tenant_data(
        self, tenant_cache, mock_redis, tenant1_id, tenant2_id
    ):
        """Tenant 1 cannot read Tenant 2's data by design."""
        # Tenant 1 tries to get data with their ID
        mock_redis.get.return_value = None  # Cache miss

        result = await tenant_cache.get(
            tenant_id=tenant1_id,
            key="signals:technical:BTC_USDT"
        )

        # Should only query tenant1's namespace
        call_key = mock_redis.get.call_args[0][0]
        assert str(tenant1_id) in call_key
        assert str(tenant2_id) not in call_key
        assert result is None


class TestSharedMarketDataAccess:
    """Test that shared market data is accessible to all tenants."""

    async def test_get_shared_ohlcv_delegates_to_shared_cache(
        self, tenant_cache, mock_shared_cache, tenant1_id
    ):
        """Getting shared OHLCV data delegates to SharedMarketCache."""
        mock_shared_cache.get_ohlcv.return_value = {
            "open": 50000, "high": 50100, "low": 49900, "close": 50050
        }

        result = await tenant_cache.get_shared_market_data(
            tenant_id=tenant1_id,
            data_type="ohlcv",
            exchange="binance",
            symbol="BTC_USDT",
            interval="1m"
        )

        # Should delegate to shared cache (no tenant_id in call)
        mock_shared_cache.get_ohlcv.assert_called_once_with(
            exchange="binance",
            symbol="BTC_USDT",
            interval="1m"
        )

        assert result is not None
        assert result["close"] == 50050

    async def test_multiple_tenants_read_same_shared_data(
        self, tenant_cache, mock_shared_cache, tenant1_id, tenant2_id
    ):
        """Multiple tenants can access the same shared market data."""
        ohlcv_data = {"close": 50050}
        mock_shared_cache.get_ohlcv.return_value = ohlcv_data

        # Tenant 1 reads
        result1 = await tenant_cache.get_shared_market_data(
            tenant_id=tenant1_id,
            data_type="ohlcv",
            exchange="binance",
            symbol="BTC_USDT",
            interval="1m"
        )

        # Tenant 2 reads
        result2 = await tenant_cache.get_shared_market_data(
            tenant_id=tenant2_id,
            data_type="ohlcv",
            exchange="binance",
            symbol="BTC_USDT",
            interval="1m"
        )

        # Both should get same data
        assert result1 == result2 == ohlcv_data

        # Shared cache should be called twice (once per tenant)
        assert mock_shared_cache.get_ohlcv.call_count == 2

    async def test_set_shared_ticker_delegates_to_shared_cache(
        self, tenant_cache, mock_shared_cache, tenant1_id
    ):
        """Setting shared ticker data delegates to SharedMarketCache."""
        ticker_data = {"last": 50050, "bid": 50040, "ask": 50060}

        await tenant_cache.set_shared_market_data(
            tenant_id=tenant1_id,
            data_type="ticker",
            exchange="binance",
            symbol="BTC_USDT",
            data=ticker_data,
            ttl=10
        )

        # Should delegate to shared cache
        mock_shared_cache.set_ticker.assert_called_once_with(
            exchange="binance",
            symbol="BTC_USDT",
            data=ticker_data,
            ttl_seconds=10
        )


class TestUsageTracking:
    """Test integration with UsageTracker (Story 4.3)."""

    async def test_set_increments_usage_counter(
        self, tenant_cache, mock_usage_tracker, mock_redis, tenant1_id
    ):
        """Setting data increments tenant usage counter."""
        data = {"key": "value" * 100}  # ~1KB
        mock_redis.setex.return_value = True

        await tenant_cache.set(
            tenant_id=tenant1_id,
            key="test_key",
            value=data,
            ttl=60
        )

        # Should track usage
        mock_usage_tracker.increment_usage.assert_called_once()
        call_args = mock_usage_tracker.increment_usage.call_args

        assert call_args[1]["tenant_id"] == tenant1_id
        assert call_args[1]["size_mb"] > 0  # Size in MB


class TestLRUTracking:
    """Test integration with LRUTracker (Story 4.4)."""

    async def test_get_tracks_lru_access(
        self, tenant_cache, mock_lru_tracker, mock_redis, tenant1_id
    ):
        """Getting data tracks LRU access."""
        mock_redis.get.return_value = b'{"value": 42}'

        await tenant_cache.get(
            tenant_id=tenant1_id,
            key="test_key"
        )

        # Should track access
        mock_lru_tracker.track_access.assert_called_once()
        call_args = mock_lru_tracker.track_access.call_args[1]

        assert call_args["tenant_id"] == tenant1_id
        assert "test_key" in call_args["key"]


class TestBatchOperations:
    """Test batch operations with tenant isolation."""

    async def test_mget_multiple_tenant_keys(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Getting multiple keys maintains tenant isolation."""
        keys = ["signals:technical:BTC_USDT", "signals:fundamental:ETH_USDT"]
        mock_redis.mget.return_value = [b'{"rsi": 65}', b'{"pe_ratio": 15}']

        await tenant_cache.mget(
            tenant_id=tenant1_id,
            keys=keys
        )

        # All keys should be tenant-namespaced
        mock_redis.mget.assert_called_once()
        call_keys = mock_redis.mget.call_args[0][0]

        for key in call_keys:
            assert str(tenant1_id) in key

    async def test_mset_multiple_tenant_keys(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Setting multiple keys maintains tenant isolation."""
        data = {
            "signals:technical:BTC_USDT": {"rsi": 65},
            "signals:fundamental:ETH_USDT": {"pe_ratio": 15}
        }

        await tenant_cache.mset(
            tenant_id=tenant1_id,
            data=data,
            ttl=300
        )

        # All keys should be tenant-namespaced
        mock_redis.setex.assert_called()
        for call in mock_redis.setex.call_args_list:
            key = call[0][0]
            assert str(tenant1_id) in key


class TestKeyExpiration:
    """Test that key expiration works correctly."""

    async def test_set_with_ttl_expires(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Setting data with TTL uses Redis SETEX."""
        await tenant_cache.set(
            tenant_id=tenant1_id,
            key="test_key",
            value={"data": "test"},
            ttl=60
        )

        # Should use setex with TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]

        ttl = call_args[1]
        assert ttl == 60

    async def test_different_data_types_different_ttls(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Different data types can have different TTLs."""
        # Signals: 5 minutes
        await tenant_cache.set(tenant1_id, "signals:test", {}, ttl=300)
        assert mock_redis.setex.call_args[0][1] == 300

        # Portfolio: 1 minute
        await tenant_cache.set(tenant1_id, "portfolio:positions", {}, ttl=60)
        assert mock_redis.setex.call_args[0][1] == 60

        # Session: 1 hour
        await tenant_cache.set(tenant1_id, "session:abc", {}, ttl=3600)
        assert mock_redis.setex.call_args[0][1] == 3600


class TestErrorHandling:
    """Test error handling in tenant-aware caching."""

    async def test_set_redis_error_raises(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Set operation propagates Redis errors."""
        mock_redis.setex.side_effect = Exception("Redis connection failed")

        with pytest.raises(Exception) as exc_info:
            await tenant_cache.set(
                tenant_id=tenant1_id,
                key="test_key",
                value={"data": "test"},
                ttl=60
            )

        assert "Redis connection failed" in str(exc_info.value)

    async def test_get_redis_error_raises(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Get operation propagates Redis errors."""
        mock_redis.get.side_effect = Exception("Redis timeout")

        with pytest.raises(Exception):
            await tenant_cache.get(
                tenant_id=tenant1_id,
                key="test_key"
            )


class TestCacheInvalidation:
    """Test cache invalidation for tenant data."""

    async def test_delete_tenant_key(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Deleting tenant key uses correct namespace."""
        mock_redis.delete.return_value = 1

        await tenant_cache.delete(
            tenant_id=tenant1_id,
            key="signals:technical:BTC_USDT"
        )

        # Should delete tenant-namespaced key
        mock_redis.delete.assert_called_once()
        call_key = mock_redis.delete.call_args[0][0]

        assert str(tenant1_id) in call_key
        assert "signals:technical:BTC_USDT" in call_key

    async def test_delete_multiple_tenant_keys_pattern(
        self, tenant_cache, mock_redis, tenant1_id
    ):
        """Deleting keys by pattern maintains tenant isolation."""
        # Mock Redis scan to return tenant keys
        async def mock_scan_iter(match):
            keys = [
                f"tenant:{tenant1_id}:signals:technical:BTC_USDT".encode(),
                f"tenant:{tenant1_id}:signals:technical:ETH_USDT".encode()
            ]
            for key in keys:
                yield key

        mock_redis.scan_iter = mock_scan_iter
        mock_redis.delete.return_value = 2

        count = await tenant_cache.delete_pattern(
            tenant_id=tenant1_id,
            pattern="signals:technical:*"
        )

        assert count == 2
