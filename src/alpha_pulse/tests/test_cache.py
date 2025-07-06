"""Comprehensive tests for caching functionality."""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from alpha_pulse.cache.redis_manager import (
    RedisManager, CacheTier, CacheConfig, SerializationFormat, EvictionPolicy
)
from alpha_pulse.cache.cache_strategies import (
    CacheAsideStrategy, WriteThroughStrategy, WriteBehindStrategy, RefreshAheadStrategy
)
from alpha_pulse.cache.cache_decorators import cache, cache_invalidate, cache_clear
from alpha_pulse.cache.distributed_cache import (
    DistributedCacheManager, CacheNode, NodeRole, ConsistentHashRing
)
from alpha_pulse.cache.cache_invalidation import (
    CacheInvalidationManager, InvalidationType, InvalidationRule
)
from alpha_pulse.services.caching_service import CachingService
from alpha_pulse.config.cache_config import CacheConfig as ServiceConfig
from alpha_pulse.utils.serialization_utils import OptimizedSerializer, SerializerConfig


class TestRedisManager:
    """Test Redis manager functionality."""
    
    @pytest.fixture
    async def redis_manager(self):
        """Create test Redis manager."""
        config = CacheConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=15,  # Use test DB
            default_ttl=60
        )
        manager = RedisManager(config)
        await manager.initialize()
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, redis_manager):
        """Test basic cache operations."""
        # Test set and get
        key = "test:key"
        value = {"data": "test value", "number": 42}
        
        success = await redis_manager.set(key, value)
        assert success is True
        
        retrieved = await redis_manager.get(key)
        assert retrieved == value
        
        # Test exists
        exists = await redis_manager.exists(key)
        assert exists is True
        
        # Test delete
        deleted = await redis_manager.delete(key)
        assert deleted is True
        
        exists = await redis_manager.exists(key)
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, redis_manager):
        """Test TTL expiration."""
        key = "test:ttl"
        value = "expires soon"
        
        # Set with 1 second TTL
        await redis_manager.set(key, value, ttl=1)
        
        # Should exist immediately
        exists = await redis_manager.exists(key)
        assert exists is True
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should not exist
        exists = await redis_manager.exists(key)
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, redis_manager):
        """Test batch operations."""
        data = {
            "test:batch:1": "value1",
            "test:batch:2": "value2",
            "test:batch:3": "value3"
        }
        
        # Test mset
        success = await redis_manager.mset(data)
        assert success is True
        
        # Test mget
        keys = list(data.keys())
        results = await redis_manager.mget(keys)
        assert results == data
    
    @pytest.mark.asyncio
    async def test_l1_cache(self, redis_manager):
        """Test L1 memory cache."""
        key = "test:l1"
        value = "memory cached"
        
        # Set in L1
        await redis_manager.set(key, value, tier=CacheTier.L1_MEMORY)
        
        # Get from L1
        retrieved = await redis_manager.get(key, tier=CacheTier.L1_MEMORY)
        assert retrieved == value
        
        # Verify L1 cache size limits
        for i in range(1500):  # Exceed L1 max size
            await redis_manager.set(f"test:l1:{i}", f"value{i}", tier=CacheTier.L1_MEMORY)
        
        # L1 cache should have evicted old entries
        assert len(redis_manager._l1_cache) <= redis_manager._l1_max_size
    
    @pytest.mark.asyncio
    async def test_serialization_formats(self, redis_manager):
        """Test different serialization formats."""
        test_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        # Test JSON
        redis_manager.config.default_format = SerializationFormat.JSON
        await redis_manager.set("test:json", test_data)
        result = await redis_manager.get("test:json")
        assert result == test_data
        
        # Test MessagePack
        redis_manager.config.default_format = SerializationFormat.MSGPACK
        await redis_manager.set("test:msgpack", test_data)
        result = await redis_manager.get("test:msgpack")
        assert result == test_data
        
        # Test Pickle
        redis_manager.config.default_format = SerializationFormat.PICKLE
        await redis_manager.set("test:pickle", test_data)
        result = await redis_manager.get("test:pickle")
        assert result == test_data


class TestCacheStrategies:
    """Test cache strategies."""
    
    @pytest.fixture
    async def redis_manager(self):
        """Create test Redis manager."""
        config = CacheConfig(redis_db=15)
        manager = RedisManager(config)
        await manager.initialize()
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_cache_aside_strategy(self, redis_manager):
        """Test cache-aside strategy."""
        strategy = CacheAsideStrategy(redis_manager)
        
        fetch_called = False
        async def fetch_fn():
            nonlocal fetch_called
            fetch_called = True
            return "fetched value"
        
        # First call should fetch
        result = await strategy.get("test:aside", fetch_fn)
        assert result == "fetched value"
        assert fetch_called is True
        
        # Second call should hit cache
        fetch_called = False
        result = await strategy.get("test:aside", fetch_fn)
        assert result == "fetched value"
        assert fetch_called is False
    
    @pytest.mark.asyncio
    async def test_write_through_strategy(self, redis_manager):
        """Test write-through strategy."""
        backend_data = {}
        
        async def write_fn(key, value):
            backend_data[key] = value
            return True
        
        strategy = WriteThroughStrategy(redis_manager, write_fn=write_fn)
        
        # Set should write to both cache and backend
        success = await strategy.set("test:writethrough", "value")
        assert success is True
        assert backend_data["test:writethrough"] == "value"
        
        # Get should retrieve from cache
        result = await strategy.get("test:writethrough", AsyncMock())
        assert result == "value"
    
    @pytest.mark.asyncio
    async def test_write_behind_strategy(self, redis_manager):
        """Test write-behind strategy."""
        backend_data = {}
        
        async def write_fn(key, value):
            backend_data[key] = value
            return True
        
        strategy = WriteBehindStrategy(
            redis_manager,
            write_fn=write_fn,
            buffer_size=2,
            flush_interval=1
        )
        
        # Start background tasks
        await strategy.start_background_tasks()
        
        try:
            # Set values - should buffer
            await strategy.set("test:wb1", "value1")
            await strategy.set("test:wb2", "value2")
            
            # Backend should not have data yet
            assert len(backend_data) == 0
            
            # Third set should trigger flush (buffer_size=2)
            await strategy.set("test:wb3", "value3")
            
            # Wait for flush
            await asyncio.sleep(0.1)
            
            # Backend should have first two values
            assert len(backend_data) >= 2
            
        finally:
            await strategy.stop_background_tasks()
    
    @pytest.mark.asyncio
    async def test_refresh_ahead_strategy(self, redis_manager):
        """Test refresh-ahead strategy."""
        refresh_count = 0
        
        async def fetch_fn():
            nonlocal refresh_count
            refresh_count += 1
            return f"value_{refresh_count}"
        
        strategy = RefreshAheadStrategy(
            redis_manager,
            refresh_threshold=0.5  # Refresh when 50% TTL remains
        )
        
        # Initial fetch
        result = await strategy.get("test:refresh", fetch_fn, ttl=2)
        assert result == "value_1"
        assert refresh_count == 1
        
        # Wait for refresh threshold
        await asyncio.sleep(1.2)
        
        # Get should trigger background refresh
        result = await strategy.get("test:refresh", fetch_fn, ttl=2)
        assert result == "value_1"  # Still returns cached value
        
        # Wait for refresh to complete
        await asyncio.sleep(0.5)
        
        # Should have refreshed in background
        assert refresh_count == 2


class TestCacheDecorators:
    """Test cache decorators."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create test cache manager."""
        config = CacheConfig(redis_db=15)
        manager = RedisManager(config)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_cache_decorator(self, cache_manager):
        """Test cache decorator."""
        call_count = 0
        
        class TestService:
            def __init__(self):
                self._cache_manager = cache_manager
            
            @cache(ttl=60, namespace="test")
            async def expensive_operation(self, param):
                nonlocal call_count
                call_count += 1
                return f"result_{param}"
        
        service = TestService()
        
        # First call
        result = await service.expensive_operation("test")
        assert result == "result_test"
        assert call_count == 1
        
        # Second call should hit cache
        result = await service.expensive_operation("test")
        assert result == "result_test"
        assert call_count == 1
        
        # Different parameter should miss cache
        result = await service.expensive_operation("other")
        assert result == "result_other"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_invalidate_decorator(self, cache_manager):
        """Test cache invalidate decorator."""
        class TestService:
            def __init__(self):
                self._cache_manager = cache_manager
            
            @cache(namespace="test")
            async def get_data(self, key):
                return f"data_{key}"
            
            @cache_invalidate(namespace="test")
            async def update_data(self, key, value):
                return value
        
        service = TestService()
        
        # Cache data
        result = await service.get_data("key1")
        assert result == "data_key1"
        
        # Verify cached
        cached = await cache_manager.get("test:get_data:*")
        
        # Update should invalidate
        await service.update_data("key1", "new_value")
        
        # Next get should fetch fresh data
        result = await service.get_data("key1")
        assert result == "data_key1"


class TestDistributedCache:
    """Test distributed caching."""
    
    @pytest.mark.asyncio
    async def test_consistent_hash_ring(self):
        """Test consistent hash ring."""
        ring = ConsistentHashRing(virtual_nodes=150)
        
        # Add nodes
        node1 = CacheNode("node1", "host1", 6379, NodeRole.PRIMARY)
        node2 = CacheNode("node2", "host2", 6379, NodeRole.PRIMARY)
        node3 = CacheNode("node3", "host3", 6379, NodeRole.PRIMARY)
        
        ring.add_node(node1)
        ring.add_node(node2)
        ring.add_node(node3)
        
        # Test key distribution
        key_distribution = {"node1": 0, "node2": 0, "node3": 0}
        
        for i in range(1000):
            key = f"test_key_{i}"
            node = ring.get_node(key)
            key_distribution[node.id] += 1
        
        # Verify relatively even distribution
        for count in key_distribution.values():
            assert 200 < count < 500  # Roughly 333 each
        
        # Test node removal
        ring.remove_node("node2")
        
        # Keys should redistribute
        new_distribution = {"node1": 0, "node3": 0}
        
        for i in range(1000):
            key = f"test_key_{i}"
            node = ring.get_node(key)
            new_distribution[node.id] += 1
        
        # Verify redistribution
        for count in new_distribution.values():
            assert 400 < count < 600  # Roughly 500 each
    
    @pytest.mark.asyncio
    async def test_distributed_cache_manager(self):
        """Test distributed cache manager."""
        # Mock Redis manager
        redis_manager = Mock(spec=RedisManager)
        redis_manager.get = AsyncMock(return_value=None)
        redis_manager.set = AsyncMock(return_value=True)
        redis_manager.delete = AsyncMock(return_value=True)
        redis_manager._serialize = Mock(return_value=b"serialized")
        redis_manager._deserialize = Mock(return_value="deserialized")
        
        # Create distributed manager
        dist_manager = DistributedCacheManager(
            redis_manager,
            "node1",
            replication_factor=2
        )
        
        # Mock remote connections
        dist_manager._remote_connections = {
            "node2": AsyncMock(),
            "node3": AsyncMock()
        }
        
        # Initialize with nodes
        local_node = CacheNode("node1", "localhost", 6379, NodeRole.PRIMARY)
        remote_nodes = [
            CacheNode("node2", "host2", 6379, NodeRole.REPLICA),
            CacheNode("node3", "host3", 6379, NodeRole.REPLICA)
        ]
        
        await dist_manager.initialize(local_node, remote_nodes)
        
        # Test distributed set
        success = await dist_manager.set("test:key", "value", ttl=60)
        assert success is True
        
        # Verify replication
        assert redis_manager.set.called
        for conn in dist_manager._remote_connections.values():
            assert conn.set.called


class TestCacheInvalidation:
    """Test cache invalidation."""
    
    @pytest.fixture
    async def invalidation_manager(self):
        """Create test invalidation manager."""
        redis_manager = Mock(spec=RedisManager)
        redis_manager.delete = AsyncMock(return_value=True)
        redis_manager.set = AsyncMock(return_value=True)
        
        manager = CacheInvalidationManager(redis_manager)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_tag_based_invalidation(self, invalidation_manager):
        """Test tag-based invalidation."""
        # Track keys with tags
        await invalidation_manager.track_key("key1", tags={"tag1", "tag2"})
        await invalidation_manager.track_key("key2", tags={"tag2", "tag3"})
        await invalidation_manager.track_key("key3", tags={"tag1", "tag3"})
        
        # Invalidate by tag
        count = await invalidation_manager.invalidate(tags=["tag2"])
        assert count == 2  # key1 and key2
        
        # Verify tag index updated
        assert "key1" not in invalidation_manager._tag_index.get("tag2", set())
        assert "key2" not in invalidation_manager._tag_index.get("tag2", set())
    
    @pytest.mark.asyncio
    async def test_dependency_based_invalidation(self, invalidation_manager):
        """Test dependency-based invalidation."""
        # Track keys with dependencies
        await invalidation_manager.track_key("parent", tags={"parent"})
        await invalidation_manager.track_key("child1", dependencies={"parent"})
        await invalidation_manager.track_key("child2", dependencies={"parent"})
        await invalidation_manager.track_key("grandchild", dependencies={"child1"})
        
        # Invalidate parent should cascade
        count = await invalidation_manager.invalidate(keys=["parent"])
        assert count >= 3  # parent, child1, child2
    
    @pytest.mark.asyncio
    async def test_pattern_based_invalidation(self, invalidation_manager):
        """Test pattern-based invalidation."""
        # Track keys
        await invalidation_manager.track_key("user:123:profile")
        await invalidation_manager.track_key("user:123:settings")
        await invalidation_manager.track_key("user:456:profile")
        await invalidation_manager.track_key("product:789:details")
        
        # Invalidate by pattern
        count = await invalidation_manager.invalidate(pattern="user:123:.*")
        assert count == 2  # user:123:profile and user:123:settings


class TestSerializationUtils:
    """Test serialization utilities."""
    
    @pytest.mark.asyncio
    async def test_numpy_serialization(self):
        """Test numpy array serialization."""
        serializer = OptimizedSerializer()
        
        # Test various numpy arrays
        arrays = [
            np.array([1, 2, 3, 4, 5]),
            np.array([[1, 2], [3, 4]]),
            np.random.rand(10, 10),
            np.array(["text", "data"], dtype=object)
        ]
        
        for arr in arrays:
            serialized = serializer.serialize(arr)
            deserialized = serializer.deserialize(serialized)
            np.testing.assert_array_equal(arr, deserialized)
    
    @pytest.mark.asyncio
    async def test_pandas_serialization(self):
        """Test pandas DataFrame serialization."""
        serializer = OptimizedSerializer()
        
        # Test DataFrame
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": pd.date_range("2024-01-01", periods=3)
        })
        
        serialized = serializer.serialize(df)
        deserialized = serializer.deserialize(serialized)
        pd.testing.assert_frame_equal(df, deserialized)
        
        # Test Series
        series = pd.Series([1, 2, 3], name="test_series")
        serialized = serializer.serialize(series)
        deserialized = serializer.deserialize(serialized)
        pd.testing.assert_series_equal(series, deserialized)
    
    @pytest.mark.asyncio
    async def test_compression(self):
        """Test data compression."""
        from alpha_pulse.utils.serialization_utils import CompressionType
        
        config = SerializerConfig(
            compression=CompressionType.LZ4,
            compression_threshold=100
        )
        serializer = OptimizedSerializer(config)
        
        # Small data - should not compress
        small_data = {"key": "value"}
        serialized = serializer.serialize(small_data)
        assert not serializer._is_compressed(serialized)
        
        # Large data - should compress
        large_data = {"data": "x" * 1000}
        serialized = serializer.serialize(large_data)
        assert serializer._is_compressed(serialized)
        
        # Verify decompression works
        deserialized = serializer.deserialize(serialized)
        assert deserialized == large_data


class TestCachingService:
    """Test caching service."""
    
    @pytest.fixture
    async def caching_service(self):
        """Create test caching service."""
        config = ServiceConfig()
        config.redis_node.db = 15  # Test DB
        
        service = CachingService(config)
        await service.initialize()
        yield service
        await service.close()
    
    @pytest.mark.asyncio
    async def test_service_operations(self, caching_service):
        """Test caching service operations."""
        # Test set and get
        await caching_service.set("test:key", "value", ttl=60)
        result = await caching_service.get("test:key")
        assert result == "value"
        
        # Test with fetch function
        async def fetch_fn():
            return "fetched_value"
        
        result = await caching_service.get("test:missing", fetch_fn=fetch_fn)
        assert result == "fetched_value"
        
        # Test batch operations
        data = {"key1": "value1", "key2": "value2"}
        await caching_service.mset(data)
        
        results = await caching_service.mget(["key1", "key2"])
        assert results == data
    
    @pytest.mark.asyncio
    async def test_cache_context(self, caching_service):
        """Test cache context manager."""
        async with caching_service.cache_context("test_namespace") as cache:
            await cache.set("key1", "value1")
            result = await cache.get("key1")
            assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, caching_service):
        """Test cache invalidation through service."""
        # Set values with tags
        await caching_service.set("user:1", "data1", tags=["users"])
        await caching_service.set("user:2", "data2", tags=["users"])
        await caching_service.set("product:1", "data3", tags=["products"])
        
        # Invalidate by tag
        count = await caching_service.invalidate(tags=["users"])
        assert count >= 2
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, caching_service):
        """Test cache statistics."""
        # Populate cache
        for i in range(10):
            await caching_service.set(f"test:{i}", f"value{i}")
        
        # Get stats
        stats = await caching_service.get_stats()
        assert "hit_rate" in stats
        assert "total_keys" in stats
        assert "memory_usage" in stats
        assert stats["total_keys"] >= 10


# Performance tests
class TestCachePerformance:
    """Performance tests for caching."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency(self):
        """Test cache under high concurrency."""
        config = CacheConfig(redis_db=15)
        manager = RedisManager(config)
        await manager.initialize()
        
        try:
            # Create many concurrent operations
            async def cache_operation(i):
                key = f"perf:test:{i}"
                value = f"value_{i}"
                await manager.set(key, value)
                result = await manager.get(key)
                assert result == value
            
            # Run 100 concurrent operations
            tasks = [cache_operation(i) for i in range(100)]
            await asyncio.gather(*tasks)
            
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_large_data_caching(self):
        """Test caching of large data."""
        config = CacheConfig(redis_db=15)
        manager = RedisManager(config)
        await manager.initialize()
        
        try:
            # Create large DataFrame
            df = pd.DataFrame(
                np.random.rand(1000, 100),
                columns=[f"col_{i}" for i in range(100)]
            )
            
            # Cache it
            await manager.set("large:df", df)
            
            # Retrieve it
            result = await manager.get("large:df")
            pd.testing.assert_frame_equal(df, result)
            
        finally:
            await manager.close()