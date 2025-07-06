"""Redis cache manager for multi-tier caching architecture."""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import redis
import redis.asyncio as redis_async
from redis.exceptions import RedisError
from redis.sentinel import Sentinel
import msgpack

from ..monitoring.metrics import MetricsCollector
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class CacheTier(Enum):
    """Cache tier levels."""
    
    L1_MEMORY = "l1_memory"
    L2_LOCAL_REDIS = "l2_local_redis"
    L3_DISTRIBUTED_REDIS = "l3_distributed_redis"


class SerializationFormat(Enum):
    """Serialization formats for cache data."""
    
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    
    LRU = "lru"
    LFU = "lfu"
    TLRU = "tlru"
    ARC = "arc"


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    
    # Redis connection settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Cluster settings
    cluster_enabled: bool = False
    cluster_nodes: List[Dict[str, Any]] = None
    
    # Sentinel settings
    sentinel_enabled: bool = False
    sentinel_hosts: List[Tuple[str, int]] = None
    sentinel_service_name: str = "mymaster"
    
    # Cache settings
    default_ttl: int = 300  # 5 minutes
    max_memory: str = "1gb"
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    
    # Serialization settings
    default_format: SerializationFormat = SerializationFormat.MSGPACK
    compression_threshold: int = 1024  # bytes
    
    # Connection pool settings
    max_connections: int = 50
    connection_timeout: int = 20
    socket_timeout: int = 5
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.1


class RedisManager:
    """Manages Redis connections and operations for multi-tier caching."""
    
    def __init__(self, config: CacheConfig, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize Redis manager."""
        self.config = config
        self.metrics = metrics_collector
        self._connections: Dict[CacheTier, Union[redis.Redis, redis_async.Redis]] = {}
        self._sentinel: Optional[Sentinel] = None
        self._connection_pools: Dict[CacheTier, redis.ConnectionPool] = {}
        self._is_initialized = False
        
        # L1 memory cache (simple dict for now)
        self._l1_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._l1_max_size = 1000
        
    async def initialize(self) -> None:
        """Initialize Redis connections."""
        try:
            # Initialize connection pools
            await self._setup_connection_pools()
            
            # Initialize Redis connections for each tier
            await self._setup_connections()
            
            # Test connections
            await self._test_connections()
            
            self._is_initialized = True
            logger.info("Redis manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis manager: {e}")
            raise
    
    async def _setup_connection_pools(self) -> None:
        """Set up Redis connection pools."""
        pool_config = {
            'max_connections': self.config.max_connections,
            'socket_timeout': self.config.socket_timeout,
            'socket_connect_timeout': self.config.connection_timeout,
            'decode_responses': False,
        }
        
        if self.config.redis_password:
            pool_config['password'] = self.config.redis_password
        
        # L2 local Redis pool
        self._connection_pools[CacheTier.L2_LOCAL_REDIS] = redis_async.ConnectionPool(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            **pool_config
        )
        
        # L3 distributed Redis pool (cluster or sentinel)
        if self.config.cluster_enabled:
            # Use Redis cluster
            from redis.asyncio.cluster import RedisCluster
            self._connection_pools[CacheTier.L3_DISTRIBUTED_REDIS] = RedisCluster(
                startup_nodes=self.config.cluster_nodes,
                decode_responses=False,
                skip_full_coverage_check=True,
                **pool_config
            )
        elif self.config.sentinel_enabled:
            # Use Redis sentinel
            self._sentinel = Sentinel(
                self.config.sentinel_hosts,
                socket_timeout=self.config.socket_timeout
            )
        else:
            # Use same pool as L2 for simplicity
            self._connection_pools[CacheTier.L3_DISTRIBUTED_REDIS] = \
                self._connection_pools[CacheTier.L2_LOCAL_REDIS]
    
    async def _setup_connections(self) -> None:
        """Set up Redis connections."""
        # L2 local Redis
        self._connections[CacheTier.L2_LOCAL_REDIS] = redis_async.Redis(
            connection_pool=self._connection_pools[CacheTier.L2_LOCAL_REDIS]
        )
        
        # L3 distributed Redis
        if self.config.sentinel_enabled and self._sentinel:
            # Get master from sentinel
            master = await asyncio.get_event_loop().run_in_executor(
                None,
                self._sentinel.master_for,
                self.config.sentinel_service_name,
                self.config.socket_timeout
            )
            self._connections[CacheTier.L3_DISTRIBUTED_REDIS] = master
        else:
            self._connections[CacheTier.L3_DISTRIBUTED_REDIS] = redis_async.Redis(
                connection_pool=self._connection_pools[CacheTier.L3_DISTRIBUTED_REDIS]
            )
    
    async def _test_connections(self) -> None:
        """Test Redis connections."""
        for tier, conn in self._connections.items():
            try:
                await conn.ping()
                logger.info(f"Redis connection for {tier.value} successful")
            except Exception as e:
                logger.error(f"Redis connection for {tier.value} failed: {e}")
                raise
    
    def _serialize(self, data: Any, format: SerializationFormat = None) -> bytes:
        """Serialize data for cache storage."""
        format = format or self.config.default_format
        
        try:
            if format == SerializationFormat.JSON:
                return json.dumps(data).encode('utf-8')
            elif format == SerializationFormat.PICKLE:
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            elif format == SerializationFormat.MSGPACK:
                return msgpack.packb(data, use_bin_type=True)
            else:
                raise ValueError(f"Unknown serialization format: {format}")
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes, format: SerializationFormat = None) -> Any:
        """Deserialize data from cache storage."""
        if not data:
            return None
        
        format = format or self.config.default_format
        
        try:
            if format == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            elif format == SerializationFormat.PICKLE:
                return pickle.loads(data)
            elif format == SerializationFormat.MSGPACK:
                return msgpack.unpackb(data, raw=False)
            else:
                raise ValueError(f"Unknown serialization format: {format}")
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def get(self, key: str, tier: CacheTier = CacheTier.L2_LOCAL_REDIS) -> Optional[Any]:
        """Get value from cache."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Check L1 memory cache first
            if tier == CacheTier.L1_MEMORY or tier == CacheTier.L2_LOCAL_REDIS:
                if key in self._l1_cache:
                    value, expiry = self._l1_cache[key]
                    if expiry > datetime.utcnow():
                        if self.metrics:
                            self.metrics.increment("cache.hit", {"tier": "L1"})
                        return value
                    else:
                        # Expired, remove from L1
                        del self._l1_cache[key]
            
            # Check Redis cache
            if tier in self._connections:
                conn = self._connections[tier]
                data = await conn.get(key)
                
                if data:
                    if self.metrics:
                        self.metrics.increment("cache.hit", {"tier": tier.value})
                    
                    value = self._deserialize(data)
                    
                    # Populate L1 cache if accessing L2
                    if tier == CacheTier.L2_LOCAL_REDIS:
                        self._update_l1_cache(key, value)
                    
                    return value
                else:
                    if self.metrics:
                        self.metrics.increment("cache.miss", {"tier": tier.value})
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("cache.error", {"operation": "get", "tier": tier.value})
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set value in cache."""
        if not self._is_initialized:
            await self.initialize()
        
        ttl = ttl or self.config.default_ttl
        
        try:
            # Update L1 memory cache
            if tier == CacheTier.L1_MEMORY or tier == CacheTier.L2_LOCAL_REDIS:
                self._update_l1_cache(key, value, ttl)
            
            # Update Redis cache
            if tier in self._connections:
                conn = self._connections[tier]
                data = self._serialize(value)
                
                await conn.set(key, data, ex=ttl)
                
                if self.metrics:
                    self.metrics.increment("cache.set", {"tier": tier.value})
                    self.metrics.gauge("cache.key_size", len(data), {"tier": tier.value})
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("cache.error", {"operation": "set", "tier": tier.value})
            return False
    
    def _update_l1_cache(self, key: str, value: Any, ttl: int = None) -> None:
        """Update L1 memory cache with LRU eviction."""
        ttl = ttl or self.config.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        
        # Implement simple LRU by removing oldest entries if at capacity
        if len(self._l1_cache) >= self._l1_max_size and key not in self._l1_cache:
            # Remove oldest entry
            oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k][1])
            del self._l1_cache[oldest_key]
        
        self._l1_cache[key] = (value, expiry)
    
    async def delete(self, key: str, tier: CacheTier = None) -> bool:
        """Delete value from cache."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            deleted = False
            
            # Delete from L1 cache
            if key in self._l1_cache:
                del self._l1_cache[key]
                deleted = True
            
            # Delete from specified tier or all tiers
            tiers_to_delete = [tier] if tier else list(self._connections.keys())
            
            for t in tiers_to_delete:
                if t in self._connections:
                    conn = self._connections[t]
                    result = await conn.delete(key)
                    if result:
                        deleted = True
                        if self.metrics:
                            self.metrics.increment("cache.delete", {"tier": t.value})
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("cache.error", {"operation": "delete"})
            return False
    
    async def mget(self, keys: List[str], tier: CacheTier = CacheTier.L2_LOCAL_REDIS) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self._is_initialized:
            await self.initialize()
        
        result = {}
        
        try:
            # Check L1 cache first
            l1_hits = []
            redis_keys = []
            
            for key in keys:
                if key in self._l1_cache:
                    value, expiry = self._l1_cache[key]
                    if expiry > datetime.utcnow():
                        result[key] = value
                        l1_hits.append(key)
                    else:
                        del self._l1_cache[key]
                        redis_keys.append(key)
                else:
                    redis_keys.append(key)
            
            if l1_hits and self.metrics:
                self.metrics.increment("cache.hit", {"tier": "L1"}, len(l1_hits))
            
            # Get remaining from Redis
            if redis_keys and tier in self._connections:
                conn = self._connections[tier]
                values = await conn.mget(redis_keys)
                
                for key, data in zip(redis_keys, values):
                    if data:
                        value = self._deserialize(data)
                        result[key] = value
                        
                        # Update L1 cache
                        if tier == CacheTier.L2_LOCAL_REDIS:
                            self._update_l1_cache(key, value)
                        
                        if self.metrics:
                            self.metrics.increment("cache.hit", {"tier": tier.value})
                    else:
                        if self.metrics:
                            self.metrics.increment("cache.miss", {"tier": tier.value})
            
            return result
            
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            if self.metrics:
                self.metrics.increment("cache.error", {"operation": "mget", "tier": tier.value})
            return {}
    
    async def mset(
        self,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L2_LOCAL_REDIS
    ) -> bool:
        """Set multiple values in cache."""
        if not self._is_initialized:
            await self.initialize()
        
        ttl = ttl or self.config.default_ttl
        
        try:
            # Update L1 cache
            for key, value in data.items():
                if tier == CacheTier.L1_MEMORY or tier == CacheTier.L2_LOCAL_REDIS:
                    self._update_l1_cache(key, value, ttl)
            
            # Update Redis cache
            if tier in self._connections:
                conn = self._connections[tier]
                
                # Serialize all values
                serialized_data = {}
                for key, value in data.items():
                    serialized_data[key] = self._serialize(value)
                
                # Use pipeline for atomic multi-set
                pipe = conn.pipeline()
                for key, value in serialized_data.items():
                    pipe.set(key, value, ex=ttl)
                
                await pipe.execute()
                
                if self.metrics:
                    self.metrics.increment("cache.set", {"tier": tier.value}, len(data))
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            if self.metrics:
                self.metrics.increment("cache.error", {"operation": "mset", "tier": tier.value})
            return False
    
    async def exists(self, key: str, tier: CacheTier = CacheTier.L2_LOCAL_REDIS) -> bool:
        """Check if key exists in cache."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Check L1 cache
            if key in self._l1_cache:
                _, expiry = self._l1_cache[key]
                if expiry > datetime.utcnow():
                    return True
                else:
                    del self._l1_cache[key]
            
            # Check Redis cache
            if tier in self._connections:
                conn = self._connections[tier]
                return bool(await conn.exists(key))
            
            return False
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int, tier: CacheTier = None) -> bool:
        """Set expiration time for key."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            expired = False
            
            # Update L1 cache expiry
            if key in self._l1_cache:
                value, _ = self._l1_cache[key]
                self._l1_cache[key] = (value, datetime.utcnow() + timedelta(seconds=ttl))
                expired = True
            
            # Update Redis expiry
            tiers_to_update = [tier] if tier else list(self._connections.keys())
            
            for t in tiers_to_update:
                if t in self._connections:
                    conn = self._connections[t]
                    result = await conn.expire(key, ttl)
                    if result:
                        expired = True
            
            return expired
            
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None, tier: CacheTier = None) -> int:
        """Clear cache entries matching pattern."""
        if not self._is_initialized:
            await self.initialize()
        
        count = 0
        
        try:
            # Clear L1 cache
            if pattern:
                keys_to_delete = [k for k in self._l1_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self._l1_cache[key]
                    count += 1
            else:
                count += len(self._l1_cache)
                self._l1_cache.clear()
            
            # Clear Redis cache
            tiers_to_clear = [tier] if tier else list(self._connections.keys())
            
            for t in tiers_to_clear:
                if t in self._connections:
                    conn = self._connections[t]
                    
                    if pattern:
                        # Use SCAN to find matching keys
                        cursor = 0
                        while True:
                            cursor, keys = await conn.scan(cursor, match=f"*{pattern}*", count=100)
                            if keys:
                                deleted = await conn.delete(*keys)
                                count += deleted
                            if cursor == 0:
                                break
                    else:
                        # Clear all (be careful with this!)
                        await conn.flushdb()
                        count += await conn.dbsize()
            
            if self.metrics:
                self.metrics.increment("cache.clear", {"pattern": pattern or "all"}, count)
            
            return count
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            if self.metrics:
                self.metrics.increment("cache.error", {"operation": "clear"})
            return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "l1_cache": {
                "size": len(self._l1_cache),
                "max_size": self._l1_max_size,
                "keys": list(self._l1_cache.keys())[:10]  # Sample keys
            },
            "redis_tiers": {}
        }
        
        try:
            for tier, conn in self._connections.items():
                info = await conn.info()
                stats["redis_tiers"][tier.value] = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "0"),
                    "total_keys": await conn.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                    "evicted_keys": info.get("evicted_keys", 0),
                    "expired_keys": info.get("expired_keys", 0)
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
        
        return stats
    
    async def close(self) -> None:
        """Close all Redis connections."""
        try:
            for conn in self._connections.values():
                await conn.close()
            
            for pool in self._connection_pools.values():
                await pool.disconnect()
            
            self._connections.clear()
            self._connection_pools.clear()
            self._l1_cache.clear()
            self._is_initialized = False
            
            logger.info("Redis manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Redis manager: {e}")