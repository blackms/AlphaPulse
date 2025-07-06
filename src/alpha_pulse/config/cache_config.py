"""Cache configuration for AlphaPulse."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..cache.redis_manager import CacheTier, SerializationFormat, EvictionPolicy
from ..cache.distributed_cache import ShardingStrategy, NodeRole
from ..utils.serialization_utils import CompressionType


@dataclass
class TierConfig:
    """Configuration for a specific cache tier."""
    
    enabled: bool = True
    ttl: int = 300  # Default TTL in seconds
    max_size: Optional[int] = None  # Maximum number of entries
    max_memory: Optional[str] = None  # e.g., "100mb", "1gb"
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    
    # Tier-specific settings
    use_cases: List[str] = field(default_factory=list)
    priority: int = 1  # Higher priority = preferred tier


@dataclass
class RedisNodeConfig:
    """Configuration for a Redis node."""
    
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


@dataclass
class RedisClusterConfig:
    """Configuration for Redis cluster."""
    
    enabled: bool = False
    nodes: List[Dict[str, any]] = field(default_factory=list)
    skip_full_coverage_check: bool = True
    max_connections_per_node: int = 50
    readonly_mode: bool = False


@dataclass
class RedisSentinelConfig:
    """Configuration for Redis Sentinel."""
    
    enabled: bool = False
    sentinels: List[Tuple[str, int]] = field(default_factory=list)
    service_name: str = "mymaster"
    socket_timeout: float = 0.2
    min_other_sentinels: int = 0


@dataclass
class SerializationConfig:
    """Configuration for data serialization."""
    
    default_format: SerializationFormat = SerializationFormat.MSGPACK
    compression: CompressionType = CompressionType.LZ4
    compression_threshold: int = 1024  # Bytes
    compression_level: int = 6  # For GZIP
    max_serialized_size: Optional[int] = 10 * 1024 * 1024  # 10MB
    
    # Type-specific settings
    numpy_optimize: bool = True
    pandas_optimize: bool = True
    use_pickle_for_complex: bool = False


@dataclass
class CacheWarmingConfig:
    """Configuration for cache warming."""
    
    enabled: bool = True
    strategies: List[str] = field(default_factory=lambda: ["market_open", "predictive"])
    
    # Market open warming
    market_open_offset: int = -300  # Seconds before market open
    market_open_keys: List[str] = field(default_factory=lambda: [
        "market:*:quote",
        "market:*:orderbook",
        "indicators:*:sma",
        "indicators:*:rsi"
    ])
    
    # Predictive warming
    predictive_enabled: bool = True
    predictive_model: str = "time_series"  # or "ml_based"
    predictive_threshold: float = 0.7  # Confidence threshold
    
    # Background warming
    background_interval: int = 300  # Seconds
    background_batch_size: int = 100


@dataclass
class InvalidationConfig:
    """Configuration for cache invalidation."""
    
    # Time-based invalidation
    default_ttl: int = 300  # 5 minutes
    ttl_variance: float = 0.1  # 10% variance to prevent thundering herd
    
    # Event-driven invalidation
    event_driven_enabled: bool = True
    event_channels: List[str] = field(default_factory=lambda: [
        "market:update",
        "trade:executed",
        "order:filled",
        "position:changed"
    ])
    
    # Dependency tracking
    track_dependencies: bool = True
    max_dependency_depth: int = 3
    
    # Version-based invalidation
    version_tracking: bool = True
    version_check_interval: int = 60  # Seconds


@dataclass
class MonitoringConfig:
    """Configuration for cache monitoring."""
    
    enabled: bool = True
    metrics_interval: int = 60  # Seconds
    
    # Metrics to track
    track_hit_rate: bool = True
    track_latency: bool = True
    track_memory_usage: bool = True
    track_key_distribution: bool = True
    
    # Alerting thresholds
    min_hit_rate: float = 0.7  # Alert if hit rate drops below
    max_latency_ms: float = 10.0  # Alert if latency exceeds
    max_memory_usage_percent: float = 0.9  # Alert if memory usage exceeds
    
    # Logging
    log_cache_operations: bool = False  # Can be verbose
    log_slow_operations_ms: float = 100.0


@dataclass
class DistributedCacheConfig:
    """Configuration for distributed caching."""
    
    enabled: bool = False
    node_id: str = "node-1"
    
    # Sharding
    sharding_strategy: ShardingStrategy = ShardingStrategy.CONSISTENT_HASH
    replication_factor: int = 3
    virtual_nodes: int = 150
    
    # Node discovery
    discovery_method: str = "static"  # or "consul", "etcd", "kubernetes"
    static_nodes: List[Dict[str, any]] = field(default_factory=list)
    
    # Health checking
    health_check_interval: int = 30  # Seconds
    unhealthy_threshold: int = 3  # Failed checks before marking unhealthy
    
    # Rebalancing
    auto_rebalance: bool = True
    rebalance_threshold: float = 0.2  # 20% node change triggers rebalance


@dataclass
class CacheConfig:
    """Main cache configuration."""
    
    # General settings
    enabled: bool = True
    cache_prefix: str = "alphapulse"
    
    # Connection settings
    connection_timeout: int = 20  # Seconds
    socket_timeout: int = 5  # Seconds
    max_connections: int = 50
    retry_on_timeout: bool = True
    max_retries: int = 3
    retry_delay: float = 0.1  # Seconds
    
    # Redis configurations
    redis_node: RedisNodeConfig = field(default_factory=RedisNodeConfig)
    redis_cluster: RedisClusterConfig = field(default_factory=RedisClusterConfig)
    redis_sentinel: RedisSentinelConfig = field(default_factory=RedisSentinelConfig)
    
    # Tier configurations
    tiers: Dict[str, TierConfig] = field(default_factory=lambda: {
        "l1_memory": TierConfig(
            enabled=True,
            ttl=60,  # 1 minute
            max_size=1000,
            eviction_policy=EvictionPolicy.LRU,
            use_cases=["hot_data", "real_time_quotes", "active_calculations"],
            priority=3
        ),
        "l2_local_redis": TierConfig(
            enabled=True,
            ttl=300,  # 5 minutes
            max_memory="1gb",
            eviction_policy=EvictionPolicy.LRU,
            use_cases=["indicators", "recent_trades", "user_sessions"],
            priority=2
        ),
        "l3_distributed_redis": TierConfig(
            enabled=False,  # Enable when using distributed setup
            ttl=3600,  # 1 hour
            max_memory="10gb",
            eviction_policy=EvictionPolicy.LFU,
            use_cases=["historical_data", "backtest_results", "model_outputs"],
            priority=1
        )
    })
    
    # Feature configurations
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    warming: CacheWarmingConfig = field(default_factory=CacheWarmingConfig)
    invalidation: InvalidationConfig = field(default_factory=InvalidationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    distributed: DistributedCacheConfig = field(default_factory=DistributedCacheConfig)
    
    # Cache patterns
    default_strategy: str = "cache_aside"
    enable_write_through: bool = False
    enable_write_behind: bool = False
    enable_refresh_ahead: bool = True
    
    # Performance tuning
    batch_size: int = 100  # For batch operations
    pipeline_size: int = 1000  # For pipelined operations
    lua_scripting: bool = True  # Enable Lua scripts for atomic operations


# Default configurations for different environments
def get_development_config() -> CacheConfig:
    """Get development environment cache configuration."""
    config = CacheConfig()
    config.monitoring.log_cache_operations = True
    config.warming.enabled = False  # Disable warming in dev
    return config


def get_testing_config() -> CacheConfig:
    """Get testing environment cache configuration."""
    config = CacheConfig()
    config.redis_node.db = 15  # Use separate DB for tests
    config.tiers["l1_memory"].max_size = 100  # Smaller cache for tests
    config.monitoring.enabled = False
    return config


def get_production_config() -> CacheConfig:
    """Get production environment cache configuration."""
    config = CacheConfig()
    
    # Enable distributed caching
    config.distributed.enabled = True
    config.tiers["l3_distributed_redis"].enabled = True
    
    # Enable clustering
    config.redis_cluster.enabled = True
    
    # Tune for production
    config.max_connections = 200
    config.serialization.compression = CompressionType.LZ4
    config.warming.predictive_enabled = True
    
    # Strict monitoring
    config.monitoring.min_hit_rate = 0.8
    config.monitoring.max_latency_ms = 5.0
    
    return config


# Cache configuration presets for specific use cases
class CachePresets:
    """Predefined cache configurations for common use cases."""
    
    @staticmethod
    def real_time_trading() -> Dict[str, any]:
        """Cache settings optimized for real-time trading."""
        return {
            "default_strategy": "refresh_ahead",
            "tiers": {
                "l1_memory": {"ttl": 30, "max_size": 5000},
                "l2_local_redis": {"ttl": 60, "max_memory": "2gb"}
            },
            "warming": {
                "strategies": ["market_open", "predictive"],
                "predictive_threshold": 0.8
            },
            "invalidation": {
                "default_ttl": 60,
                "event_driven_enabled": True
            }
        }
    
    @staticmethod
    def backtesting() -> Dict[str, any]:
        """Cache settings optimized for backtesting."""
        return {
            "default_strategy": "cache_aside",
            "tiers": {
                "l1_memory": {"ttl": 3600, "max_size": 10000},
                "l2_local_redis": {"ttl": 7200, "max_memory": "5gb"},
                "l3_distributed_redis": {"enabled": True, "ttl": 86400}
            },
            "serialization": {
                "compression": CompressionType.GZIP,
                "compression_threshold": 512
            },
            "warming": {"enabled": False},
            "invalidation": {"event_driven_enabled": False}
        }
    
    @staticmethod
    def analytics() -> Dict[str, any]:
        """Cache settings optimized for analytics workloads."""
        return {
            "default_strategy": "write_behind",
            "enable_write_behind": True,
            "tiers": {
                "l2_local_redis": {"ttl": 1800, "max_memory": "4gb"},
                "l3_distributed_redis": {"enabled": True, "ttl": 7200}
            },
            "serialization": {
                "pandas_optimize": True,
                "numpy_optimize": True,
                "compression": CompressionType.LZ4
            },
            "batch_size": 1000,
            "pipeline_size": 5000
        }