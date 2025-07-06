"""Database configuration with connection pooling settings."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List


class PoolingStrategy(Enum):
    """Database connection pooling strategies."""
    
    SESSION = "session"  # Each client gets dedicated connection
    TRANSACTION = "transaction"  # Connection returned after transaction
    STATEMENT = "statement"  # Connection returned after statement


class LoadBalancingStrategy(Enum):
    """Connection load balancing strategies."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration settings."""
    
    # Pool size settings
    min_size: int = 5  # Minimum number of connections
    max_size: int = 20  # Maximum number of connections
    overflow: int = 10  # Additional connections when pool exhausted
    
    # Timeout settings
    pool_timeout: float = 30.0  # Seconds to wait for connection
    max_overflow: int = 10  # Maximum overflow connections
    echo_pool: bool = False  # Log pool checkouts/checkins
    
    # Connection settings
    pool_pre_ping: bool = True  # Test connections before use
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    connect_timeout: int = 10  # Connection timeout in seconds
    
    # Strategy settings
    pooling_strategy: PoolingStrategy = PoolingStrategy.TRANSACTION
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN


@dataclass
class DatabaseNode:
    """Represents a database node in the cluster."""
    
    host: str
    port: int = 5432
    database: str = "alphapulse"
    username: str = "alphapulse"
    password: str = ""
    
    # Node properties
    is_master: bool = True
    weight: int = 1  # For weighted load balancing
    max_connections: int = 100
    
    # Health check settings
    health_check_interval: int = 30  # Seconds
    health_check_timeout: int = 5  # Seconds
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


@dataclass
class QueryOptimizationConfig:
    """Query optimization configuration."""
    
    # Query caching
    enable_query_cache: bool = True
    query_cache_size: int = 1000  # Number of cached queries
    query_cache_ttl: int = 300  # Cache TTL in seconds
    
    # Prepared statements
    enable_prepared_statements: bool = True
    prepared_statement_cache_size: int = 500
    
    # Query analysis
    slow_query_threshold: float = 1.0  # Seconds
    log_slow_queries: bool = True
    analyze_query_plans: bool = True
    
    # Optimization settings
    enable_query_hints: bool = True
    enable_join_optimization: bool = True
    enable_subquery_optimization: bool = True


@dataclass
class DatabaseConfig:
    """Main database configuration."""
    
    # Connection settings
    master_node: DatabaseNode = field(default_factory=lambda: DatabaseNode(
        host="localhost",
        port=5432,
        database="alphapulse",
        username="alphapulse",
        password="alphapulse123",
        is_master=True
    ))
    
    # Read replicas
    read_replicas: List[DatabaseNode] = field(default_factory=list)
    
    # Connection pooling
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    
    # Query optimization
    query_optimization: QueryOptimizationConfig = field(default_factory=QueryOptimizationConfig)
    
    # Read/write splitting
    enable_read_write_split: bool = True
    read_preference: str = "nearest"  # nearest, primary, secondary
    
    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60  # Seconds
    
    # TimescaleDB settings
    enable_timescale: bool = True
    compression_policy: Dict[str, int] = field(default_factory=lambda: {
        "market_data": 7,  # Compress after 7 days
        "trades": 30,  # Compress after 30 days
        "logs": 1  # Compress after 1 day
    })


def get_development_config() -> DatabaseConfig:
    """Get development database configuration."""
    return DatabaseConfig(
        connection_pool=ConnectionPoolConfig(
            min_size=2,
            max_size=10,
            pool_timeout=10.0
        )
    )


def get_production_config() -> DatabaseConfig:
    """Get production database configuration."""
    config = DatabaseConfig()
    
    # Add read replicas for production
    config.read_replicas = [
        DatabaseNode(
            host="db-replica-1.alphapulse.internal",
            is_master=False,
            weight=1
        ),
        DatabaseNode(
            host="db-replica-2.alphapulse.internal",
            is_master=False,
            weight=2  # This replica has more capacity
        )
    ]
    
    # Production pool settings
    config.connection_pool = ConnectionPoolConfig(
        min_size=10,
        max_size=50,
        overflow=20,
        pool_timeout=30.0,
        pool_pre_ping=True
    )
    
    return config


def get_testing_config() -> DatabaseConfig:
    """Get testing database configuration."""
    return DatabaseConfig(
        master_node=DatabaseNode(
            host="localhost",
            database="alphapulse_test",
            username="test_user",
            password="test_pass"
        ),
        connection_pool=ConnectionPoolConfig(
            min_size=1,
            max_size=5,
            pool_timeout=5.0
        )
    )