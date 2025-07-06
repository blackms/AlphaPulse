"""Database optimization components."""

from .connection_pool import ConnectionPool
from .connection_manager import (
    ConnectionValidator,
    ConnectionMonitor,
    ConnectionHealth,
    ConnectionState
)
from .query_analyzer import QueryAnalyzer, QueryPlan, QueryType
from .slow_query_detector import SlowQueryDetector, SlowQueryInfo
from .query_optimizer import (
    QueryOptimizer,
    QueryCost,
    OptimizationSuggestion,
    OptimizationType
)
from .index_advisor import (
    IndexAdvisor,
    IndexInfo,
    IndexRecommendation,
    IndexUsageStats
)
from .index_manager import IndexManager, IndexOperation
from .partition_manager import (
    PartitionManager,
    PartitionInfo,
    PartitionStrategy,
    PartitionType,
    PartitionInterval
)
from .read_write_router import (
    ReadWriteRouter,
    TransactionRouter,
    QueryIntent,
    ReplicaLagPolicy
)
from .load_balancer import (
    LoadBalancer,
    AdaptiveLoadBalancer,
    NodeStatus,
    NodeMetrics,
    NodeState
)
from .failover_manager import (
    FailoverManager,
    FailoverState,
    PromotionStrategy,
    FailoverEvent
)
from .database_monitor import (
    DatabaseMonitor,
    QueryMetrics,
    ConnectionMetrics,
    TableMetrics
)

__all__ = [
    # Connection management
    "ConnectionPool",
    "ConnectionValidator",
    "ConnectionMonitor",
    "ConnectionHealth",
    "ConnectionState",
    
    # Query optimization
    "QueryAnalyzer",
    "QueryPlan",
    "QueryType",
    "SlowQueryDetector",
    "SlowQueryInfo",
    "QueryOptimizer",
    "QueryCost",
    "OptimizationSuggestion",
    "OptimizationType",
    
    # Index management
    "IndexAdvisor",
    "IndexInfo",
    "IndexRecommendation",
    "IndexUsageStats",
    "IndexManager",
    "IndexOperation",
    
    # Partitioning
    "PartitionManager",
    "PartitionInfo",
    "PartitionStrategy",
    "PartitionType",
    "PartitionInterval",
    
    # Read/write routing
    "ReadWriteRouter",
    "TransactionRouter",
    "QueryIntent",
    "ReplicaLagPolicy",
    
    # Load balancing
    "LoadBalancer",
    "AdaptiveLoadBalancer",
    "NodeStatus",
    "NodeMetrics",
    "NodeState",
    
    # Failover
    "FailoverManager",
    "FailoverState",
    "PromotionStrategy",
    "FailoverEvent",
    
    # Monitoring
    "DatabaseMonitor",
    "QueryMetrics",
    "ConnectionMetrics",
    "TableMetrics"
]