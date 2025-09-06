"""
Mock implementations of services for testing.
These mocks provide the minimal interface needed for tests.
"""
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock


class MockCachingService:
    """Mock implementation of CachingService for testing."""
    
    def __init__(self):
        self.initialized = False
        self.closed = False
        self._metrics = {
            'hit_rate': 0.0,
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_latency_ms': 0.0,
            'memory_usage_mb': 0.0
        }
    
    @classmethod
    def create_for_trading(cls):
        """Factory method to create a caching service for trading."""
        return cls()
    
    async def initialize(self):
        """Initialize the caching service."""
        self.initialized = True
        return True
    
    async def close(self):
        """Close the caching service."""
        self.closed = True
        return True
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self._metrics
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache."""
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return True


class MockDatabaseOptimizationService:
    """Mock implementation of DatabaseOptimizationService for testing."""
    
    def __init__(self, connection_string: str = "", monitoring_interval: int = 300, 
                 enable_auto_optimization: bool = True):
        self.connection_string = connection_string
        self.monitoring_interval = monitoring_interval
        self.enable_auto_optimization = enable_auto_optimization
        self.initialized = False
        self.monitoring = False
        self.closed = False
        self._metrics = {
            'avg_query_time_ms': 5.0,
            'slow_queries_count': 0,
            'total_queries': 0,
            'cache_hit_rate': 0.0,
            'active_connections': 0,
            'optimizations_applied': 0
        }
    
    async def initialize(self):
        """Initialize the database optimization service."""
        self.initialized = True
        return True
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        return True
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        return True
    
    async def close(self):
        """Close the service."""
        self.closed = True
        return True
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        return self._metrics
    
    async def optimize_query(self, query: str) -> str:
        """Optimize a SQL query."""
        return query
    
    async def analyze_slow_queries(self) -> list:
        """Analyze slow queries."""
        return []


class MockDataAggregationService:
    """Mock implementation of DataAggregationService for testing."""
    
    def __init__(self):
        self.initialized = False
        self.closed = False
        self._aggregations = {}
    
    async def initialize(self):
        """Initialize the data aggregation service."""
        self.initialized = True
        return True
    
    async def close(self):
        """Close the service."""
        self.closed = True
        return True
    
    async def aggregate(self, data: list, method: str = "mean") -> Any:
        """Aggregate data using specified method."""
        if not data:
            return None
        
        if method == "mean":
            return sum(data) / len(data)
        elif method == "sum":
            return sum(data)
        elif method == "min":
            return min(data)
        elif method == "max":
            return max(data)
        else:
            return data
    
    async def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            'total_aggregations': len(self._aggregations),
            'methods_used': list(set(self._aggregations.values()))
        }