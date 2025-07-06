"""Cache monitoring and analytics for performance tracking."""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Deque

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary

from .redis_manager import RedisManager, CacheTier
from .distributed_cache import DistributedCacheManager
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class CacheMetricType(Enum):
    """Types of cache metrics."""
    
    HIT_RATE = "hit_rate"
    MISS_RATE = "miss_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    KEY_COUNT = "key_count"
    EVICTION_RATE = "eviction_rate"
    ERROR_RATE = "error_rate"


@dataclass
class CacheOperation:
    """Represents a cache operation for tracking."""
    
    operation_type: str  # get, set, delete, etc.
    key: str
    tier: CacheTier
    success: bool
    latency_ms: float
    size_bytes: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    
    total_latency_ms: float = 0.0
    operation_count: int = 0
    
    memory_used_bytes: int = 0
    key_count: int = 0
    eviction_count: int = 0
    
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.operation_count if self.operation_count > 0 else 0.0
    
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.errors / self.operation_count if self.operation_count > 0 else 0.0


class CacheMonitor:
    """Monitors cache performance and provides analytics."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        distributed_manager: Optional[DistributedCacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        history_window: int = 3600  # 1 hour
    ):
        """Initialize cache monitor."""
        self.redis_manager = redis_manager
        self.distributed_manager = distributed_manager
        self.metrics_collector = metrics_collector
        self.history_window = history_window
        
        # Metrics storage
        self._metrics_by_tier: Dict[CacheTier, CacheMetrics] = defaultdict(CacheMetrics)
        self._global_metrics = CacheMetrics()
        
        # Operation history
        self._operation_history: Deque[CacheOperation] = deque(maxlen=10000)
        
        # Time-series metrics
        self._time_series: Dict[str, Deque[Tuple[datetime, float]]] = {
            "hit_rate": deque(maxlen=1000),
            "latency": deque(maxlen=1000),
            "throughput": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000)
        }
        
        # Key access patterns
        self._key_access_count: Dict[str, int] = defaultdict(int)
        self._key_last_access: Dict[str, datetime] = {}
        self._hot_keys: Set[str] = set()
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
    
    def _setup_prometheus_metrics(self) -> None:
        """Set up Prometheus metrics."""
        # Counters
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['tier']
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            ['tier']
        )
        self.cache_operations = Counter(
            'cache_operations_total',
            'Total number of cache operations',
            ['operation', 'tier', 'status']
        )
        
        # Gauges
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate',
            ['tier']
        )
        self.cache_memory_usage = Gauge(
            'cache_memory_usage_bytes',
            'Cache memory usage in bytes',
            ['tier']
        )
        self.cache_key_count = Gauge(
            'cache_key_count',
            'Number of keys in cache',
            ['tier']
        )
        
        # Histograms
        self.cache_latency = Histogram(
            'cache_operation_latency_seconds',
            'Cache operation latency',
            ['operation', 'tier'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # Summary
        self.cache_key_size = Summary(
            'cache_key_size_bytes',
            'Size of cached values',
            ['tier']
        )
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        self._monitoring_task = asyncio.create_task(self._monitor_cache())
        self._analytics_task = asyncio.create_task(self._analyze_patterns())
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
    
    def record_operation(self, operation: CacheOperation) -> None:
        """Record a cache operation."""
        # Add to history
        self._operation_history.append(operation)
        
        # Update metrics
        tier_metrics = self._metrics_by_tier[operation.tier]
        
        if operation.operation_type == "get":
            if operation.success:
                tier_metrics.hits += 1
                self._global_metrics.hits += 1
                self.cache_hits.labels(tier=operation.tier.value).inc()
            else:
                tier_metrics.misses += 1
                self._global_metrics.misses += 1
                self.cache_misses.labels(tier=operation.tier.value).inc()
        elif operation.operation_type == "set":
            tier_metrics.sets += 1
            self._global_metrics.sets += 1
        elif operation.operation_type == "delete":
            tier_metrics.deletes += 1
            self._global_metrics.deletes += 1
        
        if not operation.success:
            tier_metrics.errors += 1
            self._global_metrics.errors += 1
        
        # Update latency
        tier_metrics.total_latency_ms += operation.latency_ms
        tier_metrics.operation_count += 1
        self._global_metrics.total_latency_ms += operation.latency_ms
        self._global_metrics.operation_count += 1
        
        # Update Prometheus metrics
        self.cache_operations.labels(
            operation=operation.operation_type,
            tier=operation.tier.value,
            status="success" if operation.success else "error"
        ).inc()
        
        self.cache_latency.labels(
            operation=operation.operation_type,
            tier=operation.tier.value
        ).observe(operation.latency_ms / 1000.0)
        
        if operation.size_bytes:
            self.cache_key_size.labels(tier=operation.tier.value).observe(operation.size_bytes)
        
        # Track key access
        self._key_access_count[operation.key] += 1
        self._key_last_access[operation.key] = operation.timestamp
    
    async def _monitor_cache(self) -> None:
        """Monitor cache metrics periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Get Redis stats
                stats = await self.redis_manager.get_stats()
                
                # Update metrics for each tier
                for tier_name, tier_stats in stats.get("redis_tiers", {}).items():
                    tier = CacheTier(tier_name)
                    tier_metrics = self._metrics_by_tier[tier]
                    
                    # Update memory and key count
                    tier_metrics.memory_used_bytes = self._parse_memory(
                        tier_stats.get("used_memory", "0")
                    )
                    tier_metrics.key_count = tier_stats.get("total_keys", 0)
                    tier_metrics.eviction_count = tier_stats.get("evicted_keys", 0)
                    
                    # Update Prometheus gauges
                    self.cache_hit_rate.labels(tier=tier_name).set(tier_metrics.hit_rate())
                    self.cache_memory_usage.labels(tier=tier_name).set(tier_metrics.memory_used_bytes)
                    self.cache_key_count.labels(tier=tier_name).set(tier_metrics.key_count)
                
                # Update time series
                now = datetime.utcnow()
                self._time_series["hit_rate"].append((now, self._global_metrics.hit_rate()))
                self._time_series["latency"].append((now, self._global_metrics.avg_latency_ms()))
                self._time_series["memory_usage"].append(
                    (now, sum(m.memory_used_bytes for m in self._metrics_by_tier.values()))
                )
                
                # Calculate throughput
                recent_ops = [
                    op for op in self._operation_history
                    if (now - op.timestamp).total_seconds() < 60
                ]
                throughput = len(recent_ops) / 60.0  # ops per second
                self._time_series["throughput"].append((now, throughput))
                
                # Clean up old data
                self._cleanup_old_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
    
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes."""
        try:
            if memory_str.endswith("K"):
                return int(float(memory_str[:-1]) * 1024)
            elif memory_str.endswith("M"):
                return int(float(memory_str[:-1]) * 1024 * 1024)
            elif memory_str.endswith("G"):
                return int(float(memory_str[:-1]) * 1024 * 1024 * 1024)
            else:
                return int(float(memory_str))
        except Exception:
            return 0
    
    async def _analyze_patterns(self) -> None:
        """Analyze cache access patterns."""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Identify hot keys
                access_threshold = np.percentile(
                    list(self._key_access_count.values()),
                    95
                ) if self._key_access_count else 0
                
                self._hot_keys = {
                    key for key, count in self._key_access_count.items()
                    if count > access_threshold
                }
                
                # Log hot keys
                if self._hot_keys:
                    logger.info(f"Identified {len(self._hot_keys)} hot keys")
                
                # Analyze cache efficiency
                efficiency_report = self._analyze_efficiency()
                if efficiency_report:
                    logger.info(f"Cache efficiency report: {efficiency_report}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pattern analysis error: {e}")
    
    def _analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze cache efficiency."""
        if not self._operation_history:
            return {}
        
        now = datetime.utcnow()
        recent_window = timedelta(minutes=15)
        
        recent_ops = [
            op for op in self._operation_history
            if now - op.timestamp < recent_window
        ]
        
        if not recent_ops:
            return {}
        
        # Calculate metrics by operation type
        ops_by_type = defaultdict(list)
        for op in recent_ops:
            ops_by_type[op.operation_type].append(op)
        
        # Analyze patterns
        report = {
            "window_minutes": 15,
            "total_operations": len(recent_ops),
            "operations_by_type": {
                op_type: len(ops) for op_type, ops in ops_by_type.items()
            },
            "avg_latency_by_type": {
                op_type: np.mean([op.latency_ms for op in ops])
                for op_type, ops in ops_by_type.items()
            },
            "error_rate": sum(1 for op in recent_ops if not op.success) / len(recent_ops),
            "hot_key_percentage": len(self._hot_keys) / len(self._key_access_count) if self._key_access_count else 0
        }
        
        return report
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.history_window)
        
        # Clean up key access data
        keys_to_remove = [
            key for key, last_access in self._key_last_access.items()
            if last_access < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self._key_access_count[key]
            del self._key_last_access[key]
            self._hot_keys.discard(key)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of cache metrics."""
        return {
            "global": {
                "hit_rate": self._global_metrics.hit_rate(),
                "avg_latency_ms": self._global_metrics.avg_latency_ms(),
                "error_rate": self._global_metrics.error_rate(),
                "total_operations": self._global_metrics.operation_count
            },
            "by_tier": {
                tier.value: {
                    "hit_rate": metrics.hit_rate(),
                    "avg_latency_ms": metrics.avg_latency_ms(),
                    "memory_used_mb": metrics.memory_used_bytes / (1024 * 1024),
                    "key_count": metrics.key_count,
                    "eviction_count": metrics.eviction_count
                }
                for tier, metrics in self._metrics_by_tier.items()
            },
            "hot_keys": list(self._hot_keys)[:10],  # Top 10 hot keys
            "recent_errors": [
                {
                    "key": op.key,
                    "operation": op.operation_type,
                    "tier": op.tier.value,
                    "timestamp": op.timestamp.isoformat()
                }
                for op in self._operation_history
                if not op.success
            ][-10:]  # Last 10 errors
        }
    
    def get_time_series_data(
        self,
        metric: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Tuple[datetime, float]]:
        """Get time series data for a metric."""
        if metric not in self._time_series:
            return []
        
        data = self._time_series[metric]
        
        if not start_time and not end_time:
            return list(data)
        
        filtered_data = []
        for timestamp, value in data:
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_data.append((timestamp, value))
        
        return filtered_data
    
    def get_cache_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for cache optimization."""
        recommendations = []
        
        # Check hit rate
        for tier, metrics in self._metrics_by_tier.items():
            hit_rate = metrics.hit_rate()
            if hit_rate < 0.7:  # Less than 70% hit rate
                recommendations.append({
                    "type": "low_hit_rate",
                    "tier": tier.value,
                    "current_hit_rate": hit_rate,
                    "recommendation": "Consider increasing cache size or TTL",
                    "severity": "medium"
                })
        
        # Check latency
        if self._global_metrics.avg_latency_ms() > 10:  # More than 10ms average
            recommendations.append({
                "type": "high_latency",
                "avg_latency_ms": self._global_metrics.avg_latency_ms(),
                "recommendation": "Consider using faster serialization or local caching",
                "severity": "high"
            })
        
        # Check hot keys
        if len(self._hot_keys) > 100:
            recommendations.append({
                "type": "many_hot_keys",
                "hot_key_count": len(self._hot_keys),
                "recommendation": "Consider using dedicated caching for hot keys",
                "severity": "low"
            })
        
        # Check error rate
        error_rate = self._global_metrics.error_rate()
        if error_rate > 0.01:  # More than 1% errors
            recommendations.append({
                "type": "high_error_rate",
                "error_rate": error_rate,
                "recommendation": "Investigate cache errors and connection issues",
                "severity": "high"
            })
        
        return recommendations


class CacheAnalytics:
    """Advanced analytics for cache optimization."""
    
    def __init__(self, monitor: CacheMonitor):
        """Initialize cache analytics."""
        self.monitor = monitor
    
    def analyze_key_patterns(self) -> Dict[str, Any]:
        """Analyze key access patterns."""
        access_counts = self.monitor._key_access_count
        
        if not access_counts:
            return {}
        
        # Calculate access distribution
        counts = list(access_counts.values())
        
        return {
            "total_unique_keys": len(access_counts),
            "access_distribution": {
                "min": min(counts),
                "max": max(counts),
                "mean": np.mean(counts),
                "median": np.median(counts),
                "p95": np.percentile(counts, 95),
                "p99": np.percentile(counts, 99)
            },
            "skewness": self._calculate_skewness(counts),
            "recommendation": self._recommend_caching_strategy(counts)
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of distribution."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _recommend_caching_strategy(self, access_counts: List[int]) -> str:
        """Recommend caching strategy based on access patterns."""
        skewness = self._calculate_skewness(access_counts)
        
        if skewness > 2:
            return "High skewness detected. Consider using a multi-tier cache with hot key optimization."
        elif skewness > 1:
            return "Moderate skewness. Consider using LFU eviction policy for better performance."
        else:
            return "Uniform access pattern. Current LRU policy is appropriate."
    
    def predict_cache_size_requirements(
        self,
        target_hit_rate: float = 0.9,
        growth_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Predict cache size requirements."""
        current_metrics = self.monitor.get_metrics_summary()
        
        recommendations = {}
        
        for tier, metrics in current_metrics["by_tier"].items():
            current_hit_rate = metrics["hit_rate"]
            current_size = metrics["memory_used_mb"]
            
            if current_hit_rate < target_hit_rate:
                # Estimate required size increase
                size_multiplier = target_hit_rate / max(current_hit_rate, 0.1)
                recommended_size = current_size * size_multiplier * (1 + growth_rate)
                
                recommendations[tier] = {
                    "current_size_mb": current_size,
                    "current_hit_rate": current_hit_rate,
                    "recommended_size_mb": recommended_size,
                    "size_increase_percent": (recommended_size - current_size) / current_size * 100
                }
        
        return recommendations
    
    def analyze_cache_efficiency_trends(self) -> Dict[str, Any]:
        """Analyze cache efficiency trends over time."""
        # Get time series data
        hit_rate_series = self.monitor.get_time_series_data("hit_rate")
        latency_series = self.monitor.get_time_series_data("latency")
        
        if len(hit_rate_series) < 10 or len(latency_series) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        hit_rates = [value for _, value in hit_rate_series[-100:]]
        latencies = [value for _, value in latency_series[-100:]]
        
        # Simple linear regression for trend
        x = np.arange(len(hit_rates))
        hit_rate_trend = np.polyfit(x, hit_rates, 1)[0]
        latency_trend = np.polyfit(x, latencies, 1)[0]
        
        return {
            "hit_rate": {
                "current": hit_rates[-1],
                "trend": "improving" if hit_rate_trend > 0 else "declining",
                "change_per_hour": hit_rate_trend * 60
            },
            "latency": {
                "current_ms": latencies[-1],
                "trend": "improving" if latency_trend < 0 else "worsening",
                "change_per_hour_ms": latency_trend * 60
            },
            "recommendations": self._generate_trend_recommendations(hit_rate_trend, latency_trend)
        }
    
    def _generate_trend_recommendations(
        self,
        hit_rate_trend: float,
        latency_trend: float
    ) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []
        
        if hit_rate_trend < -0.001:  # Declining hit rate
            recommendations.append("Hit rate is declining. Consider cache warming or increasing TTL.")
        
        if latency_trend > 0.1:  # Increasing latency
            recommendations.append("Latency is increasing. Check network connectivity and Redis performance.")
        
        if not recommendations:
            recommendations.append("Cache performance is stable.")
        
        return recommendations