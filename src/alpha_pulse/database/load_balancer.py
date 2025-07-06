"""Database connection load balancing."""

import asyncio
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from ..config.database_config import DatabaseNode, LoadBalancingStrategy
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class NodeStatus(Enum):
    """Status of a database node."""
    
    ACTIVE = "active"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DRAINING = "draining"  # Preparing to remove


@dataclass
class NodeMetrics:
    """Metrics for a database node."""
    
    node_id: str
    active_connections: int = 0
    total_connections: int = 0
    queries_per_second: float = 0
    average_response_time: float = 0
    error_rate: float = 0
    cpu_usage: float = 0
    memory_usage: float = 0
    disk_io: float = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def load_score(self) -> float:
        """Calculate overall load score (0-100)."""
        # Weighted combination of metrics
        connection_score = min(self.active_connections / max(self.total_connections, 1) * 100, 100)
        response_score = min(self.average_response_time * 10, 100)  # Assume 10s is max
        error_score = min(self.error_rate * 100, 100)
        resource_score = (self.cpu_usage + self.memory_usage + self.disk_io) / 3
        
        # Weighted average
        return (
            connection_score * 0.3 +
            response_score * 0.3 +
            error_score * 0.2 +
            resource_score * 0.2
        )


@dataclass
class NodeState:
    """State of a database node."""
    
    node: DatabaseNode
    node_id: str
    status: NodeStatus = NodeStatus.ACTIVE
    metrics: NodeMetrics = field(default_factory=lambda: NodeMetrics(""))
    consecutive_failures: int = 0
    last_failure: Optional[datetime] = None
    circuit_breaker_open: bool = False
    circuit_breaker_opens_at: Optional[datetime] = None
    request_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def __post_init__(self):
        """Initialize metrics with node_id."""
        if not self.metrics.node_id:
            self.metrics = NodeMetrics(self.node_id)


class LoadBalancer:
    """Load balances database connections across nodes."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize load balancer."""
        self.strategy = strategy
        self.metrics = metrics_collector
        
        # Node tracking
        self._nodes: Dict[str, NodeState] = {}
        self._active_nodes: List[str] = []
        
        # Round-robin state
        self._round_robin_index = 0
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 5  # failures
        self.circuit_breaker_timeout = 60  # seconds
        
        # Load tracking
        self._request_counts: Dict[str, int] = {}
        self._last_reset = time.time()
        
    def add_node(self, node: DatabaseNode, node_id: str):
        """Add a node to the load balancer."""
        state = NodeState(node=node, node_id=node_id)
        self._nodes[node_id] = state
        self._update_active_nodes()
        
        logger.info(f"Added node {node_id} to load balancer")
    
    def remove_node(self, node_id: str):
        """Remove a node from the load balancer."""
        if node_id in self._nodes:
            # Mark as draining first
            self._nodes[node_id].status = NodeStatus.DRAINING
            self._update_active_nodes()
            
            # Actually remove after a delay
            asyncio.create_task(self._drain_and_remove(node_id))
    
    async def _drain_and_remove(self, node_id: str, drain_time: int = 30):
        """Drain connections and remove node."""
        logger.info(f"Draining node {node_id}")
        await asyncio.sleep(drain_time)
        
        del self._nodes[node_id]
        self._update_active_nodes()
        logger.info(f"Removed node {node_id}")
    
    def select_node(self, request_metadata: Optional[Dict] = None) -> Optional[str]:
        """Select a node based on load balancing strategy."""
        if not self._active_nodes:
            return None
        
        # Update request counts
        self._update_request_rates()
        
        # Select based on strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            node_id = self._select_round_robin()
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            node_id = self._select_least_connections()
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            node_id = self._select_weighted()
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            node_id = self._select_random()
        else:
            # Default to round-robin
            node_id = self._select_round_robin()
        
        if node_id:
            # Track request
            self._track_request(node_id, request_metadata)
            
            # Update metrics
            if self.metrics:
                self.metrics.increment(
                    "db.loadbalancer.requests",
                    {"node": node_id, "strategy": self.strategy.value}
                )
        
        return node_id
    
    def _select_round_robin(self) -> Optional[str]:
        """Select node using round-robin."""
        if not self._active_nodes:
            return None
        
        node_id = self._active_nodes[self._round_robin_index % len(self._active_nodes)]
        self._round_robin_index += 1
        
        return node_id
    
    def _select_least_connections(self) -> Optional[str]:
        """Select node with least connections."""
        if not self._active_nodes:
            return None
        
        # Find node with minimum active connections
        min_connections = float('inf')
        selected_node = None
        
        for node_id in self._active_nodes:
            state = self._nodes[node_id]
            connections = state.metrics.active_connections
            
            if connections < min_connections:
                min_connections = connections
                selected_node = node_id
        
        return selected_node
    
    def _select_weighted(self) -> Optional[str]:
        """Select node based on weights."""
        if not self._active_nodes:
            return None
        
        # Calculate total weight
        weights = []
        total_weight = 0
        
        for node_id in self._active_nodes:
            state = self._nodes[node_id]
            weight = state.node.weight
            
            # Adjust weight based on node health
            if state.status == NodeStatus.DEGRADED:
                weight *= 0.5
            
            weights.append((node_id, weight))
            total_weight += weight
        
        # Weighted random selection
        if total_weight > 0:
            rand = random.uniform(0, total_weight)
            cumulative = 0
            
            for node_id, weight in weights:
                cumulative += weight
                if rand <= cumulative:
                    return node_id
        
        return self._select_random()
    
    def _select_random(self) -> Optional[str]:
        """Select random node."""
        if not self._active_nodes:
            return None
        
        return random.choice(self._active_nodes)
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update metrics for a node."""
        if node_id not in self._nodes:
            return
        
        state = self._nodes[node_id]
        node_metrics = state.metrics
        
        # Update metrics
        node_metrics.active_connections = metrics.get("active_connections", 0)
        node_metrics.total_connections = metrics.get("total_connections", 0)
        node_metrics.average_response_time = metrics.get("avg_response_time", 0)
        node_metrics.error_rate = metrics.get("error_rate", 0)
        node_metrics.cpu_usage = metrics.get("cpu_usage", 0)
        node_metrics.memory_usage = metrics.get("memory_usage", 0)
        node_metrics.disk_io = metrics.get("disk_io", 0)
        node_metrics.last_updated = datetime.utcnow()
        
        # Check if node should be marked as degraded
        if node_metrics.load_score > 80:
            state.status = NodeStatus.DEGRADED
        elif state.status == NodeStatus.DEGRADED and node_metrics.load_score < 60:
            state.status = NodeStatus.ACTIVE
        
        self._update_active_nodes()
    
    def report_failure(self, node_id: str, error: Optional[Exception] = None):
        """Report a node failure."""
        if node_id not in self._nodes:
            return
        
        state = self._nodes[node_id]
        state.consecutive_failures += 1
        state.last_failure = datetime.utcnow()
        
        logger.warning(
            f"Node {node_id} failure #{state.consecutive_failures}: {error}"
        )
        
        # Check circuit breaker
        if state.consecutive_failures >= self.circuit_breaker_threshold:
            self._open_circuit_breaker(node_id)
        
        # Update metrics
        if self.metrics:
            self.metrics.increment(
                "db.loadbalancer.failures",
                {"node": node_id}
            )
    
    def report_success(self, node_id: str, response_time: float):
        """Report a successful request."""
        if node_id not in self._nodes:
            return
        
        state = self._nodes[node_id]
        state.consecutive_failures = 0
        
        # Track response time
        state.request_history.append({
            "timestamp": time.time(),
            "response_time": response_time,
            "success": True
        })
        
        # Check if circuit breaker can be closed
        if state.circuit_breaker_open:
            if (datetime.utcnow() - state.circuit_breaker_opens_at).seconds > self.circuit_breaker_timeout:
                self._close_circuit_breaker(node_id)
    
    def _open_circuit_breaker(self, node_id: str):
        """Open circuit breaker for a node."""
        state = self._nodes[node_id]
        state.circuit_breaker_open = True
        state.circuit_breaker_opens_at = datetime.utcnow()
        state.status = NodeStatus.UNAVAILABLE
        
        self._update_active_nodes()
        
        logger.error(f"Circuit breaker opened for node {node_id}")
        
        # Schedule circuit breaker check
        asyncio.create_task(
            self._check_circuit_breaker(node_id, self.circuit_breaker_timeout)
        )
    
    def _close_circuit_breaker(self, node_id: str):
        """Close circuit breaker for a node."""
        state = self._nodes[node_id]
        state.circuit_breaker_open = False
        state.consecutive_failures = 0
        state.status = NodeStatus.ACTIVE
        
        self._update_active_nodes()
        
        logger.info(f"Circuit breaker closed for node {node_id}")
    
    async def _check_circuit_breaker(self, node_id: str, timeout: int):
        """Check if circuit breaker can be closed."""
        await asyncio.sleep(timeout)
        
        if node_id in self._nodes:
            state = self._nodes[node_id]
            if state.circuit_breaker_open:
                # Try a test request
                logger.info(f"Testing node {node_id} after circuit breaker timeout")
                # In real implementation, would make a test query
                # For now, just close the breaker
                self._close_circuit_breaker(node_id)
    
    def _update_active_nodes(self):
        """Update list of active nodes."""
        self._active_nodes = [
            node_id for node_id, state in self._nodes.items()
            if state.status == NodeStatus.ACTIVE and not state.circuit_breaker_open
        ]
    
    def _track_request(self, node_id: str, metadata: Optional[Dict]):
        """Track request for metrics."""
        self._request_counts[node_id] = self._request_counts.get(node_id, 0) + 1
        
        if metadata:
            state = self._nodes[node_id]
            state.request_history.append({
                "timestamp": time.time(),
                "metadata": metadata
            })
    
    def _update_request_rates(self):
        """Update request rates for all nodes."""
        current_time = time.time()
        elapsed = current_time - self._last_reset
        
        if elapsed > 60:  # Update every minute
            for node_id, count in self._request_counts.items():
                if node_id in self._nodes:
                    self._nodes[node_id].metrics.queries_per_second = count / elapsed
            
            self._request_counts.clear()
            self._last_reset = current_time
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get status of all nodes."""
        status = {
            "strategy": self.strategy.value,
            "total_nodes": len(self._nodes),
            "active_nodes": len(self._active_nodes),
            "nodes": {}
        }
        
        for node_id, state in self._nodes.items():
            status["nodes"][node_id] = {
                "status": state.status.value,
                "host": state.node.host,
                "weight": state.node.weight,
                "circuit_breaker": state.circuit_breaker_open,
                "consecutive_failures": state.consecutive_failures,
                "metrics": {
                    "active_connections": state.metrics.active_connections,
                    "load_score": state.metrics.load_score,
                    "qps": state.metrics.queries_per_second,
                    "avg_response_time": state.metrics.average_response_time,
                    "error_rate": state.metrics.error_rate
                }
            }
        
        return status
    
    def rebalance(self):
        """Rebalance load across nodes."""
        logger.info("Rebalancing load across nodes")
        
        # Reset round-robin index
        self._round_robin_index = 0
        
        # Clear request history to start fresh
        for state in self._nodes.values():
            state.request_history.clear()
        
        # Update active nodes
        self._update_active_nodes()


class AdaptiveLoadBalancer(LoadBalancer):
    """Load balancer that adapts strategy based on conditions."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize adaptive load balancer."""
        super().__init__(LoadBalancingStrategy.WEIGHTED, metrics_collector)
        
        # Adaptation settings
        self.adaptation_interval = 300  # 5 minutes
        self._last_adaptation = time.time()
        self._performance_history: List[Dict] = []
    
    def select_node(self, request_metadata: Optional[Dict] = None) -> Optional[str]:
        """Select node with adaptive strategy."""
        # Check if we should adapt strategy
        if time.time() - self._last_adaptation > self.adaptation_interval:
            self._adapt_strategy()
        
        return super().select_node(request_metadata)
    
    def _adapt_strategy(self):
        """Adapt load balancing strategy based on performance."""
        # Analyze recent performance
        total_nodes = len(self._active_nodes)
        
        if total_nodes == 0:
            return
        
        # Calculate variance in load scores
        load_scores = [
            self._nodes[node_id].metrics.load_score 
            for node_id in self._active_nodes
        ]
        
        if not load_scores:
            return
        
        avg_load = sum(load_scores) / len(load_scores)
        variance = sum((score - avg_load) ** 2 for score in load_scores) / len(load_scores)
        
        # Adapt strategy based on variance
        if variance < 10:
            # Load is well balanced - use round robin
            self.strategy = LoadBalancingStrategy.ROUND_ROBIN
        elif variance < 25:
            # Moderate imbalance - use least connections
            self.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        else:
            # High imbalance - use weighted
            self.strategy = LoadBalancingStrategy.WEIGHTED
        
        logger.info(f"Adapted load balancing strategy to {self.strategy.value}")
        
        self._last_adaptation = time.time()