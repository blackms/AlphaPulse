"""
Distributed supervision coordinator for managing agent clusters.
"""
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio
from collections import defaultdict
from loguru import logger

from ..interfaces import AgentState, Task
from ..optimization.ml_optimizer import MLOptimizer
from ..monitoring.performance_monitor import PerformanceMonitor


class AgentCluster:
    """Represents a group of related agents."""
    def __init__(self, cluster_id: str, config: Optional[Dict[str, Any]] = None):
        self.cluster_id = cluster_id
        self.config = config or {}
        self.agent_ids: Set[str] = set()
        self.load_metrics: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.last_update = datetime.now()


class DistributedCoordinator:
    """
    Coordinates multiple agent clusters and handles distributed supervision.
    Implements load balancing and cluster-wide optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize distributed coordinator."""
        self.config = config or {}
        self._clusters: Dict[str, AgentCluster] = {}
        self._agent_assignments: Dict[str, str] = {}  # agent_id -> cluster_id
        self._cluster_optimizers: Dict[str, MLOptimizer] = {}
        self._cluster_monitors: Dict[str, PerformanceMonitor] = {}
        self._load_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
    async def register_cluster(
        self,
        cluster_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new agent cluster.
        
        Args:
            cluster_id: Unique identifier for the cluster
            config: Cluster configuration parameters
        """
        if cluster_id in self._clusters:
            raise ValueError(f"Cluster {cluster_id} already exists")
            
        self._clusters[cluster_id] = AgentCluster(cluster_id, config)
        self._cluster_optimizers[cluster_id] = MLOptimizer(config)
        self._cluster_monitors[cluster_id] = PerformanceMonitor(config)
        
        logger.info(f"Registered new cluster: {cluster_id}")
        
    async def assign_agent(
        self,
        agent_id: str,
        preferred_cluster: Optional[str] = None
    ) -> str:
        """
        Assign an agent to a cluster using load balancing.
        
        Args:
            agent_id: Agent to assign
            preferred_cluster: Preferred cluster ID (optional)
            
        Returns:
            Assigned cluster ID
        """
        if agent_id in self._agent_assignments:
            return self._agent_assignments[agent_id]
            
        # Use preferred cluster if specified and available
        if preferred_cluster and preferred_cluster in self._clusters:
            cluster = self._clusters[preferred_cluster]
            if await self._check_cluster_capacity(cluster):
                cluster.agent_ids.add(agent_id)
                self._agent_assignments[agent_id] = cluster.cluster_id
                return cluster.cluster_id
                
        # Find least loaded cluster
        assigned_cluster = await self._find_best_cluster(agent_id)
        if not assigned_cluster:
            raise RuntimeError("No available clusters for agent assignment")
            
        assigned_cluster.agent_ids.add(agent_id)
        self._agent_assignments[agent_id] = assigned_cluster.cluster_id
        logger.info(f"Assigned agent {agent_id} to cluster {assigned_cluster.cluster_id}")
        
        return assigned_cluster.cluster_id
        
    async def update_metrics(
        self,
        agent_id: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Update agent metrics and get monitoring results.
        
        Args:
            agent_id: Agent identifier
            metrics: Current performance metrics
            
        Returns:
            Monitoring results including alerts and recommendations
        """
        if agent_id not in self._agent_assignments:
            raise ValueError(f"Agent {agent_id} not assigned to any cluster")
            
        cluster_id = self._agent_assignments[agent_id]
        cluster = self._clusters[cluster_id]
        
        # Update cluster metrics
        cluster.performance_metrics[agent_id] = metrics
        cluster.last_update = datetime.now()
        
        # Monitor performance
        monitor = self._cluster_monitors[cluster_id]
        monitoring_result = await monitor.monitor_performance(
            agent_id,
            metrics
        )
        
        # Check if optimization is needed
        if monitoring_result['alert_level'] in ['critical', 'warning']:
            await self._trigger_cluster_optimization(cluster_id)
            
        return monitoring_result
        
    async def optimize_cluster(
        self,
        cluster_id: str
    ) -> Dict[str, Any]:
        """
        Optimize all agents in a cluster.
        
        Args:
            cluster_id: Cluster to optimize
            
        Returns:
            Optimization results
        """
        if cluster_id not in self._clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
            
        cluster = self._clusters[cluster_id]
        optimizer = self._cluster_optimizers[cluster_id]
        
        try:
            # Collect cluster-wide performance data
            performance_data = []
            for agent_id in cluster.agent_ids:
                if agent_id in cluster.performance_metrics:
                    performance_data.append({
                        'agent_id': agent_id,
                        'metrics': cluster.performance_metrics[agent_id]
                    })
                    
            if not performance_data:
                return {'status': 'no_data'}
                
            # Optimize cluster parameters
            cluster_params = await self._get_cluster_parameters(cluster_id)
            optimized_params = await optimizer.optimize_parameters(
                cluster_params,
                performance_data
            )
            
            # Apply optimized parameters
            await self._apply_cluster_parameters(
                cluster_id,
                optimized_params
            )
            
            return {
                'status': 'success',
                'optimized_params': optimized_params,
                'agents_affected': len(cluster.agent_ids)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cluster {cluster_id}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
            
    async def get_cluster_analytics(
        self,
        cluster_id: str
    ) -> Dict[str, Any]:
        """
        Get analytics for a cluster.
        
        Args:
            cluster_id: Cluster to analyze
            
        Returns:
            Cluster analytics
        """
        if cluster_id not in self._clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
            
        cluster = self._clusters[cluster_id]
        monitor = self._cluster_monitors[cluster_id]
        
        analytics = {
            'cluster_id': cluster_id,
            'agent_count': len(cluster.agent_ids),
            'last_update': cluster.last_update,
            'performance': {},
            'load_metrics': cluster.load_metrics.copy()
        }
        
        # Get performance analytics for each agent
        for agent_id in cluster.agent_ids:
            agent_analytics = await monitor.get_performance_analytics(agent_id)
            if agent_analytics:
                analytics['performance'][agent_id] = agent_analytics
                
        return analytics
        
    async def _find_best_cluster(self, agent_id: str) -> Optional[AgentCluster]:
        """Find the best cluster for an agent based on load."""
        if not self._clusters:
            return None
            
        # Calculate load scores
        cluster_scores = []
        for cluster in self._clusters.values():
            load_score = await self._calculate_cluster_load(cluster)
            capacity_score = 1 - (len(cluster.agent_ids) / 10)  # Assume max 10 agents
            performance_score = await self._calculate_cluster_performance(cluster)
            
            total_score = (
                load_score * 0.4 +
                capacity_score * 0.3 +
                performance_score * 0.3
            )
            
            cluster_scores.append((cluster, total_score))
            
        # Return cluster with highest score
        return max(cluster_scores, key=lambda x: x[1])[0] if cluster_scores else None
        
    async def _check_cluster_capacity(self, cluster: AgentCluster) -> bool:
        """Check if cluster has capacity for new agents."""
        load = await self._calculate_cluster_load(cluster)
        return load < self._load_thresholds['high']
        
    async def _calculate_cluster_load(self, cluster: AgentCluster) -> float:
        """Calculate cluster load score."""
        if not cluster.load_metrics:
            return 0.0
            
        # Consider CPU, memory, and network load
        cpu_load = cluster.load_metrics.get('cpu_usage', 0) / 100
        memory_load = cluster.load_metrics.get('memory_usage', 0) / 100
        network_load = cluster.load_metrics.get('network_usage', 0) / 100
        
        return (cpu_load * 0.4 + memory_load * 0.4 + network_load * 0.2)
        
    async def _calculate_cluster_performance(self, cluster: AgentCluster) -> float:
        """Calculate cluster performance score."""
        if not cluster.performance_metrics:
            return 0.0
            
        # Average performance scores across agents
        scores = []
        for metrics in cluster.performance_metrics.values():
            if 'performance_score' in metrics:
                scores.append(metrics['performance_score'])
                
        return np.mean(scores) if scores else 0.0
        
    async def _trigger_cluster_optimization(self, cluster_id: str) -> None:
        """Trigger cluster optimization if needed."""
        try:
            await self.optimize_cluster(cluster_id)
        except Exception as e:
            logger.error(f"Error triggering optimization for cluster {cluster_id}: {str(e)}")
            
    async def _get_cluster_parameters(self, cluster_id: str) -> Dict[str, float]:
        """Get current cluster parameters."""
        # Implementation depends on specific parameters to optimize
        return {}
        
    async def _apply_cluster_parameters(
        self,
        cluster_id: str,
        parameters: Dict[str, float]
    ) -> None:
        """Apply optimized parameters to cluster."""
        # Implementation depends on specific parameters to optimize
        pass