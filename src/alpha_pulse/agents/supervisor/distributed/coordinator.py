"""
Distributed coordination for self-supervised agents.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger


class ClusterCoordinator:
    """
    Coordinates distributed agent clusters and manages their lifecycle.
    """
    
    def __init__(self):
        """Initialize cluster coordinator."""
        self._clusters: Dict[str, Dict[str, Any]] = {}
        self._active_cluster_id: Optional[str] = None
        
    def register_cluster(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new agent cluster.
        
        Args:
            config: Optional cluster configuration
            
        Returns:
            Cluster ID
        """
        cluster_id = f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._clusters[cluster_id] = {
            "id": cluster_id,
            "created_at": datetime.now(),
            "config": config or {},
            "status": "active",
            "agents": {},
            "metrics": {
                "total_signals": 0,
                "active_agents": 0,
                "performance_score": 1.0
            }
        }
        
        self._active_cluster_id = cluster_id
        logger.info(f"Registered new cluster: {cluster_id}")
        return cluster_id
        
    def get_active_cluster(self) -> Optional[Dict[str, Any]]:
        """Get currently active cluster configuration."""
        if self._active_cluster_id:
            return self._clusters.get(self._active_cluster_id)
        return None
        
    async def update_metrics(
        self,
        agent_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update metrics for an agent.
        
        Args:
            agent_id: ID of the agent
            metrics: New metric values
        """
        if self._active_cluster_id:
            cluster = self._clusters[self._active_cluster_id]
            if agent_id in cluster["agents"]:
                cluster["agents"][agent_id]["metrics"] = metrics
                
    async def assign_agent(
        self,
        agent_id: str,
        preferred_cluster: Optional[str] = None
    ) -> None:
        """
        Assign an agent to a cluster.
        
        Args:
            agent_id: ID of the agent to assign
            preferred_cluster: Optional preferred cluster ID
        """
        cluster_id = preferred_cluster or self._active_cluster_id
        if not cluster_id or cluster_id not in self._clusters:
            cluster_id = self.register_cluster()
            
        self.register_agent(
            cluster_id=cluster_id,
            agent_id=agent_id,
            agent_type="unknown",  # Type will be updated with metrics
            config={}
        )
        
    def update_cluster_metrics(
        self,
        cluster_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update metrics for a specific cluster.
        
        Args:
            cluster_id: ID of the cluster to update
            metrics: New metric values
        """
        if cluster_id in self._clusters:
            self._clusters[cluster_id]["metrics"].update(metrics)
            
    def register_agent(
        self,
        cluster_id: str,
        agent_id: str,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an agent with a cluster.
        
        Args:
            cluster_id: ID of the cluster
            agent_id: ID of the agent
            agent_type: Type of the agent
            config: Optional agent configuration
        """
        if cluster_id in self._clusters:
            self._clusters[cluster_id]["agents"][agent_id] = {
                "id": agent_id,
                "type": agent_type,
                "config": config or {},
                "status": "active",
                "registered_at": datetime.now()
            }
            self._clusters[cluster_id]["metrics"]["active_agents"] += 1
            
    def deregister_agent(self, cluster_id: str, agent_id: str) -> None:
        """
        Remove an agent from a cluster.
        
        Args:
            cluster_id: ID of the cluster
            agent_id: ID of the agent to remove
        """
        if (cluster_id in self._clusters and 
            agent_id in self._clusters[cluster_id]["agents"]):
            del self._clusters[cluster_id]["agents"][agent_id]
            self._clusters[cluster_id]["metrics"]["active_agents"] -= 1
            
    def get_cluster_agents(self, cluster_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all agents in a cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary of agent configurations
        """
        if cluster_id in self._clusters:
            return self._clusters[cluster_id]["agents"].copy()
        return {}
        
    def get_cluster_metrics(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Cluster metrics
        """
        if cluster_id in self._clusters:
            return self._clusters[cluster_id]["metrics"].copy()
        return {}