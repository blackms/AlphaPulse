"""
Main supervisor agent implementation with ML optimization and monitoring.
"""
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
from loguru import logger

from .interfaces import (
    AgentState,
    Task,
    AgentHealth,
    ISelfSupervisedAgent
)
from .managers import (
    LifecycleManager,
    TaskManager,
    MetricsMonitor
)
from .optimization.ml_optimizer import MLOptimizer
from .monitoring.performance_monitor import PerformanceMonitor
from .distributed.coordinator import ClusterCoordinator


class SupervisorAgent:
    """
    Main supervisor agent that orchestrates self-supervised agents.
    Implements the Singleton pattern for global access.
    """
    
    _instance = None
    
    @classmethod
    def instance(cls) -> 'SupervisorAgent':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        """Initialize supervisor components."""
        if SupervisorAgent._instance is not None:
            raise RuntimeError("Use SupervisorAgent.instance() to get the supervisor")
            
        self.lifecycle_manager = LifecycleManager()
        self.task_manager = TaskManager()
        self.metrics_monitor = MetricsMonitor()
        
        # Advanced components
        self.ml_optimizer = MLOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.distributed_coordinator = ClusterCoordinator()
        
        self._monitoring_interval = 60  # seconds
        self._optimization_interval = 3600  # seconds
        self._last_optimization = datetime.now()
        self._monitoring_task = None
        self._is_running = False
        self._cluster_id = None
        
    async def start(self) -> None:
        """Start the supervisor agent."""
        if self._is_running:
            logger.warning("Supervisor agent is already running")
            return
            
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Initialize cluster
        self._cluster_id = f"cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.distributed_coordinator.register_cluster(self._cluster_id)
        
        logger.info("Supervisor agent started")
        
    async def stop(self) -> None:
        """Stop the supervisor agent."""
        if not self._is_running:
            return
            
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            
        # Stop all agents
        for agent_id in list(self.lifecycle_manager._agents.keys()):
            await self.lifecycle_manager.stop_agent(agent_id)
            
        logger.info("Supervisor agent stopped")
        
    async def register_agent(
        self,
        agent_id: str,
        config: Dict[str, Any]
    ) -> ISelfSupervisedAgent:
        """
        Register and initialize a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Agent configuration parameters
            
        Returns:
            Initialized agent instance
        """
        # Initialize agent
        agent = await self.lifecycle_manager.initialize_agent(agent_id, config)
        await self.lifecycle_manager.start_agent(agent_id)
        
        # Assign to cluster
        if self._cluster_id:
            await self.distributed_coordinator.assign_agent(
                agent_id,
                preferred_cluster=self._cluster_id
            )
        
        logger.info(f"Registered agent {agent_id}")
        return agent
        
    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister and stop an agent.
        
        Args:
            agent_id: Agent identifier to unregister
        """
        await self.lifecycle_manager.stop_agent(agent_id)
        # Cleanup agent resources
        if agent_id in self.lifecycle_manager._agents:
            del self.lifecycle_manager._agents[agent_id]
        if agent_id in self.lifecycle_manager._agent_configs:
            del self.lifecycle_manager._agent_configs[agent_id]
        logger.info(f"Unregistered agent {agent_id}")
        
    async def delegate_task(
        self,
        agent_id: str,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 0
    ) -> Task:
        """
        Delegate a task to an agent.
        
        Args:
            agent_id: Target agent identifier
            task_type: Type of task to execute
            parameters: Task parameters
            priority: Task priority (higher number = higher priority)
            
        Returns:
            Created task
        """
        task = await self.task_manager.create_task(
            agent_id,
            task_type,
            parameters,
            priority
        )
        await self.task_manager.assign_task(task)
        return task
        
    async def get_agent_status(self, agent_id: str) -> AgentHealth:
        """
        Get current health status of an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent health status
        """
        agent = self.lifecycle_manager._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        return await agent.get_health_status()
        
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status including ML-based analytics.
        
        Returns:
            System health metrics and analytics
        """
        base_status = await self.metrics_monitor.get_system_health()
        
        # Get cluster analytics
        cluster_analytics = {}
        if self._cluster_id:
            cluster_analytics = await self.distributed_coordinator.get_cluster_analytics(
                self._cluster_id
            )
            
        # Get performance analytics for all agents
        performance_analytics = {}
        for agent_id in self.lifecycle_manager._agents:
            analytics = await self.performance_monitor.get_performance_analytics(agent_id)
            if analytics:
                performance_analytics[agent_id] = analytics
                
        return {
            **base_status,
            'cluster_analytics': cluster_analytics,
            'performance_analytics': performance_analytics
        }
        
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop with ML-based analysis."""
        while self._is_running:
            try:
                await self._check_agent_health()
                await self._optimize_if_needed()
                await asyncio.sleep(self._monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
    async def _check_agent_health(self) -> None:
        """Check health of all agents using ML-based monitoring."""
        for agent_id, agent in self.lifecycle_manager._agents.items():
            try:
                # Get agent health status
                health = await agent.get_health_status()
                
                # Monitor performance
                monitoring_result = await self.performance_monitor.monitor_performance(
                    agent_id,
                    health.metrics
                )
                
                # Update distributed coordinator
                if self._cluster_id:
                    await self.distributed_coordinator.update_metrics(
                        agent_id,
                        health.metrics
                    )
                
                # Handle alerts
                if monitoring_result['alert_level'] != 'normal':
                    logger.warning(
                        f"Detected anomalies for agent {agent_id}: "
                        f"{monitoring_result}"
                    )
                    
                    # Handle different alert levels
                    if monitoring_result['alert_level'] == 'critical':
                        await self.lifecycle_manager.restart_agent(agent_id)
                    elif monitoring_result['alert_level'] == 'warning':
                        # Pause agent temporarily
                        await self.lifecycle_manager.stop_agent(agent_id)
                        await asyncio.sleep(30)  # Cool-down period
                        await self.lifecycle_manager.start_agent(agent_id)
                        
            except Exception as e:
                logger.error(f"Error checking health for agent {agent_id}: {str(e)}")
                
    async def _optimize_if_needed(self) -> None:
        """Check if optimization is needed using ML-based analysis."""
        if (datetime.now() - self._last_optimization).total_seconds() < self._optimization_interval:
            return
            
        try:
            # Optimize cluster if available
            if self._cluster_id:
                optimization_result = await self.distributed_coordinator.optimize_cluster(
                    self._cluster_id
                )
                logger.info(f"Cluster optimization result: {optimization_result}")
                
            # Optimize individual agents
            for agent_id, agent in self.lifecycle_manager._agents.items():
                # Get performance history
                analytics = await self.performance_monitor.get_performance_analytics(
                    agent_id
                )
                
                if analytics.get('trends', {}).get('performance_score', 0) < 0:
                    logger.info(f"Triggering optimization for agent {agent_id}")
                    await self.delegate_task(
                        agent_id,
                        "optimize",
                        {},
                        priority=1
                    )
                    
            self._last_optimization = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in optimization check: {str(e)}")