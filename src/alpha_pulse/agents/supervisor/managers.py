"""
Core manager implementations for the supervisor agent system.
"""
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

from loguru import logger

from .interfaces import (
    ILifecycleManager,
    ITaskManager,
    IMetricsMonitor,
    ISelfSupervisedAgent,
    AgentState,
    Task,
    AgentHealth
)


class LifecycleManager(ILifecycleManager):
    """Manages the lifecycle of self-supervised agents."""
    
    def __init__(self):
        self._agents: Dict[str, ISelfSupervisedAgent] = {}
        self._agent_configs: Dict[str, Dict[str, Any]] = {}
        
    async def initialize_agent(
        self,
        agent_id: str,
        config: Dict[str, Any]
    ) -> ISelfSupervisedAgent:
        """Initialize a new agent with configuration."""
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' already exists")
            
        # Agent creation should be handled by a factory
        from .factory import AgentFactory
        # Extract agent type from config
        agent_type = config.get("type", "technical")  # Default to technical if not specified
        agent = await AgentFactory.create_agent(agent_type, agent_id, config)
        
        self._agents[agent_id] = agent
        self._agent_configs[agent_id] = config
        
        await agent.initialize(config)
        logger.info(f"Initialized agent '{agent_id}'")
        
        return agent
        
    async def start_agent(self, agent_id: str) -> None:
        """Start an initialized agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' not found")
            
        agent = self._agents[agent_id]
        await agent.resume()
        logger.info(f"Started agent '{agent_id}'")
        
    async def stop_agent(self, agent_id: str) -> None:
        """Stop a running agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self._agents[agent_id]
        await agent.pause()
        logger.info(f"Stopped agent '{agent_id}'")
        
    async def restart_agent(self, agent_id: str) -> None:
        """Restart an agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' not found")
            
        await self.stop_agent(agent_id)
        
        # Re-initialize with stored config
        config = self._agent_configs[agent_id]
        agent = self._agents[agent_id]
        await agent.initialize(config)
        
        await self.start_agent(agent_id)
        logger.info(f"Restarted agent '{agent_id}'")
        
    async def get_agent_status(self, agent_id: str) -> AgentState:
        """Get current state of an agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' not found")
            
        agent = self._agents[agent_id]
        health = await agent.get_health_status()
        return health.state


class TaskManager(ITaskManager):
    """Manages tasks for self-supervised agents."""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._agent_tasks: Dict[str, List[str]] = defaultdict(list)
        self._task_queue: List[Task] = []
        
    async def create_task(
        self,
        agent_id: str,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 0
    ) -> Task:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            priority=priority,
            parameters=parameters
        )
        
        self._tasks[task_id] = task
        self._agent_tasks[agent_id].append(task_id)
        
        # Add to priority queue
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created task '{task_id}' for agent '{agent_id}'")
        return task
        
    async def assign_task(self, task: Task) -> None:
        """Assign task to an agent."""
        if task.task_id not in self._tasks:
            raise ValueError(f"Task '{task.task_id}' not found")
            
        # Get agent from lifecycle manager
        from .supervisor import SupervisorAgent
        agent = SupervisorAgent.instance().lifecycle_manager._agents.get(task.agent_id)
        
        if not agent:
            raise ValueError(f"Agent '{task.agent_id}' not found")
            
        task.status = "running"
        await agent.execute_task(task)
        
        # Update task status based on execution
        self._tasks[task.task_id] = task
        logger.info(f"Completed task '{task.task_id}' for agent '{task.agent_id}'")
        
    async def get_task_status(self, task_id: str) -> str:
        """Get current status of a task."""
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found")
        return self._tasks[task_id].status
        
    async def cancel_task(self, task_id: str) -> None:
        """Cancel a pending or running task."""
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found")
            
        task = self._tasks[task_id]
        if task.status in ["pending", "running"]:
            task.status = "cancelled"
            task.completed_at = datetime.now()
            
            # Remove from queue if pending
            self._task_queue = [t for t in self._task_queue if t.task_id != task_id]
            
            logger.info(f"Cancelled task '{task_id}'")


class MetricsMonitor(IMetricsMonitor):
    """Monitors metrics and health of self-supervised agents."""
    
    def __init__(self):
        self._metrics_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._health_history: Dict[str, List[AgentHealth]] = defaultdict(list)
        self._anomaly_thresholds = {
            "error_rate": 0.1,
            "memory_usage": 1000,  # MB
            "cpu_usage": 80.0,     # percent
            "performance_drop": 0.3
        }
        
    async def collect_metrics(self, agent_id: str) -> Dict[str, float]:
        """Collect current metrics from an agent."""
        # Get agent from lifecycle manager
        from .supervisor import SupervisorAgent
        agent = SupervisorAgent.instance().lifecycle_manager._agents.get(agent_id)
        
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")
            
        health = await agent.get_health_status()
        metrics = health.metrics
        
        self._metrics_history[agent_id].append(metrics)
        self._health_history[agent_id].append(health)
        
        return metrics
        
    async def analyze_performance(self, agent_id: str) -> Dict[str, Any]:
        """Analyze agent performance over time."""
        if agent_id not in self._metrics_history:
            return {}
            
        metrics_history = self._metrics_history[agent_id]
        if not metrics_history:
            return {}
            
        # Calculate performance trends
        performance_scores = [m.get("performance_score", 0) for m in metrics_history]
        
        return {
            "current_score": performance_scores[-1],
            "average_score": sum(performance_scores) / len(performance_scores),
            "score_trend": "improving" if len(performance_scores) > 1 and
                          performance_scores[-1] > performance_scores[-2] else "declining",
            "metrics_collected": len(metrics_history)
        }
        
    async def detect_anomalies(self, agent_id: str) -> List[Dict[str, Any]]:
        """Detect anomalies in agent behavior."""
        if agent_id not in self._health_history:
            return []
            
        health_history = self._health_history[agent_id]
        if not health_history:
            return []
            
        anomalies = []
        latest_health = health_history[-1]
        
        # Check error rate
        error_rate = latest_health.error_count / len(health_history)
        if error_rate > self._anomaly_thresholds["error_rate"]:
            anomalies.append({
                "type": "high_error_rate",
                "value": error_rate,
                "threshold": self._anomaly_thresholds["error_rate"]
            })
            
        # Check resource usage
        if latest_health.memory_usage > self._anomaly_thresholds["memory_usage"]:
            anomalies.append({
                "type": "high_memory_usage",
                "value": latest_health.memory_usage,
                "threshold": self._anomaly_thresholds["memory_usage"]
            })
            
        if latest_health.cpu_usage > self._anomaly_thresholds["cpu_usage"]:
            anomalies.append({
                "type": "high_cpu_usage",
                "value": latest_health.cpu_usage,
                "threshold": self._anomaly_thresholds["cpu_usage"]
            })
            
        # Check performance drops
        metrics_history = self._metrics_history[agent_id]
        if len(metrics_history) > 1:
            current_score = metrics_history[-1].get("performance_score", 0)
            previous_score = metrics_history[-2].get("performance_score", 0)
            
            if (previous_score - current_score) > self._anomaly_thresholds["performance_drop"]:
                anomalies.append({
                    "type": "performance_drop",
                    "value": previous_score - current_score,
                    "threshold": self._anomaly_thresholds["performance_drop"]
                })
                
        return anomalies
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        # Get all agents from lifecycle manager
        from .supervisor import SupervisorAgent
        agents = SupervisorAgent.instance().lifecycle_manager._agents
        
        total_memory = 0
        total_cpu = 0
        total_errors = 0
        active_agents = 0
        
        for agent_id, agent in agents.items():
            health = await agent.get_health_status()
            
            if health.state == AgentState.ACTIVE:
                active_agents += 1
                
            total_memory += health.memory_usage
            total_cpu += health.cpu_usage
            total_errors += health.error_count
            
        return {
            "total_agents": len(agents),
            "active_agents": active_agents,
            "total_memory_mb": total_memory,
            "average_cpu_percent": total_cpu / len(agents) if agents else 0,
            "total_errors": total_errors,
            "timestamp": datetime.now()
        }