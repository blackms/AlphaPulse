"""
Interfaces for the supervisor agent system.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from ..interfaces import ITradeAgent, TradeSignal


class AgentState(Enum):
    """Possible states of a self-supervised agent."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OPTIMIZING = "optimizing"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class AgentHealth:
    """Health status of a self-supervised agent."""
    state: AgentState
    last_active: datetime
    error_count: int
    last_error: Optional[str]
    memory_usage: float  # in MB
    cpu_usage: float    # percentage
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Task:
    """Task to be executed by a self-supervised agent."""
    task_id: str
    agent_id: str
    task_type: str
    priority: int
    parameters: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ISelfSupervisedAgent(ITradeAgent):
    """
    Extended interface for self-supervised agents.
    Adds capabilities for self-evaluation, optimization, and health monitoring.
    """
    
    @abstractmethod
    async def self_evaluate(self) -> Dict[str, float]:
        """
        Evaluate agent's own performance and return metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        pass
    
    @abstractmethod
    async def optimize(self) -> None:
        """
        Self-optimize based on performance metrics.
        May involve parameter tuning, model retraining, or strategy adjustment.
        """
        pass
    
    @abstractmethod
    async def get_health_status(self) -> AgentHealth:
        """
        Report agent's current health metrics.
        
        Returns:
            AgentHealth object with current status
        """
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> None:
        """
        Execute a specific task assigned by the supervisor.
        
        Args:
            task: Task object with details of what to execute
        """
        pass
    
    @abstractmethod
    async def pause(self) -> None:
        """Temporarily pause agent operations."""
        pass
    
    @abstractmethod
    async def resume(self) -> None:
        """Resume agent operations."""
        pass


class ILifecycleManager(ABC):
    """Interface for managing agent lifecycles."""
    
    @abstractmethod
    async def initialize_agent(self, agent_id: str, config: Dict[str, Any]) -> ISelfSupervisedAgent:
        """Initialize a new agent."""
        pass
    
    @abstractmethod
    async def start_agent(self, agent_id: str) -> None:
        """Start an initialized agent."""
        pass
    
    @abstractmethod
    async def stop_agent(self, agent_id: str) -> None:
        """Stop a running agent."""
        pass
    
    @abstractmethod
    async def restart_agent(self, agent_id: str) -> None:
        """Restart an agent (stop and start)."""
        pass
    
    @abstractmethod
    async def get_agent_status(self, agent_id: str) -> AgentState:
        """Get current state of an agent."""
        pass


class ITaskManager(ABC):
    """Interface for managing agent tasks."""
    
    @abstractmethod
    async def create_task(
        self,
        agent_id: str,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 0
    ) -> Task:
        """Create a new task."""
        pass
    
    @abstractmethod
    async def assign_task(self, task: Task) -> None:
        """Assign task to an agent."""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> str:
        """Get current status of a task."""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> None:
        """Cancel a pending or running task."""
        pass


class IMetricsMonitor(ABC):
    """Interface for monitoring agent metrics."""
    
    @abstractmethod
    async def collect_metrics(self, agent_id: str) -> Dict[str, float]:
        """Collect current metrics from an agent."""
        pass
    
    @abstractmethod
    async def analyze_performance(self, agent_id: str) -> Dict[str, Any]:
        """Analyze agent performance over time."""
        pass
    
    @abstractmethod
    async def detect_anomalies(self, agent_id: str) -> List[Dict[str, Any]]:
        """Detect anomalies in agent behavior."""
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        pass