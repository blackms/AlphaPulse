"""
Supervisor module for self-supervised agents.
"""
from .base import BaseSelfSupervisedAgent
from .factory import AgentFactory, create_self_supervised_agent
from .technical_agent import SelfSupervisedTechnicalAgent
from .sentiment_agent import SelfSupervisedSentimentAgent
from .supervisor import SupervisorAgent
from .distributed.coordinator import ClusterCoordinator
from .interfaces import (
    AgentState,
    AgentHealth,
    Task,
    ISelfSupervisedAgent,
    ILifecycleManager,
    ITaskManager,
    IMetricsMonitor
)

__all__ = [
    'BaseSelfSupervisedAgent',
    'AgentFactory',
    'create_self_supervised_agent',
    'SelfSupervisedTechnicalAgent',
    'SelfSupervisedSentimentAgent',
    'SupervisorAgent',
    'ClusterCoordinator',
    'AgentState',
    'AgentHealth',
    'Task',
    'ISelfSupervisedAgent',
    'ILifecycleManager',
    'ITaskManager',
    'IMetricsMonitor'
]