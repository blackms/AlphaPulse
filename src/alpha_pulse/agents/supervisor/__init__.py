"""
Supervisor module initialization.
"""
from .base import BaseSelfSupervisedAgent
from .factory import AgentFactory, create_self_supervised_agent
from .supervisor import SupervisorAgent
from .interfaces import AgentState, AgentHealth, Task, TradeSignal
from .sentiment_agent import SelfSupervisedSentimentAgent

__all__ = [
    'BaseSelfSupervisedAgent',
    'AgentFactory',
    'create_self_supervised_agent',
    'SupervisorAgent',
    'AgentState',
    'AgentHealth',
    'Task',
    'TradeSignal',
    'SelfSupervisedSentimentAgent'
]