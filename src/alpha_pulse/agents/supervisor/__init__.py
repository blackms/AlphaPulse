"""
Supervisor agent system for AlphaPulse.

Example usage:

```python
from alpha_pulse.agents.supervisor import (
    SupervisorAgent,
    AgentFactory,
    AgentState,
    Task
)

async def main():
    # Get supervisor instance
    supervisor = SupervisorAgent.instance()
    await supervisor.start()
    
    try:
        # Register a technical agent
        technical_config = {
            "type": "technical",
            "optimization_threshold": 0.7,
            "indicators": ["SMA", "RSI", "MACD"]
        }
        technical_agent = await supervisor.register_agent(
            "tech_agent_1",
            technical_config
        )
        
        # Register a fundamental agent
        fundamental_config = {
            "type": "fundamental",
            "optimization_threshold": 0.8,
            "metrics": ["PE", "PB", "ROE"]
        }
        fundamental_agent = await supervisor.register_agent(
            "fund_agent_1",
            fundamental_config
        )
        
        # Delegate tasks
        task = await supervisor.delegate_task(
            "tech_agent_1",
            "optimize",
            {"full_optimization": True},
            priority=1
        )
        
        # Monitor system health
        system_status = await supervisor.get_system_status()
        print(f"System status: {system_status}")
        
        # Monitor specific agent
        agent_health = await supervisor.get_agent_status("tech_agent_1")
        print(f"Agent health: {agent_health}")
        
    finally:
        # Cleanup
        await supervisor.stop()

# Run example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```
"""

from .interfaces import (
    ISelfSupervisedAgent,
    ILifecycleManager,
    ITaskManager,
    IMetricsMonitor,
    AgentState,
    AgentHealth,
    Task
)

from .base import BaseSelfSupervisedAgent
from .factory import AgentFactory
from .managers import (
    LifecycleManager,
    TaskManager,
    MetricsMonitor
)
from .supervisor import SupervisorAgent

# Initialize additional agent types
from .factory import register_additional_agent_types
register_additional_agent_types()

__all__ = [
    # Interfaces
    'ISelfSupervisedAgent',
    'ILifecycleManager',
    'ITaskManager',
    'IMetricsMonitor',
    'AgentState',
    'AgentHealth',
    'Task',
    
    # Base classes
    'BaseSelfSupervisedAgent',
    
    # Factory
    'AgentFactory',
    
    # Managers
    'LifecycleManager',
    'TaskManager',
    'MetricsMonitor',
    
    # Main supervisor
    'SupervisorAgent',
]