"""
Common test fixtures and configuration for supervisor tests.
"""
import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

from alpha_pulse.agents.interfaces import MarketData
from alpha_pulse.agents.supervisor.base import BaseSelfSupervisedAgent
from alpha_pulse.agents.supervisor.interfaces import AgentState, AgentHealth, Task
from alpha_pulse.agents.supervisor.supervisor import SupervisorAgent

class MockAgent(BaseSelfSupervisedAgent):
    """Mock agent for testing."""
    def __init__(self, agent_id: str, config=None):
        super().__init__(agent_id, config or {})

@pytest.fixture
def market_data():
    """Create mock market data for testing."""
    return MarketData(
        prices=Mock(columns=['BTC/USD']),
        volumes=Mock(columns=['BTC/USD']),
        timestamp=datetime.now()
    )

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent("test_agent", {
        "optimization_threshold": 0.7,
        "monitoring_interval": 30,
        "max_errors": 5
    })

@pytest.fixture
def mock_health():
    """Create mock health status for testing."""
    return AgentHealth(
        state=AgentState.ACTIVE,
        last_active=datetime.now(),
        error_count=0,
        last_error=None,
        memory_usage=100.0,
        cpu_usage=50.0,
        metrics={"performance_score": 0.8}
    )

@pytest.fixture
def mock_metrics():
    """Create mock metrics for testing."""
    return Mock(
        signal_accuracy=0.6,
        profit_factor=1.2,
        sharpe_ratio=1.0,
        max_drawdown=0.2
    )

@pytest_asyncio.fixture
async def mock_async_agent():
    """Create a mock agent with async methods for testing."""
    agent = AsyncMock()
    agent.agent_id = "test_agent"
    agent._state = AgentState.ACTIVE
    
    # Configure health status
    health = AgentHealth(
        state=AgentState.ACTIVE,
        last_active=datetime.now(),
        error_count=0,
        last_error=None,
        memory_usage=100.0,
        cpu_usage=50.0,
        metrics={"performance_score": 0.8}
    )
    agent.get_health_status.return_value = health
    
    # Configure other async methods
    agent.execute_task = AsyncMock(side_effect=lambda task: setattr(task, 'status', 'completed'))
    agent.optimize = AsyncMock()
    agent.initialize = AsyncMock()
    agent.stop = AsyncMock()
    
    return agent

@pytest_asyncio.fixture
async def mock_supervisor():
    """Create a mock supervisor for testing."""
    supervisor = AsyncMock()
    supervisor._is_running = True
    supervisor._cluster_id = "test_cluster"
    supervisor._monitoring_task = None
    
    # Configure managers
    supervisor.lifecycle_manager = AsyncMock()
    supervisor.lifecycle_manager._agents = {}
    
    supervisor.task_manager = AsyncMock()
    task = Task(
        task_id="test_task",
        agent_id="test_agent",
        task_type="optimize",
        priority=1,
        parameters={}
    )
    supervisor.task_manager.create_task.return_value = task
    
    supervisor.metrics_monitor = AsyncMock()
    supervisor.distributed_coordinator = AsyncMock()
    
    # Configure methods
    supervisor.get_system_status.return_value = {
        "total_agents": 2,
        "active_agents": 2,
        "cluster_analytics": {},
        "performance_analytics": {}
    }
    
    supervisor.get_agent_status.side_effect = lambda agent_id: (
        AgentHealth(
            state=AgentState.ACTIVE,
            last_active=datetime.now(),
            error_count=0,
            last_error=None,
            memory_usage=100.0,
            cpu_usage=50.0,
            metrics={"performance_score": 0.8}
        ) if agent_id in supervisor.lifecycle_manager._agents
        else ValueError(f"Agent '{agent_id}' not found")
    )
    
    # Configure register_agent method
    async def mock_register_agent(agent_id, config):
        agent = AsyncMock()
        agent.agent_id = agent_id
        agent._state = AgentState.ACTIVE
        agent.get_health_status.return_value = AgentHealth(
            state=AgentState.ACTIVE,
            last_active=datetime.now(),
            error_count=0,
            last_error=None,
            memory_usage=100.0,
            cpu_usage=50.0,
            metrics={"performance_score": 0.8}
        )
        supervisor.lifecycle_manager._agents[agent_id] = agent
        return agent
    supervisor.register_agent.side_effect = mock_register_agent
    
    # Configure unregister_agent method
    async def mock_unregister_agent(agent_id):
        if agent_id in supervisor.lifecycle_manager._agents:
            del supervisor.lifecycle_manager._agents[agent_id]
    supervisor.unregister_agent.side_effect = mock_unregister_agent
    
    # Configure stop method
    async def mock_stop():
        supervisor._is_running = False
        if supervisor._monitoring_task:
            supervisor._monitoring_task.cancel()
            supervisor._monitoring_task = None
        # Stop all agents
        for agent in supervisor.lifecycle_manager._agents.values():
            await agent.stop()
    supervisor.stop.side_effect = mock_stop
    
    # Configure _check_agent_health method
    async def mock_check_agent_health():
        for agent_id, agent in supervisor.lifecycle_manager._agents.items():
            health = await agent.get_health_status()
            if health.metrics.get("performance_score", 1.0) < 0.5:
                await supervisor.task_manager.create_task(
                    agent_id,
                    "optimize",
                    {},
                    priority=1
                )
    supervisor._check_agent_health = AsyncMock(side_effect=mock_check_agent_health)
    
    return supervisor

@pytest_asyncio.fixture
async def supervisor():
    """Create and start a supervisor agent for testing."""
    # Clear singleton instance
    SupervisorAgent._instance = None
    supervisor = SupervisorAgent.instance()
    await supervisor.start()
    
    try:
        yield supervisor
    finally:
        # Cleanup
        await supervisor.stop()
        SupervisorAgent._instance = None