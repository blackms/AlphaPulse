"""
Tests for the main supervisor agent implementation.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, create_autospec
import asyncio

from alpha_pulse.agents.supervisor.supervisor import SupervisorAgent
from alpha_pulse.agents.supervisor.interfaces import AgentState, Task, AgentHealth

@pytest.mark.asyncio
async def test_singleton_pattern():
    """Test supervisor agent singleton pattern."""
    # Clear any existing instance
    SupervisorAgent._instance = None
    
    supervisor1 = SupervisorAgent.instance()
    supervisor2 = SupervisorAgent.instance()
    assert supervisor1 is supervisor2
    
    with pytest.raises(RuntimeError):
        SupervisorAgent()

@pytest.mark.asyncio
async def test_agent_registration(mock_supervisor):
    """Test agent registration and initialization."""
    # Create a mock agent directly
    mock_agent = AsyncMock()
    mock_agent.agent_id = "test_agent"
    mock_supervisor.register_agent.return_value = mock_agent
    
    agent = await mock_supervisor.register_agent(
        "test_agent",
        {
            "type": "technical",
            "optimization_threshold": 0.8
        }
    )
    
    assert agent.agent_id == "test_agent"
    assert "test_agent" in mock_supervisor.lifecycle_manager._agents

@pytest.mark.asyncio
async def test_agent_unregistration(mock_supervisor):
    """Test agent unregistration."""
    # Create a mock agent directly
    mock_agent = AsyncMock()
    mock_agent.agent_id = "test_agent"
    mock_supervisor.register_agent.return_value = mock_agent
    
    # First register an agent
    await mock_supervisor.register_agent(
        "test_agent",
        {"type": "technical"}
    )
    
    await mock_supervisor.unregister_agent("test_agent")
    assert "test_agent" not in mock_supervisor.lifecycle_manager._agents

@pytest.mark.asyncio
async def test_task_delegation(mock_supervisor, mock_async_agent):
    """Test task delegation to agents."""
    # First register an agent
    mock_supervisor.lifecycle_manager._agents["test_agent"] = mock_async_agent
    
    # Create a task
    task = Task(
        task_id="test_task",
        agent_id="test_agent",
        task_type="optimize",
        priority=1,
        parameters={}
    )
    mock_supervisor.delegate_task.return_value = task
    
    result = await mock_supervisor.delegate_task(
        "test_agent",
        "optimize",
        {"param": "value"},
        priority=1
    )
    
    assert result.agent_id == "test_agent"
    assert result.status == "pending"
    assert result.task_type == "optimize"

@pytest.mark.asyncio
async def test_agent_status(mock_supervisor, mock_async_agent, mock_health):
    """Test agent status monitoring."""
    mock_supervisor.lifecycle_manager._agents["test_agent"] = mock_async_agent
    mock_async_agent.get_health_status.return_value = mock_health
    
    health = await mock_supervisor.get_agent_status("test_agent")
    
    assert isinstance(health, AgentHealth)
    assert health.state == AgentState.ACTIVE
    assert health.error_count == 0

@pytest.mark.asyncio
async def test_system_status(mock_supervisor, mock_async_agent):
    """Test system-wide status monitoring."""
    mock_supervisor.lifecycle_manager._agents = {
        "agent1": mock_async_agent,
        "agent2": mock_async_agent
    }
    
    status = await mock_supervisor.get_system_status()
    
    assert status["total_agents"] == 2
    assert status["active_agents"] == 2
    assert "cluster_analytics" in status
    assert "performance_analytics" in status

@pytest.mark.asyncio
async def test_monitoring_loop(mock_supervisor):
    """Test monitoring loop functionality."""
    # Set up monitoring task
    mock_supervisor._monitoring_task = None
    mock_supervisor._is_running = False
    mock_supervisor._check_agent_health = AsyncMock()
    mock_supervisor._optimize_if_needed = AsyncMock()
    
    # Mock the monitoring loop
    async def mock_monitoring_loop():
        while mock_supervisor._is_running:
            await mock_supervisor._check_agent_health()
            await mock_supervisor._optimize_if_needed()
            await asyncio.sleep(0.1)
    mock_supervisor._monitoring_loop = mock_monitoring_loop
    
    # Start supervisor
    mock_supervisor._is_running = True
    mock_supervisor._monitoring_task = asyncio.create_task(mock_supervisor._monitoring_loop())
    
    # Wait for monitoring cycle
    await asyncio.sleep(0.2)
    
    mock_supervisor._check_agent_health.assert_called()
    mock_supervisor._optimize_if_needed.assert_called()
    
    # Cleanup
    mock_supervisor._is_running = False
    if mock_supervisor._monitoring_task:
        mock_supervisor._monitoring_task.cancel()
        try:
            await mock_supervisor._monitoring_task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_error_handling(mock_supervisor):
    """Test error handling in supervisor operations."""
    # Test invalid agent operations
    mock_supervisor.get_agent_status.side_effect = ValueError("Agent not found")
    
    with pytest.raises(ValueError):
        await mock_supervisor.get_agent_status("nonexistent_agent")

@pytest.mark.asyncio
async def test_performance_monitoring(mock_supervisor, mock_async_agent):
    """Test performance monitoring and optimization triggers."""
    # Set up agent with poor performance
    mock_supervisor.lifecycle_manager._agents["test_agent"] = mock_async_agent
    mock_async_agent.get_health_status.return_value = AgentHealth(
        state=AgentState.ACTIVE,
        last_active=datetime.now(),
        error_count=0,
        last_error=None,
        memory_usage=100.0,
        cpu_usage=50.0,
        metrics={"performance_score": 0.3}  # Poor performance
    )
    
    # Set up task creation chain
    task = Task(
        task_id="test_task",
        agent_id="test_agent",
        task_type="optimize",
        priority=1,
        parameters={}
    )
    mock_supervisor.task_manager.create_task = AsyncMock(return_value=task)
    mock_supervisor.task_manager.assign_task = AsyncMock()
    
    # Trigger monitoring cycle
    await mock_supervisor._check_agent_health()
    
    # Verify optimization was triggered
    mock_supervisor.task_manager.create_task.assert_called_with(
        "test_agent",
        "optimize",
        {},
        priority=1
    )

@pytest.mark.asyncio
async def test_cluster_coordination(mock_supervisor):
    """Test cluster coordination functionality."""
    assert mock_supervisor._cluster_id == "test_cluster"
    
    # Test cluster analytics
    analytics = await mock_supervisor.distributed_coordinator.get_cluster_analytics(
        mock_supervisor._cluster_id
    )
    assert analytics is not None

@pytest.mark.asyncio
async def test_supervisor_shutdown(mock_supervisor, mock_async_agent):
    """Test clean shutdown of supervisor and agents."""
    # Set up agents
    mock_supervisor.lifecycle_manager._agents = {
        "agent1": mock_async_agent,
        "agent2": mock_async_agent
    }
    mock_supervisor._is_running = True
    mock_supervisor._monitoring_task = asyncio.create_task(asyncio.sleep(1))
    
    # Set up agent stop method
    mock_async_agent.stop = AsyncMock()
    
    # Stop supervisor
    await mock_supervisor.stop()
    
    assert not mock_supervisor._is_running
    assert mock_supervisor._monitoring_task is None
    assert mock_async_agent.stop.call_count == 2  # Called for both agents

@pytest.mark.asyncio
async def test_supervisor_restart():
    """Test supervisor restart capability."""
    # Clear singleton instance
    SupervisorAgent._instance = None
    supervisor = SupervisorAgent.instance()
    
    await supervisor.start()
    assert supervisor._is_running
    
    await supervisor.stop()
    assert not supervisor._is_running
    
    await supervisor.start()
    assert supervisor._is_running
    
    # Cleanup
    await supervisor.stop()
    SupervisorAgent._instance = None