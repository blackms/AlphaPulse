"""
Tests for the supervisor agent system managers.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import uuid

from alpha_pulse.agents.supervisor.managers import (
    LifecycleManager,
    TaskManager,
    MetricsMonitor
)
from alpha_pulse.agents.supervisor.interfaces import (
    AgentState,
    Task,
    AgentHealth
)

# Lifecycle Manager Tests
@pytest.mark.asyncio
async def test_lifecycle_manager_initialization():
    """Test lifecycle manager initialization."""
    manager = LifecycleManager()
    assert len(manager._agents) == 0
    assert len(manager._agent_configs) == 0

@pytest.mark.asyncio
async def test_initialize_agent():
    """Test agent initialization through lifecycle manager."""
    manager = LifecycleManager()
    config = {"type": "technical", "optimization_threshold": 0.8}
    
    agent = await manager.initialize_agent("test_agent", config)
    
    assert agent.agent_id == "test_agent"
    assert agent._state == AgentState.ACTIVE
    assert "test_agent" in manager._agents
    assert manager._agent_configs["test_agent"] == config

@pytest.mark.asyncio
async def test_duplicate_agent_initialization():
    """Test error handling for duplicate agent initialization."""
    manager = LifecycleManager()
    await manager.initialize_agent("test_agent", {})
    
    with pytest.raises(ValueError) as exc_info:
        await manager.initialize_agent("test_agent", {})
    assert "already exists" in str(exc_info.value)

@pytest.mark.asyncio
async def test_agent_lifecycle_operations():
    """Test agent lifecycle operations (start, stop, restart)."""
    manager = LifecycleManager()
    agent = await manager.initialize_agent("test_agent", {})
    
    await manager.stop_agent("test_agent")
    assert (await manager.get_agent_status("test_agent")) == AgentState.INACTIVE
    
    await manager.start_agent("test_agent")
    assert (await manager.get_agent_status("test_agent")) == AgentState.ACTIVE
    
    await manager.restart_agent("test_agent")
    assert (await manager.get_agent_status("test_agent")) == AgentState.ACTIVE

# Task Manager Tests
@pytest.mark.asyncio
async def test_task_creation():
    """Test task creation and validation."""
    manager = TaskManager()
    
    task = await manager.create_task(
        "test_agent",
        "optimize",
        {"param": "value"},
        priority=1
    )
    
    assert task.agent_id == "test_agent"
    assert task.task_type == "optimize"
    assert task.priority == 1
    assert task.status == "pending"

@pytest.mark.asyncio
async def test_invalid_task_type():
    """Test error handling for invalid task types."""
    manager = TaskManager()
    
    with pytest.raises(ValueError) as exc_info:
        await manager.create_task(
            "test_agent",
            "invalid_type",
            {}
        )
    assert "Unknown task type" in str(exc_info.value)

@pytest.mark.asyncio
async def test_task_assignment(mock_async_agent):
    """Test task assignment to agent."""
    task_manager = TaskManager()
    
    # Patch SupervisorAgent.instance() to return our mock
    with patch('alpha_pulse.agents.supervisor.supervisor.SupervisorAgent.instance') as mock_supervisor:
        mock_supervisor.return_value.lifecycle_manager._agents = {
            "test_agent": mock_async_agent
        }
        
        task = await task_manager.create_task(
            "test_agent",
            "optimize",
            {}
        )
        
        await task_manager.assign_task(task)
        
        mock_async_agent.execute_task.assert_called_once_with(task)
        assert task.status == "completed"

@pytest.mark.asyncio
async def test_task_cancellation():
    """Test task cancellation."""
    manager = TaskManager()
    
    task = await manager.create_task(
        "test_agent",
        "optimize",
        {}
    )
    
    await manager.cancel_task(task.task_id)
    assert task.status == "cancelled"
    assert task.completed_at is not None

# Metrics Monitor Tests
@pytest.mark.asyncio
async def test_metrics_collection(mock_async_agent):
    """Test metrics collection from agent."""
    monitor = MetricsMonitor()
    
    # Patch SupervisorAgent.instance()
    with patch('alpha_pulse.agents.supervisor.supervisor.SupervisorAgent.instance') as mock_supervisor:
        mock_supervisor.return_value.lifecycle_manager._agents = {
            "test_agent": mock_async_agent
        }
        
        metrics = await monitor.collect_metrics("test_agent")
        
        assert metrics["performance_score"] == 0.8
        assert len(monitor._metrics_history["test_agent"]) == 1
        assert len(monitor._health_history["test_agent"]) == 1

@pytest.mark.asyncio
async def test_anomaly_detection(mock_async_agent):
    """Test anomaly detection in metrics."""
    monitor = MetricsMonitor()
    
    # Set up concerning metrics
    mock_async_agent.get_health_status.return_value = AgentHealth(
        state=AgentState.ACTIVE,
        last_active=datetime.now(),
        error_count=5,
        last_error=None,
        memory_usage=1200.0,  # Above threshold
        cpu_usage=90.0,       # Above threshold
        metrics={"performance_score": 0.4}
    )
    
    # Patch SupervisorAgent.instance()
    with patch('alpha_pulse.agents.supervisor.supervisor.SupervisorAgent.instance') as mock_supervisor:
        mock_supervisor.return_value.lifecycle_manager._agents = {
            "test_agent": mock_async_agent
        }
        
        await monitor.collect_metrics("test_agent")
        anomalies = await monitor.detect_anomalies("test_agent")
        
        assert len(anomalies) > 0
        assert any(a["type"] == "high_memory_usage" for a in anomalies)
        assert any(a["type"] == "high_cpu_usage" for a in anomalies)

@pytest.mark.asyncio
async def test_system_health(mock_async_agent):
    """Test system-wide health metrics collection."""
    monitor = MetricsMonitor()
    
    # Create multiple mock agents
    mock_agents = {
        "agent1": mock_async_agent,
        "agent2": mock_async_agent
    }
    
    # Patch SupervisorAgent.instance()
    with patch('alpha_pulse.agents.supervisor.supervisor.SupervisorAgent.instance') as mock_supervisor:
        mock_supervisor.return_value.lifecycle_manager._agents = mock_agents
        
        health = await monitor.get_system_health()
        
        assert health["total_agents"] == 2
        assert health["active_agents"] == 2
        assert health["total_memory_mb"] == 200.0
        assert health["average_cpu_percent"] == 50.0
        assert health["total_errors"] == 0