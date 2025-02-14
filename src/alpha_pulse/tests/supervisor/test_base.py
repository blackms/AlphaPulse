"""
Tests for the base self-supervised agent implementation.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, patch

from alpha_pulse.agents.supervisor.base import BaseSelfSupervisedAgent
from alpha_pulse.agents.supervisor.interfaces import AgentState, Task

@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization with config."""
    agent = BaseSelfSupervisedAgent("test_agent", {
        "optimization_threshold": 0.8,
        "monitoring_interval": 45
    })
    
    assert agent.agent_id == "test_agent"
    assert agent._state == AgentState.INITIALIZING
    assert agent._optimization_threshold == 0.8
    assert agent.config["monitoring_interval"] == 45
    assert len(agent._performance_history) == 0

@pytest.mark.asyncio
async def test_agent_lifecycle(mock_agent):
    """Test agent lifecycle state transitions."""
    # Test initialization
    await mock_agent.initialize({})
    assert mock_agent._state == AgentState.ACTIVE
    
    # Test pause/resume
    await mock_agent.pause()
    assert mock_agent._state == AgentState.INACTIVE
    
    await mock_agent.resume()
    assert mock_agent._state == AgentState.ACTIVE
    
    # Test stop
    await mock_agent.stop()
    assert mock_agent._state == AgentState.STOPPED

@pytest.mark.asyncio
async def test_self_evaluation(mock_agent):
    """Test agent self-evaluation logic."""
    # Mock metrics with values that will result in a score between 0 and 1
    mock_agent.metrics = Mock(
        signal_accuracy=0.6,
        profit_factor=1.2,
        sharpe_ratio=1.0,
        max_drawdown=0.2
    )
    
    metrics = await mock_agent.self_evaluate()
    
    assert "performance_score" in metrics
    assert 0 <= metrics["performance_score"] <= 1
    assert len(mock_agent._performance_history) == 1

@pytest.mark.asyncio
async def test_optimization_trigger(mock_agent, market_data):
    """Test optimization triggering based on performance."""
    # Mock poor performance metrics
    mock_agent.metrics = Mock(
        signal_accuracy=0.4,
        profit_factor=0.8,
        sharpe_ratio=0.5,
        max_drawdown=0.3
    )
    
    with patch.object(mock_agent, 'optimize') as mock_optimize:
        await mock_agent.generate_signals(market_data)
        mock_optimize.assert_called_once()

@pytest.mark.asyncio
async def test_health_status(mock_agent):
    """Test health status reporting."""
    health = await mock_agent.get_health_status()
    
    assert health.state == mock_agent._state
    assert isinstance(health.memory_usage, float)
    assert isinstance(health.cpu_usage, float)
    assert isinstance(health.error_count, int)

@pytest.mark.asyncio
async def test_task_execution(mock_agent):
    """Test task execution handling."""
    task = Task(
        task_id="test_task",
        agent_id=mock_agent.agent_id,
        task_type="optimize",
        priority=1,
        parameters={}
    )
    
    with patch.object(mock_agent, 'optimize') as mock_optimize:
        await mock_agent.execute_task(task)
        mock_optimize.assert_called_once()
        assert task.status == "completed"

@pytest.mark.asyncio
async def test_error_handling(mock_agent, market_data):
    """Test error handling and state transitions."""
    # Mock generate_signals in parent class to raise an exception
    with patch('alpha_pulse.agents.interfaces.BaseTradeAgent.generate_signals',
              side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await mock_agent.generate_signals(market_data)
        
        assert mock_agent._state == AgentState.ERROR
        assert mock_agent._error_count == 1
        assert "Test error" in mock_agent._last_error

@pytest.mark.asyncio
async def test_performance_history(mock_agent):
    """Test performance history tracking."""
    # Mock metrics with reasonable values
    mock_agent.metrics = Mock(
        signal_accuracy=0.6,
        profit_factor=1.2,
        sharpe_ratio=1.0,
        max_drawdown=0.2
    )
    
    for _ in range(3):
        await mock_agent.self_evaluate()
    
    assert len(mock_agent._performance_history) == 3
    assert all("performance_score" in metrics for metrics in mock_agent._performance_history)
    assert all(0 <= metrics["performance_score"] <= 1 
              for metrics in mock_agent._performance_history)

@pytest.mark.asyncio
async def test_config_defaults(mock_agent):
    """Test configuration defaults and overrides."""
    assert mock_agent.config["max_errors"] == 5  # From fixture
    assert mock_agent.config["memory_limit_mb"] == 1000  # Default
    assert mock_agent._optimization_threshold == 0.7  # From fixture