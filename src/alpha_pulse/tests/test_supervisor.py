"""
Tests for the supervisor agent system.
"""
import pytest
import pytest_asyncio
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from ..agents.supervisor import (
    SupervisorAgent,
    AgentFactory,
    AgentState,
    AgentHealth,
    Task,
    ISelfSupervisedAgent,
    BaseSelfSupervisedAgent
)
from ..agents.interfaces import MarketData, TradeSignal, SignalDirection


class TestTradeAgent(BaseSelfSupervisedAgent):
    """Test agent implementation."""
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate test signals."""
        return [
            TradeSignal(
                agent_id=self.agent_id,
                symbol="BTC/USD",
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now()
            )
        ]


@pytest_asyncio.fixture(scope="function")
async def market_data():
    """Fixture for market data."""
    return MarketData(
        prices=pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102]
        }),
        volumes=pd.DataFrame({
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    )


class TestSupervisorAgent:
    """Test cases for SupervisorAgent."""
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test supervisor singleton pattern."""
        supervisor1 = SupervisorAgent.instance()
        supervisor2 = SupervisorAgent.instance()
        assert supervisor1 is supervisor2
        
        with pytest.raises(RuntimeError):
            SupervisorAgent()
            
    @pytest.mark.asyncio
    async def test_agent_registration(self, supervisor):
        """Test agent registration and initialization."""
        config = {
            "type": "technical",
            "optimization_threshold": 0.7
        }
        
        agent = await supervisor.register_agent("test_agent", config)
        assert isinstance(agent, ISelfSupervisedAgent)
        
        status = await supervisor.get_agent_status("test_agent")
        assert status.state == AgentState.ACTIVE
        
        await supervisor.unregister_agent("test_agent")
        with pytest.raises(ValueError):
            await supervisor.get_agent_status("test_agent")
            
    @pytest.mark.asyncio
    async def test_task_delegation(self, supervisor):
        """Test task creation and delegation."""
        # Register an agent
        config = {"type": "technical"}
        agent = await supervisor.register_agent("test_agent", config)
        assert isinstance(agent, TestTradeAgent)
        
        # Create and delegate task
        task = await supervisor.delegate_task(
            "test_agent",
            "optimize",
            {"param": "value"},
            priority=1
        )
        
        assert isinstance(task, Task)
        assert task.status == "completed"
        assert task.agent_id == "test_agent"
        
        # Cleanup
        await supervisor.unregister_agent("test_agent")
        
    @pytest.mark.asyncio
    async def test_health_monitoring(self, supervisor):
        """Test health monitoring functionality."""
        # Register multiple agents
        agents = [
            ("tech_agent", {"type": "technical"}),
            ("fund_agent", {"type": "fundamental"}),
            ("sent_agent", {"type": "sentiment"})
        ]
        
        for agent_id, config in agents:
            await supervisor.register_agent(agent_id, config)
            
        # Get system health
        health = await supervisor.get_system_status()
        assert health["total_agents"] == 3
        assert health["active_agents"] > 0
        
        # Check individual agent health
        for agent_id, _ in agents:
            status = await supervisor.get_agent_status(agent_id)
            assert isinstance(status, AgentHealth)
            assert status.state in AgentState
            
        # Cleanup
        for agent_id, _ in agents:
            await supervisor.unregister_agent(agent_id)
            
    @pytest.mark.asyncio
    async def test_agent_optimization(self, supervisor, market_data):
        """Test agent optimization process."""
        # Register agent
        config = {
            "type": "technical",
            "optimization_threshold": 0.5  # Low threshold to trigger optimization
        }
        agent = await supervisor.register_agent("test_agent", config)
        assert isinstance(agent, TestTradeAgent)
        
        # Generate signals to trigger optimization
        signals = await agent.generate_signals(market_data)
        assert len(signals) > 0
        assert isinstance(signals[0], TradeSignal)
        
        # Force optimization
        task = await supervisor.delegate_task(
            "test_agent",
            "optimize",
            {},
            priority=1
        )
        assert task.status == "completed"
        
        # Verify agent state during optimization
        status = await supervisor.get_agent_status("test_agent")
        assert status.state in [AgentState.ACTIVE, AgentState.OPTIMIZING]
        
        # Cleanup
        await supervisor.unregister_agent("test_agent")
        
    @pytest.mark.asyncio
    async def test_error_handling(self, supervisor):
        """Test error handling and recovery."""
        # Register agent
        config = {"type": "technical"}
        agent = await supervisor.register_agent("test_agent", config)
        assert isinstance(agent, TestTradeAgent)
        
        # Simulate errors by sending invalid tasks
        with pytest.raises(ValueError):
            await supervisor.delegate_task(
                "test_agent",
                "invalid_task_type",
                {}
            )
            
        # Check error count in health status
        status = await supervisor.get_agent_status("test_agent")
        assert status.error_count > 0
        assert status.last_error is not None
        
        # Verify agent is still functional
        status = await supervisor.get_agent_status("test_agent")
        assert status.state == AgentState.ACTIVE
        
        # Cleanup
        await supervisor.unregister_agent("test_agent")
        
    @pytest.mark.asyncio
    async def test_integration_with_existing_system(self, supervisor, market_data):
        """Test integration with existing agent system."""
        # Create test agent with config
        config = {
            "type": "technical",
            "optimization_threshold": 0.7
        }
        test_agent = TestTradeAgent("test_agent", config)
        
        # Upgrade to self-supervised
        new_agent = await AgentFactory.upgrade_to_self_supervised(
            test_agent,
            config
        )
        
        # Register with supervisor
        agent = await supervisor.register_agent(new_agent.agent_id, config)
        assert isinstance(agent, TestTradeAgent)
        
        # Verify functionality
        signals = await agent.generate_signals(market_data)
        assert len(signals) > 0
        assert isinstance(signals[0], TradeSignal)
        
        status = await supervisor.get_agent_status(agent.agent_id)
        assert isinstance(status, AgentHealth)
        
        # Cleanup
        await supervisor.unregister_agent(agent.agent_id)
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, supervisor):
        """Test concurrent agent operations."""
        # Register multiple agents
        agent_configs = [
            ("agent1", {"type": "technical"}),
            ("agent2", {"type": "fundamental"}),
            ("agent3", {"type": "sentiment"})
        ]
        
        # Start agents concurrently
        agents = await asyncio.gather(
            *[supervisor.register_agent(id, cfg) for id, cfg in agent_configs]
        )
        assert len(agents) == 3
        assert all(isinstance(agent, TestTradeAgent) for agent in agents)
        
        # Delegate tasks concurrently
        tasks = await asyncio.gather(
            *[
                supervisor.delegate_task(
                    f"agent{i}",
                    "optimize",
                    {},
                    priority=i
                )
                for i in range(1, 4)
            ]
        )
        assert len(tasks) == 3
        assert all(t.status == "completed" for t in tasks)
        
        # Get health status concurrently
        statuses = await asyncio.gather(
            *[supervisor.get_agent_status(f"agent{i}") for i in range(1, 4)]
        )
        assert len(statuses) == 3
        assert all(isinstance(s, AgentHealth) for s in statuses)
        
        # Cleanup
        await asyncio.gather(
            *[supervisor.unregister_agent(f"agent{i}") for i in range(1, 4)]
        )