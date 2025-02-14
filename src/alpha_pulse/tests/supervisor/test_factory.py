import pytest
from unittest.mock import Mock, patch

from alpha_pulse.agents.supervisor.factory import AgentFactory
from alpha_pulse.agents.supervisor.technical_agent import SelfSupervisedTechnicalAgent
from alpha_pulse.agents.supervisor.sentiment_agent import SelfSupervisedSentimentAgent
from alpha_pulse.agents.supervisor.base import BaseSelfSupervisedAgent

class MockTestAgent(BaseSelfSupervisedAgent):
    """Mock agent class for testing."""
    pass

@pytest.mark.asyncio
async def test_create_technical_agent():
    """Test creation of technical agent."""
    agent = await AgentFactory.create_agent(
        "technical",
        "tech_agent_1",
        {"optimization_threshold": 0.8}
    )
    
    assert isinstance(agent, SelfSupervisedTechnicalAgent)
    assert agent.agent_id == "tech_agent_1"
    assert agent._optimization_threshold == 0.8

@pytest.mark.asyncio
async def test_create_sentiment_agent():
    """Test creation of sentiment agent."""
    agent = await AgentFactory.create_agent(
        "sentiment",
        "sent_agent_1",
        {"news_weight": 0.4}
    )
    
    assert isinstance(agent, SelfSupervisedSentimentAgent)
    assert agent.agent_id == "sent_agent_1"
    assert agent.sentiment_sources["news"] == 0.4

@pytest.mark.asyncio
async def test_unknown_agent_type():
    """Test error handling for unknown agent type."""
    with pytest.raises(ValueError) as exc_info:
        await AgentFactory.create_agent(
            "unknown_type",
            "test_agent",
            {}
        )
    assert "Unknown agent type" in str(exc_info.value)

@pytest.mark.asyncio
async def test_test_mode():
    """Test factory test mode with mock agent."""
    AgentFactory.enable_test_mode(MockTestAgent)
    
    try:
        agent = await AgentFactory.create_agent(
            "technical",  # Type doesn't matter in test mode
            "test_agent",
            {"test_param": True}
        )
        
        assert isinstance(agent, MockTestAgent)
        assert agent.agent_id == "test_agent"
        assert agent.config["test_param"] is True
        
    finally:
        AgentFactory.disable_test_mode()

@pytest.mark.asyncio
async def test_upgrade_to_self_supervised():
    """Test upgrading existing agent to self-supervised."""
    existing_agent = MockTestAgent("existing_agent", {})
    
    upgraded_agent = await AgentFactory.upgrade_to_self_supervised(
        existing_agent,
        {"type": "technical", "optimization_threshold": 0.9}
    )
    
    assert isinstance(upgraded_agent, SelfSupervisedTechnicalAgent)
    assert upgraded_agent.agent_id == "existing_agent"
    assert upgraded_agent._optimization_threshold == 0.9

@pytest.mark.asyncio
async def test_upgrade_in_test_mode():
    """Test upgrade behavior in test mode."""
    AgentFactory.enable_test_mode(MockTestAgent)
    
    try:
        existing_agent = MockTestAgent("test_agent", {})
        upgraded_agent = await AgentFactory.upgrade_to_self_supervised(
            existing_agent,
            {"type": "technical"}
        )
        
        # In test mode, should return original agent
        assert upgraded_agent is existing_agent
        
    finally:
        AgentFactory.disable_test_mode()

@pytest.mark.asyncio
async def test_create_agent_with_minimal_config():
    """Test agent creation with minimal configuration."""
    agent = await AgentFactory.create_agent(
        "technical",
        "minimal_agent"
    )
    
    assert isinstance(agent, SelfSupervisedTechnicalAgent)
    assert agent.agent_id == "minimal_agent"
    assert agent._optimization_threshold == 0.7  # Default value

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during agent creation."""
    with patch('alpha_pulse.agents.supervisor.factory.SelfSupervisedTechnicalAgent', 
              side_effect=Exception("Initialization error")):
        
        with pytest.raises(Exception) as exc_info:
            await AgentFactory.create_agent(
                "technical",
                "error_agent"
            )
        assert "Initialization error" in str(exc_info.value)