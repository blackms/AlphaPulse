"""
Factory for creating self-supervised agents.
"""
from typing import Dict, Any, Type
from loguru import logger

from ..interfaces import BaseTradeAgent
from .base import BaseSelfSupervisedAgent
from .interfaces import ISelfSupervisedAgent
from .technical_agent import SelfSupervisedTechnicalAgent
from .fundamental_agent import SelfSupervisedFundamentalAgent
from .sentiment_agent import SelfSupervisedSentimentAgent


class AgentFactory:
    """Factory for creating self-supervised agents."""
    
    # Registry of available agent types
    _agent_types: Dict[str, Type[ISelfSupervisedAgent]] = {
        "technical": SelfSupervisedTechnicalAgent,
        "fundamental": SelfSupervisedFundamentalAgent,
        "sentiment": SelfSupervisedSentimentAgent
    }

    _test_mode = False
    _test_agent_class = None
    
    @classmethod
    def register_agent_type(
        cls,
        agent_type: str,
        agent_class: Type[ISelfSupervisedAgent]
    ) -> None:
        """
        Register a new agent type.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class implementation that extends ISelfSupervisedAgent
            
        Raises:
            ValueError: If agent_class doesn't implement ISelfSupervisedAgent
        """
        if not issubclass(agent_class, ISelfSupervisedAgent):
            raise ValueError(
                f"Agent class must implement ISelfSupervisedAgent interface"
            )
        cls._agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")

    @classmethod
    def enable_test_mode(cls, test_agent_class: Type[ISelfSupervisedAgent]) -> None:
        """
        Enable test mode to use test agent class.
        
        Args:
            test_agent_class: Test agent class to use during testing
        """
        cls._test_mode = True
        cls._test_agent_class = test_agent_class
        logger.info("Enabled test mode with test agent class")

    @classmethod
    def disable_test_mode(cls) -> None:
        """Disable test mode."""
        cls._test_mode = False
        cls._test_agent_class = None
        logger.info("Disabled test mode")
        
    @classmethod
    async def create_self_supervised_agent(
        cls,
        agent_id: str,
        config: Dict[str, Any]
    ) -> ISelfSupervisedAgent:
        """
        Create a new self-supervised agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Agent configuration parameters
            
        Returns:
            Initialized agent instance
            
        Raises:
            ValueError: If agent type is not supported
        """
        if cls._test_mode and cls._test_agent_class:
            agent = cls._test_agent_class(agent_id, config)
            logger.info(f"Created test agent with ID: {agent_id}")
            return agent

        agent_type = config.get("type", "technical")  # Default to technical
        
        if agent_type not in cls._agent_types:
            raise ValueError(
                f"Unsupported agent type: {agent_type}. "
                f"Available types: {list(cls._agent_types.keys())}"
            )
            
        agent_class = cls._agent_types[agent_type]
        agent = agent_class(agent_id, config)
        
        logger.info(f"Created {agent_type} agent with ID: {agent_id}")
        return agent
        
    @classmethod
    def get_available_agent_types(cls) -> Dict[str, Type[ISelfSupervisedAgent]]:
        """
        Get dictionary of available agent types.
        
        Returns:
            Dictionary mapping agent type names to their classes
        """
        return cls._agent_types.copy()
        
    @classmethod
    async def upgrade_to_self_supervised(
        cls,
        agent: BaseTradeAgent,
        config: Dict[str, Any]
    ) -> ISelfSupervisedAgent:
        """
        Upgrade an existing trade agent to a self-supervised agent.
        
        Args:
            agent: Existing trade agent instance
            config: Additional configuration for self-supervision
            
        Returns:
            New self-supervised agent instance
            
        Note:
            This method creates a new agent instance with the same ID and
            base configuration as the input agent, but adds self-supervision
            capabilities.
        """
        # Create new config combining existing and new settings
        new_config = {
            "type": config.get("type", "technical"),
            **agent.config,
            **config
        }
        
        # Create new self-supervised agent
        new_agent = await cls.create_self_supervised_agent(
            agent.agent_id,
            new_config
        )
        
        # Copy relevant state if needed
        if hasattr(agent, "metrics"):
            new_agent.metrics = agent.metrics
            
        logger.info(
            f"Upgraded agent {agent.agent_id} to "
            f"self-supervised {new_config['type']} agent"
        )
        return new_agent


# Register any additional agent types here
def register_additional_agent_types():
    """Register custom agent types."""
    # Example:
    # AgentFactory.register_agent_type("custom", CustomSelfSupervisedAgent)
    pass