"""
Factory for creating self-supervised agents.
"""
from typing import Dict, Any, Type
from loguru import logger

from ..interfaces import BaseTradeAgent
from .base import BaseSelfSupervisedAgent
from .interfaces import ISelfSupervisedAgent


class SelfSupervisedTechnicalAgent(BaseSelfSupervisedAgent):
    """Self-supervised technical analysis agent."""
    
    async def optimize(self) -> None:
        """Optimize technical indicators and parameters."""
        await super().optimize()
        # Add technical-specific optimization logic
        logger.info(f"Optimizing technical parameters for agent {self.agent_id}")


class SelfSupervisedFundamentalAgent(BaseSelfSupervisedAgent):
    """Self-supervised fundamental analysis agent."""
    
    async def optimize(self) -> None:
        """Optimize fundamental analysis parameters."""
        await super().optimize()
        # Add fundamental-specific optimization logic
        logger.info(f"Optimizing fundamental parameters for agent {self.agent_id}")


class SelfSupervisedSentimentAgent(BaseSelfSupervisedAgent):
    """Self-supervised sentiment analysis agent."""
    
    async def optimize(self) -> None:
        """Optimize sentiment analysis parameters."""
        await super().optimize()
        # Add sentiment-specific optimization logic
        logger.info(f"Optimizing sentiment parameters for agent {self.agent_id}")


class AgentFactory:
    """Factory for creating self-supervised agents."""
    
    # Registry of available agent types
    _agent_types: Dict[str, Type[ISelfSupervisedAgent]] = {
        "technical": SelfSupervisedTechnicalAgent,
        "fundamental": SelfSupervisedFundamentalAgent,
        "sentiment": SelfSupervisedSentimentAgent
    }
    
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
            agent_class: Agent class implementation
        """
        if not issubclass(agent_class, ISelfSupervisedAgent):
            raise ValueError(
                f"Agent class must implement ISelfSupervisedAgent interface"
            )
        cls._agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
        
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