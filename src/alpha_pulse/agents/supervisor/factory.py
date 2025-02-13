"""
Factory for creating self-supervised agents.
"""
from typing import Optional, Dict, Any
from loguru import logger

from .technical_agent import SelfSupervisedTechnicalAgent
from .sentiment_agent import SelfSupervisedSentimentAgent
from .base import BaseSelfSupervisedAgent


class AgentFactory:
    """Factory class for creating self-supervised agents."""
    
    _test_mode = False
    _test_agent_class = None
    
    @classmethod
    def enable_test_mode(cls, test_agent_class):
        """Enable test mode with a specific test agent class."""
        cls._test_mode = True
        cls._test_agent_class = test_agent_class
        
    @classmethod
    def disable_test_mode(cls):
        """Disable test mode."""
        cls._test_mode = False
        cls._test_agent_class = None
    
    @classmethod
    async def create_agent(
        cls,
        agent_type: str,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseSelfSupervisedAgent:
        """
        Create a self-supervised agent of the specified type.
        
        Args:
            agent_type: Type of agent to create ('technical', 'sentiment', etc.)
            agent_id: Unique identifier for the agent
            config: Optional configuration parameters
            
        Returns:
            Initialized self-supervised agent
            
        Raises:
            ValueError: If agent_type is not recognized
        """
        try:
            # Use test agent if in test mode
            if cls._test_mode and cls._test_agent_class:
                agent = cls._test_agent_class(agent_id, config)
                logger.info(f"Created test agent with ID: {agent_id}")
                return agent
            
            # Normal production mode
            if agent_type == "technical":
                agent = SelfSupervisedTechnicalAgent(agent_id, config)
                logger.info(f"Created technical agent with ID: {agent_id}")
                return agent
                
            elif agent_type == "sentiment":
                agent = SelfSupervisedSentimentAgent(agent_id, config)
                logger.info(f"Created sentiment agent with ID: {agent_id}")
                return agent
                
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
                
        except Exception as e:
            logger.error(f"Error creating agent {agent_id} of type {agent_type}: {str(e)}")
            raise


# For backward compatibility
create_self_supervised_agent = AgentFactory.create_agent