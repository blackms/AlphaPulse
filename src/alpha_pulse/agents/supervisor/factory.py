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
    
    @staticmethod
    async def create_agent(
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