"""
Agent factory for creating and managing trading agents.
"""
from typing import Dict, Any, Type, List

from .interfaces import BaseTradeAgent
from .activist_agent import ActivistAgent
from .value_agent import ValueAgent
from .fundamental_agent import FundamentalAgent
from .sentiment_agent import SentimentAgent
from .technical_agent import TechnicalAgent
from .valuation_agent import ValuationAgent


class AgentFactory:
    """Factory class for creating and managing trading agents."""
    
    AGENT_TYPES = {
        'activist': ActivistAgent,
        'value': ValueAgent,
        'fundamental': FundamentalAgent,
        'sentiment': SentimentAgent,
        'technical': TechnicalAgent,
        'valuation': ValuationAgent
    }
    
    @classmethod
    async def create_agent(
        cls,
        agent_type: str,
        config: Dict[str, Any] = None
    ) -> BaseTradeAgent:
        """
        Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            config: Agent configuration parameters
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent type is unknown
        """
        agent_class = cls.AGENT_TYPES.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent = agent_class(config)
        await agent.initialize(config or {})
        return agent
        
    @classmethod
    async def create_all_agents(
        cls,
        config: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, BaseTradeAgent]:
        """
        Create instances of all available agents.
        
        Args:
            config: Configuration for each agent type
            
        Returns:
            Dictionary of agent instances
        """
        agents = {}
        config = config or {}
        
        for agent_type in cls.AGENT_TYPES:
            agent_config = config.get(agent_type, {})
            agents[agent_type] = await cls.create_agent(agent_type, agent_config)
            
        return agents
        
    @classmethod
    def get_available_agents(cls) -> List[str]:
        """
        Get list of available agent types.
        
        Returns:
            List of agent type names
        """
        return list(cls.AGENT_TYPES.keys())
        
    @classmethod
    def get_agent_class(cls, agent_type: str) -> Type[BaseTradeAgent]:
        """
        Get agent class by type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Agent class
            
        Raises:
            ValueError: If agent type is unknown
        """
        agent_class = cls.AGENT_TYPES.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class
        
    @classmethod
    def get_default_config(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get default configuration for agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Default configuration dictionary
            
        Raises:
            ValueError: If agent type is unknown
        """
        agent_class = cls.get_agent_class(agent_type)
        
        if agent_type == 'activist':
            return {
                "min_market_cap": 1e9,
                "max_market_cap": 50e9,
                "min_ownership_target": 0.05,
                "holding_period": 360,
                "target_sectors": [
                    "Technology",
                    "Consumer",
                    "Industrial",
                    "Healthcare"
                ]
            }
        elif agent_type == 'value':
            return {
                "min_roe": 0.15,
                "min_roic": 0.12,
                "max_debt_to_equity": 0.5,
                "min_interest_coverage": 5,
                "min_operating_margin": 0.15,
                "holding_period": 720
            }
        elif agent_type == 'fundamental':
            return {
                "min_revenue_growth": 0.1,
                "min_gross_margin": 0.3,
                "min_ebitda_margin": 0.15,
                "min_net_margin": 0.1,
                "min_current_ratio": 1.5,
                "analysis_timeframes": {
                    "short_term": 90,
                    "medium_term": 180,
                    "long_term": 360
                }
            }
        elif agent_type == 'sentiment':
            return {
                "news_weight": 0.3,
                "social_media_weight": 0.25,
                "market_data_weight": 0.25,
                "analyst_weight": 0.2,
                "momentum_window": 14,
                "volume_window": 5
            }
        elif agent_type == 'technical':
            return {
                "trend_weight": 0.3,
                "momentum_weight": 0.2,
                "volatility_weight": 0.2,
                "volume_weight": 0.15,
                "pattern_weight": 0.15,
                "timeframes": {
                    "short": 14,
                    "medium": 50,
                    "long": 200
                }
            }
        elif agent_type == 'valuation':
            return {
                "dcf_weight": 0.3,
                "multiples_weight": 0.3,
                "asset_weight": 0.2,
                "dividend_weight": 0.2,
                "discount_rate": 0.1,
                "growth_rates": {
                    "high_growth": 0.15,
                    "moderate_growth": 0.08,
                    "stable_growth": 0.03
                }
            }
        else:
            return {}