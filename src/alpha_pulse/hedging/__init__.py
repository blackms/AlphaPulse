"""
AlphaPulse Hedging Module

This module provides tools and utilities for managing hedging strategies,
particularly focusing on spot-futures hedging with support for grid trading.

The module follows SOLID principles:
- Single Responsibility: Each component has a focused purpose
- Open/Closed: New strategies can be added without modifying existing code
- Liskov Substitution: Components implement well-defined interfaces
- Interface Segregation: Focused interfaces for specific functionality
- Dependency Inversion: Components depend on abstractions
"""

from .models import (
    SpotPosition,
    FuturesPosition,
    GridBotParams,
    HedgeAdjustment,
    HedgeRecommendation
)
from .hedge_config import HedgeConfig
from .interfaces import (
    IHedgeAnalyzer,
    IPositionFetcher,
    IOrderExecutor,
    IExecutionStrategy
)
from .basic_futures_hedge import BasicFuturesHedgeAnalyzer
from .llm_hedge_analyzer import LLMHedgeAnalyzer
from .position_fetcher import ExchangePositionFetcher
from .execution import (
    BasicExecutionStrategy,
    ExchangeOrderExecutor
)
from .hedge_manager import HedgeManager

__all__ = [
    # Models
    'SpotPosition',
    'FuturesPosition',
    'GridBotParams',
    'HedgeAdjustment',
    'HedgeRecommendation',
    
    # Configuration
    'HedgeConfig',
    
    # Interfaces
    'IHedgeAnalyzer',
    'IPositionFetcher',
    'IOrderExecutor',
    'IExecutionStrategy',
    
    # Implementations
    'BasicFuturesHedgeAnalyzer',
    'LLMHedgeAnalyzer',  # Added LLM-enhanced analyzer
    'ExchangePositionFetcher',
    'BasicExecutionStrategy',
    'ExchangeOrderExecutor',
    'HedgeManager'
]

__version__ = '0.1.0'