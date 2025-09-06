"""Experiments module for empirical validation."""

from .baselines import (
    BaselineComparison,
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    RandomStrategy
)
from .run_validation import EmpiricalValidationExperiment

__all__ = [
    'BaselineComparison',
    'BuyAndHoldStrategy', 
    'MovingAverageCrossoverStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'RandomStrategy',
    'EmpiricalValidationExperiment'
]