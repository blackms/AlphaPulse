"""
Portfolio allocation strategy interfaces and base classes.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class AllocationResult:
    """Result of portfolio allocation calculation."""
    weights: Dict[str, Decimal]  # Target weights for each asset
    expected_return: Decimal  # Expected portfolio return
    expected_risk: Decimal  # Expected portfolio risk (volatility)
    sharpe_ratio: Decimal  # Sharpe ratio of the portfolio
    rebalance_score: Decimal  # Score indicating how far current allocation is from target (0-1)


class AllocationStrategy(ABC):
    """Base class for portfolio allocation strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize allocation strategy.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    @abstractmethod
    def calculate_allocation(
        self,
        returns: pd.DataFrame,  # Historical returns (assets in columns)
        current_weights: Dict[str, Decimal],  # Current portfolio weights
        constraints: Optional[Dict] = None  # Optional constraints
    ) -> AllocationResult:
        """Calculate optimal portfolio allocation.
        
        Args:
            returns: DataFrame of historical returns
            current_weights: Current portfolio weights
            constraints: Optional allocation constraints
                - min_weight: Minimum weight per asset
                - max_weight: Maximum weight per asset
                - target_return: Target portfolio return
                - max_risk: Maximum portfolio risk
        
        Returns:
            AllocationResult with target weights and metrics
        """
        pass
    
    def calculate_rebalance_score(
        self,
        current_weights: Dict[str, Decimal],
        target_weights: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate how far current allocation is from target.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Score between 0 (needs rebalancing) and 1 (perfectly balanced)
        """
        # Ensure all assets are in both dictionaries
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        current = np.array([float(current_weights.get(asset, 0)) for asset in all_assets])
        target = np.array([float(target_weights.get(asset, 0)) for asset in all_assets])
        
        # Calculate absolute differences and sum
        diff_sum = np.sum(np.abs(current - target))
        
        # Convert to score between 0 and 1
        # 0 means maximum difference (2.0 for completely different allocations)
        # 1 means perfect match
        score = 1 - (diff_sum / 2.0)
        return Decimal(str(max(0, min(1, score))))
    
    def _validate_constraints(
        self,
        constraints: Dict,
        n_assets: int
    ) -> Dict:
        """Validate and set default constraints.
        
        Args:
            constraints: User-provided constraints
            n_assets: Number of assets
            
        Returns:
            Validated constraints with defaults
        """
        if constraints is None:
            constraints = {}
        
        defaults = {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'target_return': None,
            'max_risk': None
        }
        
        # Update with provided constraints
        for key, default in defaults.items():
            if key not in constraints:
                constraints[key] = default
        
        # Validate constraints
        if constraints['min_weight'] < 0:
            raise ValueError("Minimum weight cannot be negative")
        if constraints['max_weight'] > 1:
            raise ValueError("Maximum weight cannot exceed 1")
        if constraints['min_weight'] > constraints['max_weight']:
            raise ValueError("Minimum weight cannot exceed maximum weight")
        if constraints['target_return'] is not None and constraints['target_return'] < 0:
            raise ValueError("Target return cannot be negative")
        if constraints['max_risk'] is not None and constraints['max_risk'] < 0:
            raise ValueError("Maximum risk cannot be negative")
        
        return constraints