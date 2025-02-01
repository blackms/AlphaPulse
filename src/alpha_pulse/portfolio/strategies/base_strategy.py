"""
Base implementation of portfolio rebalancing strategy.
Provides common utilities and default implementations for strategy interface.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..interfaces import IRebalancingStrategy


class BaseStrategy(IRebalancingStrategy):
    """Base class for portfolio rebalancing strategies."""

    def __init__(self, config: Dict):
        """
        Initialize base strategy with configuration.

        Args:
            config: Strategy configuration dictionary containing:
                - min_position_size: Minimum position size as fraction
                - max_position_size: Maximum position size as fraction
                - rebalancing_threshold: Minimum deviation to trigger rebalancing
                - stablecoin_fraction: Target stablecoin allocation
                - allowed_assets: List of tradeable assets
        """
        self.config = config
        self.min_position = config.get('min_position_size', 0.05)
        self.max_position = config.get('max_position_size', 0.4)
        self.rebalancing_threshold = config.get('rebalancing_threshold', 0.1)
        self.stablecoin_target = config.get('stablecoin_fraction', 0.3)
        self.allowed_assets = set(config.get('allowed_assets', []))
        self.stablecoins = {'USDT', 'USDC', 'DAI', 'BUSD'}

    def validate_constraints(self, allocation: Dict[str, float]) -> bool:
        """
        Validate if allocation meets all strategy constraints.

        Args:
            allocation: Portfolio allocation to validate

        Returns:
            Boolean indicating if allocation is valid
        """
        if not allocation:
            return False

        # Check sum of weights is approximately 1
        if not np.isclose(sum(allocation.values()), 1.0, rtol=1e-5):
            return False

        # Check position size limits
        for weight in allocation.values():
            if weight < self.min_position or weight > self.max_position:
                return False

        # Check stablecoin allocation
        stablecoin_total = sum(
            weight for asset, weight in allocation.items()
            if asset in self.stablecoins
        )
        if not np.isclose(stablecoin_total, self.stablecoin_target, rtol=0.1):
            return False

        # Check only allowed assets are included
        if not all(asset in self.allowed_assets for asset in allocation):
            return False

        return True

    def needs_rebalancing(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> bool:
        """
        Determine if portfolio needs rebalancing based on deviation threshold.

        Args:
            current: Current portfolio weights
            target: Target portfolio weights

        Returns:
            Boolean indicating if rebalancing is needed
        """
        if not current or not target:
            return True

        # Check absolute deviation from targets
        for asset in set(current) | set(target):
            current_weight = current.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            if abs(current_weight - target_weight) > self.rebalancing_threshold:
                return True

        return False

    def compute_returns(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute asset returns from historical price data.

        Args:
            historical_data: DataFrame with asset prices

        Returns:
            DataFrame of asset returns
        """
        return historical_data.pct_change().dropna()

    def compute_covariance(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 180
    ) -> pd.DataFrame:
        """
        Compute asset covariance matrix with exponential weighting.

        Args:
            returns: Asset returns DataFrame
            lookback_days: Number of days for lookback window

        Returns:
            Covariance matrix as DataFrame
        """
        # Use exponential weighting to give more weight to recent data
        decay_factor = 0.94  # Corresponds to ~30-day half-life
        weights = np.exp(np.arange(lookback_days) * np.log(decay_factor))
        weights = weights / weights.sum()

        # Compute weighted covariance
        returns = returns.iloc[-lookback_days:]
        weighted_returns = returns - returns.mean()
        weighted_returns = weighted_returns * np.sqrt(weights[:, np.newaxis])
        return weighted_returns.T @ weighted_returns

    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: pd.DataFrame,
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Base implementation should be overridden by specific strategies.

        Args:
            current_allocation: Current portfolio weights
            historical_data: Historical price data
            risk_constraints: Dictionary of risk limits

        Returns:
            Target portfolio weights
        """
        raise NotImplementedError(
            "compute_target_allocation must be implemented by specific strategy"
        )