"""
Base implementation of portfolio rebalancing strategy.
Provides common utilities and default implementations for strategy interface.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger
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
        
        logger.info(
            f"Initialized base strategy with: min_position={self.min_position}, "
            f"max_position={self.max_position}, stablecoin_target={self.stablecoin_target}"
        )
        logger.debug(f"Allowed assets: {self.allowed_assets}")

    def get_constraint_violations(self, allocation: Dict[str, float]) -> List[str]:
        """
        Get list of constraint violations in the allocation.

        Args:
            allocation: Portfolio allocation to validate

        Returns:
            List of constraint violation descriptions
        """
        violations = []
        
        if not allocation:
            violations.append("Empty allocation")
            return violations

        # Check sum of weights is approximately 1
        total_weight = sum(allocation.values())
        if not np.isclose(total_weight, 1.0, rtol=1e-3):
            violations.append(f"Total weight {total_weight} is not close to 1.0")

        # Check position size limits
        for asset, weight in allocation.items():
            if weight < 0 or weight > self.max_position:
                violations.append(
                    f"Position size violation for {asset}: {weight} "
                    f"(min: 0, max: {self.max_position})"
                )

        # Check stablecoin allocation with tolerance
        stablecoin_total = sum(
            weight for asset, weight in allocation.items()
            if asset in self.stablecoins
        )
        min_stable = self.stablecoin_target * 0.8
        max_stable = self.stablecoin_target * 1.2
        
        if not (min_stable <= stablecoin_total <= max_stable):
            violations.append(
                f"Stablecoin allocation {stablecoin_total} outside target range "
                f"[{min_stable}, {max_stable}]"
            )

        # Check only allowed assets are included
        if self.allowed_assets:
            disallowed = [
                asset for asset in allocation
                if asset not in self.allowed_assets and asset not in self.stablecoins
            ]
            if disallowed:
                violations.append(f"Disallowed assets found: {disallowed}")

        return violations

    def validate_constraints(self, allocation: Dict[str, float]) -> bool:
        """
        Validate if allocation meets all strategy constraints.

        Args:
            allocation: Portfolio allocation to validate

        Returns:
            Boolean indicating if allocation is valid
        """
        logger.debug(f"Validating allocation: {allocation}")
        
        if not allocation:
            logger.warning("Empty allocation")
            return False

        # Check sum of weights is approximately 1
        total_weight = sum(allocation.values())
        if not np.isclose(total_weight, 1.0, rtol=1e-3):
            logger.warning(f"Total weight {total_weight} is not close to 1.0")
            return False

        # Check position size limits
        for asset, weight in allocation.items():
            if weight < 0 or weight > self.max_position:
                logger.warning(
                    f"Position size violation for {asset}: {weight} "
                    f"(min: 0, max: {self.max_position})"
                )
                return False

        # Check stablecoin allocation with tolerance
        stablecoin_total = sum(
            weight for asset, weight in allocation.items()
            if asset in self.stablecoins
        )
        min_stable = self.stablecoin_target * 0.8
        max_stable = self.stablecoin_target * 1.2
        
        if not (min_stable <= stablecoin_total <= max_stable):
            logger.warning(
                f"Stablecoin allocation {stablecoin_total} outside target range "
                f"[{min_stable}, {max_stable}]"
            )
            return False

        # Check only allowed assets are included, if allowed_assets is specified
        if self.allowed_assets and not all(
            asset in self.allowed_assets or asset in self.stablecoins 
            for asset in allocation
        ):
            disallowed = [
                asset for asset in allocation 
                if asset not in self.allowed_assets and asset not in self.stablecoins
            ]
            logger.warning(f"Disallowed assets found: {disallowed}")
            return False

        logger.info("Allocation passed all constraints")
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
        logger.debug(f"Checking rebalancing need - Current: {current}, Target: {target}")
        
        if not current or not target:
            logger.info("Empty allocation, rebalancing needed")
            return True

        # Check absolute deviation from targets
        max_deviation = 0.0
        deviating_assets = []
        
        for asset in set(current) | set(target):
            current_weight = current.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            deviation = abs(current_weight - target_weight)
            max_deviation = max(max_deviation, deviation)
            
            if deviation > self.rebalancing_threshold:
                deviating_assets.append(
                    f"{asset}: {current_weight:.2%} -> {target_weight:.2%} "
                    f"(Δ{deviation:.2%})"
                )

        if deviating_assets:
            logger.info(
                f"Rebalancing needed. Assets exceeding threshold "
                f"({self.rebalancing_threshold:.2%}): {', '.join(deviating_assets)}"
            )
            return True
            
        logger.info(
            f"No rebalancing needed. Maximum deviation {max_deviation:.2%} "
            f"below threshold {self.rebalancing_threshold:.2%}"
        )
        return False

    def compute_returns(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute asset returns from historical price data.

        Args:
            historical_data: DataFrame with asset prices

        Returns:
            DataFrame of asset returns
        """
        logger.debug(f"Computing returns from data shape: {historical_data.shape}")
        returns = historical_data.pct_change().dropna()
        logger.debug(f"Computed returns shape: {returns.shape}")
        return returns

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
        logger.debug(f"Computing covariance with {lookback_days} day lookback")
        
        # Use exponential weighting to give more weight to recent data
        decay_factor = 0.94  # Corresponds to ~30-day half-life
        weights = np.exp(np.arange(lookback_days) * np.log(decay_factor))
        weights = weights / weights.sum()

        # Compute weighted covariance
        returns = returns.iloc[-lookback_days:]
        weighted_returns = returns - returns.mean()
        weighted_returns = weighted_returns * np.sqrt(weights[:, np.newaxis])
        cov = weighted_returns.T @ weighted_returns
        
        logger.debug(f"Computed covariance matrix shape: {cov.shape}")
        return cov

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