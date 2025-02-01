"""
Modern Portfolio Theory (MPT) implementation of portfolio rebalancing strategy.
Uses mean-variance optimization to find efficient portfolio allocations.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .base_strategy import BaseStrategy


class MPTStrategy(BaseStrategy):
    """Modern Portfolio Theory based portfolio optimization strategy."""

    def __init__(self, config: Dict):
        """
        Initialize MPT strategy with configuration.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # Annual risk-free rate
        self.target_return = config.get('target_return', None)
        self.optimization_objective = config.get('optimization_objective', 'sharpe')

    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: pd.DataFrame,
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio allocation using MPT.

        Args:
            current_allocation: Current portfolio weights
            historical_data: Historical price data
            risk_constraints: Dictionary of risk limits

        Returns:
            Target portfolio weights
        """
        returns = self.compute_returns(historical_data)
        mu = returns.mean() * 252  # Annualized returns
        cov = self.compute_covariance(returns) * 252  # Annualized covariance

        # Initial guess: equal weights or current allocation
        n_assets = len(returns.columns)
        if current_allocation:
            x0 = [current_allocation.get(asset, 1.0/n_assets) for asset in returns.columns]
        else:
            x0 = [1.0/n_assets] * n_assets

        # Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]

        # Add minimum return constraint if specified
        if self.target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.dot(mu, x) - self.target_return
            })

        # Position size bounds
        bounds = [(self.min_position, self.max_position) for _ in range(n_assets)]

        # Optimize based on selected objective
        if self.optimization_objective == 'sharpe':
            result = minimize(
                lambda x: -self._sharpe_ratio(x, mu, cov),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:  # Minimize volatility
            result = minimize(
                lambda x: self._portfolio_volatility(x, cov),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

        if not result.success:
            raise ValueError(f"Portfolio optimization failed: {result.message}")

        # Convert optimal weights to dictionary
        optimal_weights = {
            asset: weight for asset, weight in zip(returns.columns, result.x)
            if weight > self.min_position  # Filter out tiny positions
        }

        # Normalize weights to ensure they sum to 1
        total_weight = sum(optimal_weights.values())
        optimal_weights = {
            k: v/total_weight for k, v in optimal_weights.items()
        }

        return optimal_weights

    def _sharpe_ratio(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Calculate portfolio Sharpe ratio.

        Args:
            weights: Array of portfolio weights
            returns: Array of expected returns
            covariance: Covariance matrix

        Returns:
            Sharpe ratio
        """
        portfolio_return = np.dot(returns, weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        return (portfolio_return - self.risk_free_rate) / portfolio_vol

    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility.

        Args:
            weights: Array of portfolio weights
            covariance: Covariance matrix

        Returns:
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

    def _portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate key portfolio metrics.

        Args:
            weights: Array of portfolio weights
            returns: Array of expected returns
            covariance: Covariance matrix

        Returns:
            Dictionary of portfolio metrics
        """
        portfolio_return = np.dot(returns, weights)
        portfolio_vol = self._portfolio_volatility(weights, covariance)
        sharpe = self._sharpe_ratio(weights, returns, covariance)

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }