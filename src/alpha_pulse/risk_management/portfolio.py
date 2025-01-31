"""
Portfolio optimization implementations for AlphaPulse.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass

from loguru import logger

from .interfaces import IPortfolioOptimizer


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    max_total_weight: float = 1.0  # Maximum sum of weights (for leverage control)
    risk_free_rate: float = 0.0  # Risk-free rate for optimization


class MeanVarianceOptimizer(IPortfolioOptimizer):
    """Mean-Variance portfolio optimization using Modern Portfolio Theory."""

    def __init__(
        self,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        risk_aversion: float = 1.0,
    ):
        """
        Initialize Mean-Variance optimizer.

        Args:
            target_return: Target portfolio return (for minimum risk optimization)
            target_risk: Target portfolio risk (for maximum Sharpe optimization)
            risk_aversion: Risk aversion parameter for utility optimization
        """
        self.target_return = target_return
        self.target_risk = target_risk
        self.risk_aversion = risk_aversion
        logger.info(
            f"Initialized MeanVarianceOptimizer (target_return={target_return}, "
            f"target_risk={target_risk}, risk_aversion={risk_aversion})"
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Calculate optimal portfolio weights using mean-variance optimization."""
        if constraints is None:
            constraints = PortfolioConstraints()

        # Calculate expected returns and covariance
        exp_returns = returns.mean()
        cov_matrix = returns.cov()

        # Define optimization objective based on parameters
        if self.target_return is not None:
            # Minimum risk portfolio with target return
            objective = self._minimum_risk_objective
            additional_constraints = [
                {
                    'type': 'eq',
                    'fun': lambda w: np.sum(w * exp_returns) - self.target_return
                }
            ]
        elif self.target_risk is not None:
            # Maximum Sharpe ratio portfolio
            objective = self._maximum_sharpe_objective
            additional_constraints = []
        else:
            # Maximum utility portfolio
            objective = self._maximum_utility_objective
            additional_constraints = []

        # Set up constraints
        base_constraints = [
            # Weights sum constraint
            {
                'type': 'eq',
                'fun': lambda w: np.sum(w) - constraints.max_total_weight
            }
        ]

        # Optimize portfolio weights
        n_assets = len(returns.columns)
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )

        result = minimize(
            objective,
            x0=np.array([1.0/n_assets] * n_assets),  # Equal weight start
            args=(exp_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=base_constraints + additional_constraints
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        # Return optimized weights as dictionary
        return dict(zip(returns.columns, result.x))

    def _minimum_risk_objective(
        self,
        weights: np.ndarray,
        exp_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
    ) -> float:
        """Objective function for minimum risk portfolio."""
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return portfolio_risk

    def _maximum_sharpe_objective(
        self,
        weights: np.ndarray,
        exp_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
    ) -> float:
        """Objective function for maximum Sharpe ratio portfolio."""
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if portfolio_risk == 0:
            return -np.inf
            
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        return -sharpe_ratio  # Minimize negative Sharpe ratio

    def _maximum_utility_objective(
        self,
        weights: np.ndarray,
        exp_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
    ) -> float:
        """Objective function for maximum utility portfolio."""
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_risk**2
        return -utility  # Minimize negative utility


class RiskParityOptimizer(IPortfolioOptimizer):
    """Risk Parity (Equal Risk Contribution) portfolio optimization."""

    def __init__(self, target_risk: Optional[float] = None):
        """
        Initialize Risk Parity optimizer.

        Args:
            target_risk: Target portfolio risk level
        """
        self.target_risk = target_risk
        logger.info(f"Initialized RiskParityOptimizer (target_risk={target_risk})")

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Calculate optimal weights using risk parity approach."""
        if constraints is None:
            constraints = PortfolioConstraints()

        cov_matrix = returns.cov()
        n_assets = len(returns.columns)

        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            asset_rc = weights * (cov_matrix @ weights) / portfolio_risk
            asset_rc_mean = np.mean(asset_rc)
            return np.sum((asset_rc - asset_rc_mean) ** 2)

        # Optimize to achieve equal risk contribution
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        constraints_list = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - constraints.max_total_weight
        }]

        result = minimize(
            risk_parity_objective,
            x0=np.array([1.0/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )

        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")

        # Scale weights to match target risk if specified
        weights = result.x
        if self.target_risk is not None:
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            scaling_factor = self.target_risk / portfolio_risk
            weights = weights * scaling_factor

        return dict(zip(returns.columns, weights))


class AdaptivePortfolioOptimizer(IPortfolioOptimizer):
    """
    Adaptive portfolio optimizer that combines multiple strategies based on market conditions.
    """

    def __init__(
        self,
        mv_optimizer: Optional[MeanVarianceOptimizer] = None,
        rp_optimizer: Optional[RiskParityOptimizer] = None,
        volatility_threshold: float = 0.2,  # Annualized volatility threshold
    ):
        """
        Initialize adaptive portfolio optimizer.

        Args:
            mv_optimizer: Mean-variance optimizer instance
            rp_optimizer: Risk parity optimizer instance
            volatility_threshold: Threshold for switching strategies
        """
        self.mv_optimizer = mv_optimizer or MeanVarianceOptimizer()
        self.rp_optimizer = rp_optimizer or RiskParityOptimizer()
        self.volatility_threshold = volatility_threshold
        logger.info(
            f"Initialized AdaptivePortfolioOptimizer "
            f"(vol_threshold={volatility_threshold})"
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Calculate optimal weights using adaptive strategy selection.
        
        Uses risk parity in high volatility regimes and mean-variance in low
        volatility regimes.
        """
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(
            (returns.std() ** 2).mean()
        ) * np.sqrt(252)  # Annualized

        # Select optimization strategy based on volatility
        if portfolio_vol > self.volatility_threshold:
            logger.info(
                f"High volatility regime detected ({portfolio_vol:.2%}), "
                "using risk parity optimization"
            )
            weights = self.rp_optimizer.optimize(
                returns,
                risk_free_rate,
                constraints
            )
        else:
            logger.info(
                f"Low volatility regime detected ({portfolio_vol:.2%}), "
                "using mean-variance optimization"
            )
            weights = self.mv_optimizer.optimize(
                returns,
                risk_free_rate,
                constraints
            )

        return weights