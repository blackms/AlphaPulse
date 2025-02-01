"""
Black-Litterman implementation of portfolio rebalancing strategy.
Combines market equilibrium returns with investor views to generate optimal portfolios.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .base_strategy import BaseStrategy


class BlackLittermanStrategy(BaseStrategy):
    """Black-Litterman based portfolio optimization strategy."""

    def __init__(self, config: Dict):
        """
        Initialize Black-Litterman strategy with configuration.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        self.market_cap_weights = config.get('market_cap_weights', {})
        self.risk_aversion = config.get('risk_aversion', 2.5)
        self.tau = config.get('tau', 0.05)  # Uncertainty in prior
        self.views: List[Dict] = config.get('views', [])
        self.view_confidences: List[float] = config.get('view_confidences', [])

    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: pd.DataFrame,
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio allocation using Black-Litterman model.

        Args:
            current_allocation: Current portfolio weights
            historical_data: Historical price data
            risk_constraints: Dictionary of risk limits

        Returns:
            Target portfolio weights
        """
        returns = self.compute_returns(historical_data)
        assets = returns.columns.tolist()
        
        # Compute covariance matrix
        sigma = self.compute_covariance(returns) * 252  # Annualized
        
        # Get market equilibrium returns
        pi = self._get_market_implied_returns(sigma)
        
        # Prepare views matrix P and views vector Q
        P, Q, omega = self._prepare_views(assets, sigma)
        
        if P is not None and Q is not None:
            # Compute posterior expected returns
            posterior_returns = self._compute_posterior(pi, P, Q, omega, sigma)
        else:
            posterior_returns = pi
            
        # Optimize portfolio using posterior returns
        weights = self._optimize_portfolio(posterior_returns, sigma)
        
        # Convert to dictionary and filter small positions
        allocation = {
            asset: weight for asset, weight in zip(assets, weights)
            if weight >= self.min_position
        }
        
        # Normalize weights
        total = sum(allocation.values())
        allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation

    def _get_market_implied_returns(self, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate market implied returns using reverse optimization.

        Args:
            sigma: Covariance matrix

        Returns:
            Array of implied returns
        """
        # If market cap weights not provided, use equal weights
        if not self.market_cap_weights:
            n = sigma.shape[0]
            weights = np.ones(n) / n
        else:
            weights = np.array(list(self.market_cap_weights.values()))
            
        # Implied returns = λΣw where λ is risk aversion
        return self.risk_aversion * sigma @ weights

    def _prepare_views(
        self,
        assets: List[str],
        sigma: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare views matrix P, views vector Q, and uncertainty matrix omega.

        Args:
            assets: List of asset names
            sigma: Covariance matrix

        Returns:
            Tuple of (P, Q, omega) matrices
        """
        if not self.views:
            return None, None, None
            
        n_assets = len(assets)
        n_views = len(self.views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, view in enumerate(self.views):
            for asset, weight in view['weights'].items():
                asset_idx = assets.index(asset)
                P[i, asset_idx] = weight
            Q[i] = view['return']
            
        # Construct diagonal uncertainty matrix
        if self.view_confidences:
            confidence_scales = np.array(self.view_confidences)
        else:
            confidence_scales = np.ones(n_views)
            
        omega = np.diag(confidence_scales * np.diag(P @ sigma @ P.T))
        
        return P, Q, omega

    def _compute_posterior(
        self,
        pi: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        Compute posterior expected returns using Black-Litterman formula.

        Args:
            pi: Prior (market implied) returns
            P: Views matrix
            Q: Views vector
            omega: Views uncertainty matrix
            sigma: Market covariance matrix

        Returns:
            Posterior expected returns
        """
        # Compute posterior parameters
        tau_sigma = self.tau * sigma
        inv_omega = np.linalg.inv(omega)
        inv_sigma = np.linalg.inv(tau_sigma)
        
        # Black-Litterman formula
        A = np.linalg.inv(inv_sigma + P.T @ inv_omega @ P)
        B = inv_sigma @ pi + P.T @ inv_omega @ Q
        
        return A @ B

    def _optimize_portfolio(
        self,
        returns: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance optimization.

        Args:
            returns: Expected returns
            sigma: Covariance matrix

        Returns:
            Array of optimal weights
        """
        n = len(returns)
        
        # Define objective function (maximize utility)
        def objective(w):
            portfolio_return = np.dot(w, returns)
            portfolio_var = np.dot(w.T, np.dot(sigma, w))
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_var
            return -utility  # Minimize negative utility
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Position size bounds
        bounds = [(self.min_position, self.max_position) for _ in range(n)]
        
        # Initial guess: equal weights
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            raise ValueError(f"Portfolio optimization failed: {result.message}")
            
        return result.x