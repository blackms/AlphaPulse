"""
Black-Litterman portfolio optimization strategy.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy


class BlackLittermanStrategy(BaseStrategy):
    """Black-Litterman portfolio optimization strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Black-Litterman strategy."""
        super().__init__(config)
        self.risk_aversion = config.get("risk_aversion", 2.5)
        self.tau = config.get("tau", 0.05)  # Uncertainty in prior
        self.market_risk_premium = config.get("market_risk_premium", 0.06)
        self.stablecoin_weight = config.get("stablecoin_weight", 0.3)  # Target stablecoin allocation
        
    def _adjust_allocation(self, weights: np.ndarray, assets: pd.Index) -> np.ndarray:
        """Adjust allocation to maintain stablecoin balance."""
        # Find stablecoin indices
        stablecoin_idx = [i for i, asset in enumerate(assets) if asset.endswith('USDT')]
        if not stablecoin_idx:
            return weights
            
        # Ensure minimum stablecoin allocation
        adjusted = weights.copy()
        current_stable = sum(adjusted[i] for i in stablecoin_idx)
        
        if current_stable < self.stablecoin_weight:
            # Increase stablecoin allocation
            for i in stablecoin_idx:
                adjusted[i] = self.stablecoin_weight / len(stablecoin_idx)
                
            # Reduce other positions proportionally
            other_idx = [i for i in range(len(weights)) if i not in stablecoin_idx]
            if other_idx:
                remaining = 1 - self.stablecoin_weight
                other_sum = sum(adjusted[i] for i in other_idx)
                if other_sum > 0:
                    for i in other_idx:
                        adjusted[i] *= remaining / other_sum
                        
        return adjusted
        
    def compute_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute covariance matrix from returns.
        
        Args:
            returns: Asset returns DataFrame
            
        Returns:
            Covariance matrix
        """
        if returns.empty:
            # Return empty DataFrame with same structure
            return pd.DataFrame(
                0, 
                index=returns.columns,
                columns=returns.columns
            )
            
        return returns.cov()
        
    def compute_equilibrium_returns(
        self,
        returns: pd.DataFrame,
        market_weights: Dict[str, float]
    ) -> pd.Series:
        """
        Compute equilibrium returns using CAPM.
        
        Args:
            returns: Asset returns DataFrame
            market_weights: Market capitalization weights
            
        Returns:
            Series of equilibrium returns
        """
        if returns.empty:
            return pd.Series(0, index=returns.columns)
            
        try:
            # Convert market weights to array with explicit type
            w = pd.Series(market_weights, dtype=float)
            w = w.reindex(returns.columns).fillna(0.0)
            
            # Compute covariance matrix
            sigma = self.compute_covariance(returns)
            
            # Compute equilibrium returns
            pi = self.risk_aversion * sigma.dot(w)
            
            return pi
            
        except Exception as e:
            logger.error(f"Error computing equilibrium returns: {str(e)}")
            return pd.Series(0, index=returns.columns)
        
    def incorporate_views(
        self,
        equilibrium_returns: pd.Series,
        sigma: pd.DataFrame,
        views: Dict[str, Dict[str, float]],
        view_confidences: Dict[str, float]
    ) -> pd.Series:
        """
        Incorporate investor views using Black-Litterman model.
        
        Args:
            equilibrium_returns: Prior equilibrium returns
            sigma: Covariance matrix
            views: Dictionary of views on relative performance
            view_confidences: Confidence in each view
            
        Returns:
            Updated expected returns
        """
        n_assets = len(equilibrium_returns)
        
        if not views:
            return equilibrium_returns
            
        try:
            # Construct view matrix P and view vector q
            P = np.zeros((len(views), n_assets))
            q = np.zeros(len(views))
            omega = np.zeros((len(views), len(views)))
            
            for i, (view_name, view_dict) in enumerate(views.items()):
                confidence = view_confidences.get(view_name, 1.0)
                for asset, weight in view_dict.items():
                    if asset in equilibrium_returns.index:
                        j = equilibrium_returns.index.get_loc(asset)
                        P[i, j] = weight
                q[i] = 0  # Relative views sum to zero
                omega[i, i] = (1 - confidence) * 0.1  # Scale by confidence
                
            # Convert to matrices
            P = pd.DataFrame(P, columns=equilibrium_returns.index)
            q = pd.Series(q)
            omega = pd.DataFrame(omega)
            
            # Compute posterior
            tau = self.tau
            sigma_inv = np.linalg.inv(sigma.values)
            omega_inv = np.linalg.inv(omega.values)
            
            # Black-Litterman formula
            post_cov = np.linalg.inv(tau * sigma_inv + P.T.dot(omega_inv).dot(P))
            post_ret = post_cov.dot(tau * sigma_inv.dot(equilibrium_returns) + 
                                  P.T.dot(omega_inv).dot(q))
                                  
            return pd.Series(post_ret, index=equilibrium_returns.index)
            
        except Exception as e:
            logger.error(f"Error incorporating views: {str(e)}")
            return equilibrium_returns
        
    def optimize_weights(
        self,
        returns: pd.Series,
        sigma: pd.DataFrame,
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            returns: Expected returns
            sigma: Covariance matrix
            risk_constraints: Risk constraints
            
        Returns:
            Optimal portfolio weights
        """
        n_assets = len(returns)
        
        if n_assets == 0:
            return {}
            
        try:
            # Handle zero or near-zero variance
            min_variance = 1e-8
            sigma_diag = np.diag(sigma)
            if np.any(sigma_diag < min_variance):
                logger.warning("Near-zero variance detected, adjusting covariance matrix")
                sigma = sigma + np.eye(n_assets) * min_variance
            
            # Solve mean-variance optimization
            sigma_inv = np.linalg.inv(sigma.values)
            ones = np.ones(n_assets)
            
            # Compute optimal weights
            A = ones.T.dot(sigma_inv).dot(ones)
            B = ones.T.dot(sigma_inv).dot(returns)
            C = returns.T.dot(sigma_inv).dot(returns)
            
            # Global minimum variance portfolio with stablecoin adjustment
            w_min = sigma_inv.dot(ones) / A
            w_min = self._adjust_allocation(w_min, returns.index)
            
            # Tangency portfolio with improved numerical stability
            w_tan = sigma_inv.dot(returns)
            w_sum = w_tan.sum()
            
            # Check if tangency portfolio is valid
            if abs(w_sum) > 1e-8 and not np.any(np.isnan(w_tan)):
                w_tan = w_tan / w_sum
                w_tan = self._adjust_allocation(w_tan, returns.index)
                
                # Verify the weights are reasonable
                if np.all(np.abs(w_tan) < 10):  # Sanity check for extreme weights
                    w = (1 - self.risk_aversion) * w_min + self.risk_aversion * w_tan
                    w = self._adjust_allocation(w, returns.index)  # Final adjustment
                else:
                    logger.info("Using adjusted minimum variance portfolio")
                    w = w_min
            else:
                logger.info("Using adjusted minimum variance portfolio")
                w = w_min
            
            # Convert to dictionary
            weights = dict(zip(returns.index, w))
            
            # Apply position limits
            weights = {
                k: max(min(v, self.max_position), 0)
                for k, v in weights.items()
            }
            
            # Normalize
            total = sum(weights.values())
            if total > 0:
                weights = {
                    k: v / total
                    for k, v in weights.items()
                }
                
            return weights
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {str(e)}")
            return {}
            
    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: Dict[str, Any],
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute target allocation using Black-Litterman model.
        
        Args:
            current_allocation: Current portfolio allocation
            historical_data: Historical market data
            risk_constraints: Risk constraints
            
        Returns:
            Target allocation
        """
        try:
            # Compute returns
            returns = self.compute_returns(historical_data)
            if returns.empty:
                logger.warning("No return data available")
                return current_allocation
                
            # Compute covariance matrix
            sigma = self.compute_covariance(returns) * 252  # Annualized
            
            # Use current allocation as market weights
            market_weights = current_allocation or {
                col: 1.0 / len(returns.columns)
                for col in returns.columns
            }
            
            # Compute equilibrium returns
            eq_returns = self.compute_equilibrium_returns(returns, market_weights)
            
            # Generate views based on technical signals
            views = {}  # Would be populated by signal analysis
            view_confidences = {}
            
            # Incorporate views
            expected_returns = self.incorporate_views(
                eq_returns,
                sigma,
                views,
                view_confidences
            )
            
            # Optimize portfolio
            weights = self.optimize_weights(
                expected_returns,
                sigma,
                risk_constraints
            )
            
            # Validate allocation
            if not self.validate_constraints(weights):
                logger.warning("Target allocation violates constraints")
                return current_allocation
                
            return weights
            
        except Exception as e:
            logger.error(f"Error computing target allocation: {str(e)}")
            return current_allocation