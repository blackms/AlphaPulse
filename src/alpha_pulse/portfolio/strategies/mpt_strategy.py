"""
Modern Portfolio Theory strategy implementation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.optimize import minimize

from .base_strategy import BaseStrategy


class MPTStrategy(BaseStrategy):
    """Modern Portfolio Theory portfolio optimization strategy."""
    
    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute returns from price data."""
        # Handle stablecoins
        for col in prices.columns:
            if col.endswith(('USDT', 'USDC', 'DAI', 'BUSD')):
                prices[col] = 1.0
        return prices.pct_change().fillna(0).dropna()
    
    def compute_covariance(self, returns: pd.DataFrame, correlation_analyzer=None) -> pd.DataFrame:
        """
        Compute covariance matrix from returns.
        
        If correlation_analyzer is provided, use advanced correlation methods.
        """
        if correlation_analyzer:
            # Use correlation analyzer for more sophisticated correlation calculation
            corr_result = correlation_analyzer.calculate_correlation_matrix(
                returns, 
                method=correlation_analyzer.config.correlation_methods[0]
            )
            # Convert correlation matrix to covariance
            std_devs = returns.std()
            correlation_matrix = pd.DataFrame(
                corr_result.matrix,
                index=returns.columns,
                columns=returns.columns
            )
            # Covariance = Correlation * StdDev_i * StdDev_j
            covariance = correlation_matrix * np.outer(std_devs, std_devs)
            return covariance
        else:
            return returns.cov()
    
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
        
        # Use correlation analyzer if provided
        correlation_analyzer = risk_constraints.get('correlation_analyzer')
        cov = self.compute_covariance(returns, correlation_analyzer) * 252  # Annualized covariance
        
        # Optimize portfolio using mean-variance optimization
        n_assets = len(mu)
        target_vol = risk_constraints.get('volatility_target', 0.15)
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        
        def portfolio_ret(weights):
            return np.sum(mu * weights)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda x: x - self.min_position},  # Min position
            {'type': 'ineq', 'fun': lambda x: self.max_position - x}  # Max position
        ]
        
        # Initial guess: equal weights
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_vol,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(self.min_position, self.max_position)] * n_assets
        )
        
        if not result.success:
            # Fallback to current allocation
            return current_allocation
            
        # Convert result to dictionary
        return {
            asset: float(weight)
            for asset, weight in zip(historical_data.columns, result.x)
        }