"""
Modern Portfolio Theory (MPT) allocation strategy implementation.
"""
from decimal import Decimal
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger

from alpha_pulse.portfolio.allocation_strategy import AllocationStrategy, AllocationResult


class MPTStrategy(AllocationStrategy):
    """Modern Portfolio Theory allocation strategy."""
    
    def calculate_allocation(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, Decimal],
        constraints: Optional[Dict] = None
    ) -> AllocationResult:
        """Calculate optimal portfolio allocation using MPT.
        
        Args:
            returns: DataFrame of historical returns
            current_weights: Current portfolio weights
            constraints: Optional allocation constraints
        
        Returns:
            AllocationResult with target weights and metrics
        """
        try:
            # Validate inputs
            if returns.empty:
                raise ValueError("Returns data is empty")
            
            constraints = self._validate_constraints(constraints, len(returns.columns))
            
            # Calculate expected returns and covariance matrix
            exp_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Initial guess (equal weights)
            n_assets = len(returns.columns)
            init_weights = np.array([1.0/n_assets] * n_assets)
            
            # Define optimization constraints
            bounds = [(constraints['min_weight'], constraints['max_weight'])] * n_assets
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Add target return constraint if specified
            if constraints['target_return'] is not None:
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda x: np.sum(exp_returns * x) - constraints['target_return']
                })
            
            # Add maximum risk constraint if specified
            if constraints['max_risk'] is not None:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints['max_risk'] - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
                })
            
            # Optimize for maximum Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(exp_returns * weights)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
                return -sharpe  # Minimize negative Sharpe ratio
            
            # Run optimization
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
            
            # Calculate portfolio metrics
            optimal_weights = result.x
            portfolio_return = np.sum(exp_returns * optimal_weights)
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            # Convert weights to dictionary
            target_weights = {
                asset: Decimal(str(weight))
                for asset, weight in zip(returns.columns, optimal_weights)
            }
            
            # Calculate rebalance score
            rebalance_score = self.calculate_rebalance_score(current_weights, target_weights)
            
            return AllocationResult(
                weights=target_weights,
                expected_return=Decimal(str(portfolio_return)),
                expected_risk=Decimal(str(portfolio_risk)),
                sharpe_ratio=Decimal(str(sharpe_ratio)),
                rebalance_score=rebalance_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating MPT allocation: {e}")
            raise
    
    def _get_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 100
    ) -> pd.DataFrame:
        """Calculate the efficient frontier.
        
        Args:
            returns: DataFrame of historical returns
            n_points: Number of points to calculate
            
        Returns:
            DataFrame with risk and return for each point
        """
        exp_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Calculate minimum and maximum returns
        min_ret = min(exp_returns)
        max_ret = max(exp_returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        efficient_frontier = []
        for target_ret in target_returns:
            constraints = {'target_return': target_ret}
            result = self.calculate_allocation(returns, {}, constraints)
            efficient_frontier.append({
                'return': float(result.expected_return),
                'risk': float(result.expected_risk),
                'sharpe': float(result.sharpe_ratio)
            })
        
        return pd.DataFrame(efficient_frontier)