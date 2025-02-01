"""
Hierarchical Risk Parity (HRP) allocation strategy implementation.

Based on "Building Diversified Portfolios that Outperform Out of Sample" by
Marcos Lopez de Prado (2016).
"""
from decimal import Decimal
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from loguru import logger

from alpha_pulse.portfolio.allocation_strategy import AllocationStrategy, AllocationResult


class HRPStrategy(AllocationStrategy):
    """Hierarchical Risk Parity allocation strategy."""
    
    def calculate_allocation(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, Decimal],
        constraints: Optional[Dict] = None
    ) -> AllocationResult:
        """Calculate optimal portfolio allocation using HRP.
        
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
            
            # Calculate correlation and covariance matrices
            corr = returns.corr()
            cov = returns.cov()
            
            # Calculate distance matrix
            dist = self._get_distance_matrix(corr)
            
            # Perform hierarchical clustering
            link = linkage(squareform(dist), 'single')
            
            # Get quasi-diagonal matrix
            sorted_idx = self._get_quasi_diag(link)
            ret = returns.iloc[:, sorted_idx]
            cov_sorted = cov.iloc[sorted_idx, sorted_idx]
            
            # Calculate HRP weights
            weights = self._get_recursive_bisection(cov_sorted)
            
            # Reorder weights to match original asset order
            inv_idx = np.argsort(sorted_idx)
            weights = weights[inv_idx]
            
            # Apply constraints
            weights = self._apply_weight_constraints(weights, constraints)
            
            # Calculate portfolio metrics
            target_weights = {
                asset: Decimal(str(weight))
                for asset, weight in zip(returns.columns, weights)
            }
            
            portfolio_return = float(np.sum(returns.mean() * weights))
            portfolio_risk = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
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
            logger.error(f"Error calculating HRP allocation: {e}")
            raise
    
    def _get_distance_matrix(self, corr: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance matrix from correlation matrix.
        
        Args:
            corr: Correlation matrix
            
        Returns:
            Distance matrix
        """
        # Convert correlations to distances
        dist = np.sqrt((1 - corr) / 2)
        return dist
    
    def _get_quasi_diag(self, link: np.ndarray) -> np.ndarray:
        """Get quasi-diagonal matrix from linkage matrix.
        
        Args:
            link: Linkage matrix from hierarchical clustering
            
        Returns:
            Array of indices for quasi-diagonal matrix
        """
        # Sort clustered items by distance
        link = link.astype(int)
        sorted_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        for i in range(num_items - 2):
            sorted_idx.index = range(0, len(sorted_idx))
            df0 = sorted_idx[sorted_idx >= num_items]
            i = df0.index[0]
            j = link[df0.iloc[0] - num_items, 0:2]
            sorted_idx.loc[i] = j[0]
            sorted_idx.loc[len(sorted_idx)] = j[1]
        
        return sorted_idx.astype(int).values
    
    def _get_recursive_bisection(self, cov: pd.DataFrame) -> np.ndarray:
        """Calculate weights using recursive bisection.
        
        Args:
            cov: Covariance matrix
            
        Returns:
            Array of portfolio weights
        """
        # Calculate inverse-variance weights
        ivars = 1. / np.diag(cov)
        weights = ivars / ivars.sum()
        
        # Recursive bisection
        clustered_alphas = [weights]
        num_items = len(weights)
        
        while len(clustered_alphas) < num_items:
            clustered_alphas = [
                item for alpha in clustered_alphas
                for item in self._get_bisection(alpha, cov)
            ]
        
        return np.array(clustered_alphas)
    
    def _get_bisection(self, weights: np.ndarray, cov: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one level of recursive bisection.
        
        Args:
            weights: Current level weights
            cov: Covariance matrix
            
        Returns:
            Tuple of left and right subset weights
        """
        w = weights.copy()
        n = len(w)
        if n <= 1:
            return [w]
        
        # Find index to split on
        i = n // 2
        
        # Split weights
        left_w = np.zeros(n)
        right_w = np.zeros(n)
        
        # Normalize left and right subsets
        left_raw = w[:i]
        right_raw = w[i:]
        
        left_var = np.dot(left_raw.T, np.dot(cov.iloc[:i, :i], left_raw))
        right_var = np.dot(right_raw.T, np.dot(cov.iloc[i:, i:], right_raw))
        
        alpha = 1 - left_var / (left_var + right_var)
        
        left_w[:i] = w[:i] * alpha
        right_w[i:] = w[i:] * (1 - alpha)
        
        return [left_w, right_w]
    
    def _apply_weight_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict
    ) -> np.ndarray:
        """Apply minimum and maximum weight constraints.
        
        Args:
            weights: Original weights
            constraints: Weight constraints
            
        Returns:
            Constrained weights
        """
        # Apply min/max constraints
        weights = np.clip(weights, constraints['min_weight'], constraints['max_weight'])
        
        # Renormalize to sum to 1
        weights = weights / weights.sum()
        
        return weights