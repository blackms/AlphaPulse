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
            
            # Calculate distance matrix and ensure symmetry
            dist = self._get_distance_matrix(corr)
            
            # Convert to condensed form for linkage
            n = len(dist)
            condensed_dist = []
            for i in range(n):
                for j in range(i + 1, n):
                    condensed_dist.append(dist.iloc[i, j])
            
            # Perform hierarchical clustering
            link = linkage(condensed_dist, 'single')
            
            # Get quasi-diagonal matrix
            sorted_idx = self._get_quasi_diag(link)
            ret = returns.iloc[:, sorted_idx]
            cov_sorted = cov.iloc[sorted_idx, sorted_idx]
            
            # Calculate HRP weights
            weights = self._get_recursive_bisection(cov_sorted)
            
            # Reorder weights to match original asset order
            inv_idx = np.argsort(sorted_idx)
            weights = weights[inv_idx]
            
            logger.debug(f"Initial weights: {weights}")
            
            # Ensure weights are valid numbers
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                logger.error("Invalid weights detected before constraints")
                weights = np.ones(len(weights)) / len(weights)
            
            # First pass: clip to constraints
            weights = np.clip(weights, constraints['min_weight'], constraints['max_weight'])
            logger.debug(f"Weights after initial clipping: {weights}")
            
            # Calculate total and identify which weights need adjustment
            total = np.sum(weights)
            if total != 1.0:
                # If total > 1, reduce proportionally from weights above min
                if total > 1.0:
                    excess = total - 1.0
                    reducible_mask = weights > constraints['min_weight']
                    if np.any(reducible_mask):
                        reduction_weights = weights[reducible_mask]
                        reduction = excess * (reduction_weights / np.sum(reduction_weights))
                        weights[reducible_mask] -= reduction
                
                # If total < 1, add proportionally to weights below max
                else:
                    shortfall = 1.0 - total
                    increasable_mask = weights < constraints['max_weight']
                    if np.any(increasable_mask):
                        room_to_max = constraints['max_weight'] - weights[increasable_mask]
                        increase = shortfall * (room_to_max / np.sum(room_to_max))
                        weights[increasable_mask] += increase
            
            # Final normalization to ensure sum is exactly 1
            weights = weights / np.sum(weights)
            
            logger.debug(f"Final adjusted weights: {weights}")
            
            # Convert to decimals with proper rounding
            target_weights = {}
            for asset, weight in zip(returns.columns, weights):
                # Round to 8 decimal places to avoid floating point issues
                decimal_weight = Decimal(str(round(float(weight), 8)))
                target_weights[asset] = decimal_weight
            
            # Final normalization to ensure exact sum of 1
            total = sum(target_weights.values())
            if total != Decimal('1.0'):
                for asset in target_weights:
                    target_weights[asset] = target_weights[asset] / total
            
            logger.debug(f"Final target weights: {target_weights}")
            
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
        # Handle any NaN or infinite values in correlation matrix
        corr = corr.fillna(0)  # Replace NaN with 0 correlation
        corr = np.clip(corr, -1, 1)  # Ensure correlations are in [-1, 1]
        
        # Convert correlations to distances
        dist = np.sqrt((1 - corr) / 2)
        
        # Ensure the diagonal is 0 (no distance to self)
        np.fill_diagonal(dist.values, 0)
        
        # Replace any remaining non-finite values
        dist = dist.fillna(1)  # Maximum distance for invalid correlations
        
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
        # Calculate inverse-variance weights with small constant to prevent division by zero
        variances = np.diag(cov)
        # Add small constant (epsilon) to prevent division by zero
        epsilon = 1e-8
        ivars = 1. / (variances + epsilon)
        weights = ivars / ivars.sum()
        
        # Recursive bisection
        num_items = len(weights)
        if num_items <= 1:
            return weights
            
        # Split into two clusters
        i = num_items // 2
        left_cluster = self._get_recursive_bisection(cov.iloc[:i, :i])
        right_cluster = self._get_recursive_bisection(cov.iloc[i:, i:])
        
        # Calculate cluster variances
        left_var = np.dot(left_cluster.T, np.dot(cov.iloc[:i, :i], left_cluster))
        right_var = np.dot(right_cluster.T, np.dot(cov.iloc[i:, i:], right_cluster))
        
        # Calculate allocation between clusters
        alpha = 1 - left_var / (left_var + right_var)
        
        # Combine weights
        final_weights = np.zeros(num_items)
        final_weights[:i] = left_cluster * alpha
        final_weights[i:] = right_cluster * (1 - alpha)
        
        return final_weights
    
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