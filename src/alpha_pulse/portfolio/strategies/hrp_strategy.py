"""
Hierarchical Risk Parity (HRP) implementation of portfolio rebalancing strategy.
Uses clustering-based approach to build a diversified portfolio based on the 
hierarchical correlation structure of assets.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from .base_strategy import BaseStrategy


class HRPStrategy(BaseStrategy):
    """Hierarchical Risk Parity based portfolio optimization strategy."""

    def __init__(self, config: Dict):
        """
        Initialize HRP strategy with configuration.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        self.linkage_method = config.get('linkage_method', 'single')
        self.risk_measure = config.get('risk_measure', 'variance')

    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: pd.DataFrame,
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio allocation using HRP.

        Args:
            current_allocation: Current portfolio weights
            historical_data: Historical price data
            risk_constraints: Dictionary of risk limits

        Returns:
            Target portfolio weights
        """
        returns = self.compute_returns(historical_data)
        
        # Compute correlation and distance matrices
        corr = returns.corr()
        dist = self._compute_distance_matrix(corr)
        
        # Perform hierarchical clustering
        link = linkage(squareform(dist), method=self.linkage_method)
        
        # Get quasi-diagonal correlation matrix
        sorted_assets = self._get_quasi_diag(link, returns.columns)
        returns_sorted = returns[sorted_assets]
        
        # Compute HRP weights
        weights = self._get_recursive_bisection(returns_sorted)
        
        # Create dictionary of weights
        allocation = dict(zip(sorted_assets, weights))
        
        # Adjust for minimum position size and normalize
        allocation = {
            k: v for k, v in allocation.items()
            if v >= self.min_position
        }
        total = sum(allocation.values())
        allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation

    def _compute_distance_matrix(self, corr: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.

        Args:
            corr: Correlation matrix

        Returns:
            Distance matrix
        """
        # Convert correlations to distances
        dist = np.sqrt((1 - corr) / 2)
        return dist

    def _get_quasi_diag(
        self,
        link: np.ndarray,
        labels: List[str]
    ) -> List[str]:
        """
        Return quasi-diagonal ordering of assets based on hierarchical clustering.

        Args:
            link: Linkage matrix from hierarchical clustering
            labels: List of asset names

        Returns:
            List of assets in quasi-diagonal order
        """
        link = link.astype(int)
        sorted_idx = self._get_recursive_order(link, labels)
        return [labels[i] for i in sorted_idx]

    def _get_recursive_order(
        self,
        link: np.ndarray,
        labels: List[str]
    ) -> List[int]:
        """
        Recursively order assets based on hierarchical clustering structure.

        Args:
            link: Linkage matrix
            labels: List of asset names

        Returns:
            List of indices in quasi-diagonal order
        """
        n = len(labels)
        
        if n == 1:
            return [0]
            
        order = []
        clustered = set()
        
        for i in range(len(link)):
            c1, c2 = link[i, 0], link[i, 1]
            
            if c1 < n and c1 not in clustered:
                order.append(int(c1))
                clustered.add(c1)
                
            if c2 < n and c2 not in clustered:
                order.append(int(c2))
                clustered.add(c2)
                
        return order

    def _get_recursive_bisection(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute HRP weights through recursive bisection.

        Args:
            returns: Asset returns DataFrame

        Returns:
            Array of portfolio weights
        """
        n = len(returns.columns)
        if n == 1:
            return np.array([1.0])

        # Split portfolio into two clusters
        mid = n // 2
        left_weights = self._get_recursive_bisection(returns.iloc[:, :mid])
        right_weights = self._get_recursive_bisection(returns.iloc[:, mid:])

        # Calculate cluster variances
        left_var = self._compute_cluster_variance(returns.iloc[:, :mid])
        right_var = self._compute_cluster_variance(returns.iloc[:, mid:])

        # Combine weights based on inverse variance
        alpha = 1 - (left_var / (left_var + right_var))
        return np.concatenate([
            left_weights * alpha,
            right_weights * (1 - alpha)
        ])

    def _compute_cluster_variance(self, returns: pd.DataFrame) -> float:
        """
        Compute variance for a cluster of assets.

        Args:
            returns: Returns DataFrame for cluster

        Returns:
            Cluster variance
        """
        if self.risk_measure == 'variance':
            return np.var(returns.mean(axis=1))
        else:  # MAD - Mean Absolute Deviation
            return np.mean(np.abs(returns.mean(axis=1)))