"""
Hierarchical Risk Parity (HRP) implementation of portfolio rebalancing strategy.
Uses clustering-based approach to build a diversified portfolio based on the 
hierarchical correlation structure of assets.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist
from decimal import Decimal
from loguru import logger
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
        logger.info(f"Initialized HRP strategy with linkage_method={self.linkage_method}, risk_measure={self.risk_measure}")

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
        logger.info("Starting HRP allocation computation")
        logger.debug(f"Input data shape: {historical_data.shape}")
        logger.debug(f"Current allocation: {current_allocation}")

        try:
            # Filter for allowed assets only and exclude stablecoins
            allowed_cols = [
                col for col in historical_data.columns 
                if col in self.allowed_assets and col not in self.stablecoins
            ]
            if not allowed_cols:
                logger.warning("No valid assets found in historical data")
                return current_allocation
                
            # Convert data to float64
            returns = self.compute_returns(historical_data[allowed_cols].astype(np.float64))
            logger.debug(f"Computed returns shape: {returns.shape}")
            
            # Handle case with only one asset
            if len(returns.columns) == 1:
                logger.info("Only one asset found, returning single-asset allocation")
                return self._allocate_with_stablecoins({returns.columns[0]: 1.0})
            
            # Compute correlation matrix
            corr = returns.corr()
            logger.debug(f"Correlation matrix:\n{corr}")
            
            # Ensure correlation matrix is symmetric and contains doubles
            corr = np.array(corr, dtype=np.float64)
            corr = (corr + corr.T) / 2.0
            np.fill_diagonal(corr, 1.0)
            
            # Replace any NaN values with 0 correlation
            corr = np.nan_to_num(corr, nan=0.0)
            
            # Compute distance matrix using correlation
            dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
            logger.debug(f"Distance matrix:\n{dist}")
            
            # Convert to condensed form
            n = len(dist)
            condensed = []
            for i in range(n):
                for j in range(i + 1, n):
                    condensed.append(dist[i, j])
            condensed = np.array(condensed, dtype=np.float64)
            
            # Compute linkage
            link = linkage(condensed, method=self.linkage_method)
            logger.debug(f"Linkage matrix shape: {link.shape}")
            
            # Get quasi-diagonal correlation matrix
            sorted_assets = self._get_quasi_diag(link, list(returns.columns))
            returns_sorted = returns[sorted_assets]
            logger.debug(f"Sorted assets order: {sorted_assets}")
            
            # Compute HRP weights
            weights = self._get_recursive_bisection(returns_sorted)
            logger.debug(f"Initial weights: {weights}")
            
            # Create dictionary of weights for non-stablecoin assets
            allocation = dict(zip(sorted_assets, weights))
            
            # Add stablecoins and normalize
            final_allocation = self._allocate_with_stablecoins(allocation)
            logger.info(f"Final target allocation: {final_allocation}")
            return final_allocation
            
        except Exception as e:
            logger.error(f"Error in HRP allocation computation: {str(e)}")
            logger.exception(e)
            return current_allocation

    def _allocate_with_stablecoins(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Add stablecoins to allocation and normalize weights."""
        try:
            # Convert all values to float
            allocation = {k: float(v) for k, v in allocation.items()}
            
            # Scale non-stablecoin weights to (1 - stablecoin_target)
            total = sum(allocation.values())
            if total > 0:
                target_weight = 1.0 - self.stablecoin_target
                allocation = {
                    k: v * target_weight / total 
                    for k, v in allocation.items()
                }
            
            # Add stablecoins
            stables = [coin for coin in self.stablecoins if coin in self.allowed_assets]
            if stables:
                # Distribute stablecoin allocation equally
                stable_weight = self.stablecoin_target / len(stables)
                for coin in stables:
                    allocation[coin] = stable_weight
            
            # Final normalization
            total = sum(allocation.values())
            if total > 0:
                allocation = {k: v/total for k, v in allocation.items()}
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error in stablecoin allocation: {str(e)}")
            logger.exception(e)
            return allocation

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
        try:
            # Ensure linkage matrix contains doubles
            link = np.array(link, dtype=np.float64)
            
            # Get the order of leaves from the dendrogram
            d = dendrogram(link, labels=labels, no_plot=True)
            logger.debug(f"Dendrogram leaf ordering: {d['ivl']}")
            return d['ivl']  # Return leaf labels in order
        except Exception as e:
            logger.error(f"Error in quasi-diagonal computation: {str(e)}")
            logger.exception(e)
            return labels

    def _get_recursive_bisection(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute HRP weights through recursive bisection.

        Args:
            returns: Asset returns DataFrame

        Returns:
            Array of portfolio weights
        """
        try:
            n = len(returns.columns)
            if n == 1:
                return np.array([1.0], dtype=np.float64)

            # Split portfolio into two clusters
            mid = n // 2
            left_weights = self._get_recursive_bisection(returns.iloc[:, :mid])
            right_weights = self._get_recursive_bisection(returns.iloc[:, mid:])

            # Calculate cluster variances
            left_var = self._compute_cluster_variance(returns.iloc[:, :mid])
            right_var = self._compute_cluster_variance(returns.iloc[:, mid:])

            logger.debug(f"Cluster variances - Left: {left_var}, Right: {right_var}")

            # Handle zero variances
            if left_var == 0 and right_var == 0:
                alpha = 0.5  # Equal weights if both variances are zero
            else:
                # Combine weights based on inverse variance
                alpha = 1.0 - (left_var / (left_var + right_var))
                
            # Ensure alpha is between 0 and 1
            alpha = np.clip(alpha, 0.0, 1.0)
            logger.debug(f"Bisection alpha: {alpha}")
            
            # Combine weights
            left_weights = left_weights.astype(np.float64) * alpha
            right_weights = right_weights.astype(np.float64) * (1.0 - alpha)
            
            return np.concatenate([left_weights, right_weights])
            
        except Exception as e:
            logger.error(f"Error in recursive bisection: {str(e)}")
            logger.exception(e)
            # Return equal weights as fallback
            return np.ones(n, dtype=np.float64) / n

    def _compute_cluster_variance(self, returns: pd.DataFrame) -> float:
        """
        Compute variance for a cluster of assets.

        Args:
            returns: Returns DataFrame for cluster

        Returns:
            Cluster variance
        """
        if len(returns.columns) == 0:
            return 0.0
            
        try:
            # Convert to float64 for numerical stability
            returns = returns.astype(np.float64)
            
            if self.risk_measure == 'variance':
                variance = float(np.var(returns.mean(axis=1)))
                logger.debug(f"Cluster variance: {variance}")
                return variance
            else:  # MAD - Mean Absolute Deviation
                mad = float(np.mean(np.abs(returns.mean(axis=1))))
                logger.debug(f"Cluster MAD: {mad}")
                return mad
                
        except Exception as e:
            logger.error(f"Error computing cluster variance: {str(e)}")
            logger.exception(e)
            return 0.0