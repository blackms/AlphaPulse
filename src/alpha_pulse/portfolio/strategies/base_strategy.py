"""
Base portfolio strategy implementation.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from ..interfaces import IRebalancingStrategy


class BaseStrategy(IRebalancingStrategy):
    """Base portfolio strategy implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.min_position = config.get("min_position", 0.05)
        self.max_position = config.get("max_position", 0.4)
        self.stablecoin_target = config.get("stablecoin_target", 0.3)
        
        logger.info(
            f"Initialized base strategy with: "
            f"min_position={self.min_position} "
            f"max_position={self.max_position} "
            f"stablecoin_target={self.stablecoin_target}"
        )

    def compute_returns(self, historical_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute returns from historical data.
        
        Args:
            historical_data: Dictionary of historical market data by symbol
            
        Returns:
            DataFrame of asset returns
        """
        try:
            # Convert market data to price DataFrame
            prices_data = {}
            timestamps = set()

            # First pass: collect all timestamps and prices
            for symbol, data_list in historical_data.items():
                if not data_list:  # Skip empty data
                    continue
                for data in data_list:
                    timestamps.add(data.timestamp)
                    if symbol not in prices_data:
                        prices_data[symbol] = []
                    prices_data[symbol].append({
                        'timestamp': data.timestamp,
                        'price': float(data.close)
                    })

            if not timestamps:  # No data available
                logger.warning("No historical data available")
                return pd.DataFrame()

            # Sort timestamps
            sorted_timestamps = sorted(timestamps)

            # Create price DataFrame
            price_data = {}
            for symbol, data_list in prices_data.items():
                # Create temporary DataFrame for this symbol
                symbol_df = pd.DataFrame(data_list)
                symbol_df.set_index('timestamp', inplace=True)
                symbol_df.sort_index(inplace=True)

                # Reindex to include all timestamps
                symbol_df = symbol_df.reindex(sorted_timestamps)
                
                # Forward fill missing values
                symbol_df = symbol_df.ffill()  # Use ffill() instead of fillna(method='ffill')
                
                # Extract price series
                price_data[symbol] = symbol_df['price']

            # Create final DataFrame
            prices_df = pd.DataFrame(price_data, index=sorted_timestamps)

            # Compute returns
            returns = prices_df.pct_change().dropna()
            
            logger.debug(f"Computed returns shape: {returns.shape}")
            return returns

        except Exception as e:
            logger.error(f"Error computing returns: {str(e)}")
            return pd.DataFrame()

    def validate_constraints(self, allocation: Dict[str, float]) -> bool:
        """
        Validate allocation constraints.
        
        Args:
            allocation: Target allocation
            
        Returns:
            True if constraints are satisfied
        """
        if not allocation:
            return False
            
        # Check allocation sum
        total_allocation = sum(allocation.values())
        if not 0.99 <= total_allocation <= 1.01:  # Allow small rounding errors
            logger.warning(f"Invalid total allocation: {total_allocation:.2f}")
            return False
            
        # Check position limits
        for asset, weight in allocation.items():
            if weight < 0:  # No short positions
                logger.warning(f"Negative weight for {asset}: {weight:.2f}")
                return False
                
            # Allow higher allocation for stablecoins
            max_position = self.max_position * 2 if asset.endswith('USDT') else self.max_position
                
            if weight > max_position:
                logger.warning(
                    f"Position size exceeds limit for {asset}: "
                    f"{weight:.2f} > {max_position:.2f}"
                )
                return False
                
            if 0 < weight < self.min_position:
                logger.warning(
                    f"Position size below minimum for {asset}: "
                    f"{weight:.2f} < {self.min_position:.2f}"
                )
                return False
                
        return True

    def get_constraint_violations(self, allocation: Dict[str, float]) -> Dict[str, str]:
        """
        Get description of constraint violations.
        
        Args:
            allocation: Target allocation
            
        Returns:
            Dictionary of violation descriptions
        """
        violations = {}
        
        if not allocation:
            return {"allocation": "Empty allocation"}
            
        # Check allocation sum
        total_allocation = sum(allocation.values())
        if not 0.99 <= total_allocation <= 1.01:
            violations["total"] = f"Total allocation {total_allocation:.2f} != 1.0"
            
        # Check position limits
        for asset, weight in allocation.items():
            if weight < 0:
                violations[asset] = f"Negative weight: {weight:.2f}"
                
            # Allow higher allocation for stablecoins
            max_position = self.max_position * 2 if asset.endswith('USDT') else self.max_position
                
            if weight > max_position:
                violations[asset] = (
                    f"Exceeds max position: "
                    f"{weight:.2f} > {max_position:.2f}"
                )
                
            if 0 < weight < self.min_position:
                violations[asset] = (
                    f"Below min position: "
                    f"{weight:.2f} < {self.min_position:.2f}"
                )
                
        return violations

    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: Dict[str, Any],
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute target allocation.
        
        Args:
            current_allocation: Current portfolio allocation
            historical_data: Historical market data
            risk_constraints: Risk management constraints
            
        Returns:
            Target allocation
        """
        raise NotImplementedError("Subclasses must implement compute_target_allocation")