# AlphaPulse: AI-Driven Hedge Fund System
# Copyright (C) 2024 AlphaPulse Trading System
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Core interfaces for the portfolio management system.
Defines the contract for implementing various portfolio strategies and risk management components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from decimal import Decimal


class IExchange(ABC):
    """Interface for exchange operations."""
    
    @abstractmethod
    def get_account_balances(self) -> Dict[str, Decimal]:
        """
        Get current account balances.

        Returns:
            Dictionary mapping asset symbols to their balances
        """
        pass

    @abstractmethod
    def get_ticker_prices(self, assets: List[str]) -> Dict[str, Decimal]:
        """
        Get current prices for assets in base currency.

        Args:
            assets: List of asset symbols

        Returns:
            Dictionary mapping asset symbols to their prices
        """
        pass

    @abstractmethod
    def get_portfolio_value(self) -> Decimal:
        """
        Get total portfolio value in base currency.

        Returns:
            Total portfolio value
        """
        pass

    @abstractmethod
    def execute_trade(
        self,
        asset: str,
        amount: Decimal,
        side: str,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Execute a trade on the exchange.

        Args:
            asset: Asset symbol
            amount: Trade amount in base currency
            side: Trade side ('buy' or 'sell')
            order_type: Order type (default: 'market')

        Returns:
            Dictionary containing trade execution details
        """
        pass

    @abstractmethod
    def get_historical_data(
        self,
        assets: List[str],
        start_time: pd.Timestamp,
        end_time: Optional[pd.Timestamp] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for assets.

        Args:
            assets: List of asset symbols
            start_time: Start time for historical data
            end_time: Optional end time (defaults to current time)
            interval: Data interval (e.g., '1d', '1h')

        Returns:
            DataFrame with historical price data
        """
        pass


class IRebalancingStrategy(ABC):
    """Interface for portfolio rebalancing strategies."""
    
    @abstractmethod
    def compute_target_allocation(
        self,
        current_allocation: Dict[str, float],
        historical_data: pd.DataFrame,
        risk_constraints: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute target portfolio allocation based on strategy-specific logic.

        Args:
            current_allocation: Current portfolio weights {asset: weight}
            historical_data: Historical price/return data for analysis
            risk_constraints: Dictionary of risk limits and constraints

        Returns:
            Dictionary of target portfolio weights {asset: weight}
        """
        pass

    @abstractmethod
    def validate_constraints(self, allocation: Dict[str, float]) -> bool:
        """
        Validate if allocation meets all strategy constraints.

        Args:
            allocation: Portfolio allocation to validate

        Returns:
            Boolean indicating if allocation is valid
        """
        pass


class IRiskScorer(ABC):
    """Interface for portfolio risk scoring."""

    @abstractmethod
    def compute_risk_metrics(
        self,
        allocation: Dict[str, float],
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute comprehensive risk metrics for a portfolio allocation.

        Args:
            allocation: Portfolio weights {asset: weight}
            historical_data: Historical price/return data

        Returns:
            Dictionary of risk metrics (volatility, VaR, etc.)
        """
        pass

    @abstractmethod
    def score_allocation(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        historical_data: pd.DataFrame
    ) -> float:
        """
        Score the risk-adjusted benefit of rebalancing from current to target allocation.

        Args:
            current: Current portfolio weights
            target: Target portfolio weights
            historical_data: Historical price/return data

        Returns:
            Risk-adjusted score for the rebalancing action
        """
        pass


class IPortfolioOptimizer(ABC):
    """Interface for portfolio optimization algorithms."""

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Dict[str, Any],
        initial_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Optimize portfolio weights given return data and constraints.

        Args:
            returns: Asset returns data
            constraints: Optimization constraints
            initial_weights: Optional starting weights

        Returns:
            Tuple of (optimal weights, optimization metrics)
        """
        pass


class ILLMStrategy(ABC):
    """Interface for LLM-assisted portfolio strategies."""

    @abstractmethod
    def analyze_allocation(
        self,
        current_allocation: Dict[str, float],
        proposed_allocation: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> Tuple[Dict[str, float], str]:
        """
        Analyze and potentially adjust portfolio allocation using LLM insights.

        Args:
            current_allocation: Current portfolio weights
            proposed_allocation: Proposed new weights
            market_data: Additional market context/data

        Returns:
            Tuple of (adjusted allocation, reasoning)
        """
        pass

    @abstractmethod
    def get_market_sentiment(self, assets: List[str]) -> Dict[str, float]:
        """
        Get LLM-based market sentiment scores for assets.

        Args:
            assets: List of asset symbols

        Returns:
            Dictionary of sentiment scores per asset
        """
        pass