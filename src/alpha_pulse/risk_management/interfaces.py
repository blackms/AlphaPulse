"""
Risk management interfaces for AlphaPulse.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from loguru import logger


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: float
    max_portfolio_risk: float
    max_correlation: float
    max_leverage: float
    stop_loss_pct: float
    max_drawdown: float


@dataclass
class PositionLimits:
    """Position-specific limits."""
    symbol: str
    max_position_size: float
    max_leverage: float
    min_position_size: float
    max_concentration: float


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    volatility: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    size: float  # Recommended position size in base currency
    confidence: float  # Confidence level in the recommendation (0-1)
    metrics: Dict[str, float]  # Additional metrics used in calculation


class IPositionSizer(ABC):
    """Interface for position sizing strategies."""

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        signal_strength: float,
        historical_returns: Optional[pd.Series] = None,
        risk_budget: Optional[Dict[str, float]] = None,
    ) -> PositionSizeResult:
        """
        Calculate recommended position size.

        Args:
            symbol: Trading symbol
            current_price: Current asset price
            portfolio_value: Total portfolio value
            volatility: Asset volatility (e.g., standard deviation of returns)
            signal_strength: Trading signal strength (0-1)
            historical_returns: Optional historical returns for more advanced calculations

        Returns:
            PositionSizeResult with recommended size and metrics
        """
        pass


class IRiskAnalyzer(ABC):
    """Interface for risk analysis."""

    @abstractmethod
    def calculate_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> RiskMetrics:
        """
        Calculate risk metrics for a return series.

        Args:
            returns: Series of asset returns
            risk_free_rate: Risk-free rate for ratio calculations

        Returns:
            RiskMetrics containing calculated metrics
        """
        pass

    @abstractmethod
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Series of asset returns
            confidence_level: VaR confidence level (default: 95%)
            method: VaR calculation method ("historical", "parametric", or "monte_carlo")

        Returns:
            VaR value at specified confidence level
        """
        pass

    @abstractmethod
    def calculate_drawdown(
        self,
        prices: pd.Series,
    ) -> pd.Series:
        """
        Calculate drawdown series.

        Args:
            prices: Series of asset prices

        Returns:
            Series of drawdown values
        """
        pass


class IPortfolioOptimizer(ABC):
    """Interface for portfolio optimization."""

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights.

        Args:
            returns: DataFrame of asset returns (columns are assets)
            risk_free_rate: Risk-free rate for optimization
            constraints: Optional constraints (e.g., max weight per asset)

        Returns:
            Dictionary mapping asset names to optimal weights
        """
        pass


class IRiskManager(ABC):
    """Interface for comprehensive risk management."""

    @abstractmethod
    def evaluate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
    ) -> bool:
        """
        Evaluate if a trade meets risk management criteria.

        Args:
            symbol: Trading symbol
            side: Trade side ("buy" or "sell")
            quantity: Trade quantity
            current_price: Current asset price
            portfolio_value: Total portfolio value
            current_positions: Dictionary of current positions

        Returns:
            True if trade is acceptable, False otherwise
        """
        pass

    @abstractmethod
    def update_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        asset_returns: Dict[str, pd.Series],
    ) -> None:
        """
        Update risk metrics with new data.

        Args:
            portfolio_returns: Series of portfolio returns
            asset_returns: Dictionary mapping symbols to their return series
        """
        pass

    @abstractmethod
    def get_stop_loss_prices(
        self,
        positions: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Calculate stop-loss prices for current positions.

        Args:
            positions: Dictionary mapping symbols to position details

        Returns:
            Dictionary mapping symbols to stop-loss prices
        """
        pass

    @abstractmethod
    def get_position_limits(
        self,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Get maximum position sizes for risk management.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary of maximum position sizes by symbol
        """
        pass