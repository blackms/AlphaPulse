"""
Risk analysis implementations for AlphaPulse.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from loguru import logger

from .interfaces import IRiskAnalyzer, RiskMetrics


class RiskAnalyzer(IRiskAnalyzer):
    """Implementation of risk analysis calculations."""

    def __init__(
        self,
        rolling_window: int = 252,  # Default to 1 year of daily data
        var_confidence: float = 0.95,
        monte_carlo_sims: int = 10000,
    ):
        """
        Initialize risk analyzer.

        Args:
            rolling_window: Window size for rolling calculations
            var_confidence: Confidence level for VaR calculations
            monte_carlo_sims: Number of simulations for Monte Carlo VaR
        """
        self.rolling_window = rolling_window
        self.var_confidence = var_confidence
        self.monte_carlo_sims = monte_carlo_sims
        logger.info(
            f"Initialized RiskAnalyzer (window={rolling_window}, "
            f"var_conf={var_confidence}, mc_sims={monte_carlo_sims})"
        )

    def calculate_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        if len(returns) < 2:
            raise ValueError("Insufficient data for risk calculations")

        # Calculate basic statistics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        excess_returns = returns - risk_free_rate
        
        # Calculate VaR and CVaR
        var_95 = self.calculate_var(returns, self.var_confidence)
        cvar_95 = self.calculate_cvar(returns, self.var_confidence)
        
        # Calculate maximum drawdown
        prices = (1 + returns).cumprod()
        max_drawdown = self.calculate_drawdown(prices).min()
        
        # Calculate ratios
        sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(returns, risk_free_rate)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
        )

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """Calculate Value at Risk using specified method."""
        if method == "historical":
            return self._historical_var(returns, confidence_level)
        elif method == "parametric":
            return self._parametric_var(returns, confidence_level)
        elif method == "monte_carlo":
            return self._monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self.calculate_var(returns, confidence_level)
        return -returns[returns <= -var].mean()

    def calculate_drawdown(
        self,
        prices: pd.Series,
    ) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = prices.expanding().max()
        drawdowns = prices / rolling_max - 1
        return drawdowns

    def _historical_var(
        self,
        returns: pd.Series,
        confidence_level: float,
    ) -> float:
        """Calculate historical VaR."""
        return -np.percentile(returns, 100 * (1 - confidence_level))

    def _parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float,
    ) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        z_score = stats.norm.ppf(confidence_level)
        return -(returns.mean() - z_score * returns.std())  # Fixed sign in calculation

    def _monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float,
    ) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate simulated returns
        simulated_returns = np.random.normal(
            mu,
            sigma,
            size=self.monte_carlo_sims
        )
        
        return -np.percentile(simulated_returns, 100 * (1 - confidence_level))

    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float,
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float,
    ) -> float:
        """Calculate Sortino ratio using downside deviation."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / downside_std

    def _calculate_calmar_ratio(
        self,
        returns: pd.Series,
        max_drawdown: float,
    ) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0
        annual_return = returns.mean() * 252
        return -annual_return / max_drawdown


class RollingRiskAnalyzer(RiskAnalyzer):
    """Extension of RiskAnalyzer with rolling window calculations."""

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: Optional[int] = None,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.

        Args:
            returns: Return series
            window: Rolling window size (default: self.rolling_window)
            risk_free_rate: Risk-free rate for calculations

        Returns:
            DataFrame with rolling risk metrics
        """
        window = window or self.rolling_window
        metrics = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            window_metrics = self.calculate_metrics(window_returns, risk_free_rate)
            metrics.append({
                'timestamp': returns.index[i-1],
                'volatility': window_metrics.volatility,
                'var_95': window_metrics.var_95,
                'cvar_95': window_metrics.cvar_95,
                'sharpe_ratio': window_metrics.sharpe_ratio,
                'sortino_ratio': window_metrics.sortino_ratio,
            })
        
        return pd.DataFrame(metrics).set_index('timestamp')

    def calculate_rolling_var(
        self,
        returns: pd.Series,
        window: Optional[int] = None,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> pd.Series:
        """
        Calculate rolling VaR.

        Args:
            returns: Return series
            window: Rolling window size
            confidence_level: VaR confidence level
            method: VaR calculation method

        Returns:
            Series of rolling VaR values
        """
        window = window or self.rolling_window
        var_values = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            var = self.calculate_var(
                window_returns,
                confidence_level,
                method
            )
            var_values.append(var)
        
        return pd.Series(
            var_values,
            index=returns.index[window-1:],
            name='VaR'
        )

    def calculate_rolling_drawdown(
        self,
        returns: pd.Series,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling drawdown metrics.

        Args:
            returns: Return series
            window: Rolling window size

        Returns:
            DataFrame with drawdown metrics
        """
        window = window or self.rolling_window
        prices = (1 + returns).cumprod()
        
        rolling_max_drawdown = []
        rolling_drawdown_duration = []
        
        for i in range(window, len(returns) + 1):
            window_prices = prices.iloc[i-window:i]
            drawdown = self.calculate_drawdown(window_prices)
            
            rolling_max_drawdown.append(drawdown.min())
            
            # Calculate drawdown duration
            underwater = drawdown < 0
            if underwater.any():
                duration = underwater.groupby(
                    (~underwater).cumsum()
                ).cumsum().max()
            else:
                duration = 0
            rolling_drawdown_duration.append(duration)
        
        return pd.DataFrame({
            'max_drawdown': rolling_max_drawdown,
            'drawdown_duration': rolling_drawdown_duration
        }, index=returns.index[window-1:])