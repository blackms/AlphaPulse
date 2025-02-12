"""
Performance monitoring and metrics collection.
"""
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time
import functools
from collections import defaultdict

from ..portfolio.data_models import PortfolioData

# Global metrics storage
API_LATENCY = defaultdict(list)


def track_latency(api_name: str):
    """
    Decorator to track API call latencies.
    
    Args:
        api_name: Name of the API being called
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                latency = end_time - start_time
                API_LATENCY[api_name].append(latency)
        return wrapper
    return decorator


@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    tracking_error: float
    information_ratio: float
    calmar_ratio: float
    beta: float
    alpha: float
    r_squared: float
    treynor_ratio: float
    timestamp: datetime


class MetricsCollector:
    """Collects and calculates portfolio performance metrics."""

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize metrics collector.

        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self._metrics_history: Dict[datetime, PerformanceMetrics] = {}

    def collect_metrics(self, portfolio_data: PortfolioData) -> Dict[str, float]:
        """
        Collect and calculate performance metrics.

        Args:
            portfolio_data: Current portfolio state

        Returns:
            Dictionary of calculated metrics
        """
        # Calculate returns if we have position history
        if not portfolio_data.positions:
            return {}

        # Calculate portfolio returns
        returns = self._calculate_returns(portfolio_data)
        if returns.empty:
            return {}

        # Calculate metrics
        metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'var_95': self._calculate_var(returns, confidence=0.95),
            'expected_shortfall': self._calculate_expected_shortfall(returns),
            'tracking_error': self._calculate_tracking_error(returns),
            'information_ratio': self._calculate_information_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'beta': self._calculate_beta(returns),
            'alpha': self._calculate_alpha(returns),
            'r_squared': self._calculate_r_squared(returns),
            'treynor_ratio': self._calculate_treynor_ratio(returns)
        }

        # Store metrics history
        self._metrics_history[datetime.now()] = PerformanceMetrics(
            **metrics,
            timestamp=datetime.now()
        )

        return metrics

    def _calculate_returns(self, portfolio_data: PortfolioData) -> pd.Series:
        """Calculate portfolio returns series."""
        # Calculate daily returns from position values
        values = []
        for position in portfolio_data.positions:
            values.append(position.quantity * position.current_price)
        
        if not values:
            return pd.Series()
            
        total_values = pd.Series(values)
        returns = total_values.pct_change().dropna()
        
        return returns

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty:
            return 0.0
            
        excess_returns = returns - self.risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
            
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if returns.empty:
            return 0.0
            
        excess_returns = returns - self.risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
            
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std == 0:
            return 0.0
            
        return np.sqrt(252) * (excess_returns.mean() / downside_std)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        
        return abs(drawdowns.min())

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if returns.empty:
            return 0.0
            
        return abs(np.percentile(returns, (1 - confidence) * 100))

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        if returns.empty:
            return 0.0
            
        var = self._calculate_var(returns, confidence)
        return abs(returns[returns <= -var].mean())

    def _calculate_tracking_error(self, returns: pd.Series) -> float:
        """Calculate tracking error."""
        if returns.empty:
            return 0.0
            
        return returns.std() * np.sqrt(252)

    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio."""
        if returns.empty:
            return 0.0
            
        tracking_error = self._calculate_tracking_error(returns)
        if tracking_error == 0:
            return 0.0
            
        return (returns.mean() - self.risk_free_rate) / tracking_error

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if returns.empty:
            return 0.0
            
        max_drawdown = self._calculate_max_drawdown(returns)
        if max_drawdown == 0:
            return 0.0
            
        return (returns.mean() * 252) / max_drawdown

    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate portfolio beta."""
        if returns.empty:
            return 0.0
            
        # Using a simple market proxy (can be enhanced with actual market data)
        market_returns = returns  # Placeholder
        
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 0.0
            
        return covariance / market_variance

    def _calculate_alpha(self, returns: pd.Series) -> float:
        """Calculate portfolio alpha."""
        if returns.empty:
            return 0.0
            
        beta = self._calculate_beta(returns)
        market_returns = returns  # Placeholder
        
        return returns.mean() - (self.risk_free_rate + beta * (market_returns.mean() - self.risk_free_rate))

    def _calculate_r_squared(self, returns: pd.Series) -> float:
        """Calculate R-squared."""
        if returns.empty:
            return 0.0
            
        market_returns = returns  # Placeholder
        correlation = returns.corr(market_returns)
        
        return correlation ** 2

    def _calculate_treynor_ratio(self, returns: pd.Series) -> float:
        """Calculate Treynor ratio."""
        if returns.empty:
            return 0.0
            
        beta = self._calculate_beta(returns)
        if beta == 0:
            return 0.0
            
        return (returns.mean() - self.risk_free_rate) / beta

    def get_metrics_history(self) -> Dict[datetime, PerformanceMetrics]:
        """Get historical metrics."""
        return self._metrics_history

    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics."""
        if not self._metrics_history:
            return None
            
        latest_time = max(self._metrics_history.keys())
        return self._metrics_history[latest_time]

    def get_api_latencies(self) -> Dict[str, Dict[str, float]]:
        """Get API latency statistics."""
        stats = {}
        for api_name, latencies in API_LATENCY.items():
            if latencies:
                stats[api_name] = {
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'max': max(latencies),
                    'min': min(latencies),
                    'count': len(latencies)
                }
        return stats