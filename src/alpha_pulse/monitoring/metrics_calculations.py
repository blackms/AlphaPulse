"""
Metrics calculation functions for the enhanced metrics collector.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import numpy as np

from ..portfolio.data_models import PortfolioData, Position


def calculate_performance_metrics(
    portfolio_data: PortfolioData, risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        portfolio_data: Current portfolio state
        risk_free_rate: Risk-free rate for calculations
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate returns if we have position history
    if not portfolio_data.positions:
        return {}
    
    # Calculate portfolio returns
    returns = calculate_returns(portfolio_data)
    if returns.size == 0:
        return {}
    
    # Calculate metrics
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(returns),
        'var_95': calculate_var(returns, confidence=0.95),
        'expected_shortfall': calculate_expected_shortfall(returns),
        'tracking_error': calculate_tracking_error(returns),
        'information_ratio': calculate_information_ratio(returns, risk_free_rate),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'beta': calculate_beta(returns),
        'alpha': calculate_alpha(returns, risk_free_rate),
        'r_squared': calculate_r_squared(returns),
        'treynor_ratio': calculate_treynor_ratio(returns, risk_free_rate),
        'total_return': calculate_total_return(portfolio_data),
        'annualized_return': calculate_annualized_return(portfolio_data),
        'volatility': calculate_volatility(returns)
    }
    
    return metrics


def calculate_risk_metrics(
    portfolio_data: PortfolioData
) -> Dict[str, Any]:
    """
    Calculate portfolio risk metrics.
    
    Args:
        portfolio_data: Current portfolio state
        
    Returns:
        Dictionary of risk metrics
    """
    if not portfolio_data.positions:
        return {}
    
    # Calculate position concentration
    total_value = sum(p.quantity * p.current_price for p in portfolio_data.positions)
    if total_value == 0:
        return {}
    
    position_weights = {
        p.symbol: (p.quantity * p.current_price) / total_value 
        for p in portfolio_data.positions
    }
    
    # Calculate Herfindahl-Hirschman Index (HHI) for concentration
    hhi = sum(w**2 for w in position_weights.values())
    
    # Calculate leverage
    leverage = total_value / portfolio_data.cash if portfolio_data.cash > 0 else 0
    
    # Calculate other risk metrics
    metrics = {
        'position_count': len(portfolio_data.positions),
        'concentration_hhi': hhi,
        'largest_position_pct': max(position_weights.values()) if position_weights else 0,
        'leverage': leverage,
        'cash_pct': portfolio_data.cash / (portfolio_data.cash + total_value) if (portfolio_data.cash + total_value) > 0 else 0,
        'portfolio_value': total_value + portfolio_data.cash
    }
    
    return metrics


def calculate_trade_metrics(
    trade_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate trade execution metrics.
    
    Args:
        trade_data: Trade execution data
        
    Returns:
        Dictionary of trade metrics
    """
    # Extract basic trade information
    metrics = {
        'symbol': trade_data.get('symbol', ''),
        'side': trade_data.get('side', ''),
        'quantity': trade_data.get('quantity', 0),
        'price': trade_data.get('price', 0),
        'order_type': trade_data.get('order_type', ''),
        'execution_time': trade_data.get('execution_time', 0)
    }
    
    # Calculate slippage if expected price is available
    expected_price = trade_data.get('expected_price')
    if expected_price and metrics['price'] > 0:
        if metrics['side'] == 'buy':
            slippage = (metrics['price'] - expected_price) / expected_price
        else:  # sell
            slippage = (expected_price - metrics['price']) / expected_price
        metrics['slippage'] = slippage
    
    # Calculate fill rate if requested quantity is available
    requested_qty = trade_data.get('requested_quantity')
    if requested_qty and requested_qty > 0:
        metrics['fill_rate'] = metrics['quantity'] / requested_qty
    
    # Calculate cost
    commission = trade_data.get('commission', 0)
    if commission > 0:
        metrics['commission'] = commission
        metrics['commission_pct'] = commission / (metrics['price'] * metrics['quantity']) if metrics['price'] * metrics['quantity'] > 0 else 0
    
    return metrics


def calculate_returns(portfolio_data: PortfolioData) -> np.ndarray:
    """
    Calculate portfolio returns series.
    
    Args:
        portfolio_data: Portfolio data
        
    Returns:
        NumPy array of returns
    """
    # Calculate daily returns from position values
    values = []
    for position in portfolio_data.positions:
        values.append(position.quantity * position.current_price)
    
    if not values:
        return np.array([])
        
    total_values = np.array(values)
    returns = np.diff(total_values) / total_values[:-1] if len(total_values) > 1 else np.array([])
    
    return returns


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if returns.size == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
        
    return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sortino ratio
    """
    if returns.size == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if downside_returns.size == 0:
        return 0.0
        
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0:
        return 0.0
        
    return np.sqrt(252) * (np.mean(excess_returns) / downside_std)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown
    """
    if returns.size == 0:
        return 0.0
        
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / rolling_max - 1
    
    return abs(np.min(drawdowns))


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Array of returns
        confidence: Confidence level
        
    Returns:
        Value at Risk
    """
    if returns.size == 0:
        return 0.0
        
    return abs(np.percentile(returns, (1 - confidence) * 100))


def calculate_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (CVaR).
    
    Args:
        returns: Array of returns
        confidence: Confidence level
        
    Returns:
        Expected Shortfall
    """
    if returns.size == 0:
        return 0.0
        
    var = calculate_var(returns, confidence)
    return abs(np.mean(returns[returns <= -var]))


def calculate_tracking_error(returns: np.ndarray) -> float:
    """
    Calculate tracking error.
    
    Args:
        returns: Array of returns
        
    Returns:
        Tracking error
    """
    if returns.size == 0:
        return 0.0
        
    return np.std(returns) * np.sqrt(252)


def calculate_information_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate information ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Information ratio
    """
    if returns.size == 0:
        return 0.0
        
    tracking_error = calculate_tracking_error(returns)
    if tracking_error == 0:
        return 0.0
        
    return (np.mean(returns) - risk_free_rate) / tracking_error


def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        returns: Array of returns
        
    Returns:
        Calmar ratio
    """
    if returns.size == 0:
        return 0.0
        
    max_drawdown = calculate_max_drawdown(returns)
    if max_drawdown == 0:
        return 0.0
        
    return (np.mean(returns) * 252) / max_drawdown


def calculate_beta(returns: np.ndarray) -> float:
    """
    Calculate portfolio beta.
    
    Args:
        returns: Array of returns
        
    Returns:
        Beta
    """
    if returns.size == 0:
        return 0.0
        
    # Using a simple market proxy (can be enhanced with actual market data)
    market_returns = returns  # Placeholder
    
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0.0
        
    return covariance / market_variance


def calculate_alpha(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate portfolio alpha.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Alpha
    """
    if returns.size == 0:
        return 0.0
        
    beta = calculate_beta(returns)
    market_returns = returns  # Placeholder
    
    return np.mean(returns) - (risk_free_rate + beta * (np.mean(market_returns) - risk_free_rate))


def calculate_r_squared(returns: np.ndarray) -> float:
    """
    Calculate R-squared.
    
    Args:
        returns: Array of returns
        
    Returns:
        R-squared
    """
    if returns.size == 0:
        return 0.0
        
    market_returns = returns  # Placeholder
    correlation = np.corrcoef(returns, market_returns)[0, 1]
    
    return correlation ** 2


def calculate_treynor_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Treynor ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Treynor ratio
    """
    if returns.size == 0:
        return 0.0
        
    beta = calculate_beta(returns)
    if beta == 0:
        return 0.0
        
    return (np.mean(returns) - risk_free_rate) / beta


def calculate_total_return(portfolio_data: PortfolioData) -> float:
    """
    Calculate total return.
    
    Args:
        portfolio_data: Portfolio data
        
    Returns:
        Total return
    """
    if not portfolio_data.positions or not hasattr(portfolio_data, 'initial_value') or portfolio_data.initial_value == 0:
        return 0.0
        
    current_value = sum(p.quantity * p.current_price for p in portfolio_data.positions) + portfolio_data.cash
    return (current_value / portfolio_data.initial_value) - 1 if portfolio_data.initial_value > 0 else 0.0


def calculate_annualized_return(portfolio_data: PortfolioData) -> float:
    """
    Calculate annualized return.
    
    Args:
        portfolio_data: Portfolio data
        
    Returns:
        Annualized return
    """
    if not portfolio_data.positions or not hasattr(portfolio_data, 'initial_value') or portfolio_data.initial_value == 0:
        return 0.0
        
    if not hasattr(portfolio_data, 'start_date'):
        return 0.0
        
    # Calculate years elapsed
    start_date = portfolio_data.start_date
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    
    years = (datetime.now(timezone.utc) - start_date).days / 365.25
    if years <= 0:
        return 0.0
        
    total_return = calculate_total_return(portfolio_data)
    return ((1 + total_return) ** (1 / years)) - 1


def calculate_volatility(returns: np.ndarray) -> float:
    """
    Calculate portfolio volatility.
    
    Args:
        returns: Array of returns
        
    Returns:
        Annualized volatility
    """
    if returns.size == 0:
        return 0.0
        
    return np.std(returns) * np.sqrt(252)