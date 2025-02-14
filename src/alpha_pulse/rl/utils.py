"""
Utility functions for RL trading.

This module provides helper functions for reward shaping, data preprocessing,
and other common RL operations.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from loguru import logger


@dataclass
class RewardParams:
    """Parameters for reward shaping."""
    profit_aim: float = 0.02  # Target profit (2%)
    risk_free_rate: float = 0.0  # Risk-free rate for Sharpe calculation
    max_position_time: int = 100  # Maximum candles to hold position
    time_decay_factor: float = 0.1  # Position holding time penalty
    risk_penalty_factor: float = 0.2  # Risk-taking penalty
    transaction_fee: float = 0.001  # Trading fee (0.1%)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio of a return series.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
        
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino ratio of a return series.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        float: Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
        
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)


def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio of a return series.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        float: Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
        
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    
    if max_drawdown == 0:
        return 0.0
        
    annual_return = (cumulative_returns[-1] ** (periods_per_year / len(returns))) - 1
    return annual_return / abs(max_drawdown)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of a cumulative return series.
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        float: Maximum drawdown
    """
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return np.min(drawdowns)


def shape_reward(
    profit: float,
    holding_time: int,
    volatility: float,
    params: RewardParams
) -> float:
    """
    Shape the reward signal using various factors.
    
    Args:
        profit: Realized or unrealized profit
        holding_time: Number of periods position was held
        volatility: Current volatility estimate
        params: Reward shaping parameters
        
    Returns:
        float: Shaped reward
    """
    # Base reward from profit
    reward = profit
    
    # Time decay penalty
    time_penalty = (holding_time / params.max_position_time) * params.time_decay_factor
    reward *= (1 - time_penalty)
    
    # Risk penalty based on volatility
    risk_penalty = volatility * params.risk_penalty_factor
    reward *= (1 - risk_penalty)
    
    # Transaction cost penalty
    if profit != 0:  # Only apply to trades
        reward -= params.transaction_fee
        
    return reward


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Preprocess data for RL training.
    
    Args:
        df: DataFrame with features
        
    Returns:
        tuple containing:
        - preprocessed data array
        - dictionary of preprocessing parameters
    """
    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].std() == 0]
    df = df.drop(columns=constant_cols)
    
    # Calculate means and stds for normalization
    means = df.mean()
    stds = df.std()
    stds[stds == 0] = 1  # Prevent division by zero
    
    # Normalize data
    normalized = (df - means) / stds
    
    # Replace inf and nan
    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    normalized = normalized.fillna(0)
    
    # Store preprocessing params
    params = {
        'means': means.to_dict(),
        'stds': stds.to_dict(),
        'constant_cols': constant_cols
    }
    
    return normalized.values, params


def detect_outliers(
    data: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers using z-score method.
    
    Args:
        data: Input data array
        threshold: Z-score threshold
        
    Returns:
        Boolean mask of outlier positions
    """
    z_scores = stats.zscore(data, nan_policy='omit')
    return np.abs(z_scores) > threshold


def calculate_position_size(
    confidence: float,
    volatility: float,
    max_position: float = 1.0,
    min_position: float = 0.1
) -> float:
    """
    Calculate position size based on confidence and volatility.
    
    Args:
        confidence: Signal confidence score (0-1)
        volatility: Current volatility estimate
        max_position: Maximum position size
        min_position: Minimum position size
        
    Returns:
        float: Position size
    """
    # Adjust confidence by volatility
    vol_adjusted_conf = confidence * (1 - volatility)
    
    # Scale between min and max position size
    position_size = min_position + (max_position - min_position) * vol_adjusted_conf
    
    return np.clip(position_size, min_position, max_position)


def exponential_decay(
    initial_value: float,
    decay_rate: float,
    time_steps: int
) -> np.ndarray:
    """
    Calculate exponential decay series.
    
    Args:
        initial_value: Starting value
        decay_rate: Decay rate per step
        time_steps: Number of steps
        
    Returns:
        Array of decayed values
    """
    time = np.arange(time_steps)
    return initial_value * np.exp(-decay_rate * time)


def calculate_trade_statistics(
    trades: List[Dict]
) -> Dict[str, float]:
    """
    Calculate trading performance statistics.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of trading statistics
    """
    if not trades:
        return {}
        
    profits = [t['profit'] for t in trades]
    durations = [t['duration'] for t in trades]
    
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    stats = {
        'total_trades': len(trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'avg_profit': np.mean(profits) if profits else 0,
        'avg_win': np.mean(winning_trades) if winning_trades else 0,
        'avg_loss': np.mean(losing_trades) if losing_trades else 0,
        'profit_factor': abs(np.sum(winning_trades) / np.sum(losing_trades)) if losing_trades else 0,
        'avg_duration': np.mean(durations) if durations else 0,
        'sharpe_ratio': calculate_sharpe_ratio(np.array(profits)) if profits else 0,
        'sortino_ratio': calculate_sortino_ratio(np.array(profits)) if profits else 0,
        'max_drawdown': calculate_max_drawdown(np.cumsum(profits)) if profits else 0
    }
    
    return stats