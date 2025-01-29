"""
Data generation module for creating sample financial data.

This module provides utilities for generating synthetic financial data
for testing and demonstration purposes.
"""
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_data(
    days: int = 365,
    start_date: Optional[datetime] = None,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Create sample OHLCV price data for demonstration.

    Args:
        days: Number of days of data to generate
        start_date: Starting date for the data (defaults to days before current date)
        base_price: Initial base price for the asset
        volatility: Daily volatility for price movements
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    if seed is not None:
        np.random.seed(seed)
    
    start_date = start_date or (datetime.now() - timedelta(days=days-1))
    dates = pd.date_range(
        start=start_date,
        periods=days,
        freq='D'
    )
    
    # Generate random walk for close prices
    rw = np.random.normal(0, volatility, len(dates)).cumsum()
    close_prices = base_price * np.exp(rw)
    
    # Generate other OHLCV data
    df = pd.DataFrame({
        'close': close_prices,
        'open': close_prices * (1 + np.random.normal(0, volatility/2, len(dates))),
        'high': close_prices * (1 + abs(np.random.normal(0, volatility, len(dates)))),
        'low': close_prices * (1 - abs(np.random.normal(0, volatility, len(dates)))),
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)
    
    # Ensure high is highest and low is lowest for each day
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def create_target_variable(
    df: pd.DataFrame,
    forward_returns_days: int = 1,
    column: str = 'close'
) -> pd.Series:
    """
    Create target variable based on forward returns.

    Args:
        df: DataFrame containing price data
        forward_returns_days: Number of days to look ahead
        column: Column to calculate returns from

    Returns:
        Series containing forward returns
    """
    prices = df[column]
    forward_returns = prices.shift(-forward_returns_days) / prices - 1
    return forward_returns