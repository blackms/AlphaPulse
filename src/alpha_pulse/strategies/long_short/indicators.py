#!/usr/bin/env python3
"""
Indicator Calculation Module for the Long/Short S&P 500 Strategy.

Provides functions to calculate technical indicators like Moving Average, RSI,
and detect volatility regimes based on VIX.
"""

from loguru import logger # Use loguru
import pandas as pd
import numpy as np # Added import
import ta
from typing import Optional, Tuple

# Loguru logger is imported directly
# logger = logger.bind(name="LongShortIndicators") # Optional binding

def calculate_moving_average(data: pd.DataFrame, window: int = 40, price_column: str = 'SP500_Adj_Close') -> Optional[pd.Series]:
    """
    Calculates the Simple Moving Average (SMA) for a given price column.

    Args:
        data: DataFrame containing the price data. Must have a DatetimeIndex.
        window: The moving average window period (e.g., 40 for 40-week MA on weekly data).
        price_column: The name of the column containing the price data.

    Returns:
        A pandas Series containing the calculated SMA, or None if calculation fails.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None.")
        return None
    if price_column not in data.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame.")
        return None
    if window <= 0:
        logger.error("Window must be a positive integer.")
        return None

    try:
        logger.debug(f"Calculating {window}-period SMA on column '{price_column}'") # DEBUG
        # Ensure input is a Series
        price_series = data[price_column]
        if isinstance(price_series, pd.DataFrame):
             price_series = price_series.squeeze() # Convert single-column DataFrame to Series
        if not isinstance(price_series, pd.Series):
             raise TypeError(f"Expected a pandas Series for SMA calculation, got {type(price_series)}")

        sma = ta.trend.sma_indicator(price_series, window=window, fillna=False)
        return sma
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return None

def calculate_rsi(data: pd.DataFrame, window: int = 14, price_column: str = 'SP500_Adj_Close') -> Optional[pd.Series]:
    """
    Calculates the Relative Strength Index (RSI) for a given price column.

    Args:
        data: DataFrame containing the price data. Must have a DatetimeIndex.
        window: The RSI window period (e.g., 14).
        price_column: The name of the column containing the price data.

    Returns:
        A pandas Series containing the calculated RSI, or None if calculation fails.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None.")
        return None
    if price_column not in data.columns:
        logger.error(f"Price column '{price_column}' not found in DataFrame.")
        return None
    if window <= 0:
        logger.error("Window must be a positive integer.")
        return None

    try:
        logger.debug(f"Calculating {window}-period RSI on column '{price_column}'") # DEBUG
        # Ensure input is a Series
        price_series = data[price_column]
        if isinstance(price_series, pd.DataFrame):
             price_series = price_series.squeeze()
        if not isinstance(price_series, pd.Series):
             raise TypeError(f"Expected a pandas Series for RSI calculation, got {type(price_series)}")

        rsi = ta.momentum.rsi(price_series, window=window, fillna=False)
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None

def detect_volatility_regime(data: pd.DataFrame, vix_column: str = 'VIX_Adj_Close', threshold: float = 25.0) -> Optional[pd.Series]:
    """
    Detects the volatility regime based on the VIX level.

    Args:
        data: DataFrame containing the VIX data.
        vix_column: The name of the column containing the VIX data.
        threshold: The VIX level threshold to distinguish between low and high volatility.

    Returns:
        A pandas Series containing the volatility regime (e.g., 0 for low, 1 for high),
        or None if calculation fails.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None.")
        return None
    if vix_column not in data.columns:
        logger.error(f"VIX column '{vix_column}' not found in DataFrame.")
        return None
    if threshold <= 0:
        logger.error("Volatility threshold must be positive.")
        return None

    try:
        logger.debug(f"Detecting volatility regime using VIX column '{vix_column}' and threshold {threshold}") # DEBUG
        # High volatility regime = 1, Low volatility regime = 0
        volatility_regime = (data[vix_column] > threshold).astype(int)
        return volatility_regime
    except Exception as e:
        logger.error(f"Error detecting volatility regime: {e}")
        return None


def calculate_atr(data: pd.DataFrame, window: int = 14,
                  high_col: str = 'SP500_High', low_col: str = 'SP500_Low', close_col: str = 'SP500_Close') -> Optional[pd.Series]:
    """
    Calculates the Average True Range (ATR).

    Args:
        data: DataFrame containing High, Low, and Close price data.
        window: The ATR window period (e.g., 14).
        high_col: Name of the High price column.
        low_col: Name of the Low price column.
        close_col: Name of the Close price column.

    Returns:
        A pandas Series containing the calculated ATR, or None if calculation fails.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None for ATR calculation.")
        return None
    required_cols = [high_col, low_col, close_col]
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Missing one or more required columns for ATR: {required_cols}")
        return None
    if window <= 0:
        logger.error("Window must be a positive integer for ATR.")
        return None

    try:
        logger.debug(f"Calculating {window}-period ATR using columns '{high_col}', '{low_col}', '{close_col}'") # DEBUG

        # Check for sufficient data length BEFORE accessing/squeezing
        if len(data) < window:
             logger.warning(f"Insufficient data ({len(data)} rows) for ATR calculation with window {window}. Returning NaNs.")
             # Return a Series of NaNs with the same index as the input data
             return pd.Series(np.nan, index=data.index)

        # Ensure inputs are Series
        high_series = data[high_col]
        low_series = data[low_col]
        close_series = data[close_col]

        if isinstance(high_series, pd.DataFrame): high_series = high_series.squeeze()
        if isinstance(low_series, pd.DataFrame): low_series = low_series.squeeze()
        if isinstance(close_series, pd.DataFrame): close_series = close_series.squeeze()

        if not all(isinstance(s, pd.Series) for s in [high_series, low_series, close_series]):
             raise TypeError("Expected pandas Series for High, Low, Close in ATR calculation.")

        atr = ta.volatility.average_true_range(high_series, low_series, close_series, window=window, fillna=False)
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None


def add_indicators_to_data(data: pd.DataFrame, ma_window: int = 40, rsi_window: int = 14, atr_window: int = 14,
                           vix_threshold: float = 25.0, price_column: str = 'SP500_Close', # Default to Close
                           high_col: str = 'SP500_High', low_col: str = 'SP500_Low', close_col: str = 'SP500_Close',
                           vix_column: str = 'VIX_Close') -> Optional[pd.DataFrame]: # Default to Close
    """
    Adds MA, RSI, ATR, and Volatility Regime indicators to the input DataFrame.

    Args:
        data: The input DataFrame (should be resampled with OHLC columns).
        ma_window: Window for Moving Average calculation.
        rsi_window: Window for RSI calculation.
        atr_window: Window for ATR calculation.
        vix_threshold: Threshold for VIX volatility regime detection.
        price_column: Column name for price data (used for MA, RSI - typically Adj Close or Close).
        high_col: Column name for High prices (used for ATR).
        low_col: Column name for Low prices (used for ATR).
        close_col: Column name for Closing prices (used for ATR).
        vix_column: Column name for VIX prices.

    Returns:
        DataFrame with added indicator columns (e.g., 'MA_40', 'RSI_14', 'ATR_14', 'Vol_Regime'),
        or None if any indicator calculation fails.
    """
    if data is None:
        logger.error("Input data is None.")
        return None

    data_with_indicators = data.copy()

    # Calculate MA
    ma_series = calculate_moving_average(data_with_indicators, window=ma_window, price_column=price_column)
    if ma_series is None: return None # Error logged in calculate_moving_average
    data_with_indicators[f'MA_{ma_window}'] = ma_series

    # Calculate RSI
    rsi_series = calculate_rsi(data_with_indicators, window=rsi_window, price_column=price_column)
    if rsi_series is None: return None # Error logged in calculate_rsi
    data_with_indicators[f'RSI_{rsi_window}'] = rsi_series

    # Detect Volatility Regime
    vol_regime_series = detect_volatility_regime(data_with_indicators, vix_column=vix_column, threshold=vix_threshold)
    if vol_regime_series is None: return None # Error logged in detect_volatility_regime
    data_with_indicators['Vol_Regime'] = vol_regime_series

    # Calculate ATR
    atr_series = calculate_atr(data_with_indicators, window=atr_window, high_col=high_col, low_col=low_col, close_col=close_col)
    if atr_series is None: return None # Error logged in calculate_atr
    data_with_indicators[f'ATR_{atr_window}'] = atr_series

    logger.debug("Successfully added MA, RSI, Volatility Regime, and ATR indicators.") # DEBUG
    return data_with_indicators

# Removed the __main__ block as it depended on data_handler