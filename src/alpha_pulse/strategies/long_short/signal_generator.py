#!/usr/bin/env python3
"""
Signal Generation Module for the Long/Short S&P 500 Strategy.

Combines trend-following, mean reversion, and volatility signals
to generate a composite trading signal.
"""

from loguru import logger # Use loguru
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Loguru logger is imported directly
# logger = logger.bind(name="LongShortSignalGenerator") # Optional binding

def generate_trend_signal(data: pd.DataFrame, price_column: str, ma_column: str) -> Optional[pd.Series]:
    """
    Generates a trend-following signal based on price vs. moving average.

    Signal: 1 (long) if price > MA, -1 (short) if price < MA.

    Args:
        data: DataFrame containing price and MA data.
        price_column: Name of the price column.
        ma_column: Name of the moving average column.

    Returns:
        A pandas Series with the trend signal (1, -1, or 0 if MA is NaN), or None on error.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None for trend signal.")
        return None
    if price_column not in data.columns or ma_column not in data.columns:
        logger.error(f"Required columns ('{price_column}', '{ma_column}') not found for trend signal.")
        return None

    try:
        logger.debug(f"Generating trend signal based on '{price_column}' vs '{ma_column}'") # DEBUG
        # Initialize signal with 0
        signal = pd.Series(0.0, index=data.index) # Initialize with float
        price_series = data[price_column]
        ma_series = data[ma_column]

        # Align series to handle potential index mismatches (e.g., due to NaNs in MA)
        price_aligned, ma_aligned = price_series.align(ma_series, join='inner', copy=False)

        # Long signal where price > MA (only on aligned indices)
        long_condition = price_aligned > ma_aligned
        signal.loc[long_condition[long_condition].index] = 1.0

        # Short signal where price < MA (only on aligned indices)
        short_condition = price_aligned < ma_aligned
        signal.loc[short_condition[short_condition].index] = -1.0

        # Set signal to 0 where MA is NaN (typically at the start) - Use original ma_series for isna()
        signal[ma_series.isna()] = 0.0
        return signal
    except Exception as e:
        logger.error(f"Error generating trend signal: {e}")
        return None

def generate_mean_reversion_adjustment(data: pd.DataFrame, rsi_column: str,
                                       overbought_threshold: float = 70.0, oversold_threshold: float = 30.0) -> Optional[pd.Series]:
    """
    Generates a mean reversion adjustment factor based on RSI.

    Factor: -1 if RSI > overbought, 1 if RSI < oversold, 0 otherwise.
    This factor is intended to *reduce* the strength of the trend signal in extreme conditions.

    Args:
        data: DataFrame containing RSI data.
        rsi_column: Name of the RSI column.
        overbought_threshold: RSI level considered overbought.
        oversold_threshold: RSI level considered oversold.

    Returns:
        A pandas Series with the adjustment factor (-1, 1, 0), or None on error.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None for mean reversion adjustment.")
        return None
    if rsi_column not in data.columns:
        logger.error(f"Required column '{rsi_column}' not found for mean reversion adjustment.")
        return None

    try:
        logger.debug(f"Generating mean reversion adjustment based on '{rsi_column}' (OB: {overbought_threshold}, OS: {oversold_threshold})") # DEBUG
        adjustment = pd.Series(0, index=data.index)
        # Reduce long strength if overbought
        adjustment[data[rsi_column] > overbought_threshold] = -1
        # Reduce short strength if oversold
        adjustment[data[rsi_column] < oversold_threshold] = 1
        # Set adjustment to 0 where RSI is NaN
        adjustment[data[rsi_column].isna()] = 0
        return adjustment
    except Exception as e:
        logger.error(f"Error generating mean reversion adjustment: {e}")
        return None

def generate_volatility_adjustment(data: pd.DataFrame, vol_regime_column: str) -> Optional[pd.Series]:
    """
    Generates a volatility adjustment factor based on the volatility regime.

    Factor: 0.5 if high volatility (regime=1), 1.0 if low volatility (regime=0).
    This factor is intended to scale down the position size during high volatility.

    Args:
        data: DataFrame containing volatility regime data.
        vol_regime_column: Name of the volatility regime column (0 or 1).

    Returns:
        A pandas Series with the volatility adjustment factor (0.5 or 1.0), or None on error.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None for volatility adjustment.")
        return None
    if vol_regime_column not in data.columns:
        logger.error(f"Required column '{vol_regime_column}' not found for volatility adjustment.")
        return None

    try:
        logger.debug(f"Generating volatility adjustment based on '{vol_regime_column}'") # DEBUG
        # Default to 1.0 (low volatility)
        adjustment = pd.Series(1.0, index=data.index)
        # Reduce factor to 0.5 if high volatility (regime == 1)
        adjustment[data[vol_regime_column] == 1] = 0.5
        # Handle potential NaNs in regime column if necessary (shouldn't happen with current indicator logic)
        adjustment[data[vol_regime_column].isna()] = 1.0 # Default to no adjustment if regime is unknown
        return adjustment
    except Exception as e:
        logger.error(f"Error generating volatility adjustment: {e}")
        return None

def generate_composite_signal(
    data: pd.DataFrame,
    weights: Dict[str, float] = None,
    ma_window: int = 40, # Add MA window parameter
    rsi_window: int = 14  # Add RSI window parameter
) -> Optional[pd.DataFrame]:
    """
    Generates the final composite signal by combining individual signals and adjustments.

    Steps:
    1. Calculate base trend signal (1 or -1).
    2. Calculate mean reversion adjustment (-1, 0, 1).
    3. Calculate volatility adjustment (0.5 or 1.0).
    4. Combine:
       - Apply mean reversion: If trend is long (1) and MR is overbought (-1), signal becomes weaker (e.g., 0.5).
                               If trend is short (-1) and MR is oversold (1), signal becomes weaker (e.g., -0.5).
                               (Simple approach: Add MR adjustment scaled by a weight)
       - Apply volatility: Multiply the resulting signal strength by the volatility factor.

    Args:
        data: DataFrame containing price, MA, RSI, and Volatility Regime columns.
        weights: Dictionary defining weights for combining signals (e.g., {'trend': 1.0, 'mean_reversion': 0.3}).

    Returns:
        DataFrame with intermediate signals ('Trend_Signal', 'MR_Adjustment'),
        the volatility adjustment factor ('Vol_Adjustment'), and the core signal
        strength ('Core_Signal') before volatility scaling, or None on error.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None for composite signal generation.")
        return None

    # Default weights if not provided
    if weights is None:
        weights = {'trend': 1.0, 'mean_reversion': 0.3, 'volatility': 1.0} # Volatility weight applied multiplicatively

    # --- Determine correct column names ---
    price_col = 'SP500_Close' # Assuming this prefix is consistent from _prepare_input_data
    ma_col = f'MA_{ma_window}' # Dynamic MA column name
    rsi_col = f'RSI_{rsi_window}' # Dynamic RSI column name
    vol_regime_col = 'Vol_Regime' # This one is likely fixed
    required_cols = [price_col, ma_col, rsi_col, vol_regime_col]
    # --- End Determine correct column names ---

    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        logger.error(f"Missing one or more required columns for composite signal: {missing}. Available: {data.columns.tolist()}")
        # Attempt to infer column names if defaults are not present (more robust)
        # This part could be enhanced based on how indicators.py names columns
        # For now, rely on default names from indicators.py
        return None

    signal_df = pd.DataFrame(index=data.index)

    # 1. Generate Trend Signal (using correct columns)
    signal_df['Trend_Signal'] = generate_trend_signal(data, price_column=price_col, ma_column=ma_col)
    if signal_df['Trend_Signal'] is None: return None

    # 2. Generate Mean Reversion Adjustment (using correct columns)
    signal_df['MR_Adjustment'] = generate_mean_reversion_adjustment(data, rsi_column=rsi_col)
    if signal_df['MR_Adjustment'] is None: return None

    # 3. Generate Volatility Adjustment Factor (using correct columns)
    signal_df['Vol_Adjustment'] = generate_volatility_adjustment(data, vol_regime_column=vol_regime_col)
    if signal_df['Vol_Adjustment'] is None: return None

    # 4. Combine Signals
    logger.debug("Combining signals into core signal...") # DEBUG
    try:
        # Calculate core signal based on Trend and Mean Reversion weights
        core_signal = signal_df['Trend_Signal'] * weights.get('trend', 1.0)
        core_signal += signal_df['MR_Adjustment'] * weights.get('mean_reversion', 0.3)

        # Clip the core signal strength between -1 and 1
        core_signal = np.clip(core_signal, -1.0, 1.0)

        # Store the core signal (before volatility adjustment)
        signal_df['Core_Signal'] = core_signal
        # Vol_Adjustment is already in signal_df

        # Handle NaNs introduced at the start due to indicators
        # Apply fillna individually to avoid overwriting Vol_Adjustment with 0
        signal_df['Trend_Signal'].fillna(0, inplace=True)
        signal_df['MR_Adjustment'].fillna(0, inplace=True)
        signal_df['Core_Signal'].fillna(0, inplace=True)
        # Vol_Adjustment should already handle its NaNs (defaulting to 1.0)

        # --- TRACE LOGGING ---
        if not signal_df.empty:
             latest_signals = signal_df.iloc[-1]
             logger.trace(f"Latest Signals: Trend={latest_signals.get('Trend_Signal', 'N/A'):.2f}, MR_Adj={latest_signals.get('MR_Adjustment', 'N/A'):.2f}, Vol_Adj={latest_signals.get('Vol_Adjustment', 'N/A'):.2f}, Core={latest_signals.get('Core_Signal', 'N/A'):.4f}") # TRACE
        # --- END TRACE LOGGING ---

        logger.debug("Core signal and Volatility Adjustment generated successfully.") # DEBUG
        return signal_df # Return DataFrame containing Core_Signal and Vol_Adjustment

    except Exception as e:
        logger.error(f"Error generating composite signal: {e}")
        return None

# Removed the __main__ block