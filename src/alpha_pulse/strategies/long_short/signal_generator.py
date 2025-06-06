#!/usr/bin/env python3
"""
Signal Generation Module for the Long/Short S&P 500 Strategy.

Combines trend-following, mean reversion, and volatility signals
to generate a composite trading signal.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LongShortSignalGenerator")

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
        logger.info(f"Generating trend signal based on '{price_column}' vs '{ma_column}'")
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
        logger.info(f"Generating mean reversion adjustment based on '{rsi_column}' (OB: {overbought_threshold}, OS: {oversold_threshold})")
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
        logger.info(f"Generating volatility adjustment based on '{vol_regime_column}'")
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

def generate_composite_signal(data: pd.DataFrame, weights: Dict[str, float] = None) -> Optional[pd.DataFrame]:
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
        DataFrame with intermediate signals and the final 'Composite_Signal' column
        (ranging potentially from -1.0 to 1.0), or None on error.
    """
    if data is None or data.empty:
        logger.error("Input data is empty or None for composite signal generation.")
        return None

    # Default weights if not provided
    if weights is None:
        weights = {'trend': 1.0, 'mean_reversion': 0.3, 'volatility': 1.0} # Volatility weight applied multiplicatively

    # --- Determine correct column names ---
    # Assuming default indicator names generated by indicators.py
    # TODO: Make this more robust by passing window sizes or getting names dynamically
    price_col = 'SP500_Close' # Use Close instead of Adj Close
    ma_col = 'MA_40'
    rsi_col = 'RSI_14'
    vol_regime_col = 'Vol_Regime'
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
    logger.info("Combining signals into composite score...")
    try:
        # Start with the base trend signal
        composite = signal_df['Trend_Signal'] * weights.get('trend', 1.0)

        # Apply mean reversion adjustment additively (scaled)
        # If trend is long (1) and RSI is overbought (MR=-1), composite decreases.
        # If trend is short (-1) and RSI is oversold (MR=1), composite increases (becomes less negative).
        composite += signal_df['MR_Adjustment'] * weights.get('mean_reversion', 0.3)

        # Clip the signal strength between -1 and 1 after MR adjustment
        composite = np.clip(composite, -1.0, 1.0)

        # Apply volatility adjustment multiplicatively
        # This scales the position size based on volatility
        composite *= signal_df['Vol_Adjustment'] # No weight needed here, it's a direct scaling factor

        # Final composite signal
        signal_df['Composite_Signal'] = composite

        # Handle NaNs introduced at the start due to indicators
        signal_df.fillna(0, inplace=True) # Default to neutral signal if components are NaN

        # --- DEBUG LOGGING ---
        if not signal_df.empty:
             latest_signals = signal_df.iloc[-1]
             logger.debug(f"Latest Signals: Trend={latest_signals.get('Trend_Signal', 'N/A'):.2f}, MR_Adj={latest_signals.get('MR_Adjustment', 'N/A'):.2f}, Vol_Adj={latest_signals.get('Vol_Adjustment', 'N/A'):.2f}, Composite={latest_signals.get('Composite_Signal', 'N/A'):.4f}")
        # --- END DEBUG LOGGING ---

        logger.info("Composite signal generated successfully.")
        return signal_df

    except Exception as e:
        logger.error(f"Error generating composite signal: {e}")
        return None

# Removed the __main__ block