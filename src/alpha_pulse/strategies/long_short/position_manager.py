#!/usr/bin/env python3
"""
Position Management Module for the Long/Short S&P 500 Strategy.

Determines target position direction and size based on the composite signal.
"""

from loguru import logger # Use loguru
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Loguru logger is imported directly
# logger = logger.bind(name="LongShortPositionManager") # Optional binding

class PositionManager:
    """
    Manages target position calculation based on trading signals.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the Position Manager.

        Args:
            config: Configuration dictionary (optional). Can include thresholds.
                    Example: {'long_threshold': 0.1, 'short_threshold': -0.1}
        """
        self.config = config if config else {}
        self.long_threshold = self.config.get('long_threshold', 0.1)
        self.short_threshold = self.config.get('short_threshold', -0.1)
        # Note: Position sizing logic might depend on portfolio context (e.g., capital)
        # For now, target_position represents desired allocation (-1 to 1)
        logger.debug(f"PositionManager initialized. Long Threshold: {self.long_threshold}, Short Threshold: {self.short_threshold}") # DEBUG

    def calculate_target_position(self, signal_data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculates the target position allocation based on the core signal and volatility adjustment.

        The target position represents the desired allocation fraction (-1.0 to 1.0).
        - Core_Signal > long_threshold: Long position considered.
        - Core_Signal < short_threshold: Short position considered.
        - Otherwise: Flat position (0).
        - Final position size is scaled by Vol_Adjustment.

        Args:
            signal_data: DataFrame containing 'Core_Signal' and 'Vol_Adjustment' columns.

        Returns:
            A pandas Series representing the target position allocation (-1.0 to 1.0)
            for each period, or None on error.
        """
        if signal_data is None or signal_data.empty:
            logger.error("Input signal data is empty or None.")
            return None
        required_cols = ['Core_Signal', 'Vol_Adjustment']
        if not all(col in signal_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in signal_data.columns]
            logger.error(f"Missing required columns {missing} in signal_data.")
            return None

        try:
            logger.debug("Calculating target position based on Core_Signal and Vol_Adjustment") # DEBUG
            core_signals = signal_data['Core_Signal']
            vol_adjustments = signal_data['Vol_Adjustment']
            initial_target_position = pd.Series(0.0, index=core_signals.index) # Default to flat

            # Apply thresholds to Core_Signal
            long_mask = core_signals > self.long_threshold
            short_mask = core_signals < self.short_threshold

            # Initial target position is the Core_Signal value where thresholds are met
            initial_target_position[long_mask] = core_signals[long_mask]
            initial_target_position[short_mask] = core_signals[short_mask]

            # Clip initial target between -1 and 1 (should already be the case from signal gen)
            initial_target_position = np.clip(initial_target_position, -1.0, 1.0)

            # Apply volatility adjustment to scale the position size
            target_position = initial_target_position * vol_adjustments

            # --- TRACE LOGGING ---
            if not target_position.empty:
                 latest_target = target_position.iloc[-1]
                 latest_core_signal = core_signals.iloc[-1]
                 latest_vol_adj = vol_adjustments.iloc[-1]
                 logger.trace(f"Latest Target Position: {latest_target:.2f} (Core Signal: {latest_core_signal:.4f}, Vol Adj: {latest_vol_adj:.2f}, Thresh: L={self.long_threshold}, S={self.short_threshold})") # TRACE
            # --- END TRACE LOGGING ---

            logger.debug("Target position calculation complete.") # DEBUG
            return target_position

        except Exception as e:
            logger.error(f"Error calculating target position: {e}")
            return None

# Removed the __main__ block as it depended on data_handler