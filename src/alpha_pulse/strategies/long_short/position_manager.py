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

    def calculate_target_position(self, signal_data: pd.DataFrame, signal_column: str = 'Composite_Signal') -> Optional[pd.Series]:
        """
        Calculates the target position allocation based on the composite signal.

        The target position represents the desired allocation fraction (-1.0 to 1.0).
        - Signal > long_threshold: Long position, size proportional to signal strength.
        - Signal < short_threshold: Short position, size proportional to signal strength.
        - Otherwise: Flat position (0).

        Args:
            signal_data: DataFrame containing the composite signal.
            signal_column: Name of the column containing the composite signal.

        Returns:
            A pandas Series representing the target position allocation (-1.0 to 1.0)
            for each period, or None on error.
        """
        if signal_data is None or signal_data.empty:
            logger.error("Input signal data is empty or None.")
            return None
        if signal_column not in signal_data.columns:
            logger.error(f"Signal column '{signal_column}' not found in DataFrame.")
            return None

        try:
            logger.debug(f"Calculating target position based on signal '{signal_column}'") # DEBUG
            signals = signal_data[signal_column]
            target_position = pd.Series(0.0, index=signals.index) # Default to flat

            # Apply thresholds
            long_mask = signals > self.long_threshold
            short_mask = signals < self.short_threshold

            # Target position is directly the signal value where thresholds are met
            # The signal value already incorporates adjustments (MR, Volatility)
            target_position[long_mask] = signals[long_mask]
            target_position[short_mask] = signals[short_mask]

            # Ensure values are clipped between -1 and 1 (should already be the case from signal gen)
            target_position = np.clip(target_position, -1.0, 1.0)

            # --- TRACE LOGGING ---
            if not target_position.empty:
                 latest_target = target_position.iloc[-1]
                 latest_signal = signal_data[signal_column].iloc[-1] if signal_column in signal_data else 'N/A'
                 logger.trace(f"Latest Target Position: {latest_target:.2f} (based on signal: {latest_signal:.4f}, thresholds: L={self.long_threshold}, S={self.short_threshold})") # TRACE
            # --- END TRACE LOGGING ---

            logger.debug("Target position calculation complete.") # DEBUG
            return target_position

        except Exception as e:
            logger.error(f"Error calculating target position: {e}")
            return None

# Removed the __main__ block as it depended on data_handler