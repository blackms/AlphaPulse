#!/usr/bin/env python3
"""
Position Management Module for the Long/Short S&P 500 Strategy.

Determines target position direction and size based on the composite signal.
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
logger = logging.getLogger("LongShortPositionManager")

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
        logger.info(f"PositionManager initialized. Long Threshold: {self.long_threshold}, Short Threshold: {self.short_threshold}")

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
            logger.info(f"Calculating target position based on signal '{signal_column}'")
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

            logger.info("Target position calculation complete.")
            return target_position

        except Exception as e:
            logger.error(f"Error calculating target position: {e}")
            return None

if __name__ == '__main__':
    # Example Usage (requires data_handler, indicators, signal_generator)
    try:
        from data_handler import LongShortDataHandler
        from indicators import add_indicators_to_data
        from signal_generator import generate_composite_signal
    except ImportError:
        print("Run this script from the 'src/strategies/long_short' directory or ensure modules are in PYTHONPATH.")
        exit()

    config_example = {"cache_dir": "./data/cache/long_short_test"}
    data_handler = LongShortDataHandler(config_example)

    start = "2010-01-01"
    end = "2023-12-31"

    daily_data = data_handler.get_combined_data(start, end)
    if daily_data is not None:
        weekly_data = data_handler.resample_data(daily_data, timeframe='W')
        if weekly_data is not None:
            weekly_data_with_indicators = add_indicators_to_data(weekly_data)
            if weekly_data_with_indicators is not None:
                signal_results = generate_composite_signal(weekly_data_with_indicators)
                if signal_results is not None:
                    print("\n--- Generated Signals ---")
                    print(signal_results.tail())

                    # Initialize Position Manager
                    pos_manager = PositionManager(config={'long_threshold': 0.1, 'short_threshold': -0.1})

                    # Calculate Target Positions
                    target_positions = pos_manager.calculate_target_position(signal_results)

                    if target_positions is not None:
                        print("\n--- Calculated Target Positions ---")
                        # Merge for context
                        final_df = pd.merge(weekly_data_with_indicators, signal_results, left_index=True, right_index=True)
                        final_df['Target_Position'] = target_positions
                        print(final_df[['SP500_Adj_Close', 'Composite_Signal', 'Target_Position']].tail(20))

                        print("\nTarget Position Distribution:")
                        print(final_df['Target_Position'].describe())
                        print(final_df['Target_Position'].value_counts(bins=10, sort=False))

                        print("\nPosition Management Example Complete.")
                    else:
                        print("Failed to calculate target positions.")
                else:
                    print("Failed to generate signals.")
            else:
                print("Failed to add indicators.")
        else:
            print("Failed to resample data.")
    else:
        print("Failed to fetch data.")