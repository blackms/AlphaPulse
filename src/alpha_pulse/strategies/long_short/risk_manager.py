#!/usr/bin/env python3
"""
Risk Management Module for the Long/Short S&P 500 Strategy.

Defines rules for stop-loss calculation and potentially drawdown control.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LongShortRiskManager")

class RiskManager:
    """
    Manages risk parameters like stop-loss levels.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the Risk Manager.

        Args:
            config: Configuration dictionary (optional). Example:
                    {
                        'stop_loss_type': 'atr', # 'percentage' or 'atr'
                        'stop_loss_pct': 0.02,   # e.g., 2% stop loss
                        'stop_loss_atr_multiplier': 2.0 # e.g., 2 * ATR
                    }
        """
        self.config = config if config else {}
        self.stop_loss_type = self.config.get('stop_loss_type', 'atr') # Default to ATR
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.stop_loss_atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)

        if self.stop_loss_type not in ['percentage', 'atr']:
            logger.warning(f"Invalid stop_loss_type '{self.stop_loss_type}'. Defaulting to 'atr'.")
            self.stop_loss_type = 'atr'

        logger.info(f"RiskManager initialized. Stop Loss Type: {self.stop_loss_type}, "
                    f"Percent: {self.stop_loss_pct}, ATR Multiplier: {self.stop_loss_atr_multiplier}")

    def calculate_stop_loss(self, entry_price: float, direction: Literal['long', 'short'],
                            current_atr: Optional[float] = None) -> Optional[float]:
        """
        Calculates the stop-loss price based on the configured method.

        Args:
            entry_price: The price at which the position was entered.
            direction: The direction of the trade ('long' or 'short').
            current_atr: The current ATR value (required if stop_loss_type is 'atr').

        Returns:
            The calculated stop-loss price, or None if calculation is not possible.
        """
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            logger.error(f"Invalid entry_price: {entry_price}")
            return None
        if direction not in ['long', 'short']:
            logger.error(f"Invalid direction: {direction}")
            return None

        stop_loss_price = None

        if self.stop_loss_type == 'percentage':
            if direction == 'long':
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            else: # short
                stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            logger.debug(f"Calculated percentage stop loss: {stop_loss_price} for {direction} entry at {entry_price}")

        elif self.stop_loss_type == 'atr':
            if current_atr is None or not isinstance(current_atr, (int, float)) or current_atr <= 0:
                logger.error(f"ATR stop loss selected, but valid current_atr was not provided or is invalid: {current_atr}")
                return None
            atr_offset = self.stop_loss_atr_multiplier * current_atr
            if direction == 'long':
                stop_loss_price = entry_price - atr_offset
            else: # short
                stop_loss_price = entry_price + atr_offset
            logger.debug(f"Calculated ATR stop loss: {stop_loss_price} (ATR={current_atr}, Mult={self.stop_loss_atr_multiplier}) for {direction} entry at {entry_price}")

        # Ensure stop loss is not negative (though unlikely for SP500)
        if stop_loss_price is not None and stop_loss_price < 0:
            stop_loss_price = 0.0

        return stop_loss_price

    def update_trailing_stop(
        self,
        current_stop_loss: float,
        high_since_entry: float,
        low_since_entry: float,
        direction: Literal['long', 'short'],
        current_atr: Optional[float] = None
    ) -> float:
        """
        Updates the trailing stop loss level based on the high/low price reached.

        The stop loss only moves in the direction of the trade (up for longs, down for shorts).

        Args:
            current_stop_loss: The current stop loss level for the position.
            high_since_entry: The highest price reached since the position was opened.
            low_since_entry: The lowest price reached since the position was opened.
            direction: The direction of the trade ('long' or 'short').
            current_atr: The current ATR value (required if stop_loss_type is 'atr').

        Returns:
            The updated (or original) stop loss level.
        """
        if pd.isna(current_stop_loss): # Cannot trail if no initial stop exists
             return current_stop_loss

        new_potential_stop = None

        if self.stop_loss_type == 'percentage':
            if direction == 'long':
                new_potential_stop = high_since_entry * (1 - self.stop_loss_pct)
            else: # short
                new_potential_stop = low_since_entry * (1 + self.stop_loss_pct)
            logger.debug(f"Trailing % Stop: Potential new stop={new_potential_stop:.2f} based on high/low={high_since_entry:.2f}/{low_since_entry:.2f}")

        elif self.stop_loss_type == 'atr':
            if current_atr is None or not isinstance(current_atr, (int, float)) or current_atr <= 0:
                logger.warning(f"Cannot update ATR trailing stop: valid current_atr not provided or invalid: {current_atr}")
                return current_stop_loss # Return current stop if ATR is invalid
            atr_offset = self.stop_loss_atr_multiplier * current_atr
            if direction == 'long':
                new_potential_stop = high_since_entry - atr_offset
            else: # short
                new_potential_stop = low_since_entry + atr_offset
            logger.debug(f"Trailing ATR Stop: Potential new stop={new_potential_stop:.2f} based on high/low={high_since_entry:.2f}/{low_since_entry:.2f}, ATR={current_atr:.2f}")

        # Ensure potential stop is not negative
        if new_potential_stop is not None and new_potential_stop < 0:
            new_potential_stop = 0.0

        # Update stop only if it moves favorably
        updated_stop_loss = current_stop_loss
        if new_potential_stop is not None:
            if direction == 'long' and new_potential_stop > current_stop_loss:
                updated_stop_loss = new_potential_stop
                logger.debug(f"Trailing Stop Updated (Long): {current_stop_loss:.2f} -> {updated_stop_loss:.2f}")
            elif direction == 'short' and new_potential_stop < current_stop_loss:
                updated_stop_loss = new_potential_stop
                logger.debug(f"Trailing Stop Updated (Short): {current_stop_loss:.2f} -> {updated_stop_loss:.2f}")

        return updated_stop_loss

    def check_drawdown(self, equity_curve: pd.Series) -> bool:
        """
        Placeholder for drawdown control logic.

        In a real implementation, this would analyze the equity curve
        and potentially trigger risk reduction measures.

        Args:
            equity_curve: A pandas Series representing the portfolio equity over time.

        Returns:
            Boolean indicating if a drawdown limit has been breached (example).
        """
        # TODO: Implement actual drawdown calculation and threshold check
        logger.warning("Drawdown control check is currently a placeholder.")
        # Example: Check if drawdown exceeds 20%
        # peak = equity_curve.expanding(min_periods=1).max()
        # drawdown = (equity_curve - peak) / peak
        # max_drawdown = drawdown.min()
        # if max_drawdown < -0.20:
        #     logger.warning(f"Drawdown limit breached: {max_drawdown:.2%}")
        #     return True
        return False

    # Potential future addition: Position Sizing Limits
    # def check_position_size_limit(self, current_position_size, max_allowed_size):
    #     pass

# Removed the __main__ block