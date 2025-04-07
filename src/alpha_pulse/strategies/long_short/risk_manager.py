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