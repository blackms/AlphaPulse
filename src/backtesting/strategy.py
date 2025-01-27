"""
Trading strategy definitions for the AlphaPulse backtesting framework.
"""
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from .backtester import Position


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def should_enter(self, signal: float) -> bool:
        """
        Determine if a new position should be entered.

        Args:
            signal: The current trading signal value

        Returns:
            bool: True if a position should be entered, False otherwise
        """
        pass

    @abstractmethod
    def should_exit(self, signal: float, position: Position) -> bool:
        """
        Determine if the current position should be exited.

        Args:
            signal: The current trading signal value
            position: The current open position

        Returns:
            bool: True if the position should be exited, False otherwise
        """
        pass


class DefaultStrategy(BaseStrategy):
    """
    Default trading strategy implementation.
    Goes long when signal is positive, exits when signal turns negative.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Initialize the default strategy.

        Args:
            threshold: Signal threshold for trade decisions (default: 0.0)
        """
        self.threshold = threshold
        logger.info(f"Initialized DefaultStrategy with threshold {threshold}")

    def should_enter(self, signal: float) -> bool:
        """Enter long position when signal is above threshold."""
        return signal > self.threshold

    def should_exit(self, signal: float, position: Position) -> bool:
        """Exit long position when signal drops below threshold."""
        return signal <= self.threshold


class MeanReversionStrategy(BaseStrategy):
    """
    Example mean reversion strategy.
    Goes long when signal is below lower threshold, exits above upper threshold.
    """

    def __init__(self, lower_threshold: float = -1.0, upper_threshold: float = 1.0):
        """
        Initialize the mean reversion strategy.

        Args:
            lower_threshold: Signal threshold for entries (default: -1.0)
            upper_threshold: Signal threshold for exits (default: 1.0)
        """
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        logger.info(
            f"Initialized MeanReversionStrategy with thresholds: "
            f"lower={lower_threshold}, upper={upper_threshold}"
        )

    def should_enter(self, signal: float) -> bool:
        """Enter long position when signal drops below lower threshold."""
        return signal < self.lower_threshold

    def should_exit(self, signal: float, position: Position) -> bool:
        """Exit long position when signal rises above upper threshold."""
        return signal > self.upper_threshold


class TrendFollowingStrategy(BaseStrategy):
    """
    Example trend following strategy.
    Goes long when signal crosses above threshold, exits when crosses below.
    """

    def __init__(
        self, 
        entry_threshold: float = 0.5, 
        exit_threshold: float = -0.5
    ):
        """
        Initialize the trend following strategy.

        Args:
            entry_threshold: Signal threshold for entries (default: 0.5)
            exit_threshold: Signal threshold for exits (default: -0.5)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        logger.info(
            f"Initialized TrendFollowingStrategy with thresholds: "
            f"entry={entry_threshold}, exit={exit_threshold}"
        )

    def should_enter(self, signal: float) -> bool:
        """Enter long position when signal exceeds entry threshold."""
        return signal > self.entry_threshold

    def should_exit(self, signal: float, position: Position) -> bool:
        """Exit long position when signal drops below exit threshold."""
        return signal < self.exit_threshold