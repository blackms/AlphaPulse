"""
Core backtesting implementation for AlphaPulse trading strategies.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from .strategy import BaseStrategy, DefaultStrategy
from .models import Position


@dataclass
class BacktestResult:
    """Contains the results of a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    positions: List[Position]
    equity_curve: pd.Series

    def __str__(self) -> str:
        """Returns a formatted string of the backtest results."""
        return (
            f"Backtest Results:\n"
            f"Total Return: {self.total_return:.2%}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Total Trades: {self.total_trades}\n"
            f"Win Rate: {self.win_rate:.2%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Avg Win: {self.avg_win:.2%}\n"
            f"Avg Loss: {self.avg_loss:.2%}\n"
        )


class Backtester:
    """Main backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        commission: float = 0.001,  # 0.1% commission per trade
        initial_capital: float = 100000.0,
        position_size: float = 1.0,  # Fraction of capital to risk per trade
    ):
        """
        Initialize the backtester with trading parameters.

        Args:
            commission: Trading commission as a fraction (e.g., 0.001 for 0.1%)
            initial_capital: Starting capital for the backtest
            position_size: Fraction of capital to risk per trade (0.0 to 1.0)
        """
        if position_size < 0.0 or position_size > 1.0:
            raise ValueError("Position size must be between 0.0 and 1.0")
        
        self.commission = commission
        self.initial_capital = initial_capital
        self.position_size = position_size
        self._reset()

    def _reset(self) -> None:
        """Reset the backtester state."""
        self.equity = self.initial_capital
        self.positions: List[Position] = []
        self.current_position: Optional[Position] = None
        self.equity_curve: Dict[datetime, float] = {}

    def _calculate_position_size(self, price: float) -> float:
        """Calculate the position size based on current equity."""
        if self.position_size == 0.0:
            return 0.0
        return (self.equity * self.position_size) / price

    def _update_equity(self, timestamp: datetime, pnl: float = 0.0) -> None:
        """Update equity with realized PnL."""
        self.equity += pnl
        self.equity_curve[timestamp] = self.equity

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate the Sharpe ratio from a series of returns.
        
        Args:
            returns: Array of trade returns

        Returns:
            float: Annualized Sharpe ratio (assuming daily data)
        """
        if len(returns) < 2:  # Need at least 2 returns for std calculation
            return 0.0
        
        std_dev = np.std(returns, ddof=1)  # Use sample standard deviation
        if std_dev == 0:
            return 0.0
            
        return np.sqrt(252) * np.mean(returns) / std_dev

    def backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        strategy: Optional[BaseStrategy] = None,
    ) -> BacktestResult:
        """
        Run a backtest using price data and predicted signals.

        Args:
            prices: Time series of asset prices
            signals: Time series of trading signals (same index as prices)
            strategy: Trading strategy to use (defaults to DefaultStrategy)

        Returns:
            BacktestResult containing performance metrics and trade history
        """
        logger.info("Starting backtest...")
        self._reset()
        
        if strategy is None:
            strategy = DefaultStrategy()

        if not prices.index.equals(signals.index):
            raise ValueError("Price and signal data must have matching timestamps")

        if (prices <= 0).any():
            raise ValueError("Prices must be positive")

        # Skip backtesting if position size is zero
        if self.position_size == 0.0:
            logger.warning("Position size is zero, no trades will be executed")
            equity_curve = pd.Series(
                self.initial_capital,
                index=prices.index
            )
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                positions=[],
                equity_curve=equity_curve
            )

        returns = []
        
        # Initialize equity curve with starting capital
        self._update_equity(prices.index[0])
        
        for timestamp, price in prices.items():
            signal = signals[timestamp]
            
            # Check for exit of existing position
            if self.current_position is not None:
                if strategy.should_exit(signal, self.current_position):
                    # Calculate PnL including commission
                    exit_value = (
                        self.current_position.size * price * (1 - self.commission)
                    )
                    entry_value = (
                        self.current_position.size 
                        * self.current_position.entry_price 
                        * (1 + self.commission)
                    )
                    pnl = exit_value - entry_value
                    
                    self.current_position.exit_price = price
                    self.current_position.exit_time = timestamp
                    self.current_position.pnl = pnl
                    
                    self._update_equity(timestamp, pnl)
                    self.positions.append(self.current_position)
                    self.current_position = None
                    
                    if entry_value != 0:  # Avoid division by zero
                        returns.append(pnl / entry_value)
                else:
                    # Update equity curve with unrealized PnL
                    unrealized_pnl = (
                        self.current_position.size * 
                        (price - self.current_position.entry_price)
                    )
                    self._update_equity(timestamp)
            else:
                # Check for new entry
                if strategy.should_enter(signal):
                    size = self._calculate_position_size(price)
                    if size > 0:  # Only enter if position size is positive
                        self.current_position = Position(
                            entry_price=price,
                            entry_time=timestamp,
                            size=size
                        )
                self._update_equity(timestamp)

        # Close any remaining position at the end
        if self.current_position is not None:
            last_price = prices.iloc[-1]
            exit_value = (
                self.current_position.size * last_price * (1 - self.commission)
            )
            entry_value = (
                self.current_position.size 
                * self.current_position.entry_price 
                * (1 + self.commission)
            )
            pnl = exit_value - entry_value
            
            self.current_position.exit_price = last_price
            self.current_position.exit_time = prices.index[-1]
            self.current_position.pnl = pnl
            
            self._update_equity(prices.index[-1], pnl)
            self.positions.append(self.current_position)
            
            if entry_value != 0:  # Avoid division by zero
                returns.append(pnl / entry_value)

        # Calculate performance metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        
        if not returns:  # No trades made
            logger.warning("No trades were executed during the backtest")
            equity_curve = pd.Series(
                self.initial_capital,
                index=prices.index
            )
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                positions=[],
                equity_curve=equity_curve
            )

        returns = np.array(returns)
        winning_trades = len([p for p in self.positions if p.pnl > 0])
        losing_trades = len([p for p in self.positions if p.pnl < 0])
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Create equity curve series
        equity_curve = pd.Series(self.equity_curve)
        
        # Calculate maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Calculate average win/loss
        wins = [p.pnl for p in self.positions if p.pnl > 0]
        losses = [p.pnl for p in self.positions if p.pnl < 0]
        
        avg_win = np.mean(wins) / self.initial_capital if wins else 0.0
        avg_loss = abs(np.mean(losses)) / self.initial_capital if losses else 0.0
        
        # Calculate profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        logger.info(f"Backtest completed. Total return: {total_return:.2%}")
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(self.positions),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=winning_trades / len(self.positions) if self.positions else 0.0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            positions=self.positions,
            equity_curve=equity_curve
        )