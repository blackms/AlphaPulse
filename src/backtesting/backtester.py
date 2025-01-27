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


@dataclass
class Position:
    """Represents a trading position."""
    entry_price: float
    entry_time: datetime
    size: float
    pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None


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
        self.commission = commission
        self.initial_capital = initial_capital
        self.position_size = position_size
        self._reset()

    def _reset(self) -> None:
        """Reset the backtester state."""
        self.equity = self.initial_capital
        self.positions: List[Position] = []
        self.current_position: Optional[Position] = None
        self.equity_curve: List[float] = [self.initial_capital]

    def _calculate_position_size(self, price: float) -> float:
        """Calculate the position size based on current equity."""
        return (self.equity * self.position_size) / price

    def _update_equity(self, pnl: float) -> None:
        """Update equity with realized PnL."""
        self.equity += pnl
        self.equity_curve.append(self.equity)

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

        returns = []
        
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
                    
                    self._update_equity(pnl)
                    self.positions.append(self.current_position)
                    self.current_position = None
                    
                    returns.append(pnl / entry_value)
            
            # Check for new entry
            if self.current_position is None and strategy.should_enter(signal):
                size = self._calculate_position_size(price)
                self.current_position = Position(
                    entry_price=price,
                    entry_time=timestamp,
                    size=size
                )

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
            
            self._update_equity(pnl)
            self.positions.append(self.current_position)
            returns.append(pnl / entry_value)

        # Calculate performance metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        
        if not returns:  # No trades made
            logger.warning("No trades were executed during the backtest")
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
                equity_curve=pd.Series(self.equity_curve, index=prices.index)
            )

        returns = np.array(returns)
        winning_trades = len([p for p in self.positions if p.pnl > 0])
        losing_trades = len([p for p in self.positions if p.pnl < 0])
        
        # Calculate Sharpe ratio (assuming daily data)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        
        # Calculate maximum drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
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
            win_rate=winning_trades / len(self.positions),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            positions=self.positions,
            equity_curve=pd.Series(self.equity_curve, index=prices.index)
        )