"""
Core backtesting implementation for AlphaPulse trading strategies.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

# Keep for potential future use? Or remove if BaseStrategy is not used here.
# from .strategy import BaseStrategy, DefaultStrategy
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
    avg_win: float # As percentage of initial capital
    avg_loss: float # As percentage of initial capital
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
            f"Avg Win Pct: {self.avg_win:.2%}\n"
            f"Avg Loss Pct: {self.avg_loss:.2%}\n"
        )


class Backtester:
    """Main backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        commission: float = 0.001,  # 0.1% commission per trade
        initial_capital: float = 100000.0,
        # position_size: float = 1.0, # No longer used directly, allocation comes from signals
    ):
        """
        Initialize the backtester with trading parameters.

        Args:
            commission: Trading commission as a fraction (e.g., 0.001 for 0.1%)
            initial_capital: Starting capital for the backtest
        """
        # if position_size < 0.0 or position_size > 1.0:
        #     raise ValueError("Position size must be between 0.0 and 1.0")

        self.commission = commission
        self.initial_capital = initial_capital
        # self.position_size = position_size # Removed
        self._reset()

    def _reset(self) -> None:
        """Reset the backtester state."""
        self.equity = self.initial_capital # Tracks cash + realized PnL
        self.positions: List[Position] = []
        self.current_position: Optional[Position] = None
        self.equity_curve: Dict[datetime, float] = {} # Stores mark-to-market equity

    # def _calculate_position_size(self, price: float) -> float: # No longer needed with target allocation
    #     """Calculate the position size based on current equity."""
    #     if self.position_size == 0.0:
    #         return 0.0
    #     return (self.equity * self.position_size) / price

    def _update_equity(self, timestamp: datetime, realized_pnl: float = 0.0) -> None:
        """
        Update equity (cash + realized PnL) component.
        Note: Equity curve (mark-to-market) is updated separately in the main loop.
        """
        self.equity += realized_pnl
        # We update the equity curve dict at the end of each day in the main loop
        # self.equity_curve[timestamp] = self.equity # Don't do this here

    def _calculate_sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate the annualized Sharpe ratio from a series of trade returns.

        Args:
            returns: Array of individual trade returns (e.g., pnl / cost_basis)
            periods_per_year: Number of trading periods in a year (e.g., 252 for daily)

        Returns:
            float: Annualized Sharpe ratio (assuming 0 risk-free rate)
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_dev = np.std(returns, ddof=1)

        if std_dev == 0:
             # If std dev is 0, Sharpe is undefined or infinite if mean > 0. Return 0 for simplicity.
             return 0.0

        # Annualize (assuming 0 risk-free rate)
        sharpe = mean_return / std_dev
        annualized_sharpe = sharpe * np.sqrt(periods_per_year)
        return annualized_sharpe

    def _generate_empty_result(self, index: pd.Index) -> 'BacktestResult':
        """Helper to return an empty result set."""
        equity_curve = pd.Series(self.initial_capital, index=index)
        return BacktestResult(
            total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            positions=[], equity_curve=equity_curve
        )

    def backtest(
        self,
        prices: pd.Series,
        signals: pd.Series, # Interpreted as TARGET ALLOCATION (-1.0 to +1.0)
        stop_losses: Optional[pd.Series] = None, # Optional series of stop-loss prices
    ) -> BacktestResult:
        """
        Run a backtest using price data and target allocation signals.

        Args:
            prices: Time series of asset prices (e.g., Close prices).
            signals: Time series of target allocation signals (-1.0 to +1.0).
            stop_losses: Optional time series of stop-loss prices for open positions.

        Returns:
            BacktestResult containing performance metrics and trade history.
        """
        logger.info("Starting backtest...")
        self._reset()

        # --- Input Alignment and Validation ---
        if not isinstance(prices, pd.Series) or not isinstance(signals, pd.Series):
             raise TypeError("Prices and signals must be pandas Series.")
        if not isinstance(prices.index, pd.DatetimeIndex) or not isinstance(signals.index, pd.DatetimeIndex):
             raise TypeError("Prices and signals must have a DatetimeIndex.")

        if not prices.index.equals(signals.index):
            logger.warning("Price and signal indices do not match. Aligning using inner join...")
            prices, signals = prices.align(signals, join='inner', copy=False)
            if prices.empty:
                raise ValueError("Price and signal data have no matching timestamps after alignment.")

        if stop_losses is not None:
            if not isinstance(stop_losses, pd.Series) or not isinstance(stop_losses.index, pd.DatetimeIndex):
                 raise TypeError("Stop losses must be a pandas Series with a DatetimeIndex.")
            if not prices.index.equals(stop_losses.index):
                logger.warning("Price and stop_loss indices do not match. Aligning...")
                # Align stop_losses to prices index, keeping all price points
                stop_losses = stop_losses.reindex(prices.index)
                # Forward fill stop losses after alignment
                stop_losses = stop_losses.ffill()
            else:
                 # Forward fill even if indices match initially, to handle potential NaNs
                 stop_losses = stop_losses.ffill()
        else:
             # Create a series of NaNs if no stop losses provided, to simplify loop logic
             stop_losses = pd.Series(np.nan, index=prices.index)

        if (prices <= 0).any():
            logger.error("Prices must be positive. Check data source.")
            first_invalid_idx = prices[prices <= 0].index[0]
            logger.error(f"First non-positive price found at: {first_invalid_idx}")
            return self._generate_empty_result(prices.index)

        if self.initial_capital <= 0:
             logger.error("Initial capital must be positive.")
             return self._generate_empty_result(prices.index)

        if not (-1.0 <= signals).all() or not (signals <= 1.0).all():
             logger.warning("Signals contain values outside the expected [-1.0, 1.0] range. Clipping.")
             signals = signals.clip(-1.0, 1.0)
        # --- End Input Validation ---


        trade_returns = [] # Stores returns of individual closed trades
        # Initialize equity curve with starting capital
        if not prices.empty:
            start_timestamp = prices.index[0]
            # Initialize equity curve dict with starting capital at the first timestamp
            self.equity_curve[start_timestamp] = self.initial_capital
            self.equity = self.initial_capital # Ensure equity starts correctly


        # --- Main Backtest Loop ---
        for timestamp, current_price in prices.items():
            # Skip first timestamp if it's the initialization point
            if timestamp == start_timestamp and len(self.equity_curve) == 1:
                 continue

            target_allocation = signals.loc[timestamp]
            stop_loss_level = stop_losses.loc[timestamp] # Already ffilled, will be NaN if not provided
            trade_executed_today = False
            realized_pnl_today = 0.0 # Track PnL realized today

            # --- 1. Check Stop Loss ---
            if self.current_position is not None and pd.notna(stop_loss_level):
                exit_price = None
                # Check if low/high price is available for more realistic SL check
                # For now, using current_price (e.g., Close) for simplicity
                if self.current_position.size > 0 and current_price <= stop_loss_level: # Long SL hit
                    exit_price = stop_loss_level # Assume execution at SL price
                    logger.info(f"[{timestamp.date()}] LONG STOP LOSS hit at {exit_price:.2f} (Entry: {self.current_position.entry_price:.2f})")
                elif self.current_position.size < 0 and current_price >= stop_loss_level: # Short SL hit
                    exit_price = stop_loss_level
                    logger.info(f"[{timestamp.date()}] SHORT STOP LOSS hit at {exit_price:.2f} (Entry: {self.current_position.entry_price:.2f})")

                if exit_price is not None:
                    # Calculate PnL for the closing trade
                    exit_value = abs(self.current_position.size) * exit_price * (1 - self.commission)
                    entry_value_basis = abs(self.current_position.size) * self.current_position.entry_price * (1 + self.commission) # Cost basis includes entry commission
                    # PnL calculation depends on direction
                    pnl = (exit_value - entry_value_basis) if self.current_position.size > 0 else (entry_value_basis - exit_value)
                    realized_pnl_today += pnl

                    # Record the closed position
                    self.current_position.exit_price = exit_price
                    self.current_position.exit_time = timestamp
                    self.current_position.pnl = pnl
                    self.positions.append(self.current_position)

                    # Update equity immediately
                    self.equity += pnl # Update equity with realized PnL
                    logger.info(f"[{timestamp.date()}] Position closed via SL. PnL: {pnl:.2f}, Equity: {self.equity:.2f}")

                    # Calculate return based on cost basis
                    if entry_value_basis != 0:
                        trade_returns.append(pnl / entry_value_basis)

                    self.current_position = None
                    trade_executed_today = True

            # --- 2. Adjust Position based on Target Allocation (if no SL triggered) ---
            if not trade_executed_today:
                current_position_size = self.current_position.size if self.current_position else 0.0
                # Calculate target size based on current equity and price
                target_position_size = (self.equity * target_allocation) / current_price if abs(current_price) > 1e-9 else 0.0
                quantity_to_trade = target_position_size - current_position_size

                # Execute trade if change is significant enough (e.g., > 0.01% of portfolio or minimum quantity/value)
                min_trade_value = 1.0 # Example: minimum $1 trade value threshold
                min_trade_qty_threshold = 1e-9 # Avoid trading dust

                if abs(quantity_to_trade * current_price) > min_trade_value and abs(quantity_to_trade) > min_trade_qty_threshold:
                    commission_cost = abs(quantity_to_trade * current_price) * self.commission
                    self.equity -= commission_cost # Commission always reduces equity

                    realized_pnl_trade = 0.0
                    entry_value_closed = 0.0
                    old_size = current_position_size
                    new_size = old_size + quantity_to_trade

                    # Check if closing part of position or flipping
                    is_reducing = abs(new_size) < abs(old_size)
                    is_flipping = np.sign(old_size) != np.sign(new_size) and abs(old_size) > 1e-9

                    if (is_reducing or is_flipping) and self.current_position:
                        # Determine the quantity being closed
                        if is_flipping:
                            closed_quantity = abs(old_size)
                        else: # Reducing size
                            closed_quantity = abs(quantity_to_trade)

                        closed_quantity = min(closed_quantity, abs(old_size)) # Ensure we don't close more than held

                        if closed_quantity > 1e-9:
                            exit_value_closed = closed_quantity * current_price * (1 - self.commission)
                            entry_value_closed = closed_quantity * self.current_position.entry_price * (1 + self.commission) # Cost basis of the closed part
                            pnl_closed = (exit_value_closed - entry_value_closed) if old_size > 0 else (entry_value_closed - exit_value_closed)
                            realized_pnl_trade += pnl_closed
                            logger.debug(f"[{timestamp.date()}] Realized PnL on partial close/flip: {pnl_closed:.2f}")
                            # Update equity immediately with realized PnL
                            self.equity += realized_pnl_trade
                            realized_pnl_today += realized_pnl_trade # Add to daily total
                            # Calculate return for closed portion
                            if entry_value_closed != 0:
                                trade_returns.append(pnl_closed / entry_value_closed)


                    # Update or create position object
                    if self.current_position is None: # Entering new position
                        if abs(quantity_to_trade) > min_trade_qty_threshold: # Ensure non-dust entry
                            self.current_position = Position(
                                entry_price=current_price, entry_time=timestamp, size=quantity_to_trade
                            )
                            logger.info(f"[{timestamp.date()}] ENTER {'LONG' if quantity_to_trade > 0 else 'SHORT'}. Size: {quantity_to_trade:.4f} @ {current_price:.2f}. Target: {target_allocation:.2f}. Equity: {self.equity:.2f}")
                            trade_executed_today = True
                    else: # Adjusting existing position
                        if abs(new_size) < 1e-9: # Position closed completely
                            self.current_position.exit_price = current_price
                            self.current_position.exit_time = timestamp
                            self.current_position.pnl = realized_pnl_trade # PnL from this closing trade
                            self.positions.append(self.current_position)
                            logger.info(f"[{timestamp.date()}] CLOSE {'LONG' if old_size > 0 else 'SHORT'}. Qty: {quantity_to_trade:.4f} @ {current_price:.2f}. PnL: {realized_pnl_trade:.2f}. Equity: {self.equity:.2f}")
                            self.current_position = None
                            trade_executed_today = True
                        elif is_flipping:
                             # Record the closed part
                            closed_pos_record = Position(
                                entry_price=self.current_position.entry_price, entry_time=self.current_position.entry_time,
                                size= -closed_quantity if old_size < 0 else closed_quantity,
                                exit_price=current_price, exit_time=timestamp, pnl=realized_pnl_trade
                            )
                            self.positions.append(closed_pos_record)
                            # Reset current_position for the new direction
                            self.current_position.entry_price = current_price
                            self.current_position.entry_time = timestamp
                            self.current_position.size = new_size # Remaining size after flip
                            self.current_position.exit_price = None
                            self.current_position.exit_time = None
                            self.current_position.pnl = None
                            logger.info(
                                f"[{timestamp.date()}] FLIP to {'LONG' if new_size > 0 else 'SHORT'}. "
                                f"Close Qty: {closed_quantity:.4f}. Open Qty: {abs(new_size):.4f} @ {current_price:.2f}. "
                                f"Realized PnL: {realized_pnl_trade:.2f}. Equity: {self.equity:.2f}"
                            )
                            trade_executed_today = True
                        else: # Adjusting size (increasing or partial close without flip)
                            # TODO: Consider updating average entry price if increasing position size
                            self.current_position.size = new_size
                            logger.info(
                                f"[{timestamp.date()}] ADJUST {'LONG' if new_size > 0 else 'SHORT'}. "
                                f"Trade Qty: {quantity_to_trade:.4f} @ {current_price:.2f}. "
                                f"New Size: {new_size:.4f}. Target: {target_allocation:.2f}. "
                                f"Realized PnL: {realized_pnl_trade:.2f}. Equity: {self.equity:.2f}"
                            )
                            trade_executed_today = True


            # --- 3. Update Equity Curve (End of Day Mark-to-Market) ---
            unrealized_pnl = 0.0
            if self.current_position:
                unrealized_pnl = self.current_position.size * (current_price - self.current_position.entry_price)
            # Equity curve reflects cash (already includes realized PnL for the day) + unrealized PnL
            current_total_equity = self.equity + unrealized_pnl
            self.equity_curve[timestamp] = current_total_equity


        # --- End of Backtest Loop ---

        # Close any remaining position at the end
        if self.current_position is not None:
            last_price = prices.iloc[-1]
            last_timestamp = prices.index[-1]
            # Calculate PnL including commission for exit
            exit_value = abs(self.current_position.size) * last_price * (1 - self.commission)
            entry_value_basis = abs(self.current_position.size) * self.current_position.entry_price * (1 + self.commission)
            # PnL calculation depends on direction
            pnl = (exit_value - entry_value_basis) if self.current_position.size > 0 else (entry_value_basis - exit_value)

            self.current_position.exit_price = last_price
            self.current_position.exit_time = last_timestamp
            self.current_position.pnl = pnl

            self._update_equity(last_timestamp, pnl) # Update equity with final PnL
            self.positions.append(self.current_position)

            if entry_value_basis != 0:
                trade_returns.append(pnl / entry_value_basis)

        # Calculate performance metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital

        if not trade_returns:  # No trades made
            logger.warning("No trades were executed during the backtest")
            return self._generate_empty_result(prices.index)

        returns_array = np.array(trade_returns)
        winning_trades = len([p for p in self.positions if p.pnl is not None and p.pnl > 0])
        losing_trades = len([p for p in self.positions if p.pnl is not None and p.pnl < 0])

        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array)

        # Create equity curve series
        equity_curve = pd.Series(self.equity_curve).sort_index() # Ensure sorted by time

        # Calculate maximum drawdown
        rolling_max = equity_curve.expanding(min_periods=1).max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0

        # Calculate average win/loss
        wins = [p.pnl for p in self.positions if p.pnl is not None and p.pnl > 0]
        losses = [p.pnl for p in self.positions if p.pnl is not None and p.pnl < 0]

        # Calculate avg win/loss based on initial capital for percentage representation
        avg_win_pct = np.mean(wins) / self.initial_capital if wins else 0.0
        avg_loss_pct = abs(np.mean(losses)) / self.initial_capital if losses else 0.0

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
            avg_win=avg_win_pct, # Return percentage value
            avg_loss=avg_loss_pct, # Return percentage value
            profit_factor=profit_factor,
            positions=self.positions,
            equity_curve=equity_curve
        )