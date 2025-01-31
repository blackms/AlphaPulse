"""
Paper trading broker implementation for simulated trading.
"""
from datetime import datetime
from typing import Dict, List, Optional
import logging
import uuid

from .broker_interface import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class RiskLimits:
    """Risk management limits configuration."""
    def __init__(
        self,
        max_position_size: float = 100000.0,  # Maximum position size in base currency
        max_portfolio_size: float = 1000000.0,  # Maximum total portfolio value
        max_drawdown_pct: float = 0.25,  # Maximum drawdown as percentage (0.25 = 25%)
        stop_loss_pct: float = 0.10,  # Stop loss percentage per position (0.10 = 10%)
    ):
        self.max_position_size = max_position_size
        self.max_portfolio_size = max_portfolio_size
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct


class PaperBroker(BrokerInterface):
    """Paper trading broker implementation."""

    def __init__(
        self,
        initial_balance: float = 100000.0,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self._cash_balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._market_prices: Dict[str, float] = {}
        self._risk_limits = risk_limits or RiskLimits()
        self._logger = logging.getLogger(__name__)

    def place_order(self, order: Order) -> Order:
        """Place a new order in the paper trading system."""
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return order

        # Generate order ID
        order.order_id = str(uuid.uuid4())
        
        # For market orders, execute immediately
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        else:
            # Store limit/stop orders for later execution
            self._orders[order.order_id] = order
            self._logger.info(f"Order placed: {order}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if order_id not in self._orders:
            return False
        
        order = self._orders.pop(order_id)
        order.status = OrderStatus.CANCELLED
        self._logger.info(f"Order cancelled: {order}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get current state of an order."""
        return self._orders.get(order_id)

    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        if symbol:
            return [o for o in self._orders.values() if o.symbol == symbol]
        return list(self._orders.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self._positions.copy()

    def get_account_balance(self) -> float:
        """Get current account cash balance."""
        return self._cash_balance

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        positions_value = sum(
            pos.quantity * self._market_prices.get(pos.symbol, 0)
            for pos in self._positions.values()
        )
        return self._cash_balance + positions_value

    def update_market_data(self, symbol: str, current_price: float) -> None:
        """Update market data and check pending orders."""
        self._market_prices[symbol] = current_price
        
        # Update position PnL
        if symbol in self._positions:
            self._update_position_pnl(symbol)
            self._check_stop_loss(symbol)

        # Check pending orders for execution
        self._check_pending_orders(symbol)

    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters and risk limits."""
        # Check if we have market data for the symbol
        if order.symbol not in self._market_prices:
            self._logger.warning(f"No market data for symbol: {order.symbol}")
            return False

        current_price = self._market_prices[order.symbol]
        order_value = order.quantity * current_price

        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            if order_value > self._cash_balance:
                self._logger.warning("Insufficient funds for order")
                return False

        # Check position size limits
        current_position = self._positions.get(order.symbol)
        new_position_size = order_value
        if current_position:
            if order.side == OrderSide.BUY:
                new_position_size += current_position.quantity * current_price
            else:
                new_position_size = abs(
                    (current_position.quantity - order.quantity) * current_price
                )

        if new_position_size > self._risk_limits.max_position_size:
            self._logger.warning("Order exceeds maximum position size")
            return False

        # Check portfolio value limits
        if self.get_portfolio_value() + order_value > self._risk_limits.max_portfolio_size:
            self._logger.warning("Order exceeds maximum portfolio size")
            return False

        return True

    def _execute_market_order(self, order: Order) -> None:
        """Execute a market order."""
        current_price = self._market_prices[order.symbol]
        
        # Update cash balance
        order_value = order.quantity * current_price
        if order.side == OrderSide.BUY:
            self._cash_balance -= order_value
        else:
            self._cash_balance += order_value

        # Update position
        position = self._positions.get(order.symbol)
        if position is None:
            if order.side == OrderSide.BUY:
                position = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_entry_price=current_price,
                )
                self._positions[order.symbol] = position
        else:
            if order.side == OrderSide.BUY:
                # Update average entry price
                total_quantity = position.quantity + order.quantity
                position.avg_entry_price = (
                    (position.quantity * position.avg_entry_price + order_value)
                    / total_quantity
                )
                position.quantity = total_quantity
            else:
                # Update position quantity and realized PnL
                position.realized_pnl += (
                    current_price - position.avg_entry_price
                ) * order.quantity
                position.quantity -= order.quantity
                if position.quantity <= 0:
                    del self._positions[order.symbol]

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = current_price
        self._logger.info(f"Order executed: {order}")

    def _check_pending_orders(self, symbol: str) -> None:
        """Check if any pending orders should be executed."""
        current_price = self._market_prices[symbol]
        
        # Find orders that should be executed
        executable_orders = []
        for order in self._orders.values():
            if order.symbol != symbol:
                continue

            if order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and current_price <= order.price) or (
                    order.side == OrderSide.SELL and current_price >= order.price
                ):
                    executable_orders.append(order)
            
            elif order.order_type == OrderType.STOP:
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or (
                    order.side == OrderSide.SELL and current_price <= order.stop_price
                ):
                    executable_orders.append(order)

        # Execute orders
        for order in executable_orders:
            if self._validate_order(order):
                self._execute_market_order(order)
            del self._orders[order.order_id]

    def _update_position_pnl(self, symbol: str) -> None:
        """Update unrealized PnL for a position."""
        position = self._positions[symbol]
        current_price = self._market_prices[symbol]
        position.unrealized_pnl = (
            current_price - position.avg_entry_price
        ) * position.quantity
        position.timestamp = datetime.now()

    def _check_stop_loss(self, symbol: str) -> None:
        """Check if stop loss should be triggered."""
        position = self._positions[symbol]
        current_price = self._market_prices[symbol]
        
        loss_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
        if abs(loss_pct) >= self._risk_limits.stop_loss_pct:
            # Create and execute stop loss order
            stop_order = Order(
                symbol=symbol,
                side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                quantity=abs(position.quantity),
                order_type=OrderType.MARKET,
            )
            self._execute_market_order(stop_order)
            self._logger.warning(f"Stop loss triggered for {symbol}")