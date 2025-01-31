"""
Unit tests for the execution module.
"""
import unittest
from datetime import datetime
from decimal import Decimal

from alpha_pulse.execution import (
    BrokerInterface,
    PaperBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    RiskLimits,
)


class TestPaperBroker(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.initial_balance = 1000000.0  # Increased for larger test trades
        self.risk_limits = RiskLimits(
            max_position_size=500000.0,  # Increased to allow multiple orders
            max_portfolio_size=1500000.0,
            max_drawdown_pct=0.5,  # More permissive for testing
            stop_loss_pct=0.5,  # Increased to prevent premature triggers
        )
        self.broker = PaperBroker(
            initial_balance=self.initial_balance,
            risk_limits=self.risk_limits,
        )
        
        # Set up test market data
        self.test_symbol = "BTC/USD"
        self.test_price = 50000.0
        self.broker.update_market_data(self.test_symbol, self.test_price)

    def test_market_order_execution(self):
        """Test basic market order execution."""
        # Create and place a market buy order
        quantity = 1.0
        order = Order(
            symbol=self.test_symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
        
        executed_order = self.broker.place_order(order)
        
        # Check order status
        self.assertEqual(executed_order.status, OrderStatus.FILLED)
        self.assertEqual(executed_order.filled_quantity, quantity)
        self.assertEqual(executed_order.filled_price, self.test_price)
        
        # Check position
        position = self.broker.get_position(self.test_symbol)
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, quantity)
        self.assertEqual(position.avg_entry_price, self.test_price)
        
        # Check cash balance
        expected_balance = self.initial_balance - (quantity * self.test_price)
        self.assertEqual(self.broker.get_account_balance(), expected_balance)

    def test_limit_order_execution(self):
        """Test limit order execution."""
        # Place limit buy order above current price (should not execute)
        limit_price = self.test_price * 0.9  # 10% below market
        order = Order(
            symbol=self.test_symbol,
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=limit_price,
        )
        
        placed_order = self.broker.place_order(order)
        self.assertEqual(placed_order.status, OrderStatus.PENDING)
        
        # Update price to trigger order
        self.broker.update_market_data(self.test_symbol, limit_price)
        
        # Check if order was executed
        executed_order = self.broker.get_order(placed_order.order_id)
        self.assertIsNone(executed_order)  # Order should be removed after execution
        
        # Verify position
        position = self.broker.get_position(self.test_symbol)
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 1.0)
        self.assertEqual(position.avg_entry_price, limit_price)

    def test_position_tracking(self):
        """Test position tracking with multiple orders."""
        # First buy
        self.broker.place_order(
            Order(
                symbol=self.test_symbol,
                side=OrderSide.BUY,
                quantity=1.0,
                order_type=OrderType.MARKET,
            )
        )
        
        # Second buy at different price
        new_price = self.test_price * 1.1
        self.broker.update_market_data(self.test_symbol, new_price)
        self.broker.place_order(
            Order(
                symbol=self.test_symbol,
                side=OrderSide.BUY,
                quantity=1.0,
                order_type=OrderType.MARKET,
            )
        )
        
        # Check position
        position = self.broker.get_position(self.test_symbol)
        self.assertEqual(position.quantity, 2.0)
        # Average entry price should be between first and second price
        self.assertTrue(
            self.test_price < position.avg_entry_price < new_price
        )

    def test_risk_limits(self):
        """Test risk management limits."""
        # Try to place order exceeding position size limit
        large_quantity = self.risk_limits.max_position_size / self.test_price + 1
        order = Order(
            symbol=self.test_symbol,
            side=OrderSide.BUY,
            quantity=large_quantity,
            order_type=OrderType.MARKET,
        )
        
        executed_order = self.broker.place_order(order)
        self.assertEqual(executed_order.status, OrderStatus.REJECTED)

    def test_stop_loss(self):
        """Test stop loss functionality."""
        # Place initial buy order
        self.broker.place_order(
            Order(
                symbol=self.test_symbol,
                side=OrderSide.BUY,
                quantity=1.0,
                order_type=OrderType.MARKET,
            )
        )
        
        # Update price to trigger stop loss
        stop_loss_price = self.test_price * (1 - self.risk_limits.stop_loss_pct - 0.01)
        self.broker.update_market_data(self.test_symbol, stop_loss_price)
        
        # Position should be closed
        position = self.broker.get_position(self.test_symbol)
        self.assertIsNone(position)

    def test_pnl_calculation(self):
        """Test PnL calculations."""
        # Place buy order
        quantity = 1.0
        self.broker.place_order(
            Order(
                symbol=self.test_symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
        )
        
        # Update price and check unrealized PnL
        new_price = self.test_price * 1.1  # 10% increase
        self.broker.update_market_data(self.test_symbol, new_price)
        
        position = self.broker.get_position(self.test_symbol)
        expected_pnl = (new_price - self.test_price) * quantity
        self.assertAlmostEqual(position.unrealized_pnl, expected_pnl, places=2)
        
        # Sell position and check realized PnL
        self.broker.place_order(
            Order(
                symbol=self.test_symbol,
                side=OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
        )
        
        # Position should be closed with realized PnL
        position = self.broker.get_position(self.test_symbol)
        self.assertIsNone(position)
        
        # Check final portfolio value includes realized PnL
        expected_portfolio_value = self.initial_balance + expected_pnl
        self.assertAlmostEqual(
            self.broker.get_portfolio_value(),
            expected_portfolio_value,
            places=2
        )

    def test_order_cancellation(self):
        """Test order cancellation."""
        # Place limit order
        order = Order(
            symbol=self.test_symbol,
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=self.test_price * 0.9,
        )
        placed_order = self.broker.place_order(order)
        
        # Cancel order
        success = self.broker.cancel_order(placed_order.order_id)
        self.assertTrue(success)
        
        # Verify order is cancelled
        cancelled_order = self.broker.get_order(placed_order.order_id)
        self.assertIsNone(cancelled_order)


if __name__ == '__main__':
    unittest.main()