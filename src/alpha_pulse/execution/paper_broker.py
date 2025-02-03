"""
Paper trading broker implementation.
"""
from decimal import Decimal
from typing import Dict, List, Optional
from loguru import logger

from .broker_interface import (
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position
)


class PaperBroker(BrokerInterface):
    """Paper trading broker implementation."""
    
    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize paper broker.
        
        Args:
            initial_balance: Initial balance in quote currency
        """
        self.balance = Decimal(str(initial_balance))
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.market_prices: Dict[str, Decimal] = {}
        
        logger.info(f"Initialized paper broker with {initial_balance} balance")
        
    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        total = self.balance
        
        for symbol, position in self.positions.items():
            if symbol in self.market_prices:
                price = self.market_prices[symbol]
                total += Decimal(str(position.quantity)) * price
                
        return total
        
    def get_available_margin(self) -> Decimal:
        """Get available margin for trading."""
        # Use 80% of portfolio value as available margin
        return self.get_portfolio_value() * Decimal('0.8')
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
        
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
        
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
        
    async def place_order(self, order: Order) -> OrderResult:
        """Place a new order."""
        try:
            # Validate order
            if not self._validate_order(order):
                return OrderResult(
                    success=False,
                    error="Invalid order parameters"
                )
                
            # Check available margin
            required_margin = (
                Decimal(str(order.quantity)) *
                Decimal(str(order.price or self.market_prices.get(order.symbol, 0)))
            )
            if required_margin > self.get_available_margin():
                return OrderResult(
                    success=False,
                    error="Insufficient margin"
                )
                
            # Store order
            self.orders[order.order_id] = order
            
            # Execute market orders immediately
            if order.order_type == OrderType.MARKET:
                return await self._execute_market_order(order)
                
            return OrderResult(
                success=True,
                order_id=order.order_id
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return OrderResult(
                success=False,
                error=str(e)
            )
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False
        
    def update_market_data(self, symbol: str, price: float) -> None:
        """Update market data."""
        self.market_prices[symbol] = Decimal(str(price))
        
        # Check for triggered orders
        self._check_triggered_orders(symbol)
        
    async def initialize_spot_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> None:
        """Initialize spot position."""
        # Create market buy order
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price
        )
        
        # Execute order
        result = await self._execute_market_order(order)
        if result.success:
            logger.info(
                f"Initialized paper spot position: {quantity} {symbol} @ {price}"
            )
        else:
            logger.error(f"Failed to initialize position: {result.error}")
            
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        try:
            if not order.symbol or not order.quantity or order.quantity <= 0:
                return False
                
            if order.order_type != OrderType.MARKET and not order.price:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {str(e)}")
            return False
            
    async def _execute_market_order(self, order: Order) -> OrderResult:
        """Execute a market order."""
        try:
            # Get execution price
            price = (
                Decimal(str(order.price))
                if order.price
                else self.market_prices.get(order.symbol)
            )
            if not price:
                return OrderResult(
                    success=False,
                    error="No price available"
                )
                
            # Update position
            position = self.positions.get(order.symbol)
            quantity = Decimal(str(order.quantity))
            
            if order.side == OrderSide.BUY:
                # Deduct balance
                required_funds = quantity * price
                if required_funds > self.balance:
                    return OrderResult(
                        success=False,
                        error="Insufficient funds"
                    )
                self.balance -= required_funds
                
                # Update position
                if position:
                    new_quantity = position.quantity + quantity
                    new_avg_price = (
                        (position.quantity * position.avg_entry_price +
                         quantity * price) / new_quantity
                    )
                    position.quantity = float(new_quantity)
                    position.avg_entry_price = float(new_avg_price)
                else:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=float(quantity),
                        avg_entry_price=float(price)
                    )
                    
            else:  # SELL
                if not position or position.quantity < quantity:
                    return OrderResult(
                        success=False,
                        error="Insufficient position"
                    )
                    
                # Add to balance
                self.balance += quantity * price
                
                # Update position
                position.quantity = float(position.quantity - quantity)
                if position.quantity == 0:
                    del self.positions[order.symbol]
                    
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = float(quantity)
            order.filled_price = float(price)
            
            logger.info(f"Order executed: {order}")
            
            return OrderResult(
                success=True,
                order_id=order.order_id,
                filled_quantity=float(quantity),
                filled_price=float(price)
            )
            
        except Exception as e:
            logger.error(f"Error executing market order: {str(e)}")
            return OrderResult(
                success=False,
                error=str(e)
            )
            
    def _check_triggered_orders(self, symbol: str) -> None:
        """Check for triggered orders."""
        if symbol not in self.market_prices:
            return
            
        current_price = self.market_prices[symbol]
        
        for order_id, order in list(self.orders.items()):
            if order.symbol != symbol:
                continue
                
            # Check stop orders
            if (order.order_type == OrderType.STOP and
                order.stop_price and
                current_price <= Decimal(str(order.stop_price))):
                asyncio.create_task(self._execute_market_order(order))
                continue
                
            # Check limit orders
            if order.order_type == OrderType.LIMIT and order.price:
                price = Decimal(str(order.price))
                if ((order.side == OrderSide.BUY and current_price <= price) or
                    (order.side == OrderSide.SELL and current_price >= price)):
                    asyncio.create_task(self._execute_market_order(order))
                    continue