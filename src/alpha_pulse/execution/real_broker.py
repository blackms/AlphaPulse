"""
Real broker implementation for live trading on exchanges.
"""
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from ..exchanges.base import BaseExchange
from ..exchanges.binance import BinanceExchange
from ..exchanges.bybit import BybitExchange
from .broker_interface import (
    BrokerInterface,
    Order,
    OrderStatus,
    Position,
)


class RealBroker(BrokerInterface):
    """Live trading broker implementation."""

    def __init__(self, exchange: BaseExchange):
        """Initialize with specific exchange instance."""
        self.exchange = exchange
        self._orders: Dict[str, Order] = {}
        logger.info(f"Initialized RealBroker with {exchange.__class__.__name__}")

    @classmethod
    def create_binance(cls, api_key: str, api_secret: str, testnet: bool = False) -> 'RealBroker':
        """Create RealBroker instance for Binance."""
        exchange = BinanceExchange(api_key, api_secret, testnet=testnet)
        return cls(exchange)

    @classmethod
    def create_bybit(cls, api_key: str, api_secret: str, testnet: bool = False) -> 'RealBroker':
        """Create RealBroker instance for Bybit."""
        exchange = BybitExchange(api_key, api_secret, testnet=testnet)
        return cls(exchange)

    def place_order(self, order: Order) -> Order:
        """Place a real order on the exchange."""
        try:
            logger.info(f"Placing order on {self.exchange.__class__.__name__}: {order}")
            result = self.exchange.place_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price
            )
            
            order.order_id = result['orderId']
            order.status = OrderStatus.PENDING
            order.timestamp = datetime.now()
            self._orders[order.order_id] = order
            
            logger.info(f"Order placed successfully: {order.order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            order.status = OrderStatus.REJECTED
            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on the exchange."""
        try:
            logger.info(f"Cancelling order {order_id}")
            result = self.exchange.cancel_order(order_id)
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get current state of an order."""
        try:
            order_info = self.exchange.get_order(order_id)
            if order_id in self._orders:
                order = self._orders[order_id]
                order.status = OrderStatus(order_info['status'].lower())
                order.filled_quantity = float(order_info.get('executedQty', 0))
                order.filled_price = float(order_info.get('avgPrice', 0))
                return order
            return None
        except Exception as e:
            logger.error(f"Error fetching order: {str(e)}")
            return None

    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders."""
        try:
            orders = self.exchange.get_open_orders(symbol)
            return [self._convert_exchange_order(o) for o in orders]
        except Exception as e:
            logger.error(f"Error fetching orders: {str(e)}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        try:
            pos = self.exchange.get_position(symbol)
            if pos and float(pos['positionAmt']) != 0:
                return Position(
                    symbol=symbol,
                    quantity=float(pos['positionAmt']),
                    avg_entry_price=float(pos['entryPrice']),
                    unrealized_pnl=float(pos.get('unrealizedPnl', 0)),
                    timestamp=datetime.now()
                )
            return None
        except Exception as e:
            logger.error(f"Error fetching position: {str(e)}")
            return None

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        try:
            positions = {}
            for pos in self.exchange.get_positions():
                if float(pos['positionAmt']) != 0:
                    positions[pos['symbol']] = Position(
                        symbol=pos['symbol'],
                        quantity=float(pos['positionAmt']),
                        avg_entry_price=float(pos['entryPrice']),
                        unrealized_pnl=float(pos.get('unrealizedPnl', 0)),
                        timestamp=datetime.now()
                    )
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            return {}

    def get_account_balance(self) -> float:
        """Get current account cash balance."""
        try:
            return float(self.exchange.get_balance())
        except Exception as e:
            logger.error(f"Error fetching balance: {str(e)}")
            return 0.0

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        try:
            return float(self.exchange.get_account_value())
        except Exception as e:
            logger.error(f"Error fetching portfolio value: {str(e)}")
            return 0.0

    def update_market_data(self, symbol: str, current_price: float) -> None:
        """Update market data (no-op for real broker as exchange handles this)."""
        pass

    def _convert_exchange_order(self, exchange_order: Dict) -> Order:
        """Convert exchange order format to internal Order object."""
        return Order(
            symbol=exchange_order['symbol'],
            side=exchange_order['side'],
            quantity=float(exchange_order['origQty']),
            order_type=exchange_order['type'],
            price=float(exchange_order.get('price', 0)) or None,
            stop_price=float(exchange_order.get('stopPrice', 0)) or None,
            order_id=exchange_order['orderId'],
            status=OrderStatus(exchange_order['status'].lower()),
            filled_quantity=float(exchange_order.get('executedQty', 0)),
            filled_price=float(exchange_order.get('avgPrice', 0)) or None,
            timestamp=datetime.fromtimestamp(exchange_order['time'] / 1000)
        )