"""
Recommendation-only broker that logs trade signals without executing them.
"""
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from .broker_interface import (
    BrokerInterface,
    Order,
    OrderStatus,
    Position,
)


class RecommendationOnlyBroker(BrokerInterface):
    """Broker that only logs recommendations without executing trades."""

    def __init__(self):
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        logger.info("Initialized RecommendationOnlyBroker - Orders will be logged but not executed")

    def place_order(self, order: Order) -> Order:
        """Log the order recommendation without execution."""
        logger.info(
            f"[RECOMMENDATION] Would place order: Symbol={order.symbol}, "
            f"Side={order.side.value}, Quantity={order.quantity}, "
            f"Type={order.order_type.value}, "
            f"Price={order.price if order.price else 'MARKET'}"
        )
        order.status = OrderStatus.REJECTED
        order.timestamp = datetime.now()
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Log the cancellation recommendation."""
        logger.info(f"[RECOMMENDATION] Would cancel order: {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Return None as orders are not tracked."""
        return None

    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Return empty list as orders are not tracked."""
        return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return None as positions are not tracked."""
        return None

    def get_positions(self) -> Dict[str, Position]:
        """Return empty dict as positions are not tracked."""
        return {}

    def get_account_balance(self) -> float:
        """Return 0 as balance is not tracked."""
        return 0.0

    def get_portfolio_value(self) -> float:
        """Return 0 as portfolio is not tracked."""
        return 0.0

    def update_market_data(self, symbol: str, current_price: float) -> None:
        """No-op as market data is not tracked."""
        pass