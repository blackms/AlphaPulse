"""
Order management for grid hedging strategy.
"""
from decimal import Decimal
from typing import Dict, List, Optional
from loguru import logger

from alpha_pulse.execution.broker_interface import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderType
)

from ..common.interfaces import OrderManager, RiskManager
from ..common.types import GridLevel, MarketState, PositionState


class GridOrderManager(OrderManager):
    """Order management implementation for grid strategy."""
    
    def __init__(
        self,
        broker: BrokerInterface,
        risk_manager: RiskManager,
        symbol: str,
        max_active_orders: int = 50
    ):
        """
        Initialize order manager.
        
        Args:
            broker: Trading broker interface
            risk_manager: Risk management interface
            symbol: Trading symbol
            max_active_orders: Maximum number of active orders
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.max_active_orders = max_active_orders
        
        # Track active orders
        self.active_orders: Dict[str, GridLevel] = {}
        self.stop_loss_order_id: Optional[str] = None
        self.take_profit_order_id: Optional[str] = None
        
        logger.info(
            f"Initialized order manager for {symbol} "
            f"(max orders: {max_active_orders})"
        )
        
    async def place_grid_orders(
        self,
        levels: List[GridLevel],
        market: MarketState
    ) -> Dict[str, GridLevel]:
        """
        Place grid orders.
        
        Args:
            levels: Grid levels to place orders for
            market: Current market state
            
        Returns:
            Dictionary of placed orders
        """
        try:
            # Validate number of orders
            if len(levels) > self.max_active_orders:
                logger.warning(
                    f"Too many levels ({len(levels)}), "
                    f"limiting to {self.max_active_orders}"
                )
                levels = levels[:self.max_active_orders]
                
            # Calculate available margin
            available_margin = self.broker.get_available_margin()
            if available_margin <= Decimal('0'):
                logger.error("No margin available for orders")
                return {}
                
            # Place orders
            placed_orders = {}
            for level in levels:
                # Calculate safe size
                size = self.risk_manager.calculate_position_size(
                    price=level.price,
                    volatility=market.volatility,
                    available_margin=available_margin
                )
                
                if size <= Decimal('0'):
                    logger.warning(f"Invalid size calculated for level {level.price}")
                    continue
                    
                # Create order
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.BUY if level.is_long else OrderSide.SELL,
                    quantity=float(size),
                    order_type=OrderType.LIMIT,
                    price=float(level.price)
                )
                
                # Place order
                result = await self.broker.place_order(order)
                if result.order_id:
                    # Update level with actual size and order ID
                    updated_level = level.update(
                        quantity=size,
                        order_id=result.order_id
                    )
                    placed_orders[result.order_id] = updated_level
                    logger.info(
                        f"Placed {order.side.value} order at {level.price} "
                        f"for {size}"
                    )
                else:
                    logger.error(f"Failed to place order at price {level.price}")
                    
            return placed_orders
            
        except Exception as e:
            logger.error(f"Error placing grid orders: {str(e)}")
            return {}
            
    async def cancel_orders(self, order_ids: List[str]) -> None:
        """
        Cancel orders.
        
        Args:
            order_ids: List of order IDs to cancel
        """
        try:
            for order_id in order_ids:
                if await self.broker.cancel_order(order_id):
                    if order_id in self.active_orders:
                        level = self.active_orders[order_id]
                        logger.info(f"Cancelled order at price {level.price}")
                        del self.active_orders[order_id]
                else:
                    logger.warning(f"Failed to cancel order {order_id}")
                    
        except Exception as e:
            logger.error(f"Error cancelling orders: {str(e)}")
            
    async def update_risk_orders(
        self,
        position: PositionState,
        market: MarketState
    ) -> None:
        """
        Update stop loss and take profit orders.
        
        Args:
            position: Current position state
            market: Current market state
        """
        try:
            # Cancel existing risk orders if no position
            if position.spot_quantity == Decimal('0'):
                await self._cancel_risk_orders()
                return
                
            # Calculate stop loss price
            stop_loss_price = self.risk_manager.calculate_stop_loss(
                position=position,
                market=market
            )
            
            if stop_loss_price <= Decimal('0'):
                logger.error("Invalid stop loss price calculated")
                return
                
            # Calculate take profit price (1.5x the risk)
            risk = abs(position.avg_entry_price - stop_loss_price)
            take_profit_price = position.avg_entry_price + (risk * Decimal('1.5'))
            
            # Update stop loss order
            if not self.stop_loss_order_id:
                stop_order = Order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=float(position.spot_quantity),
                    order_type=OrderType.STOP,
                    stop_price=float(stop_loss_price)
                )
                result = await self.broker.place_order(stop_order)
                if result.order_id:
                    self.stop_loss_order_id = result.order_id
                    logger.info(
                        f"Placed stop loss order at {stop_loss_price:.2f} "
                        f"for {position.spot_quantity}"
                    )
                    
            # Update take profit order
            if not self.take_profit_order_id:
                take_profit_order = Order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=float(position.spot_quantity),
                    order_type=OrderType.LIMIT,
                    price=float(take_profit_price)
                )
                result = await self.broker.place_order(take_profit_order)
                if result.order_id:
                    self.take_profit_order_id = result.order_id
                    logger.info(
                        f"Placed take profit order at {take_profit_price:.2f} "
                        f"for {position.spot_quantity}"
                    )
                    
        except Exception as e:
            logger.error(f"Error updating risk orders: {str(e)}")
            
    async def _cancel_risk_orders(self) -> None:
        """Cancel stop loss and take profit orders."""
        try:
            if self.stop_loss_order_id:
                if await self.broker.cancel_order(self.stop_loss_order_id):
                    logger.info("Cancelled stop loss order")
                self.stop_loss_order_id = None
                
            if self.take_profit_order_id:
                if await self.broker.cancel_order(self.take_profit_order_id):
                    logger.info("Cancelled take profit order")
                self.take_profit_order_id = None
                
        except Exception as e:
            logger.error(f"Error cancelling risk orders: {str(e)}")
            
    def get_active_orders(self) -> Dict[str, GridLevel]:
        """Get currently active orders."""
        return self.active_orders.copy()
        
    def has_risk_orders(self) -> bool:
        """Check if risk orders are active."""
        return bool(self.stop_loss_order_id or self.take_profit_order_id)