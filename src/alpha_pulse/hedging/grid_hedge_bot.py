"""
Grid-based hedging strategy implementation.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from ..execution.broker_interface import BrokerInterface, Order, OrderSide, OrderType
from .grid_hedge_config import GridHedgeConfig, GridLevel, GridDirection


class GridHedgeBot:
    """
    Implements a grid-based hedging strategy that places orders at predefined price levels.
    """

    def __init__(self, broker: BrokerInterface, config: GridHedgeConfig):
        """
        Initialize the grid hedging bot.
        
        Args:
            broker: Trading broker implementation
            config: Grid strategy configuration
        """
        self.broker = broker
        self.config = config
        self.active_orders: Dict[str, GridLevel] = {}
        self.last_rebalance: Optional[datetime] = None
        
        # Validate configuration
        self.config.validate()
        logger.info(
            f"Initialized GridHedgeBot for {config.symbol} with "
            f"{len(config.grid_levels)} levels"
        )

    def execute(self, current_price: float) -> None:
        """
        Execute the grid strategy based on current market conditions.
        
        Args:
            current_price: Current market price of the trading pair
        """
        try:
            # Check if rebalance is needed
            if self._should_rebalance():
                logger.info(f"Rebalancing grid for {self.config.symbol}")
                self._rebalance_grid(current_price)
                self.last_rebalance = datetime.now()
        except Exception as e:
            logger.error(f"Error executing grid strategy: {str(e)}")

    def _should_rebalance(self) -> bool:
        """Check if grid should be rebalanced based on time interval."""
        if not self.last_rebalance:
            return True
        
        time_since_rebalance = datetime.now() - self.last_rebalance
        return time_since_rebalance.seconds >= self.config.rebalance_interval

    def _rebalance_grid(self, current_price: float) -> None:
        """
        Rebalance the grid by cancelling invalid orders and placing new ones.
        
        Args:
            current_price: Current market price
        """
        # Get current positions and orders
        positions = self.broker.get_positions()
        current_position = positions.get(self.config.symbol)
        current_position_size = (
            current_position.quantity if current_position else 0.0
        )

        # Cancel orders that are no longer valid
        self._cancel_invalid_orders(current_price)

        # Calculate which grid levels need orders
        levels_needing_orders = self._get_levels_needing_orders(
            current_price, current_position_size
        )

        # Place new orders
        for level in levels_needing_orders:
            self._place_grid_order(level, current_price)

    def _cancel_invalid_orders(self, current_price: float) -> None:
        """Cancel orders that are no longer valid for the current price."""
        for order_id, level in list(self.active_orders.items()):
            should_cancel = False
            
            # Check if order is too far from current price
            price_diff = abs(level.price - current_price)
            if price_diff > self.config.grid_spacing * 2:
                should_cancel = True
                
            # Check if order is in wrong direction based on current price
            if level.is_long and level.price > current_price:
                should_cancel = True
            elif not level.is_long and level.price < current_price:
                should_cancel = True
                
            if should_cancel:
                if self.broker.cancel_order(order_id):
                    del self.active_orders[order_id]
                    logger.info(f"Cancelled order at price {level.price}")

    def _get_levels_needing_orders(
        self, current_price: float, current_position_size: float
    ) -> List[GridLevel]:
        """
        Determine which grid levels need new orders placed.
        
        Args:
            current_price: Current market price
            current_position_size: Current position size
            
        Returns:
            List of grid levels that need orders
        """
        levels_needing_orders = []
        
        for level in self.config.grid_levels:
            # Skip if level already has an active order
            if any(l.price == level.price for l in self.active_orders.values()):
                continue
                
            # Check if level is valid for current price
            if level.is_long:
                # Long levels must be below current price
                if level.price >= current_price:
                    continue
            else:
                # Short levels must be above current price
                if level.price <= current_price:
                    continue
                    
            # Check position limits
            if level.is_long:
                if current_position_size + level.quantity > self.config.max_position_size:
                    continue
            else:
                if current_position_size - level.quantity < -self.config.max_position_size:
                    continue
                    
            levels_needing_orders.append(level)
            
        return levels_needing_orders

    def _place_grid_order(self, level: GridLevel, current_price: float) -> None:
        """
        Place a limit order for a grid level.
        
        Args:
            level: Grid level to place order for
            current_price: Current market price
        """
        order = Order(
            symbol=self.config.symbol,
            side=OrderSide.BUY if level.is_long else OrderSide.SELL,
            quantity=level.quantity,
            order_type=OrderType.LIMIT,
            price=level.price
        )
        
        result = self.broker.place_order(order)
        if result.order_id:
            level.order_id = result.order_id
            self.active_orders[result.order_id] = level
            logger.info(
                f"Placed {order.side.value} order at {level.price} "
                f"for {level.quantity}"
            )
        else:
            logger.error(f"Failed to place order at price {level.price}")

    def get_status(self) -> Dict:
        """
        Get current status of the grid strategy.
        
        Returns:
            Dict containing current status information
        """
        return {
            "symbol": self.config.symbol,
            "active_orders": len(self.active_orders),
            "grid_levels": len(self.config.grid_levels),
            "last_rebalance": self.last_rebalance.isoformat() 
                if self.last_rebalance else None
        }