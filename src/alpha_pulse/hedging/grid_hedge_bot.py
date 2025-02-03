"""
Grid-based hedging strategy implementation.
"""
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from alpha_pulse.execution.broker_interface import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderType,
    OrderStatus
)
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider
from .grid_hedge_config import GridHedgeConfig, GridLevel, GridDirection


class GridHedgeBot:
    """
    Implements a grid-based hedging strategy that places orders at predefined price levels.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config: GridHedgeConfig,
        data_provider: Optional[ExchangeDataProvider] = None
    ):
        """
        Initialize the grid hedging bot.
        
        Args:
            broker: Trading broker implementation
            config: Grid strategy configuration
            data_provider: Optional exchange data provider for real market data
        """
        self.broker = broker
        self.config = config
        self.data_provider = data_provider
        self.active_orders: Dict[str, GridLevel] = {}
        self.stop_loss_order_id: Optional[str] = None
        self.take_profit_order_id: Optional[str] = None
        self.last_rebalance: Optional[datetime] = None
        
        # Validate configuration
        self.config.validate()
        logger.info(
            f"Initialized GridHedgeBot for {config.symbol} with "
            f"{len(config.grid_levels)} levels"
        )

    @classmethod
    async def create_for_spot_hedge(
        cls,
        broker: BrokerInterface,
        symbol: str,
        data_provider: ExchangeDataProvider,
        volatility: float = 0.02,
        spot_quantity: Optional[float] = None,
    ) -> 'GridHedgeBot':
        """
        Create a grid bot configured for hedging a spot position.
        
        Args:
            broker: Trading broker implementation
            symbol: Trading symbol
            data_provider: Exchange data provider for market data
            volatility: Volatility estimate
            spot_quantity: Optional spot quantity (if not provided, will use existing position)
            
        Returns:
            Configured GridHedgeBot instance
        """
        # Get current market price
        current_price = await data_provider.get_current_price(symbol)
        if not current_price:
            raise ValueError(f"Could not get current price for {symbol}")
        
        # Get spot position
        position = broker.get_position(symbol)
        if not position and spot_quantity is None:
            raise ValueError(f"No spot position found for {symbol}")
        
        quantity = spot_quantity if spot_quantity is not None else position.quantity
        
        # Create grid configuration
        config = await GridHedgeConfig.create_hedge_grid(
            symbol=symbol,
            current_price=float(current_price),
            spot_quantity=quantity,
            volatility=volatility,
            portfolio_value=broker.get_portfolio_value()
        )
        
        return cls(broker, config, data_provider)

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
                
            # Update stop loss and take profit orders
            self._manage_risk_orders(current_price)
            
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

    def _manage_risk_orders(self, current_price: float) -> None:
        """
        Manage stop loss and take profit orders.
        
        Args:
            current_price: Current market price
        """
        if not self.config.stop_loss_pct or not self.config.take_profit_pct:
            return
            
        position = self.broker.get_position(self.config.symbol)
        if not position:
            # Cancel any existing risk orders if no position
            self._cancel_risk_orders()
            return
            
        entry_price = position.avg_entry_price
        
        # Calculate stop loss and take profit prices
        stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
        take_profit_price = entry_price * (1 + self.config.take_profit_pct)
        
        # Check if we need to place/update stop loss order
        if not self.stop_loss_order_id:
            stop_order = Order(
                symbol=self.config.symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.STOP,
                stop_price=stop_loss_price
            )
            result = self.broker.place_order(stop_order)
            if result.order_id:
                self.stop_loss_order_id = result.order_id
                logger.info(
                    f"Placed stop loss order at {stop_loss_price:.2f} "
                    f"for {position.quantity}"
                )
        
        # Check if we need to place/update take profit order
        if not self.take_profit_order_id:
            take_profit_order = Order(
                symbol=self.config.symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.LIMIT,
                price=take_profit_price
            )
            result = self.broker.place_order(take_profit_order)
            if result.order_id:
                self.take_profit_order_id = result.order_id
                logger.info(
                    f"Placed take profit order at {take_profit_price:.2f} "
                    f"for {position.quantity}"
                )
        
        # Check if risk orders were triggered
        if self.stop_loss_order_id:
            order = self.broker.get_order(self.stop_loss_order_id)
            if order and order.status == OrderStatus.FILLED:
                logger.warning(f"Stop loss triggered at {order.filled_price}")
                self._cancel_all_orders()  # Cancel all grid orders
                self.stop_loss_order_id = None
                self.take_profit_order_id = None
                
        if self.take_profit_order_id:
            order = self.broker.get_order(self.take_profit_order_id)
            if order and order.status == OrderStatus.FILLED:
                logger.info(f"Take profit triggered at {order.filled_price}")
                self._cancel_all_orders()  # Cancel all grid orders
                self.stop_loss_order_id = None
                self.take_profit_order_id = None

    def _cancel_risk_orders(self) -> None:
        """Cancel stop loss and take profit orders."""
        if self.stop_loss_order_id:
            self.broker.cancel_order(self.stop_loss_order_id)
            self.stop_loss_order_id = None
            
        if self.take_profit_order_id:
            self.broker.cancel_order(self.take_profit_order_id)
            self.take_profit_order_id = None

    def _cancel_all_orders(self) -> None:
        """Cancel all active orders including grid and risk orders."""
        # Cancel grid orders
        for order_id in list(self.active_orders.keys()):
            if self.broker.cancel_order(order_id):
                del self.active_orders[order_id]
        
        # Cancel risk orders
        self._cancel_risk_orders()

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
        self,
        current_price: float,
        current_position_size: float
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
            "stop_loss_active": bool(self.stop_loss_order_id),
            "take_profit_active": bool(self.take_profit_order_id),
            "last_rebalance": self.last_rebalance.isoformat() 
                if self.last_rebalance else None
        }