"""
Order execution for hedging strategies.
"""
from decimal import Decimal
from typing import Dict, List, Optional
from loguru import logger

from alpha_pulse.exchanges.base import BaseExchange
from alpha_pulse.execution.broker_interface import Order, OrderSide, OrderType


class ExchangeOrderExecutor:
    """Executes orders through an exchange."""
    
    def __init__(self, exchange: BaseExchange):
        """
        Initialize order executor.
        
        Args:
            exchange: Exchange connector instance
        """
        self.exchange = exchange
        logger.info("Initialized exchange order executor")
    
    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "MARKET",
        price: Optional[Decimal] = None
    ) -> Optional[str]:
        """
        Execute a single order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type (MARKET/LIMIT)
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                quantity=float(quantity),
                order_type=OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT,
                price=float(price) if price else None
            )
            
            result = await self.exchange.place_order(order)
            if result.order_id:
                logger.info(
                    f"Executed {side} order for {quantity} {symbol} "
                    f"at {price if price else 'market'}"
                )
                return result.order_id
            else:
                logger.error(f"Failed to execute order: {result.error}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return None
    
    async def close_position(
        self,
        symbol: str,
        quantity: Decimal,
        side: str
    ) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            side: Current position side
            
        Returns:
            True if successful
        """
        try:
            # Determine closing side
            close_side = "SELL" if side == "BUY" else "BUY"
            
            # Execute market order
            order_id = await self.execute_order(
                symbol=symbol,
                side=close_side,
                quantity=quantity
            )
            
            return order_id is not None
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False