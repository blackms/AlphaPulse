from loguru import logger
from .interfaces import IExecutionStrategy, IOrderExecutor
from .models import HedgeRecommendation

class BasicExecutionStrategy(IExecutionStrategy):
    """Basic implementation of hedge execution strategy."""
    
    async def execute_hedge_adjustments(
        self,
        recommendation: HedgeRecommendation,
        executor: IOrderExecutor
    ) -> bool:
        """
        Execute hedge adjustments in order of priority.
        
        Args:
            recommendation: Hedge recommendations to execute
            executor: Order executor instance
            
        Returns:
            bool: True if all adjustments were executed successfully
        """
        if not recommendation.adjustments:
            logger.info("No hedge adjustments to execute")
            return True
        
        # Sort adjustments by priority
        sorted_adjustments = sorted(
            recommendation.adjustments,
            key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x.priority]
        )
        
        success = True
        for adj in sorted_adjustments:
            try:
                logger.info(f"Executing adjustment: {adj.recommendation}")
                
                order_id = await executor.place_futures_order(
                    symbol=adj.symbol,
                    side=adj.side,
                    quantity=float(adj.desired_delta),
                    order_type="MARKET"
                )
                
                logger.info(
                    f"Successfully executed adjustment: {adj.recommendation} "
                    f"(Order ID: {order_id})"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to execute adjustment: {adj.recommendation} "
                    f"Error: {str(e)}"
                )
                success = False
                break
        
        return success

class ExchangeOrderExecutor(IOrderExecutor):
    """Executes orders through an exchange."""
    
    def __init__(self, exchange):
        """
        Initialize the order executor.
        
        Args:
            exchange: Exchange connector instance
        """
        self.exchange = exchange
    
    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET"
    ) -> str:
        """Place a futures order through the exchange."""
        return await self.exchange.place_futures_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type
        )
    
    async def close_position(
        self,
        symbol: str,
        position_side: str,
        quantity: float
    ) -> str:
        """Close an existing position."""
        close_side = "SELL" if position_side == "LONG" else "BUY"
        return await self.place_futures_order(
            symbol=symbol,
            side=close_side,
            quantity=quantity
        )