from loguru import logger

from .interfaces import (
    IHedgeAnalyzer,
    IPositionFetcher,
    IExecutionStrategy,
    IOrderExecutor
)

class HedgeManager:
    """
    Orchestrates the hedging process using provided components.
    
    This class follows the Single Responsibility Principle by focusing solely on
    orchestrating the hedging workflow, delegating specific tasks to specialized
    components.
    """
    
    def __init__(
        self,
        hedge_analyzer: IHedgeAnalyzer,
        position_fetcher: IPositionFetcher,
        execution_strategy: IExecutionStrategy,
        order_executor: IOrderExecutor,
        execute_hedge: bool = False
    ):
        """
        Initialize the hedge manager.
        
        Args:
            hedge_analyzer: Component for analyzing positions and generating recommendations
            position_fetcher: Component for fetching position data
            execution_strategy: Strategy for executing hedge adjustments
            order_executor: Component for executing orders
            execute_hedge: Whether to execute recommended changes
        """
        self.hedge_analyzer = hedge_analyzer
        self.position_fetcher = position_fetcher
        self.execution_strategy = execution_strategy
        self.order_executor = order_executor
        self.execute_hedge = execute_hedge
    
    async def manage_hedge(self) -> None:
        """
        Main method to manage hedging process.
        
        1. Fetches current positions
        2. Analyzes hedge requirements
        3. Executes or logs recommended changes
        """
        try:
            # Get current positions
            spot_positions = await self.position_fetcher.get_spot_positions()
            futures_positions = await self.position_fetcher.get_futures_positions()
            
            # Get hedge recommendations
            # Note: analyze is now properly awaited for both sync and async implementations
            recommendation = await self.hedge_analyzer.analyze(
                spot_positions,
                futures_positions
            ) if hasattr(self.hedge_analyzer.analyze, '__await__') else \
            self.hedge_analyzer.analyze(spot_positions, futures_positions)
            
            # Log analysis results
            logger.info(f"Hedge Analysis:\n{recommendation.commentary}")
            
            if not recommendation.adjustments:
                logger.info("No hedge adjustments needed")
                return
            
            # Execute or log adjustments
            if self.execute_hedge:
                success = await self.execution_strategy.execute_hedge_adjustments(
                    recommendation,
                    self.order_executor
                )
                
                if not success:
                    logger.warning("Some hedge adjustments failed to execute")
            else:
                logger.info("Dry run mode - logging recommendations only:")
                for adj in recommendation.adjustments:
                    logger.info(f"  {adj.recommendation} (Priority: {adj.priority})")
        
        except Exception as e:
            logger.error(f"Error in hedge management: {str(e)}")
            raise
    
    async def close_all_hedges(self) -> None:
        """Close all futures positions."""
        try:
            futures_positions = await self.position_fetcher.get_futures_positions()
            
            for pos in futures_positions:
                if pos.quantity > 0:
                    try:
                        order_id = await self.order_executor.close_position(
                            symbol=pos.symbol,
                            position_side=pos.side,
                            quantity=float(pos.quantity)
                        )
                        logger.info(
                            f"Closed {pos.side} position of {pos.quantity} {pos.symbol} "
                            f"(Order ID: {order_id})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to close position {pos.symbol}: {str(e)}"
                        )
        
        except Exception as e:
            logger.error(f"Error closing hedges: {str(e)}")
            raise