import asyncio
from decimal import Decimal
import os
from loguru import logger

from alpha_pulse.exchanges.bybit import BybitExchange
from alpha_pulse.hedging.risk.config import HedgeConfig
from alpha_pulse.hedging.risk.analyzers.llm import LLMHedgeAnalyzer
from alpha_pulse.hedging.risk.manager import HedgeManager
from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
from alpha_pulse.hedging.execution.order_manager import GridOrderManager as ExchangeOrderExecutor
from alpha_pulse.hedging.common.interfaces import OrderManager as BasicExecutionStrategy

async def main():
    """
    Demonstrate the hedging system functionality with three-step LLM analysis:
    1. Basic strategy recommendations
    2. LLM-based recommendations
    3. Strategy comparison and evaluation
    """
    # Configure logging
    logger.add("logs/hedging_demo.log", rotation="1 day")
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    exchange = None
    try:
        # Create exchange connector
        # The credentials are loaded automatically from the credentials manager
        exchange = BybitExchange(
            testnet=os.getenv("ALPHA_PULSE_BYBIT_TESTNET", "true").lower() == "true"
        )
        await exchange.initialize()
        
        # Create hedge configuration
        config = HedgeConfig(
            hedge_ratio_target=Decimal('0.0'),  # Fully hedged
            max_leverage=Decimal('3.0'),
            max_margin_usage=Decimal('0.8'),
            min_position_size={'BTC': Decimal('0.001'), 'ETH': Decimal('0.01')},
            max_position_size={'BTC': Decimal('1.0'), 'ETH': Decimal('10.0')},
            grid_bot_enabled=False
        )
        
        # Create all required components
        position_fetcher = ExchangePositionFetcher(exchange)
        order_executor = ExchangeOrderExecutor(exchange)
        hedge_analyzer = LLMHedgeAnalyzer(config, openai_api_key)
        execution_strategy = BasicExecutionStrategy()
        
        # Create hedge manager with all components
        manager = HedgeManager(
            hedge_analyzer=hedge_analyzer,
            position_fetcher=position_fetcher,
            execution_strategy=execution_strategy,
            order_executor=order_executor,
            execute_hedge=False  # Set to True for live trading
        )
        
        # Get current positions
        logger.info("Fetching current positions...")
        spot_positions = await position_fetcher.get_spot_positions()
        futures_positions = await position_fetcher.get_futures_positions()
        
        logger.info("\nCurrent Positions:")
        logger.info("Spot Positions:")
        for pos in spot_positions:
            logger.info(
                f"  {pos.symbol}: {pos.quantity} @ {pos.avg_price} "
                f"(Current: {pos.current_price})"
            )
        
        logger.info("\nFutures Positions:")
        for pos in futures_positions:
            logger.info(
                f"  {pos.symbol}: {pos.quantity} {pos.side} @ {pos.entry_price} "
                f"(Current: {pos.current_price})"
            )
        
        # Run three-step analysis
        logger.info("\nRunning three-step hedge analysis...")
        recommendation = await hedge_analyzer.analyze(spot_positions, futures_positions)
        
        # Log detailed analysis results
        logger.info("\nHedge Analysis Results:")
        logger.info(recommendation.commentary)
        
        if recommendation.adjustments:
            logger.info("\nRecommended Position Adjustments:")
            for adj in recommendation.adjustments:
                logger.info(f"  {adj.recommendation} (Priority: {adj.priority})")
        
        # Example: Execute hedge adjustments
        if input("\nExecute hedge adjustments? (y/n): ").lower() == 'y':
            logger.info("Executing hedge adjustments...")
            await manager.manage_hedge()
        
        # Example: Close all hedges
        if input("\nClose all hedges? (y/n): ").lower() == 'y':
            logger.info("Closing all hedge positions...")
            await manager.close_all_hedges()
    
    except Exception as e:
        logger.error(f"Error in hedging demo: {str(e)}")
        raise
    
    finally:
        # Clean up
        if exchange:
            await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())