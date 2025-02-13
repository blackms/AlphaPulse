import asyncio
from decimal import Decimal
import os
from loguru import logger
from typing import Dict, List, Tuple

from alpha_pulse.exchanges.bybit import BybitExchange
from alpha_pulse.hedging.risk.config import HedgeConfig
from alpha_pulse.hedging.risk.analyzers.llm import LLMHedgeAnalyzer
from alpha_pulse.hedging.risk.manager import HedgeManager
from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
from alpha_pulse.hedging.execution.order_executor import ExchangeOrderExecutor
from alpha_pulse.hedging.execution.strategy import BasicExecutionStrategy
from alpha_pulse.hedging.common.types import SpotPosition, FuturesPosition

def calculate_asset_metrics(
    spot_positions: List[SpotPosition],
    futures_positions: List[FuturesPosition]
) -> Dict[str, Dict[str, Decimal]]:
    """Calculate detailed metrics for each asset."""
    metrics = {}
    
    # Process spot positions
    for spot in spot_positions:
        if not spot.current_price:
            continue
            
        symbol = spot.symbol
        spot_value = spot.quantity * spot.current_price
        
        metrics[symbol] = {
            "spot_value": spot_value,
            "spot_qty": spot.quantity,
            "futures_value": Decimal('0'),
            "futures_qty": Decimal('0'),
            "net_exposure": spot_value,
            "hedge_ratio": Decimal('1.0')  # Default to unhedged
        }
    
    # Process futures positions
    for futures in futures_positions:
        if not futures.current_price:
            continue
            
        # Extract base symbol (remove USDT suffix)
        symbol = futures.symbol.replace("USDT", "")
        futures_value = futures.quantity * futures.current_price
        
        if symbol not in metrics:
            metrics[symbol] = {
                "spot_value": Decimal('0'),
                "spot_qty": Decimal('0'),
                "futures_value": Decimal('0'),
                "futures_qty": Decimal('0'),
                "net_exposure": Decimal('0'),
                "hedge_ratio": Decimal('0')
            }
        
        # Adjust values based on position side
        futures_value_signed = futures_value * (-1 if futures.side == "SHORT" else 1)
        metrics[symbol]["futures_value"] = futures_value_signed
        metrics[symbol]["futures_qty"] = futures.quantity * (1 if futures.side == "LONG" else -1)
        
        # Calculate net exposure and hedge ratio
        spot_value = metrics[symbol]["spot_value"]
        net_exposure = spot_value + futures_value_signed
        
        # Calculate hedge ratio (1.0 = unhedged, 0.0 = fully hedged)
        hedge_ratio = Decimal('1.0')
        if spot_value != 0:
            hedge_ratio = net_exposure / spot_value
        
        metrics[symbol]["net_exposure"] = net_exposure
        metrics[symbol]["hedge_ratio"] = hedge_ratio
    
    return metrics

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
        
        # Calculate detailed metrics for each asset
        asset_metrics = calculate_asset_metrics(spot_positions, futures_positions)
        
        # Display detailed position analysis
        logger.info("\n=== Detailed Position Analysis ===")
        
        total_spot_value = Decimal('0')
        total_futures_value = Decimal('0')
        total_net_exposure = Decimal('0')
        
        # Sort assets by absolute spot value
        sorted_assets = sorted(
            asset_metrics.items(),
            key=lambda x: abs(x[1]["spot_value"]),
            reverse=True
        )
        
        for symbol, metrics in sorted_assets:
            if metrics["spot_value"] == 0 and metrics["futures_value"] == 0:
                continue
                
            logger.info(f"\n{symbol}:")
            logger.info(f"  Spot Position:    {metrics['spot_qty']:.8f} "
                       f"(${metrics['spot_value']:.2f})")
            logger.info(f"  Futures Position: {metrics['futures_qty']:.8f} "
                       f"(${metrics['futures_value']:.2f})")
            logger.info(f"  Net Exposure:     ${metrics['net_exposure']:.2f}")
            
            # Calculate and display hedge coverage
            spot_value = abs(metrics["spot_value"])
            futures_value = abs(metrics["futures_value"])
            if spot_value > 0:
                hedge_coverage = (futures_value / spot_value) * 100
                logger.info(f"  Hedge Coverage:   {hedge_coverage:.1f}% "
                          f"({'Overhedged' if hedge_coverage > 100 else 'Underhedged' if hedge_coverage < 100 else 'Fully Hedged'})")
            
            total_spot_value += metrics["spot_value"]
            total_futures_value += metrics["futures_value"]
            total_net_exposure += metrics["net_exposure"]
        
        # Display portfolio summary
        logger.info("\n=== Portfolio Summary ===")
        logger.info(f"Total Spot Value:     ${total_spot_value:.2f}")
        logger.info(f"Total Futures Value:  ${total_futures_value:.2f}")
        logger.info(f"Total Net Exposure:   ${total_net_exposure:.2f}")
        
        if total_spot_value != 0:
            portfolio_hedge_ratio = (total_net_exposure / total_spot_value) * 100
            logger.info(f"Portfolio Hedge Ratio: {portfolio_hedge_ratio:.1f}%")
        
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