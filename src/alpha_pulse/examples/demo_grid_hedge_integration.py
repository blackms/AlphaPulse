"""
Example demonstrating grid hedge bot integration with different trading modes.
"""
import os
import time
from loguru import logger

from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.hedging.grid_hedge_config import GridHedgeConfig, GridDirection
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot


def run_grid_hedge_demo(trading_mode: str = TradingMode.PAPER):
    """
    Run grid hedge bot demo with specified trading mode.
    
    Args:
        trading_mode: One of REAL, PAPER, or RECOMMENDATION
    """
    # Configure logging
    os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
    logger.add(
        f"logs/grid_hedge_{trading_mode.lower()}_{int(time.time())}.log",
        rotation="1 day"
    )
    
    # Example configuration
    symbol = "BTCUSDT"
    center_price = 40000.0  # Example center price
    grid_spacing = 100.0    # $100 between grid levels
    num_levels = 5         # 5 levels above and below center
    position_step = 0.001  # 0.001 BTC per grid level
    max_position = 0.01   # Maximum 0.01 BTC total position
    
    # Create grid configuration
    config = GridHedgeConfig.create_symmetric_grid(
        symbol=symbol,
        center_price=center_price,
        grid_spacing=grid_spacing,
        num_levels=num_levels,
        position_step_size=position_step,
        max_position_size=max_position,
        grid_direction=GridDirection.BOTH
    )
    
    # Create appropriate broker based on trading mode
    if trading_mode == TradingMode.REAL:
        # For real trading, we need exchange credentials
        api_key = os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("EXCHANGE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError(
                "EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment variables "
                "are required for real trading"
            )
        
        broker = create_broker(
            trading_mode=trading_mode,
            exchange_name="binance",  # or "bybit"
            api_key=api_key,
            api_secret=api_secret,
            testnet=True  # Use testnet for safety
        )
    else:
        # Paper trading or recommendation mode
        broker = create_broker(trading_mode=trading_mode)
        
        # Initialize market data for paper trading
        if trading_mode == TradingMode.PAPER:
            broker.update_market_data(symbol, center_price)
    
    # Create and run the grid bot
    bot = GridHedgeBot(broker, config)
    logger.info(f"Starting grid hedge bot in {trading_mode} mode")
    
    try:
        # Main loop - in real implementation this would use real market data
        current_price = center_price
        price_direction = 1  # 1 for up, -1 for down
        
        while True:
            # Simulate price movement for demo
            current_price += price_direction * (grid_spacing * 0.1)  # Move 10% of grid spacing
            if current_price > center_price + (grid_spacing * 2):
                price_direction = -1
            elif current_price < center_price - (grid_spacing * 2):
                price_direction = 1
                
            # Update market data
            if trading_mode == TradingMode.PAPER:
                broker.update_market_data(symbol, current_price)
            
            # Execute grid strategy
            bot.execute(current_price)
            
            # Get and log status
            status = bot.get_status()
            logger.info(
                f"Grid Status - Price: {current_price:.2f}, "
                f"Active Orders: {status['active_orders']}, "
                f"Last Rebalance: {status['last_rebalance']}"
            )
            
            # Wait before next iteration
            time.sleep(config.rebalance_interval)
            
    except KeyboardInterrupt:
        logger.info("Shutting down grid hedge bot")
    except Exception as e:
        logger.error(f"Error running grid hedge bot: {str(e)}")
    finally:
        logger.info("Grid hedge bot stopped")


if __name__ == "__main__":
    # Example usage:
    
    # For paper trading:
    run_grid_hedge_demo(TradingMode.PAPER)
    
    # For recommendation mode:
    # run_grid_hedge_demo(TradingMode.RECOMMENDATION)
    
    # For real trading (requires API keys):
    # run_grid_hedge_demo(TradingMode.REAL)