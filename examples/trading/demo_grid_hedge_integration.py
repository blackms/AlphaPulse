"""
Example demonstrating grid hedge bot integration with different trading modes.
"""
import asyncio
import os
import signal
from decimal import Decimal
from typing import Optional
from loguru import logger

from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.hedging.grid import GridHedgeBot


# Global flag for graceful shutdown
shutdown_flag = False


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    global shutdown_flag
    logger.info("Received shutdown signal")
    shutdown_flag = True


async def run_grid_hedge_demo(
    trading_mode: str = TradingMode.PAPER,
    symbol: str = "BTCUSDT",
    config: Optional[dict] = None
) -> None:
    """
    Run grid hedge bot demo with specified trading mode.
    
    Args:
        trading_mode: One of REAL, PAPER, or RECOMMENDATION
        symbol: Trading symbol
        config: Optional configuration parameters
    """
    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logger.add(
        f"logs/grid_hedge_{trading_mode.lower()}.log",
        rotation="1 day",
        level="INFO"
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Default configuration
    default_config = {
        "grid_spacing_pct": "0.01",    # 1% grid spacing
        "num_levels": 5,               # 5 levels each side
        "max_position_size": "1.0",    # 1 BTC max position
        "max_drawdown": "0.1",         # 10% max drawdown
        "stop_loss_pct": "0.04",       # 4% stop loss
        "var_limit": "10000",          # $10k VaR limit
        "rebalance_interval_seconds": 60
    }
    
    # Merge with provided config
    config = {**default_config, **(config or {})}
    
    bot = None
    try:
        # Create broker based on trading mode
        if trading_mode == TradingMode.REAL:
            # For real trading, we need exchange credentials
            api_key = os.getenv("EXCHANGE_API_KEY")
            api_secret = os.getenv("EXCHANGE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError(
                    "EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment "
                    "variables are required for real trading"
                )
            
            broker = create_broker(
                trading_mode=trading_mode,
                exchange_name="binance",  # or "bybit"
                api_key=api_key,
                api_secret=api_secret,
                testnet=True  # Use testnet for safety
            )
            logger.info("Created real trading broker (testnet)")
            
        else:
            # Paper trading or recommendation mode
            broker = create_broker(trading_mode=trading_mode)
            logger.info(f"Created {trading_mode} broker")
        
        # Create and start grid bot
        bot = await GridHedgeBot.create(
            broker=broker,
            symbol=symbol,
            config=config
        )
        logger.info(f"Started grid bot in {trading_mode} mode")
        
        # Main loop
        while not shutdown_flag:
            try:
                # Get current market price
                current_price = await bot.data_provider.get_current_price(symbol)
                if not current_price:
                    logger.warning("Could not get current price")
                    await asyncio.sleep(1)
                    continue
                
                # Execute strategy iteration
                await bot.execute(float(current_price))
                
                # Get and log status
                status = bot.get_status()
                position = status["position"]
                metrics = status["metrics"]
                
                logger.info(
                    f"Grid Status - "
                    f"Price: {current_price:.2f}, "
                    f"Position: {position['spot']:.8f}, "
                    f"PnL: {position['unrealized_pnl']:.2f}, "
                    f"Win Rate: {metrics['win_rate']:.2%}, "
                    f"Drawdown: {metrics['max_drawdown']:.2%}"
                )
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in grid iteration: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Error running grid hedge bot: {str(e)}")
        
    finally:
        # Clean up resources
        if bot:
            await bot.stop()
            if bot.data_provider:
                await bot.data_provider.close()
        logger.info("Grid hedge bot stopped")


def main():
    """Main entry point."""
    # Example configuration
    config = {
        "grid_spacing_pct": "0.01",    # 1% grid spacing
        "num_levels": 5,               # 5 levels each side
        "max_position_size": "1.0",    # 1 BTC max position
        "max_drawdown": "0.1",         # 10% max drawdown
        "stop_loss_pct": "0.04",       # 4% stop loss
        "var_limit": "10000",          # $10k VaR limit
        "rebalance_interval_seconds": 60
    }
    
    # Run with different modes
    modes = {
        "paper": {
            "mode": TradingMode.PAPER,
            "config": config
        },
        "recommendation": {
            "mode": TradingMode.RECOMMENDATION,
            "config": {**config, "max_position_size": "5.0"}  # Larger for demo
        },
        "real": {
            "mode": TradingMode.REAL,
            "config": {**config, "max_position_size": "0.1"}  # Smaller for safety
        }
    }
    
    # Select mode (paper trading by default)
    mode_name = os.getenv("TRADING_MODE", "paper").lower()
    if mode_name not in modes:
        logger.warning(f"Invalid mode {mode_name}, using paper trading")
        mode_name = "paper"
        
    mode = modes[mode_name]
    
    try:
        # Run the demo
        asyncio.run(run_grid_hedge_demo(
            trading_mode=mode["mode"],
            symbol="BTCUSDT",
            config=mode["config"]
        ))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
