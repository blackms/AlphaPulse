"""
Example demonstrating grid hedge bot integration with different trading modes.
"""
import os
import time
import asyncio
import signal
from loguru import logger

from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.execution.broker_interface import Order, OrderSide, OrderType
from alpha_pulse.execution.paper_broker import PaperBroker
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider
from alpha_pulse.exchanges import ExchangeType
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot


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
    volatility: float = 0.02,  # 2% daily volatility
    spot_quantity: float = 1.0,  # Initial spot position size
):
    """
    Run grid hedge bot demo with specified trading mode.
    
    Args:
        trading_mode: One of REAL, PAPER, or RECOMMENDATION
        symbol: Trading symbol
        volatility: Volatility estimate for grid calculations
        spot_quantity: Initial spot position size for paper trading
    """
    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logger.add(
        f"logs/grid_hedge_{trading_mode.lower()}_{int(time.time())}.log",
        rotation="1 day"
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Initialize components
    data_provider = None
    broker = None
    bot = None
    
    try:
        # Initialize data provider
        data_provider = ExchangeDataProvider(
            exchange_type=ExchangeType.BINANCE,
            testnet=True
        )
        await data_provider.initialize()
        
        # Start market data updates
        market_data_task = asyncio.create_task(data_provider.start([symbol]))
        
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
            
            # For paper trading, initialize spot position
            if trading_mode == TradingMode.PAPER and isinstance(broker, PaperBroker):
                # Get current price
                current_price = await data_provider.get_current_price(symbol)
                if not current_price:
                    raise ValueError(f"Could not get current price for {symbol}")
                
                # Initialize spot position
                await broker.initialize_spot_position(
                    symbol=symbol,
                    quantity=spot_quantity,
                    price=float(current_price)
                )
                logger.info(
                    f"Initialized paper spot position: {spot_quantity} {symbol} "
                    f"@ {current_price}"
                )
        
        # Create and initialize grid bot
        bot = await GridHedgeBot.create_for_spot_hedge(
            broker=broker,
            symbol=symbol,
            data_provider=data_provider,
            volatility=volatility,
            spot_quantity=spot_quantity
        )
        
        logger.info(f"Starting grid hedge bot in {trading_mode} mode")
        
        # Main loop
        while not shutdown_flag:
            try:
                # Get current market price
                current_price = data_provider.get_price(symbol)
                if not current_price:
                    current_price = await data_provider.get_current_price(symbol)
                
                if current_price:
                    # Update market data for paper trading
                    if trading_mode == TradingMode.PAPER:
                        broker.update_market_data(symbol, float(current_price))
                    
                    # Get positions for monitoring
                    spot_pos = broker.get_position(symbol)
                    futures_pos = sum(
                        p.quantity for p in broker.get_positions().values()
                        if p.symbol.endswith('PERP')  # Perpetual futures
                    )
                    
                    # Execute grid strategy
                    bot.execute(float(current_price))
                    
                    # Get and log status
                    status = bot.get_status()
                    logger.info(
                        f"Grid Status - "
                        f"Price: {current_price:.2f}, "
                        f"Spot: {spot_pos.quantity if spot_pos else 0:.8f}, "
                        f"Futures: {futures_pos:.8f}, "
                        f"Active Orders: {status['active_orders']}, "
                        f"Stop Loss: {'Active' if status['stop_loss_active'] else 'Inactive'}, "
                        f"Take Profit: {'Active' if status['take_profit_active'] else 'Inactive'}, "
                        f"Last Rebalance: {status['last_rebalance']}"
                    )
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in grid iteration: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Error running grid hedge bot: {str(e)}")
    finally:
        logger.info("Grid hedge bot stopped")
        if data_provider:
            await data_provider.close()


if __name__ == "__main__":
    # Example usage:
    
    # For paper trading:
    asyncio.run(run_grid_hedge_demo(
        trading_mode=TradingMode.PAPER,
        symbol="BTCUSDT",
        volatility=0.02,  # 2% daily volatility
        spot_quantity=1.0  # 1 BTC spot position
    ))
    
    # For recommendation mode:
    # asyncio.run(run_grid_hedge_demo(TradingMode.RECOMMENDATION))
    
    # For real trading (requires API keys):
    # asyncio.run(run_grid_hedge_demo(TradingMode.REAL))
