"""
Example demonstrating grid hedge bot integration with different trading modes.
"""
import os
import time
import asyncio
import signal
from decimal import Decimal
from loguru import logger

from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.execution.broker_interface import Order, OrderSide, OrderType, Position
from alpha_pulse.execution.paper_broker import PaperBroker
from alpha_pulse.execution.recommendation_broker import RecommendationOnlyBroker
from alpha_pulse.hedging.grid_hedge_config import GridHedgeConfig, GridDirection
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot
from alpha_pulse.risk_management.analysis import RiskAnalyzer
from alpha_pulse.risk_management.position_sizing import VolatilityBasedSizer
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider
from alpha_pulse.exchanges import ExchangeType


# Global flag for graceful shutdown
shutdown_flag = False


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    global shutdown_flag
    logger.info("Received shutdown signal")
    shutdown_flag = True


async def initialize_paper_spot_position(
    broker,
    symbol: str,
    quantity: float,
    price: float
) -> Position:
    """Initialize a paper trading spot position."""
    # Create market buy order
    order = Order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.MARKET,
        price=price
    )
    
    # Update market data first
    broker.update_market_data(symbol, price)
    
    # Place the order
    result = broker.place_order(order)
    if result.status.value != "filled":
        raise ValueError(f"Failed to create initial spot position: {result.status}")
    
    # Verify position
    position = broker.get_position(symbol)
    if not position or position.quantity != quantity:
        raise ValueError("Failed to verify spot position")
    
    return position


async def calculate_grid_parameters(
    current_price: float,
    spot_quantity: float,
    volatility: float,
    risk_analyzer: RiskAnalyzer,
    position_sizer: VolatilityBasedSizer,
    portfolio_value: float,
    historical_returns=None,
) -> tuple:
    """
    Calculate grid parameters based on risk metrics.
    
    Returns:
        tuple: (grid_spacing, position_step, stop_loss, take_profit)
    """
    # Calculate position size per grid level using volatility-based sizing
    position_result = position_sizer.calculate_position_size(
        symbol="BTCUSDT",
        current_price=current_price,
        portfolio_value=portfolio_value,
        volatility=volatility,
        signal_strength=1.0,  # Full signal for hedging
        historical_returns=historical_returns
    )
    
    # Calculate grid spacing based on volatility
    # Use 0.5 standard deviations for grid spacing
    grid_spacing = current_price * volatility * 0.5
    
    # Calculate position step size (ensure total matches spot position)
    num_levels = 5  # Default number of grid levels
    position_step = min(
        spot_quantity / num_levels,  # Even distribution
        position_result.size / current_price  # Risk-based limit
    )
    
    # Calculate stop loss and take profit using VaR
    if historical_returns is not None:
        metrics = risk_analyzer.calculate_metrics(historical_returns)
        stop_loss = metrics.var_95  # Use 95% VaR for stop loss
        take_profit = -stop_loss * 1.5  # Target 1.5x risk/reward
    else:
        # Default to volatility-based levels if no historical data
        stop_loss = volatility * 2  # 2 standard deviations
        take_profit = stop_loss * 1.5
    
    return grid_spacing, position_step, stop_loss, take_profit


async def create_hedge_grid(
    broker,
    data_provider: ExchangeDataProvider,
    symbol: str,
    volatility: float = 0.02,  # Default 2% daily volatility
    portfolio_value: float = None,
    spot_quantity: float = 1.0,  # Default spot position size
) -> GridHedgeConfig:
    """
    Create a grid configuration based on current spot position and risk metrics.
    
    Args:
        broker: Trading broker instance
        data_provider: Exchange data provider
        symbol: Trading symbol (e.g., 'BTCUSDT')
        volatility: Current volatility estimate
        portfolio_value: Total portfolio value (optional)
        spot_quantity: Default spot quantity if no position exists
    """
    # Get current market price
    retries = 3
    current_price = None
    
    for _ in range(retries):
        try:
            current_price = await data_provider.get_current_price(symbol)
            if current_price:
                break
        except Exception as e:
            logger.warning(f"Error getting price, retrying: {e}")
            await asyncio.sleep(1)
    
    if not current_price:
        raise ValueError(f"Could not get current price for {symbol} after {retries} attempts")
    
    # Get or create spot position
    spot_position = broker.get_position(symbol)
    if not spot_position and isinstance(broker, (PaperBroker, RecommendationOnlyBroker)):
        logger.info(f"Creating paper spot position: {spot_quantity} {symbol}")
        spot_position = await initialize_paper_spot_position(
            broker=broker,
            symbol=symbol,
            quantity=spot_quantity,
            price=float(current_price)
        )
    
    if not spot_position:
        raise ValueError(f"No spot position found for {symbol}")
    
    spot_quantity = spot_position.quantity
    
    if not portfolio_value:
        portfolio_value = broker.get_portfolio_value()
    
    # Initialize risk management components
    risk_analyzer = RiskAnalyzer(
        rolling_window=20,  # 20-day window
        var_confidence=0.95,
    )
    
    position_sizer = VolatilityBasedSizer(
        target_volatility=0.01,  # 1% daily target
        max_size_pct=0.2,  # Maximum 20% of portfolio per grid level
        volatility_lookback=20,
    )
    
    # Calculate grid parameters
    grid_spacing, position_step, stop_loss, take_profit = await calculate_grid_parameters(
        current_price=float(current_price),
        spot_quantity=spot_quantity,
        volatility=volatility,
        risk_analyzer=risk_analyzer,
        position_sizer=position_sizer,
        portfolio_value=portfolio_value
    )
    
    logger.info(
        f"Creating hedge grid for {symbol} - "
        f"Spot: {spot_quantity:.8f}, "
        f"Price: {current_price:.2f}, "
        f"Grid Spacing: {grid_spacing:.2f}, "
        f"Position Step: {position_step:.8f}, "
        f"Stop Loss: {stop_loss:.2%}, "
        f"Take Profit: {take_profit:.2%}"
    )
    
    return GridHedgeConfig.create_symmetric_grid(
        symbol=symbol,
        center_price=float(current_price),
        grid_spacing=grid_spacing,
        num_levels=5,
        position_step_size=position_step,
        max_position_size=spot_quantity,  # Don't exceed spot position
        grid_direction=GridDirection.SHORT  # Only short for hedging
    )


async def run_grid_hedge_demo(
    trading_mode: str = TradingMode.PAPER,
    symbol: str = "BTCUSDT",  # Use exchange format
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
    
    # Initialize exchange data provider
    data_provider = None
    broker = None
    
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
        
        # Create initial grid based on spot position and risk metrics
        config = await create_hedge_grid(
            broker=broker,
            data_provider=data_provider,
            symbol=symbol,
            volatility=volatility,
            spot_quantity=spot_quantity
        )
        
        # Create and run the grid bot
        bot = GridHedgeBot(broker, config)
        logger.info(f"Starting grid hedge bot in {trading_mode} mode")
        
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
                        f"Grid Status - Price: {current_price:.2f}, "
                        f"Spot: {spot_pos.quantity if spot_pos else 0:.8f}, "
                        f"Futures: {futures_pos:.8f}, "
                        f"Active Orders: {status['active_orders']}, "
                        f"Last Rebalance: {status['last_rebalance']}"
                    )
                
                # Wait before next iteration
                await asyncio.sleep(config.rebalance_interval)
                
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
        symbol="BTCUSDT",  # Use exchange format
        volatility=0.02,  # 2% daily volatility
        spot_quantity=1.0  # 1 BTC spot position
    ))
    
    # For recommendation mode:
    # asyncio.run(run_grid_hedge_demo(TradingMode.RECOMMENDATION))
    
    # For real trading (requires API keys):
    # asyncio.run(run_grid_hedge_demo(TradingMode.REAL))
