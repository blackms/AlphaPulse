"""
Example script demonstrating backtesting with historical data.

This script shows how to:
1. Download and prepare historical data using the data pipeline
2. Convert data into a format suitable for backtesting
3. Run backtests with different strategies
4. Visualize and analyze results
"""
import asyncio
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt

from alpha_pulse.exchanges import ExchangeType
from alpha_pulse.data_pipeline import (
    SQLAlchemyStorage,
    ExchangeFetcher,
    HistoricalDataManager,
    StorageConfig,
    DataFetchConfig,
    validate_timeframe,
    validate_symbol
)
from alpha_pulse.backtesting import (
    Backtester,
    DefaultStrategy,
    TrendFollowingStrategy,
    MeanReversionStrategy
)


async def fetch_historical_prices(
    manager: HistoricalDataManager,
    exchange_type: ExchangeType,
    symbol: str,
    timeframe: str,
    days: int = 252  # ~1 trading year
) -> pd.Series:
    """
    Fetch historical prices and convert to pandas Series.

    Args:
        manager: Historical data manager
        exchange_type: Type of exchange
        symbol: Trading symbol
        timeframe: Candle timeframe
        days: Number of days of history

    Returns:
        Price series indexed by timestamp
    """
    # Validate inputs
    validate_symbol(symbol)
    validate_timeframe(timeframe)
    
    # Define time range
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)
    
    # Ensure data is available
    await manager.ensure_data_available(
        exchange_type=exchange_type,
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time
    )
    
    # Get historical data
    data = manager.get_historical_data(
        exchange_type=exchange_type,
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time
    )
    
    if not data:
        logger.error("❌ No historical data retrieved")
        return pd.Series(dtype=float)
    
    # Convert to price series
    dates = [candle.timestamp for candle in data]
    prices = [candle.close for candle in data]
    return pd.Series(prices, index=pd.to_datetime(dates))


def plot_backtest_results(results, prices: pd.Series, save_path: Path):
    """
    Plot equity curve and trading activity.

    Args:
        results: Backtest results
        prices: Price series
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results.equity_curve.index, results.equity_curve.values, label='Portfolio Value')
    plt.title('Backtest Results')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot price series and trade points
    plt.subplot(2, 1, 2)
    plt.plot(prices.index, prices.values, label='Price', alpha=0.7)
    
    # Plot entry points
    entries = [pos.entry_time for pos in results.positions]
    entry_prices = [pos.entry_price for pos in results.positions]
    plt.scatter(entries, entry_prices, color='g', marker='^', label='Entry')
    
    # Plot exit points
    exits = [pos.exit_time for pos in results.positions if pos.exit_time]
    exit_prices = [pos.exit_price for pos in results.positions if pos.exit_price]
    plt.scatter(exits, exit_prices, color='r', marker='v', label='Exit')
    
    plt.title('Trading Activity')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"📊 Saved backtest plots to {save_path}")


async def main():
    """Run backtesting demonstration."""
    logger.info("🚀 Starting backtesting demonstration...")
    
    # Create output directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize components
        storage = SQLAlchemyStorage(config=StorageConfig())
        fetcher = ExchangeFetcher(config=DataFetchConfig())
        manager = HistoricalDataManager(storage, fetcher)
        
        # Parameters
        exchange_type = ExchangeType.BINANCE
        symbol = "BTC/USDT"
        timeframe = "1d"
        
        # Fetch historical prices
        prices = await fetch_historical_prices(
            manager=manager,
            exchange_type=exchange_type,
            symbol=symbol,
            timeframe=timeframe
        )
        
        if prices.empty:
            logger.error("❌ Unable to retrieve historical price data")
            return
        
        # Generate dummy trading signals (values between -1 and 1)
        signals = pd.Series(
            np.random.uniform(-1, 1, len(prices)),
            index=prices.index
        )
        
        # Initialize backtester
        backtester = Backtester(
            commission=0.001,      # 0.1% commission
            initial_capital=100000,
            position_size=0.1      # Risk 10% of capital per trade
        )
        
        # Test different strategies
        strategies = {
            "Default": DefaultStrategy(threshold=0.0),
            "Trend Following": TrendFollowingStrategy(),
            "Mean Reversion": MeanReversionStrategy()
        }
        
        logger.info("\n📈 Running backtests...")
        for name, strategy in strategies.items():
            logger.info(f"\nTesting {name} strategy:")
            
            # Run backtest
            results = backtester.backtest(
                prices=prices,
                signals=signals,
                strategy=strategy
            )
            
            # Log results
            logger.info(f"Total Return: {results.total_return:.2%}")
            logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
            logger.info(f"Win Rate: {results.win_rate:.2%}")
            
            # Plot results
            plot_path = plots_dir / f"backtest_results_{name.lower().replace(' ', '_')}.png"
            plot_backtest_results(results, prices, plot_path)
        
        logger.info("\n✨ Backtesting completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during backtesting: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())