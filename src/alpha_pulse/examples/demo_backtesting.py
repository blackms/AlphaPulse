"""
Example script demonstrating the AlphaPulse backtesting framework.
"""
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from alpha_pulse.backtesting import Backtester, DefaultStrategy, TrendFollowingStrategy


def create_sample_data(days: int = 100) -> tuple[pd.Series, pd.Series]:
    """Create sample price and signal data for demonstration."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days-1),  # Adjust to get exact number of days
        end=datetime.now(),
        freq='D'
    )
    
    # Generate random walk prices
    rw = np.random.normal(0, 0.02, len(dates)).cumsum()
    prices = pd.Series(
        100 * np.exp(rw),  # Convert to price levels
        index=dates
    )
    
    # Generate sample signals (-1 to 1)
    signals = pd.Series(
        np.random.normal(0, 1, len(dates)),
        index=dates
    )
    
    return prices, signals


def plot_backtest_results(results, prices):
    """Plot equity curve and trade points."""
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results.equity_curve.index, results.equity_curve.values, label='Equity')
    plt.title('Backtest Results')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # Plot price and trades
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
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/backtest_results.png')
    logger.info("Saved backtest plots to plots/backtest_results.png")


def main():
    """Run backtesting demonstration."""
    logger.info("Starting backtesting demonstration...")

    # Create sample data
    prices, signals = create_sample_data(days=252)  # One year of daily data
    
    # Initialize backtester with default parameters
    backtester = Backtester(
        commission=0.001,  # 0.1% commission
        initial_capital=100000,
        position_size=0.1  # Risk 10% of capital per trade
    )
    
    # Run backtest with default strategy
    default_results = backtester.backtest(
        prices=prices,
        signals=signals,
        strategy=DefaultStrategy(threshold=0.0)
    )
    
    logger.info("\nDefault Strategy Results:")
    logger.info(default_results)
    
    # Run backtest with trend following strategy
    trend_results = backtester.backtest(
        prices=prices,
        signals=signals,
        strategy=TrendFollowingStrategy(entry_threshold=0.5, exit_threshold=-0.5)
    )
    
    logger.info("\nTrend Following Strategy Results:")
    logger.info(trend_results)
    
    # Plot results
    plot_backtest_results(trend_results, prices)


if __name__ == "__main__":
    main()