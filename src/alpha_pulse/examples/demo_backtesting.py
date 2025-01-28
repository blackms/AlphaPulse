"""
Example script demonstrating the AlphaPulse backtesting framework.
"""
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from alpha_pulse.backtesting import Backtester, DefaultStrategy, TrendFollowingStrategy
from alpha_pulse.data_pipeline import DataFetcher
from alpha_pulse.models import BasicModel
from alpha_pulse.features import FeatureEngineer


def create_sample_data(days: int = 100) -> tuple[pd.Series, pd.Series]:
    """Create sample price and signal data for demonstration."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )
    
    # Generate random walk prices
    prices = pd.Series(
        np.random.normal(0, 0.02, days).cumsum(),
        index=dates
    )
    prices = 100 * np.exp(prices)  # Convert to price levels
    
    # Generate sample signals (-1 to 1)
    signals = pd.Series(
        np.random.normal(0, 1, days),
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
    
    # Example of using with real data and model predictions
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        feature_engineer = FeatureEngineer()
        model = BasicModel()
        
        # Fetch some historical data
        historical_data = data_fetcher.fetch_historical_data(
            symbol="BTC/USD",
            timeframe="1d",
            start_time=datetime.now() - timedelta(days=365)
        )
        
        # Prepare features
        features = feature_engineer.create_features(historical_data)
        
        # Generate predictions
        predictions = model.predict(features)
        
        # Run backtest with real data
        real_results = backtester.backtest(
            prices=historical_data['close'],
            signals=predictions,
            strategy=DefaultStrategy()
        )
        
        logger.info("\nReal Data Backtest Results:")
        logger.info(real_results)
        
        # Plot real data results
        plot_backtest_results(real_results, historical_data['close'])
        
    except Exception as e:
        logger.warning(f"Could not run real data example: {str(e)}")
        logger.info("This is expected if you haven't set up the data pipeline yet.")


if __name__ == "__main__":
    main()