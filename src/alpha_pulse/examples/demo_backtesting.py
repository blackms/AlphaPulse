"""
Example script demonstrating the integration of historical data downloading and backtesting.
This script fetches historical market data using the HistoricalDataManager, then converts the data into
a price series for backtesting with a simple trading strategy.
"""
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from alpha_pulse.backtesting import Backtester, DefaultStrategy
from alpha_pulse.data_pipeline.storage import SQLAlchemyStorage
from alpha_pulse.data_pipeline.historical_data_manager import HistoricalDataManager
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider
from alpha_pulse.data_pipeline.models import OHLCV

async def fetch_historical_prices(exchange_id: str, symbol: str, timeframe: str, days: int = 252) -> pd.Series:
    """
    Fetch historical prices for a given symbol and timeframe.
    If data is missing, it downloads and stores the data.
    """
    storage = SQLAlchemyStorage()
    provider = ExchangeDataProvider()
    await provider.initialize()
    hdm = HistoricalDataManager(storage, provider)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    # Fetch historical data; if missing, it will be downloaded.
    data = await hdm.get_historical_data(exchange_id, symbol, timeframe, start_time, end_time)
    if not data:
        logger.error("No historical data fetched.")
        return pd.Series(dtype=float)
    # Convert list of OHLCV objects to a Pandas Series using the closing price.
    data_sorted = sorted(data, key=lambda x: x.timestamp)
    dates = [ohlcv.timestamp for ohlcv in data_sorted]
    prices = [ohlcv.close for ohlcv in data_sorted]
    price_series = pd.Series(prices, index=pd.to_datetime(dates))
    return price_series

def plot_backtest_results(results, prices):
    """
    Plot equity curve and trading activity from backtesting results.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results.equity_curve.index, results.equity_curve.values, label='Equity')
    plt.title('Backtest Results')
    plt.ylabel('Portfolio Value')
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
    exits = [pos.exit_time for pos in results.positions if pos.exit_time is not None]
    exit_prices = [pos.exit_price for pos in results.positions if pos.exit_price is not None]
    plt.scatter(exits, exit_prices, color='r', marker='v', label='Exit')
    
    plt.title('Trading Activity')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/backtest_results.png')
    logger.info("Saved backtest plots to plots/backtest_results.png")

async def main():
    logger.info("Starting historical data backtesting demonstration...")
    # Parameters for historical data
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "1d"  # Daily data
    # Fetch historical prices (252 days ~ one trading year)
    prices = await fetch_historical_prices(exchange_id, symbol, timeframe, days=252)
    if prices.empty:
        logger.error("Unable to retrieve historical price data. Exiting demo.")
        return
    
    # Generate dummy trading signals (values between -1 and 1)
    signals = pd.Series(np.random.uniform(-1, 1, len(prices)), index=prices.index)
    
    # Initialize backtester with defined trading parameters.
    backtester = Backtester(
        commission=0.001,      # 0.1% commission per trade
        initial_capital=100000,
        position_size=0.1      # Risk 10% of capital per trade
    )
    
    # Run backtest using DefaultStrategy with a simple threshold.
    results = backtester.backtest(
        prices=prices,
        signals=signals,
        strategy=DefaultStrategy(threshold=0.0)
    )
    
    logger.info("Backtest Results:")
    logger.info(results)
    plot_backtest_results(results, prices)

if __name__ == "__main__":
    asyncio.run(main())