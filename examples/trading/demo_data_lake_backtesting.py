"""
Enhanced backtesting demo using the data lake Silver layer for improved performance.

This script demonstrates:
1. Loading historical data from data lake Silver layer instead of PostgreSQL
2. Using time travel features for point-in-time analysis
3. Loading pre-computed technical features
4. Enhanced benchmark comparison with data lake stored benchmarks
5. Performance comparison between database and data lake approaches
"""
import asyncio
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import time

from alpha_pulse.backtesting.backtester import Backtester
from alpha_pulse.backtesting.strategy import BaseStrategy
from alpha_pulse.backtesting.models import Position
from alpha_pulse.backtesting.data_lake_loader import DataLakeBacktestingLoader, get_data_lake_loader
from alpha_pulse.backtesting.data_loader import load_ohlcv_data  # Original loader for comparison


class MovingAverageCrossStrategy(BaseStrategy):
    """Moving average crossover strategy for demonstration."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self.position_open = False
        logger.info(f"Initialized MA Cross Strategy ({short_window}/{long_window})")
    
    def should_enter(self, signal: pd.Series) -> bool:
        """Enter when short MA crosses above long MA."""
        if len(signal) < self.long_window:
            return False
        
        # Calculate moving averages
        short_ma = signal.rolling(window=self.short_window).mean()
        long_ma = signal.rolling(window=self.long_window).mean()
        
        # Check for crossover
        if len(short_ma) >= 2 and len(long_ma) >= 2:
            prev_short = short_ma.iloc[-2]
            prev_long = long_ma.iloc[-2]
            curr_short = short_ma.iloc[-1]
            curr_long = long_ma.iloc[-1]
            
            # Bullish crossover
            crossover = (prev_short <= prev_long) and (curr_short > curr_long)
            
            if crossover and not self.position_open:
                self.position_open = True
                return True
        
        return False
    
    def should_exit(self, signal: pd.Series, position: Position) -> bool:
        """Exit when short MA crosses below long MA."""
        if len(signal) < self.long_window:
            return False
        
        # Calculate moving averages
        short_ma = signal.rolling(window=self.short_window).mean()
        long_ma = signal.rolling(window=self.long_window).mean()
        
        # Check for bearish crossover
        if len(short_ma) >= 2 and len(long_ma) >= 2:
            prev_short = short_ma.iloc[-2]
            prev_long = long_ma.iloc[-2]
            curr_short = short_ma.iloc[-1]
            curr_long = long_ma.iloc[-1]
            
            # Bearish crossover
            crossover = (prev_short >= prev_long) and (curr_short < curr_long)
            
            if crossover and self.position_open:
                self.position_open = False
                return True
        
        return False


class TechnicalFeaturesStrategy(BaseStrategy):
    """Strategy using pre-computed technical features from data lake."""
    
    def __init__(self, rsi_buy_threshold: float = 30, rsi_sell_threshold: float = 70):
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.position_open = False
        logger.info(f"Initialized Technical Features Strategy (RSI {rsi_buy_threshold}/{rsi_sell_threshold})")
    
    def should_enter(self, signal: pd.Series) -> bool:
        """Enter when RSI indicates oversold condition."""
        # This would use pre-computed RSI from data lake features
        # For demo, we'll compute it here
        if len(signal) < 14:
            return False
        
        rsi = self._calculate_rsi(signal, 14)
        
        if not rsi.empty and rsi.iloc[-1] < self.rsi_buy_threshold and not self.position_open:
            self.position_open = True
            return True
        
        return False
    
    def should_exit(self, signal: pd.Series, position: Position) -> bool:
        """Exit when RSI indicates overbought condition."""
        if len(signal) < 14:
            return False
        
        rsi = self._calculate_rsi(signal, 14)
        
        if not rsi.empty and rsi.iloc[-1] > self.rsi_sell_threshold and self.position_open:
            self.position_open = False
            return True
        
        return False
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


async def demo_data_lake_backtesting():
    """Main demo function."""
    logger.info("Starting Data Lake Backtesting Demo")
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    timeframe = '1d'
    days_back = 365  # 1 year of data
    initial_capital = 100000
    
    # Date range
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Testing symbols: {symbols}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Initialize data lake loader
    data_lake_loader = get_data_lake_loader(
        data_lake_path="./data_lake",
        enable_spark=False  # Set to True if Spark is available
    )
    
    # 1. Compare loading performance: Database vs Data Lake
    await compare_loading_performance(
        symbols, timeframe, start_date, end_date, data_lake_loader
    )
    
    # 2. Demonstrate data lake features
    await demonstrate_data_lake_features(data_lake_loader)
    
    # 3. Run backtests with data lake data
    await run_enhanced_backtests(
        symbols, timeframe, start_date, end_date, 
        initial_capital, data_lake_loader
    )
    
    # 4. Demonstrate time travel capabilities
    await demonstrate_time_travel(
        symbols[0], timeframe, start_date, end_date, data_lake_loader
    )


async def compare_loading_performance(
    symbols, timeframe, start_date, end_date, data_lake_loader
):
    """Compare loading performance between database and data lake."""
    logger.info("=== Performance Comparison: Database vs Data Lake ===")
    
    # Test data lake loading
    start_time = time.time()
    try:
        lake_data = await data_lake_loader.load_ohlcv_from_lake(
            symbols=symbols,
            timeframe=timeframe,
            start_dt=start_date,
            end_dt=end_date
        )
        lake_time = time.time() - start_time
        lake_records = sum(len(df) for df in lake_data.values())
        logger.info(f"Data Lake: Loaded {lake_records} records in {lake_time:.2f}s")
    except Exception as e:
        logger.warning(f"Data lake loading failed: {e}")
        lake_time = float('inf')
        lake_records = 0
    
    # Test database loading (if available)
    start_time = time.time()
    try:
        db_data = await load_ohlcv_data(
            symbols=symbols,
            timeframe=timeframe,
            start_dt=start_date,
            end_dt=end_date
        )
        if db_data:
            db_time = time.time() - start_time
            db_records = sum(len(df) for df in db_data.values())
            logger.info(f"Database: Loaded {db_records} records in {db_time:.2f}s")
            
            # Calculate speedup
            if lake_time != float('inf') and db_time > 0:
                speedup = db_time / lake_time
                logger.info(f"Data Lake is {speedup:.1f}x faster than database")
        else:
            logger.warning("Database loading returned no data")
    except Exception as e:
        logger.warning(f"Database loading failed: {e}")


async def demonstrate_data_lake_features(data_lake_loader):
    """Demonstrate data lake specific features."""
    logger.info("=== Data Lake Features Demo ===")
    
    # List available datasets
    datasets = data_lake_loader.get_available_datasets()
    logger.info(f"Available datasets: {datasets}")
    
    # Get dataset info
    for dataset in datasets[:3]:  # Show info for first 3 datasets
        info = data_lake_loader.get_dataset_info(dataset)
        if info:
            logger.info(f"Dataset '{dataset}': {len(info.get('columns', []))} columns")
    
    # Cache statistics
    cache_stats = data_lake_loader.get_cache_stats()
    logger.info(f"Cache: {cache_stats['entries']} entries, {cache_stats['total_size_mb']:.1f} MB")


async def run_enhanced_backtests(
    symbols, timeframe, start_date, end_date, initial_capital, data_lake_loader
):
    """Run backtests using data lake data."""
    logger.info("=== Enhanced Backtesting with Data Lake ===")
    
    # Load data from data lake
    market_data = await data_lake_loader.load_ohlcv_from_lake(
        symbols=symbols,
        timeframe=timeframe,
        start_dt=start_date,
        end_dt=end_date
    )
    
    if not market_data or all(df.empty for df in market_data.values()):
        logger.error("No market data available for backtesting")
        return
    
    # Load benchmark data
    benchmark_data = await data_lake_loader.load_benchmark_data(
        benchmark_symbol='SPY',
        start_dt=start_date,
        end_dt=end_date,
        timeframe=timeframe
    )
    
    # Initialize backtester
    backtester = Backtester(
        commission=0.001,  # 0.1% commission
        initial_capital=initial_capital,
        slippage=0.0005,  # 0.05% slippage
        benchmark_symbol='SPY'
    )
    
    # Test strategies
    strategies = [
        ("MA Cross 20/50", MovingAverageCrossStrategy(20, 50)),
        ("MA Cross 10/30", MovingAverageCrossStrategy(10, 30)),
        ("RSI 30/70", TechnicalFeaturesStrategy(30, 70)),
    ]
    
    results = {}
    
    for strategy_name, strategy in strategies:
        logger.info(f"Running backtest: {strategy_name}")
        
        # Run backtest for first symbol
        symbol = symbols[0]
        if symbol in market_data and not market_data[symbol].empty:
            try:
                # Convert to price series for strategy
                price_series = market_data[symbol]['Close']
                
                # Add benchmark data if available
                benchmark_series = None
                if not benchmark_data.empty:
                    benchmark_series = benchmark_data['Close']
                
                result = backtester.backtest(
                    strategy=strategy,
                    signal=price_series,
                    benchmark=benchmark_series
                )
                
                results[strategy_name] = result
                logger.info(f"{strategy_name} Results:")
                logger.info(f"  Total Return: {result.total_return:.2%}")
                logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
                logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
                logger.info(f"  Win Rate: {result.win_rate:.2%}")
                
            except Exception as e:
                logger.error(f"Backtest failed for {strategy_name}: {e}")
    
    # Compare strategies
    if results:
        logger.info("=== Strategy Comparison ===")
        best_return = max(results.items(), key=lambda x: x[1].total_return)
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        
        logger.info(f"Best Return: {best_return[0]} ({best_return[1].total_return:.2%})")
        logger.info(f"Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")


async def demonstrate_time_travel(
    symbol, timeframe, start_date, end_date, data_lake_loader
):
    """Demonstrate time travel capabilities with Delta Lake."""
    logger.info("=== Time Travel Demo ===")
    
    try:
        # Load current data
        current_data = await data_lake_loader.load_ohlcv_from_lake(
            symbols=[symbol],
            timeframe=timeframe,
            start_dt=start_date,
            end_dt=end_date,
            version=None  # Latest version
        )
        
        if symbol in current_data and not current_data[symbol].empty:
            logger.info(f"Current data for {symbol}: {len(current_data[symbol])} records")
            logger.info(f"Date range: {current_data[symbol].index.min()} to {current_data[symbol].index.max()}")
        
        # Try to load historical version (if Delta Lake is enabled)
        try:
            historical_data = await data_lake_loader.load_ohlcv_from_lake(
                symbols=[symbol],
                timeframe=timeframe,
                start_dt=start_date,
                end_dt=end_date,
                version=0  # First version
            )
            
            if symbol in historical_data and not historical_data[symbol].empty:
                logger.info(f"Historical data (version 0): {len(historical_data[symbol])} records")
                
                # Compare differences
                if not current_data[symbol].empty:
                    current_count = len(current_data[symbol])
                    historical_count = len(historical_data[symbol])
                    logger.info(f"Data evolution: {historical_count} -> {current_count} records")
            
        except Exception as e:
            logger.info(f"Time travel not available (likely no Spark/Delta): {e}")
            
    except Exception as e:
        logger.error(f"Time travel demo failed: {e}")


async def load_and_test_features(data_lake_loader):
    """Load and test pre-computed technical features."""
    logger.info("=== Technical Features Demo ===")
    
    try:
        # Define feature configuration
        feature_config = {
            'trend': ['sma_20', 'sma_50', 'ema_12', 'ema_26'],
            'momentum': ['rsi_14', 'macd', 'macd_signal'],
            'volatility': ['bb_upper', 'bb_lower', 'atr_14']
        }
        
        # Load features
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=90)
        
        features = await data_lake_loader.load_features_dataset(
            symbols=['AAPL'],
            start_dt=start_date,
            end_dt=end_date,
            feature_config=feature_config
        )
        
        if 'AAPL' in features and not features['AAPL'].empty:
            feature_df = features['AAPL']
            logger.info(f"Loaded features: {list(feature_df.columns)}")
            logger.info(f"Feature data shape: {feature_df.shape}")
            logger.info(f"Sample features:\n{feature_df.head()}")
        else:
            logger.warning("No technical features found in data lake")
            
    except Exception as e:
        logger.error(f"Failed to load features: {e}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_data_lake_backtesting())