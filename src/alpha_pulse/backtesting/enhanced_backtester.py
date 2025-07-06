"""
Enhanced Backtesting Manager with Data Lake Integration.

Provides unified access to both database and data lake data sources,
with automatic fallback and performance optimization.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from enum import Enum

from .backtester import Backtester, BacktestResult
from .strategy import BaseStrategy
from .data_loader import load_ohlcv_data  # Original database loader
from .data_lake_loader import DataLakeBacktestingLoader, get_data_lake_loader


class DataSource(Enum):
    """Available data sources for backtesting."""
    DATABASE = "database"
    DATA_LAKE = "data_lake"
    AUTO = "auto"  # Automatically choose best source


class EnhancedBacktester:
    """
    Enhanced backtesting engine with data lake integration.
    
    Provides intelligent data source selection, caching, and performance optimization
    while maintaining compatibility with existing backtesting workflows.
    """
    
    def __init__(
        self,
        commission: float = 0.002,
        initial_capital: float = 100000.0,
        slippage: float = 0.001,
        benchmark_symbol: str = "SPY",
        data_source: DataSource = DataSource.AUTO,
        data_lake_path: str = "./data_lake",
        enable_spark: bool = False,
        cache_enabled: bool = True
    ):
        """
        Initialize enhanced backtester.
        
        Args:
            commission: Trading commission as fraction
            initial_capital: Starting capital
            slippage: Slippage as fraction
            benchmark_symbol: Benchmark symbol for comparison
            data_source: Preferred data source
            data_lake_path: Path to data lake
            enable_spark: Enable Spark for Delta Lake features
            cache_enabled: Enable data caching
        """
        # Initialize core backtester
        self.backtester = Backtester(
            commission=commission,
            initial_capital=initial_capital,
            slippage=slippage,
            benchmark_symbol=benchmark_symbol
        )
        
        self.data_source = data_source
        self.cache_enabled = cache_enabled
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Initialize data lake loader
        try:
            self.data_lake_loader = get_data_lake_loader(
                data_lake_path=data_lake_path,
                enable_spark=enable_spark
            )
            self.data_lake_available = True
            logger.info("Data lake integration enabled")
        except Exception as e:
            logger.warning(f"Data lake initialization failed: {e}")
            self.data_lake_loader = None
            self.data_lake_available = False
            if data_source == DataSource.DATA_LAKE:
                logger.error("Data lake required but not available")
                raise
        
        logger.info(f"EnhancedBacktester initialized with data source: {data_source.value}")
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        data_source: Optional[DataSource] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, BacktestResult]:
        """
        Run backtests for multiple symbols with intelligent data loading.
        
        Args:
            strategy: Trading strategy to test
            symbols: List of symbols to test
            timeframe: Data timeframe
            start_date: Start date for backtest
            end_date: End date for backtest
            data_source: Override default data source
            benchmark_data: Pre-loaded benchmark data
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dictionary mapping symbols to backtest results
        """
        logger.info(f"Running enhanced backtest for {len(symbols)} symbols")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Determine data source
        source = data_source or self.data_source
        
        # Load market data
        market_data = await self._load_market_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
            **kwargs
        )
        
        if not market_data:
            logger.error("No market data loaded")
            return {}
        
        # Load benchmark data if not provided
        if benchmark_data is None:
            benchmark_data = await self._load_benchmark_data(
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                source=source
            )
        
        # Run backtests for each symbol
        results = {}
        for symbol in symbols:
            if symbol not in market_data or market_data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            try:
                logger.info(f"Running backtest for {symbol}")
                
                # Extract price series
                price_series = market_data[symbol]['Close']
                
                # Prepare benchmark series
                benchmark_series = None
                if benchmark_data is not None and not benchmark_data.empty:
                    # Align benchmark with price series
                    aligned_benchmark = benchmark_data.reindex(price_series.index, method='ffill')
                    if 'Close' in aligned_benchmark.columns:
                        benchmark_series = aligned_benchmark['Close']
                
                # Run backtest
                result = self.backtester.backtest(
                    strategy=strategy,
                    signal=price_series,
                    benchmark=benchmark_series
                )
                
                results[symbol] = result
                
                logger.info(f"Backtest completed for {symbol}")
                logger.info(f"  Total Return: {result.total_return:.2%}")
                logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
                logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
                
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")
        
        return results
    
    async def _load_market_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        source: DataSource,
        **kwargs
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Load market data from the specified source."""
        
        # Try data lake first if preferred or auto
        if source in [DataSource.DATA_LAKE, DataSource.AUTO] and self.data_lake_available:
            try:
                logger.info("Loading data from data lake")
                data = await self.data_lake_loader.load_ohlcv_from_lake(
                    symbols=symbols,
                    timeframe=timeframe,
                    start_dt=start_date,
                    end_dt=end_date,
                    use_cache=self.cache_enabled,
                    **kwargs
                )
                
                if data and any(not df.empty for df in data.values()):
                    logger.info("Successfully loaded data from data lake")
                    return data
                else:
                    logger.warning("Data lake returned empty results")
                    
            except Exception as e:
                logger.warning(f"Data lake loading failed: {e}")
        
        # Try database if data lake failed or not preferred
        if source in [DataSource.DATABASE, DataSource.AUTO]:
            try:
                logger.info("Loading data from database")
                data = await load_ohlcv_data(
                    symbols=symbols,
                    timeframe=timeframe,
                    start_dt=start_date,
                    end_dt=end_date,
                    **kwargs
                )
                
                if data and any(not df.empty for df in data.values()):
                    logger.info("Successfully loaded data from database")
                    return data
                else:
                    logger.warning("Database returned empty results")
                    
            except Exception as e:
                logger.warning(f"Database loading failed: {e}")
        
        logger.error("All data sources failed")
        return None
    
    async def _load_benchmark_data(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        source: DataSource
    ) -> Optional[pd.DataFrame]:
        """Load benchmark data."""
        benchmark_symbol = self.backtester.benchmark_symbol
        
        try:
            # Try data lake first
            if source in [DataSource.DATA_LAKE, DataSource.AUTO] and self.data_lake_available:
                benchmark_data = await self.data_lake_loader.load_benchmark_data(
                    benchmark_symbol=benchmark_symbol,
                    start_dt=start_date,
                    end_dt=end_date,
                    timeframe=timeframe
                )
                
                if not benchmark_data.empty:
                    logger.info(f"Loaded benchmark {benchmark_symbol} from data lake")
                    return benchmark_data
            
            # Fallback to regular data loading
            if source in [DataSource.DATABASE, DataSource.AUTO]:
                benchmark_data = await load_ohlcv_data(
                    symbols=[benchmark_symbol],
                    timeframe=timeframe,
                    start_dt=start_date,
                    end_dt=end_date
                )
                
                if benchmark_data and benchmark_symbol in benchmark_data:
                    logger.info(f"Loaded benchmark {benchmark_symbol} from database")
                    return benchmark_data[benchmark_symbol]
                    
        except Exception as e:
            logger.warning(f"Failed to load benchmark data: {e}")
        
        return None
    
    async def run_feature_enhanced_backtest(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        feature_config: Dict[str, List[str]],
        feature_dataset: str = "technical_features"
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest with pre-computed technical features from data lake.
        
        Args:
            strategy: Trading strategy
            symbols: List of symbols
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            feature_config: Feature configuration
            feature_dataset: Dataset name for features
            
        Returns:
            Backtest results with enhanced features
        """
        if not self.data_lake_available:
            logger.error("Data lake required for feature-enhanced backtesting")
            return {}
        
        logger.info("Running feature-enhanced backtest")
        
        # Load market data and features
        market_data = await self.data_lake_loader.load_ohlcv_from_lake(
            symbols=symbols,
            timeframe=timeframe,
            start_dt=start_date,
            end_dt=end_date
        )
        
        features = await self.data_lake_loader.load_features_dataset(
            symbols=symbols,
            start_dt=start_date,
            end_dt=end_date,
            feature_config=feature_config,
            dataset_name=feature_dataset
        )
        
        # Combine market data with features
        enhanced_data = {}
        for symbol in symbols:
            if symbol in market_data and symbol in features:
                # Merge price data with features
                price_df = market_data[symbol]
                feature_df = features[symbol]
                
                # Align indices
                combined_df = price_df.join(feature_df, how='inner')
                enhanced_data[symbol] = combined_df
        
        # Run backtests with enhanced data
        # Note: This would require strategy modifications to use features
        # For now, run standard backtest
        return await self.run_backtest(
            strategy=strategy,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            data_source=DataSource.DATA_LAKE
        )
    
    async def compare_data_sources(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict[str, any]]:
        """
        Compare performance and data availability across different sources.
        
        Returns:
            Comparison results including load times, record counts, etc.
        """
        logger.info("Comparing data sources")
        comparison = {}
        
        # Test data lake
        if self.data_lake_available:
            import time
            start_time = time.time()
            
            try:
                lake_data = await self.data_lake_loader.load_ohlcv_from_lake(
                    symbols=symbols,
                    timeframe=timeframe,
                    start_dt=start_date,
                    end_dt=end_date,
                    use_cache=False  # Disable cache for fair comparison
                )
                
                lake_time = time.time() - start_time
                lake_records = sum(len(df) for df in lake_data.values()) if lake_data else 0
                
                comparison['data_lake'] = {
                    'load_time': lake_time,
                    'record_count': lake_records,
                    'symbols_loaded': len([s for s in symbols if s in lake_data and not lake_data[s].empty]) if lake_data else 0,
                    'success': True
                }
                
            except Exception as e:
                comparison['data_lake'] = {
                    'load_time': None,
                    'record_count': 0,
                    'symbols_loaded': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Test database
        import time
        start_time = time.time()
        
        try:
            db_data = await load_ohlcv_data(
                symbols=symbols,
                timeframe=timeframe,
                start_dt=start_date,
                end_dt=end_date
            )
            
            db_time = time.time() - start_time
            db_records = sum(len(df) for df in db_data.values()) if db_data else 0
            
            comparison['database'] = {
                'load_time': db_time,
                'record_count': db_records,
                'symbols_loaded': len([s for s in symbols if s in db_data and not db_data[s].empty]) if db_data else 0,
                'success': True
            }
            
        except Exception as e:
            comparison['database'] = {
                'load_time': None,
                'record_count': 0,
                'symbols_loaded': 0,
                'success': False,
                'error': str(e)
            }
        
        # Calculate speedup if both sources worked
        if ('data_lake' in comparison and 'database' in comparison and 
            comparison['data_lake']['success'] and comparison['database']['success']):
            
            lake_time = comparison['data_lake']['load_time']
            db_time = comparison['database']['load_time']
            
            if lake_time > 0 and db_time > 0:
                speedup = db_time / lake_time
                comparison['speedup'] = {
                    'data_lake_vs_database': speedup,
                    'faster_source': 'data_lake' if speedup > 1 else 'database'
                }
        
        return comparison
    
    def get_data_source_stats(self) -> Dict[str, any]:
        """Get statistics about data source usage and performance."""
        stats = {
            'data_lake_available': self.data_lake_available,
            'cache_enabled': self.cache_enabled,
            'preferred_source': self.data_source.value
        }
        
        if self.data_lake_available:
            cache_stats = self.data_lake_loader.get_cache_stats()
            stats['data_lake_cache'] = cache_stats
            
            datasets = self.data_lake_loader.get_available_datasets()
            stats['available_datasets'] = datasets
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self._data_cache.clear()
        if self.data_lake_available:
            self.data_lake_loader.clear_cache()
        logger.info("All caches cleared")


# Factory function for easy access
def create_enhanced_backtester(
    commission: float = 0.002,
    initial_capital: float = 100000.0,
    data_source: DataSource = DataSource.AUTO,
    data_lake_path: str = "./data_lake",
    **kwargs
) -> EnhancedBacktester:
    """
    Factory function to create an enhanced backtester with sensible defaults.
    
    Args:
        commission: Trading commission
        initial_capital: Starting capital
        data_source: Preferred data source
        data_lake_path: Path to data lake
        **kwargs: Additional arguments
        
    Returns:
        Configured EnhancedBacktester instance
    """
    return EnhancedBacktester(
        commission=commission,
        initial_capital=initial_capital,
        data_source=data_source,
        data_lake_path=data_lake_path,
        **kwargs
    )