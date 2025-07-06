"""
Data Lake Loader for the AlphaPulse Backtesting Framework.

Loads historical market data from the data lake Silver layer instead of direct database access.
Provides enhanced features like time travel, partitioned data access, and improved performance.
"""
import pandas as pd
import datetime as dt
from typing import Dict, List, Optional, Union
from loguru import logger
from pathlib import Path

from alpha_pulse.data_lake.lake_manager import DataLakeManager
from alpha_pulse.data_lake.storage_layers import SilverLayer
from alpha_pulse.data_lake.backends.local import LocalBackend


class DataLakeBacktestingLoader:
    """
    Enhanced data loader that connects backtesting to the data lake Silver layer.
    
    Provides improved performance, time travel capabilities, and advanced filtering
    compared to direct database access.
    """
    
    def __init__(
        self, 
        data_lake_path: str = "./data_lake",
        enable_spark: bool = False,
        cache_enabled: bool = True
    ):
        """
        Initialize the data lake backtesting loader.
        
        Args:
            data_lake_path: Path to the data lake root directory
            enable_spark: Whether to enable Spark for Delta Lake features
            cache_enabled: Whether to enable local caching of frequently accessed data
        """
        self.data_lake_path = Path(data_lake_path)
        self.enable_spark = enable_spark
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}
        
        # Initialize data lake components
        self._initialize_data_lake()
        
        logger.info(f"DataLakeBacktestingLoader initialized with path: {data_lake_path}")
    
    def _initialize_data_lake(self):
        """Initialize data lake manager and silver layer."""
        try:
            # Initialize backend
            self.backend = LocalBackend(root_path=self.data_lake_path)
            
            # Initialize Spark session if enabled
            spark_session = None
            if self.enable_spark:
                try:
                    from pyspark.sql import SparkSession
                    spark_session = SparkSession.builder \
                        .appName("AlphaPulse_Backtesting") \
                        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                        .getOrCreate()
                    logger.info("Spark session initialized for Delta Lake support")
                except ImportError:
                    logger.warning("Spark not available, falling back to Parquet")
                    self.enable_spark = False
            
            # Initialize Silver layer
            self.silver_layer = SilverLayer(
                path="silver",
                backend=self.backend,
                retention_days=365 * 5,  # 5 years retention
                spark_session=spark_session
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize data lake: {e}")
            raise
    
    async def load_ohlcv_from_lake(
        self,
        symbols: List[str],
        timeframe: str,
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        dataset_name: str = "market_data_ohlcv",
        use_cache: bool = True,
        version: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data from the data lake Silver layer.
        
        Args:
            symbols: List of ticker symbols to load
            timeframe: Timeframe string (e.g., '1d', '1h')
            start_dt: Start datetime (inclusive)
            end_dt: End datetime (exclusive)
            dataset_name: Name of the dataset in Silver layer
            use_cache: Whether to use local cache
            version: Specific dataset version for time travel (Delta Lake only)
            
        Returns:
            Dictionary with symbols as keys and DataFrames as values
        """
        logger.info(
            f"Loading data from data lake for symbols: {symbols}, timeframe: {timeframe}, "
            f"range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        )
        
        all_data: Dict[str, pd.DataFrame] = {}
        
        try:
            # Create cache key
            cache_key = f"{'-'.join(symbols)}_{timeframe}_{start_dt.date()}_{end_dt.date()}"
            
            # Check cache first
            if use_cache and self.cache_enabled and cache_key in self._cache:
                logger.debug(f"Using cached data for {cache_key}")
                cached_df = self._cache[cache_key].copy()
                return self._split_dataframe_by_symbol(cached_df, symbols)
            
            # Build filter expression for date range
            filter_expr = f"timestamp >= '{start_dt.isoformat()}' AND timestamp < '{end_dt.isoformat()}'"
            
            # Add symbol filter
            if len(symbols) == 1:
                filter_expr += f" AND symbol = '{symbols[0]}'"
            else:
                symbol_list = "', '".join(symbols)
                filter_expr += f" AND symbol IN ('{symbol_list}')"
            
            # Add timeframe filter
            filter_expr += f" AND timeframe = '{timeframe}'"
            
            # Load data from Silver layer
            df = self.silver_layer.read_dataset(
                dataset_name=dataset_name,
                columns=['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume'],
                filter_expr=filter_expr,
                version=version
            )
            
            if df.empty:
                logger.warning(f"No data found in data lake for symbols {symbols} in date range")
                return {symbol: pd.DataFrame() for symbol in symbols}
            
            # Convert timestamp to datetime and set timezone
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            elif df['timestamp'].dt.tz != dt.timezone.utc:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            
            # Rename columns to match backtesting expectations
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Cache the result
            if use_cache and self.cache_enabled:
                self._cache[cache_key] = df.copy()
                logger.debug(f"Cached data for {cache_key}")
            
            # Split by symbol
            all_data = self._split_dataframe_by_symbol(df, symbols)
            
            # Log loading statistics
            total_records = sum(len(symbol_df) for symbol_df in all_data.values())
            logger.info(f"Loaded {total_records} total records from data lake")
            
            for symbol, symbol_df in all_data.items():
                if not symbol_df.empty:
                    logger.info(f"  {symbol}: {len(symbol_df)} records")
                else:
                    logger.warning(f"  {symbol}: No data found")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Failed to load data from data lake: {e}")
            # Return empty DataFrames for all symbols
            return {symbol: pd.DataFrame() for symbol in symbols}
    
    def _split_dataframe_by_symbol(self, df: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Split a combined DataFrame by symbol."""
        all_data = {}
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if not symbol_data.empty:
                # Set timestamp as index and remove symbol column
                symbol_data = symbol_data.set_index('timestamp')
                symbol_data = symbol_data.drop(columns=['symbol', 'timeframe'], errors='ignore')
                symbol_data = symbol_data.sort_index()
            else:
                # Create empty DataFrame with proper structure
                symbol_data = pd.DataFrame(
                    columns=['Open', 'High', 'Low', 'Close', 'Volume']
                ).set_index(pd.DatetimeIndex([], name='timestamp', tz='UTC'))
            
            all_data[symbol] = symbol_data
        
        return all_data
    
    async def load_benchmark_data(
        self,
        benchmark_symbol: str,
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        timeframe: str = "1d",
        dataset_name: str = "benchmark_data"
    ) -> pd.DataFrame:
        """
        Load benchmark data for comparison.
        
        Args:
            benchmark_symbol: Benchmark symbol (e.g., 'SPY', '^GSPC')
            start_dt: Start datetime
            end_dt: End datetime
            timeframe: Data timeframe
            dataset_name: Dataset name in Silver layer
            
        Returns:
            DataFrame with benchmark data
        """
        logger.info(f"Loading benchmark data for {benchmark_symbol}")
        
        try:
            benchmark_data = await self.load_ohlcv_from_lake(
                symbols=[benchmark_symbol],
                timeframe=timeframe,
                start_dt=start_dt,
                end_dt=end_dt,
                dataset_name=dataset_name
            )
            
            return benchmark_data.get(benchmark_symbol, pd.DataFrame())
            
        except Exception as e:
            logger.error(f"Failed to load benchmark data: {e}")
            return pd.DataFrame()
    
    async def load_features_dataset(
        self,
        symbols: List[str],
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        feature_config: Dict[str, List[str]],
        dataset_name: str = "technical_features"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load pre-computed technical features from the Silver layer.
        
        Args:
            symbols: List of symbols
            start_dt: Start datetime
            end_dt: End datetime
            feature_config: Dictionary mapping feature types to lists of feature names
            dataset_name: Features dataset name
            
        Returns:
            Dictionary with symbols and their feature DataFrames
        """
        logger.info(f"Loading features for {len(symbols)} symbols")
        
        try:
            # Flatten feature names
            all_features = []
            for feature_list in feature_config.values():
                all_features.extend(feature_list)
            
            # Build column list
            columns = ['timestamp', 'symbol'] + all_features
            
            # Build filter expression
            filter_expr = f"timestamp >= '{start_dt.isoformat()}' AND timestamp < '{end_dt.isoformat()}'"
            if len(symbols) == 1:
                filter_expr += f" AND symbol = '{symbols[0]}'"
            else:
                symbol_list = "', '".join(symbols)
                filter_expr += f" AND symbol IN ('{symbol_list}')"
            
            # Load features
            df = self.silver_layer.read_dataset(
                dataset_name=dataset_name,
                columns=columns,
                filter_expr=filter_expr
            )
            
            if df.empty:
                logger.warning("No feature data found")
                return {symbol: pd.DataFrame() for symbol in symbols}
            
            # Process timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            
            # Split by symbol
            feature_data = {}
            for symbol in symbols:
                symbol_features = df[df['symbol'] == symbol].copy()
                if not symbol_features.empty:
                    symbol_features = symbol_features.set_index('timestamp')
                    symbol_features = symbol_features.drop(columns=['symbol'])
                    symbol_features = symbol_features.sort_index()
                else:
                    symbol_features = pd.DataFrame()
                
                feature_data[symbol] = symbol_features
            
            logger.info(f"Loaded features for {len(feature_data)} symbols")
            return feature_data
            
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return {symbol: pd.DataFrame() for symbol in symbols}
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets in the Silver layer."""
        try:
            datasets = self.silver_layer._list_datasets()
            logger.info(f"Available datasets: {datasets}")
            return datasets
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """Get information about a specific dataset."""
        try:
            # Read a small sample to get schema info
            sample_df = self.silver_layer.read_dataset(
                dataset_name=dataset_name,
                filter_expr="1=1 LIMIT 1"  # Get just one row
            )
            
            info = {
                'columns': list(sample_df.columns),
                'dtypes': sample_df.dtypes.astype(str).to_dict(),
                'sample_data': sample_df.head().to_dict('records') if not sample_df.empty else []
            }
            
            logger.info(f"Dataset {dataset_name} info: {len(info['columns'])} columns")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_name}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the local cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_size = sum(df.memory_usage(deep=True).sum() for df in self._cache.values())
        return {
            'entries': len(self._cache),
            'total_size_mb': total_size / (1024 * 1024),
            'keys': list(self._cache.keys())
        }


# Global loader instance for easy access
_data_lake_loader: Optional[DataLakeBacktestingLoader] = None


def get_data_lake_loader(
    data_lake_path: str = "./data_lake",
    enable_spark: bool = False
) -> DataLakeBacktestingLoader:
    """Get the global data lake loader instance."""
    global _data_lake_loader
    
    if _data_lake_loader is None:
        _data_lake_loader = DataLakeBacktestingLoader(
            data_lake_path=data_lake_path,
            enable_spark=enable_spark
        )
    
    return _data_lake_loader