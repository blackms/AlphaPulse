"""
Core components for AlphaPulse data pipeline.
"""
from alpha_pulse.data_pipeline.core.interfaces import (
    IDataStorage,
    IDataFetcher,
    IMarketDataProvider
)
from alpha_pulse.data_pipeline.core.errors import DataPipelineError
from alpha_pulse.data_pipeline.core.config import (
    StorageConfig,
    DataFetchConfig,
    MarketDataConfig,
    DataPipelineConfig
)

# Get timeframe durations from MarketDataConfig's default
_default_config = MarketDataConfig()
TIMEFRAME_DURATIONS = _default_config.timeframe_durations

# Import validation functions
from alpha_pulse.data_pipeline.core.validation import (
    validate_timeframe,
    validate_symbol,
    validate_exchange_type,
    validate_time_range,
    validate_ohlcv,
    validate_ohlcv_list
)

__all__ = [
    # Interfaces
    'IDataStorage',
    'IDataFetcher',
    'IMarketDataProvider',
    
    # Models and Configs
    'DataPipelineError',
    'StorageConfig',
    'DataFetchConfig',
    'MarketDataConfig',
    'DataPipelineConfig',
    'TIMEFRAME_DURATIONS',
    
    # Validation
    'validate_timeframe',
    'validate_symbol',
    'validate_exchange_type',
    'validate_time_range',
    'validate_ohlcv',
    'validate_ohlcv_list'
]