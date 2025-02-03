"""
AlphaPulse Data Pipeline Package.

This package provides functionality for fetching, storing, and managing
market data from cryptocurrency exchanges.

Example usage:
    # Initialize components
    storage = SQLAlchemyStorage()
    fetcher = ExchangeFetcher()
    provider = ExchangeDataProvider(ExchangeType.BINANCE, testnet=True)

    # Create managers
    historical_mgr = HistoricalDataManager(storage, fetcher)
    realtime_mgr = RealTimeDataManager(provider, storage)

    # Use managers to handle data
    await historical_mgr.ensure_data_available(
        ExchangeType.BINANCE,
        "BTC/USDT",
        "1h",
        start_time,
        end_time
    )

    await realtime_mgr.start(["BTC/USDT", "ETH/USDT"])
"""

# Core components
from alpha_pulse.data_pipeline.core.interfaces import (
    IDataStorage,
    IDataFetcher,
    IMarketDataProvider
)
from alpha_pulse.data_pipeline.core.models import (
    DataPipelineError,
    StorageConfig,
    DataFetchConfig,
    MarketDataConfig,
    TIMEFRAME_DURATIONS,
    validate_timeframe
)

# Implementation components
from alpha_pulse.data_pipeline.storage.sql import SQLAlchemyStorage
from alpha_pulse.data_pipeline.fetcher.exchange import ExchangeFetcher
from alpha_pulse.data_pipeline.providers.exchange import ExchangeDataProvider
from alpha_pulse.data_pipeline.managers.historical import HistoricalDataManager
from alpha_pulse.data_pipeline.managers.real_time import RealTimeDataManager

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
    'TIMEFRAME_DURATIONS',
    'validate_timeframe',
    
    # Implementations
    'SQLAlchemyStorage',
    'ExchangeFetcher',
    'ExchangeDataProvider',
    'HistoricalDataManager',
    'RealTimeDataManager'
]