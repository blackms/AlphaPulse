"""
Tests for the data fetcher module.
"""
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock

import pytest
from loguru import logger

from ..data_pipeline.data_fetcher import DataFetcher
from ..data_pipeline.interfaces import IExchangeFactory, IDataStorage


@pytest.fixture
def mock_exchange_factory():
    """Create a mock exchange factory."""
    factory = Mock(spec=IExchangeFactory)
    exchange = Mock()
    exchange.fetch_ohlcv.return_value = [
        [1640995200000, 100.0, 102.0, 99.0, 101.0, 1000.0],
        [1641081600000, 101.0, 103.0, 98.0, 102.0, 1100.0],
    ]
    factory.create_exchange.return_value = exchange
    return factory


@pytest.fixture
def mock_storage():
    """Create a mock storage."""
    storage = Mock(spec=IDataStorage)
    storage.save_historical_data.return_value = None
    storage.save_ohlcv.return_value = None
    return storage


def test_data_fetcher(mock_exchange_factory, mock_storage):
    """Test data fetcher functionality."""
    # Initialize data fetcher
    fetcher = DataFetcher(mock_exchange_factory, mock_storage)
    
    # Test fetching and storing recent data
    try:
        # Fetch last 24 hours of data
        fetcher.update_historical_data(
            exchange_id='binance',
            symbol='BTC/USDT',
            timeframe='1h',
            days_back=1
        )
        
        logger.info("Successfully fetched and stored historical data")
        
        # Verify mock calls
        mock_exchange_factory.create_exchange.assert_called_once_with('binance')
        mock_storage.save_ohlcv.assert_called_once()
        
        # Verify call arguments
        call_args = mock_storage.save_ohlcv.call_args[0][0]
        assert len(call_args) == 2  # Two OHLCV records
        assert call_args[0].exchange == 'binance'
        assert call_args[0].symbol == 'BTC/USDT'
        assert call_args[0].timeframe == '1h'
        
    except Exception as e:
        logger.error(f"Data fetcher test failed: {str(e)}")
        raise


if __name__ == "__main__":
    pytest.main([__file__])