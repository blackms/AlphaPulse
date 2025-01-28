"""
Tests for the data fetcher module.
"""
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock

import pytest
from loguru import logger

from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.interfaces import IExchangeFactory, IDataStorage
from alpha_pulse.data_pipeline.models import OHLCV


@pytest.fixture
def mock_exchange_factory():
    """Create a mock exchange factory."""
    factory = Mock(spec=IExchangeFactory)
    exchange = Mock()
    exchange.fetch_historical_data.return_value = {
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [99.0, 98.0],
        'close': [101.0, 102.0],
        'volume': [1000.0, 1100.0]
    }
    factory.create_exchange.return_value = exchange
    return factory


@pytest.fixture
def mock_storage():
    """Create a mock storage."""
    storage = Mock(spec=IDataStorage)
    storage.save_historical_data.return_value = None
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
        mock_storage.save_historical_data.assert_called_once()
        
        # Verify call arguments
        call_args = mock_storage.save_historical_data.call_args[1]
        assert call_args['exchange_id'] == 'binance'
        assert call_args['symbol'] == 'BTC/USDT'
        assert call_args['timeframe'] == '1h'
        assert isinstance(call_args['data'], dict)
        assert 'open' in call_args['data']
        assert len(call_args['data']['open']) > 0
        
    except Exception as e:
        logger.error(f"Data fetcher test failed: {str(e)}")
        raise


if __name__ == "__main__":
    pytest.main([__file__])