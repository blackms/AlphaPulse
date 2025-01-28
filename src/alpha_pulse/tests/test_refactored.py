"""
Integration tests for refactored components.
"""
import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch

from alpha_pulse.data_pipeline.interfaces import IExchange, IExchangeFactory, IDataStorage
from alpha_pulse.data_pipeline.exchange import CCXTExchange, CCXTExchangeFactory
from alpha_pulse.data_pipeline.storage import SQLAlchemyStorage
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.models import OHLCV


@pytest.fixture
def mock_ccxt_exchange():
    """Create a mock CCXT exchange."""
    exchange = Mock()
    exchange.fetch_ohlcv.return_value = [
        # timestamp, open, high, low, close, volume
        [1609459200000, 29000.0, 29100.0, 28900.0, 29050.0, 100.0],
        [1609462800000, 29050.0, 29200.0, 29000.0, 29150.0, 150.0],
    ]
    exchange.fetch_ticker.return_value = {
        "symbol": "BTC/USDT",
        "last": 29150.0,
    }
    return exchange


@pytest.fixture
def mock_exchange(mock_ccxt_exchange):
    """Create a mock exchange."""
    return CCXTExchange(mock_ccxt_exchange)


@pytest.fixture
def mock_storage():
    """Create a mock storage."""
    storage = Mock(spec=IDataStorage)
    storage.save_ohlcv.return_value = None
    storage.get_latest_ohlcv.return_value = None
    return storage


@pytest.fixture
def mock_exchange_factory(mock_exchange):
    """Create a mock exchange factory."""
    factory = Mock(spec=IExchangeFactory)
    factory.create_exchange.return_value = mock_exchange
    return factory


def test_ccxt_exchange_fetch_ohlcv(mock_exchange):
    """Test fetching OHLCV data."""
    data = mock_exchange.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2021, 1, 1, tzinfo=UTC),
        limit=2
    )
    
    assert len(data) == 2
    assert data[0][4] == 29050.0  # Check close price
    assert data[1][5] == 150.0    # Check volume


def test_ccxt_exchange_fetch_ticker(mock_exchange):
    """Test fetching ticker data."""
    ticker = mock_exchange.fetch_ticker("BTC/USDT")
    assert ticker["symbol"] == "BTC/USDT"
    assert ticker["last"] == 29150.0


def test_data_fetcher_fetch_ohlcv(mock_exchange_factory, mock_storage):
    """Test data fetcher's OHLCV fetching."""
    fetcher = DataFetcher(mock_exchange_factory, mock_storage)
    
    data = fetcher.fetch_ohlcv(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2021, 1, 1, tzinfo=UTC),
        limit=2
    )
    
    assert len(data) == 2
    assert isinstance(data[0], OHLCV)
    assert data[0].symbol == "BTC/USDT"
    assert data[0].close == 29050.0
    assert data[1].volume == 150.0


def test_data_fetcher_update_historical(mock_exchange_factory, mock_storage):
    """Test historical data update."""
    fetcher = DataFetcher(mock_exchange_factory, mock_storage)
    
    fetcher.update_historical_data(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        days_back=1
    )
    
    # Verify storage was called with correct data
    mock_storage.save_ohlcv.assert_called_once()
    saved_data = mock_storage.save_ohlcv.call_args[0][0]
    assert len(saved_data) == 2
    assert isinstance(saved_data[0], OHLCV)
    assert saved_data[0].exchange == "binance"
    assert saved_data[0].symbol == "BTC/USDT"


@pytest.mark.integration
def test_storage_integration():
    """Test storage integration."""
    storage = SQLAlchemyStorage()
    
    # Create test data
    test_data = [
        OHLCV(
            exchange="test_exchange",
            symbol="BTC/USDT",
            timestamp=datetime.now(UTC),
            open=29000.0,
            high=29100.0,
            low=28900.0,
            close=29050.0,
            volume=100.0
        )
    ]
    
    # Test saving and retrieving data
    storage.save_ohlcv(test_data)
    latest_ts = storage.get_latest_ohlcv("test_exchange", "BTC/USDT")
    
    assert latest_ts is not None
    assert isinstance(latest_ts, datetime)