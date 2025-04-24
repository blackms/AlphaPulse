"""
Integration tests for refactored components.
"""
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from typing import List, Optional

from alpha_pulse.exchanges.interfaces import BaseExchange, ExchangeConfiguration
from alpha_pulse.exchanges.adapters.ccxt_adapter import CCXTAdapter
from alpha_pulse.exchanges.factories import ExchangeFactory, ExchangeType
from alpha_pulse.exchanges.base import OHLCV, Balance
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher, IDataStorage


class InMemoryStorage(IDataStorage):
    """In-memory storage for testing."""
    
    def __init__(self):
        self.data: List[OHLCV] = []
        self.latest_ts: dict = {}
    
    async def save_ohlcv(self, records: List[OHLCV]) -> None:
        """Save OHLCV records."""
        self.data.extend(records)
        for record in records:
            key = f"{record.exchange}_{record.symbol}"
            if key not in self.latest_ts or record.timestamp > self.latest_ts[key]:
                self.latest_ts[key] = record.timestamp
    
    async def get_latest_ohlcv(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[datetime]:
        """Get latest OHLCV timestamp."""
        return self.latest_ts.get(f"{exchange}_{symbol}")


@pytest_asyncio.fixture
async def test_exchange():
    """Create a test exchange."""
    exchange = CCXTAdapter(
        exchange_id='binance',
        config=ExchangeConfiguration(
            api_key='test',
            api_secret='test',
            testnet=True
        )
    )
    await exchange.initialize()
    yield exchange
    await exchange.close()


@pytest_asyncio.fixture
async def test_storage():
    """Create a test storage."""
    return InMemoryStorage()


@pytest_asyncio.fixture
async def test_exchange_factory(test_exchange):
    """Create a test exchange factory."""
    class TestFactory(ExchangeFactory):
        async def create_exchange(self, *args, **kwargs):
            return test_exchange
    return TestFactory()


@pytest.mark.asyncio
async def test_ccxt_exchange_fetch_ohlcv(test_exchange):
    """Test fetching OHLCV data."""
    data = await test_exchange.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        since=int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1000),
        limit=10
    )
    
    assert len(data) > 0
    assert isinstance(data[0], OHLCV)
    assert data[0].symbol == "BTC/USDT"
    assert isinstance(data[0].close, Decimal)
    assert isinstance(data[0].volume, Decimal)


@pytest.mark.asyncio
async def test_ccxt_exchange_fetch_ticker(test_exchange):
    """Test fetching ticker data."""
    ticker = await test_exchange.get_ticker_price("BTC/USDT")
    assert isinstance(ticker, Decimal)
    assert ticker > 0


@pytest.mark.asyncio
async def test_data_fetcher_fetch_ohlcv(test_exchange_factory, test_storage):
    """Test data fetcher's OHLCV fetching."""
    fetcher = DataFetcher(test_exchange_factory, test_storage)
    
    data = await fetcher.fetch_ohlcv(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        since=datetime(2021, 1, 1, tzinfo=UTC),
        limit=10
    )
    
    assert len(data) > 0
    assert isinstance(data[0], OHLCV)
    assert data[0].symbol == "BTC/USDT"
    assert isinstance(data[0].close, Decimal)
    assert isinstance(data[0].volume, Decimal)


@pytest.mark.asyncio
async def test_data_fetcher_update_historical(test_exchange_factory, test_storage):
    """Test historical data update."""
    fetcher = DataFetcher(test_exchange_factory, test_storage)
    
    await fetcher.update_historical_data(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        days_back=1
    )
    
    # Verify storage has data
    latest_ts = await test_storage.get_latest_ohlcv("binance", "BTC/USDT")
    assert latest_ts is not None
    assert isinstance(latest_ts, datetime)
    
    # Verify stored data
    stored_data = test_storage.data
    assert len(stored_data) > 0
    assert isinstance(stored_data[0], OHLCV)
    assert stored_data[0].exchange == "binance"
    assert stored_data[0].symbol == "BTC/USDT"


@pytest.mark.asyncio
async def test_storage_operations(test_storage):
    """Test storage operations."""
    # Create test data
    test_data = [
        OHLCV(
            exchange="test_exchange",
            symbol="BTC/USDT",
            timestamp=datetime.now(UTC),
            open=Decimal('29000.0'),
            high=Decimal('29100.0'),
            low=Decimal('28900.0'),
            close=Decimal('29050.0'),
            volume=Decimal('100.0')
        )
    ]
    
    # Test saving and retrieving data
    await test_storage.save_ohlcv(test_data)
    latest_ts = await test_storage.get_latest_ohlcv("test_exchange", "BTC/USDT")
    
    assert latest_ts is not None
    assert isinstance(latest_ts, datetime)
    assert len(test_storage.data) == 1