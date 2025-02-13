"""
Tests for the data fetcher module.
"""
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from typing import Dict, List, Optional
import pytest
import pytest_asyncio
from loguru import logger

from ..exchanges.interfaces import BaseExchange
from ..exchanges.implementations.binance import BinanceExchange
from ..exchanges.base import OHLCV


class TestExchangeFactory:
    """Real exchange factory for testing."""
    
    async def create_exchange(self, exchange_id: str) -> BaseExchange:
        """Create a real exchange instance."""
        if exchange_id != 'binance':
            raise ValueError(f"Unsupported exchange: {exchange_id}")
        exchange = BinanceExchange(testnet=True)
        await exchange.initialize()
        return exchange


class InMemoryStorage:
    """In-memory storage for testing."""
    
    def __init__(self):
        self.data: Dict[str, List[OHLCV]] = {}
    
    async def save_ohlcv(self, records: List[OHLCV]) -> None:
        """Save OHLCV records in memory."""
        if not records:
            return
            
        key = f"{records[0].exchange}_{records[0].symbol}_{records[0].timeframe}"
        self.data[key] = records
    
    def get_ohlcv(self, exchange: str, symbol: str, timeframe: str) -> List[OHLCV]:
        """Get stored OHLCV records."""
        key = f"{exchange}_{symbol}_{timeframe}"
        return self.data.get(key, [])


class DataFetcher:
    """Data fetcher implementation."""
    
    def __init__(self, exchange_factory: TestExchangeFactory, storage: InMemoryStorage):
        self.exchange_factory = exchange_factory
        self.storage = storage
    
    async def update_historical_data(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        days_back: int
    ) -> None:
        """Fetch and store historical data."""
        exchange = None
        try:
            exchange = await self.exchange_factory.create_exchange(exchange_id)
            
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=days_back)
            
            # Fetch OHLCV data
            candles = await exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=None
            )
            
            # Add exchange metadata to OHLCV records
            records = []
            for candle in candles:
                record = candle
                record.exchange = exchange_id
                record.symbol = symbol
                record.timeframe = timeframe
                records.append(record)
            
            # Store the data
            await self.storage.save_ohlcv(records)
        finally:
            if exchange:
                await exchange.close()


@pytest_asyncio.fixture
async def exchange_factory():
    """Create a real exchange factory."""
    return TestExchangeFactory()


@pytest_asyncio.fixture
async def storage():
    """Create an in-memory storage."""
    return InMemoryStorage()


@pytest.mark.asyncio
async def test_data_fetcher(exchange_factory, storage):
    """Test data fetcher functionality."""
    # Initialize data fetcher
    fetcher = DataFetcher(exchange_factory, storage)
    
    try:
        # Fetch last 24 hours of data
        await fetcher.update_historical_data(
            exchange_id='binance',
            symbol='BTC/USDT',
            timeframe='1h',
            days_back=1
        )
        
        logger.info("Successfully fetched and stored historical data")
        
        # Verify stored data
        records = storage.get_ohlcv('binance', 'BTC/USDT', '1h')
        assert len(records) > 0
        assert isinstance(records[0], OHLCV)
        assert records[0].symbol == 'BTC/USDT'
        
    except Exception as e:
        logger.error(f"Data fetcher test failed: {str(e)}")
        raise