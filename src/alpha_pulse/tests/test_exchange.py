"""
Tests for exchange functionality.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from loguru import logger

from alpha_pulse.exchanges.factories import ExchangeType
from alpha_pulse.exchanges.interfaces import (
    BaseExchange,
    ExchangeConfiguration,
    MarketDataError
)
from alpha_pulse.exchanges.implementations.binance import BinanceExchange
from alpha_pulse.exchanges.base import OHLCV
from datetime import datetime


@pytest_asyncio.fixture
async def exchange():
    """Fixture for Binance exchange."""
    # Create exchange with testnet enabled
    exchange = BinanceExchange(testnet=True)
    
    # Initialize with test markets
    await exchange.initialize()
    
    # Mock the markets data
    exchange._markets = {
        'BTC/USDT': {
            'id': 'BTCUSDT',
            'symbol': 'BTC/USDT',
            'base': 'BTC',
            'quote': 'USDT',
            'active': True
        }
    }
    
    yield exchange
    await exchange.close()


@pytest.mark.asyncio
async def test_exchange_initialization(exchange):
    """Test exchange initialization."""
    assert exchange is not None
    assert isinstance(exchange, BaseExchange)
    logger.info("Successfully initialized Binance exchange")


@pytest.mark.asyncio
async def test_get_ticker_price(exchange):
    """Test ticker price fetching."""
    with patch.object(exchange.exchange, 'publicGetTickerPrice', new_callable=AsyncMock) as mock_ticker:
        mock_ticker.return_value = {'price': '50000.00'}
        
        price = await exchange.get_ticker_price('BTC/USDT')
        assert price is not None
        assert float(price) == 50000.00
        logger.info(f"Successfully fetched BTC/USDT price: {price}")


@pytest.mark.asyncio
async def test_fetch_ohlcv(exchange):
    """Test OHLCV data fetching."""
    mock_data = [
        [1625097600000, 35000.00, 35100.00, 34900.00, 35050.00, 100.00],
        [1625097900000, 35050.00, 35150.00, 34950.00, 35100.00, 150.00]
    ]
    
    with patch.object(exchange.exchange, 'fetch_ohlcv', new_callable=AsyncMock) as mock_ohlcv:
        mock_ohlcv.return_value = mock_data
        
        candles = await exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=2)
        assert len(candles) == 2
        assert all(hasattr(candle, 'timestamp') for candle in candles)
        logger.info("Successfully fetched OHLCV data")


@pytest.mark.asyncio
async def test_error_handling(exchange):
    """Test exchange error handling."""
    with patch.object(exchange.exchange, 'fetch_ticker', new_callable=AsyncMock) as mock_ticker:
        mock_ticker.side_effect = Exception("API error")
        
        with pytest.raises(MarketDataError):
            await exchange.get_ticker_price('BTC/USDT')


@pytest.mark.asyncio
async def test_market_validation(exchange):
    """Test market validation."""
    with patch.object(exchange.exchange, 'fetch_ticker', new_callable=AsyncMock) as mock_ticker:
        mock_ticker.side_effect = Exception("Invalid symbol")
        
        with pytest.raises(MarketDataError):
            await exchange.get_ticker_price('INVALID/PAIR')