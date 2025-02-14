"""
Tests for the broker factory module.
"""
import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal

from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.execution.paper_broker import PaperBroker
from alpha_pulse.execution.real_broker import RealBroker
from alpha_pulse.execution.recommendation_broker import RecommendationOnlyBroker


def test_create_paper_broker():
    """Test creating paper broker."""
    initial_balance = 50000.0
    broker = create_broker(
        trading_mode=TradingMode.PAPER,
        initial_balance=initial_balance
    )
    
    assert isinstance(broker, PaperBroker)
    assert broker.balance == Decimal(str(initial_balance))


def test_create_recommendation_broker():
    """Test creating recommendation broker."""
    broker = create_broker(trading_mode=TradingMode.RECOMMENDATION)
    assert isinstance(broker, RecommendationOnlyBroker)


@patch('alpha_pulse.execution.real_broker.BinanceExchange')
def test_create_real_broker_binance(mock_binance):
    """Test creating real broker for Binance."""
    # Setup mock
    mock_exchange = MagicMock()
    mock_binance.return_value = mock_exchange
    
    broker = create_broker(
        trading_mode=TradingMode.REAL,
        exchange_name="binance",
        api_key="test_key",
        api_secret="test_secret"
    )
    
    assert isinstance(broker, RealBroker)
    mock_binance.assert_called_once_with("test_key", "test_secret", testnet=False)


@patch('alpha_pulse.execution.real_broker.BybitExchange')
def test_create_real_broker_bybit(mock_bybit):
    """Test creating real broker for Bybit."""
    # Setup mock
    mock_exchange = MagicMock()
    mock_bybit.return_value = mock_exchange
    
    broker = create_broker(
        trading_mode=TradingMode.REAL,
        exchange_name="bybit",
        api_key="test_key",
        api_secret="test_secret"
    )
    
    assert isinstance(broker, RealBroker)
    mock_bybit.assert_called_once_with("test_key", "test_secret", testnet=False)


def test_create_real_broker_missing_credentials():
    """Test creating real broker without required credentials."""
    with pytest.raises(ValueError, match="exchange_name, api_key, and api_secret are required"):
        create_broker(trading_mode=TradingMode.REAL)


def test_create_real_broker_invalid_exchange():
    """Test creating real broker with invalid exchange."""
    with pytest.raises(ValueError, match="Unsupported exchange"):
        create_broker(
            trading_mode=TradingMode.REAL,
            exchange_name="invalid",
            api_key="test",
            api_secret="test"
        )


def test_invalid_trading_mode():
    """Test creating broker with invalid trading mode."""
    with pytest.raises(ValueError, match="Invalid trading_mode"):
        create_broker(trading_mode="INVALID")