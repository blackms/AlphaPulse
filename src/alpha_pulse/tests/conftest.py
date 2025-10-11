"""
Pytest configuration and fixtures.
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from types import ModuleType


@pytest.fixture(scope="session", autouse=True)
def mock_openai_env():
    """Mock OpenAI environment variables."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture(scope="session", autouse=True)
def mock_langchain():
    """Mock langchain components."""
    mock_response = MagicMock()
    mock_response.content = '{"recommendations": [], "risk_assessment": "test", "confidence_score": 0.9, "reasoning": "test"}'
    
    mock_chat = AsyncMock()
    mock_chat.ainvoke.return_value = mock_response

    if "langchain_openai" not in sys.modules:
        langchain_stub = ModuleType("langchain_openai")
        langchain_stub.ChatOpenAI = MagicMock()
        sys.modules["langchain_openai"] = langchain_stub
    
    with patch('langchain_openai.ChatOpenAI', return_value=mock_chat):
        yield


@pytest_asyncio.fixture
async def event_loop():
    """Create event loop for tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def mock_exchange_credentials():
    """Mock exchange credentials."""
    with patch.dict(os.environ, {
        "BINANCE_API_KEY": "test-key",
        "BINANCE_API_SECRET": "test-secret",
        "BYBIT_API_KEY": "test-key",
        "BYBIT_API_SECRET": "test-secret"
    }):
        yield


@pytest_asyncio.fixture
async def mock_exchange():
    """Create mock exchange."""
    exchange = AsyncMock()
    exchange.fetch_balance = AsyncMock(return_value={
        "total": {"USDT": 10000},
        "free": {"USDT": 9000},
        "used": {"USDT": 1000}
    })
    exchange.fetch_ticker = AsyncMock(return_value={
        "last": 50000,
        "bid": 49900,
        "ask": 50100,
        "volume": 1000
    })
    return exchange


@pytest.fixture
def mock_portfolio_data():
    """Create mock portfolio data."""
    from alpha_pulse.portfolio.data_models import PortfolioData, PortfolioPosition
    from decimal import Decimal
    
    return PortfolioData(
        total_value=Decimal("100000.00"),
        cash_balance=Decimal("20000.00"),
        positions=[
            PortfolioPosition(
                asset_id="BTC",
                quantity=Decimal("1.5"),
                current_price=Decimal("50000.00"),
                market_value=Decimal("75000.00"),
                profit_loss=Decimal("5000.00")
            )
        ],
        risk_metrics={
            "volatility": "0.25",
            "sharpe_ratio": "1.5",
            "max_drawdown": "-0.15"
        }
    )


@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    return {
        "BTC/USDT": {
            "price": 50000.0,
            "volume": 1000.0,
            "change_24h": 0.05,
            "high_24h": 51000.0,
            "low_24h": 49000.0
        },
        "ETH/USDT": {
            "price": 2000.0,
            "volume": 5000.0,
            "change_24h": 0.03,
            "high_24h": 2100.0,
            "low_24h": 1900.0
        }
    }


@pytest.fixture(autouse=True)
def mock_logger():
    """Mock logger to prevent actual logging during tests."""
    with patch('loguru.logger.info'), \
         patch('loguru.logger.debug'), \
         patch('loguru.logger.warning'), \
         patch('loguru.logger.error'):
        yield
