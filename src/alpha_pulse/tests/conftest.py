"""
Pytest configuration for AlphaPulse tests.
"""
import pytest
import pytest_asyncio
import asyncio

from ..agents.supervisor import SupervisorAgent, AgentFactory
from .test_supervisor import TestTradeAgent


@pytest_asyncio.fixture(autouse=True)
async def setup_test_environment():
    """Set up test environment before each test."""
    # Enable test mode with TestTradeAgent
    AgentFactory.enable_test_mode(TestTradeAgent)
    
    # Clear supervisor state
    supervisor = SupervisorAgent.instance()
    supervisor._instance = None
    
    yield
    
    # Cleanup after test
    AgentFactory.disable_test_mode()
    SupervisorAgent._instance = None


@pytest_asyncio.fixture
async def supervisor():
    """Fixture for supervisor agent."""
    supervisor = SupervisorAgent.instance()
    await supervisor.start()
    yield supervisor
    await supervisor.stop()
    # Clear instance after test
    SupervisorAgent._instance = None


@pytest_asyncio.fixture(scope="function")
async def market_data():
    """Fixture for market data."""
    from ..agents.interfaces import MarketData
    import pandas as pd
    
    return MarketData(
        prices=pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102]
        }),
        volumes=pd.DataFrame({
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    )


@pytest_asyncio.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_configure(config):
    """Configure pytest for async testing."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async"
    )