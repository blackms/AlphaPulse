"""
Common fixtures and utilities for agent tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from alpha_pulse.agents.interfaces import MarketData, AgentMetrics


@pytest.fixture
def mock_agent_config():
    """Create a common agent configuration for testing."""
    return {
        "min_confidence": 0.6,
        "max_positions": 10,
        "risk_per_trade": 0.02,
        "update_frequency": 300,  # 5 minutes
        "lookback_period": 90,    # 90 days
        "enable_logging": True
    }


@pytest.fixture
def generate_price_series():
    """Factory fixture to generate price series with specific characteristics."""
    def _generate(
        n_periods: int = 200,
        start_price: float = 100,
        trend: float = 0.0,
        volatility: float = 0.02,
        seed: int = None
    ) -> pd.Series:
        if seed is not None:
            np.random.seed(seed)
        
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        returns = np.random.normal(trend, volatility, n_periods)
        prices = start_price * (1 + returns).cumprod()
        
        return pd.Series(prices, index=dates)
    
    return _generate


@pytest.fixture
def generate_volume_series():
    """Factory fixture to generate volume series."""
    def _generate(
        n_periods: int = 200,
        base_volume: float = 1e6,
        volatility: float = 0.3,
        seed: int = None
    ) -> pd.Series:
        if seed is not None:
            np.random.seed(seed)
        
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        volume_multiplier = np.abs(np.random.normal(1, volatility, n_periods))
        volumes = base_volume * volume_multiplier
        
        return pd.Series(volumes, index=dates)
    
    return _generate


@pytest.fixture
def create_market_data():
    """Factory fixture to create MarketData objects."""
    def _create(
        symbols: List[str] = ['AAPL', 'GOOGL'],
        n_periods: int = 200,
        include_fundamentals: bool = False,
        include_sentiment: bool = False
    ) -> MarketData:
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        
        # Generate price data
        prices_dict = {}
        volumes_dict = {}
        
        for i, symbol in enumerate(symbols):
            base_price = 100 * (i + 1)
            prices_dict[symbol] = base_price + np.random.randn(n_periods).cumsum()
            volumes_dict[symbol] = 1e6 * (1 + np.random.rand(n_periods))
        
        prices_df = pd.DataFrame(prices_dict, index=dates)
        volumes_df = pd.DataFrame(volumes_dict, index=dates)
        
        # Optional fundamentals
        fundamentals = None
        if include_fundamentals:
            fundamentals = {
                symbol: {
                    "revenue_growth": np.random.uniform(0.05, 0.25),
                    "gross_margin": np.random.uniform(0.2, 0.5),
                    "ebitda_margin": np.random.uniform(0.1, 0.3),
                    "net_margin": np.random.uniform(0.05, 0.2),
                    "current_ratio": np.random.uniform(1.0, 2.5),
                    "pe_ratio": np.random.uniform(15, 35),
                    "sector": "Technology"
                }
                for symbol in symbols
            }
        
        # Optional sentiment
        sentiment = None
        if include_sentiment:
            sentiment = {
                symbol: {
                    "news": [
                        {"text": f"News about {symbol}", "score": np.random.uniform(-1, 1)}
                        for _ in range(3)
                    ],
                    "social_media": {
                        "twitter_sentiment": np.random.uniform(-1, 1),
                        "reddit_sentiment": np.random.uniform(-1, 1),
                        "volume": np.random.randint(1000, 10000)
                    },
                    "analyst_ratings": {
                        "buy": np.random.randint(5, 20),
                        "hold": np.random.randint(5, 15),
                        "sell": np.random.randint(0, 10)
                    }
                }
                for symbol in symbols
            }
        
        return MarketData(
            prices=prices_df,
            volumes=volumes_df,
            fundamentals=fundamentals,
            sentiment=sentiment,
            timestamp=datetime.now()
        )
    
    return _create


@pytest.fixture
def sample_agent_metrics():
    """Create sample agent metrics for testing."""
    return AgentMetrics(
        signal_accuracy=0.65,
        profit_factor=1.8,
        sharpe_ratio=1.2,
        max_drawdown=0.15,
        win_rate=0.58,
        avg_profit_per_signal=125.50,
        total_signals=150,
        timestamp=datetime.now()
    )


@pytest.fixture
def performance_data_generator():
    """Factory fixture to generate performance data."""
    def _generate(
        n_trades: int = 100,
        win_rate: float = 0.6,
        avg_win: float = 200,
        avg_loss: float = -100,
        seed: int = None
    ) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)
        
        # Generate wins and losses based on win rate
        outcomes = np.random.choice([1, 0], size=n_trades, p=[win_rate, 1-win_rate])
        
        profits = []
        for outcome in outcomes:
            if outcome:
                profit = avg_win * (1 + np.random.uniform(-0.5, 0.5))
            else:
                profit = avg_loss * (1 + np.random.uniform(-0.5, 0.5))
            profits.append(profit)
        
        return pd.DataFrame({
            'profit': profits,
            'timestamp': pd.date_range(end=datetime.now(), periods=n_trades, freq='H')
        })
    
    return _generate


@pytest.fixture
def mock_external_api():
    """Mock external API calls for testing."""
    class MockAPI:
        async def get_market_data(self, symbol: str) -> Dict:
            return {
                "price": 150.0,
                "volume": 1000000,
                "change": 0.02
            }
        
        async def get_fundamentals(self, symbol: str) -> Dict:
            return {
                "pe_ratio": 25.0,
                "revenue_growth": 0.15,
                "gross_margin": 0.35
            }
        
        async def get_sentiment(self, symbol: str) -> Dict:
            return {
                "overall": 0.65,
                "news": 0.7,
                "social": 0.6
            }
    
    return MockAPI()


def assert_valid_signal(signal):
    """Helper function to validate a trading signal."""
    from alpha_pulse.agents.interfaces import TradeSignal, SignalDirection
    
    assert isinstance(signal, TradeSignal)
    assert signal.agent_id is not None
    assert signal.symbol is not None
    assert signal.direction in SignalDirection
    assert 0 <= signal.confidence <= 1
    assert signal.timestamp is not None
    
    if signal.target_price is not None:
        assert signal.target_price > 0
    
    if signal.stop_loss is not None:
        assert signal.stop_loss > 0
    
    assert isinstance(signal.metadata, dict)