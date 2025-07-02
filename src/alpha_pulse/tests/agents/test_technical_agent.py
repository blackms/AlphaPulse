"""
Unit tests for Technical Analysis Agent.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from alpha_pulse.agents.technical_agent import TechnicalAgent
from alpha_pulse.agents.interfaces import (
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


@pytest.fixture
def technical_agent():
    """Create a technical agent instance for testing."""
    config = {
        "trend_weight": 0.3,
        "momentum_weight": 0.2,
        "volatility_weight": 0.2,
        "volume_weight": 0.15,
        "pattern_weight": 0.15,
        "timeframes": {
            "short": 14,
            "medium": 50,
            "long": 180
        }
    }
    return TechnicalAgent(config)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Generate 200 days of price data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    
    # Create realistic price movements
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 200)
    prices = base_price * (1 + returns).cumprod()
    
    # Create volume data with correlation to price movements
    base_volume = 1000000
    volume_noise = np.random.normal(1, 0.2, 200)
    volumes = base_volume * volume_noise * (1 + np.abs(returns) * 10)
    
    prices_df = pd.DataFrame({
        'AAPL': prices,
        'GOOGL': prices * 1.2
    }, index=dates)
    
    volumes_df = pd.DataFrame({
        'AAPL': volumes,
        'GOOGL': volumes * 0.8
    }, index=dates)
    
    return MarketData(
        prices=prices_df,
        volumes=volumes_df,
        timestamp=datetime.now()
    )


@pytest.fixture
def trending_market_data():
    """Create market data with a clear uptrend."""
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    
    # Create uptrending price data
    trend = np.linspace(100, 150, 200)
    noise = np.random.normal(0, 2, 200)
    prices = trend + noise
    
    # Volume increases with trend
    volumes = np.linspace(1000000, 1500000, 200) + np.random.normal(0, 100000, 200)
    
    prices_df = pd.DataFrame({
        'AAPL': prices
    }, index=dates)
    
    volumes_df = pd.DataFrame({
        'AAPL': volumes
    }, index=dates)
    
    return MarketData(
        prices=prices_df,
        volumes=volumes_df,
        timestamp=datetime.now()
    )


@pytest.fixture
def downtrending_market_data():
    """Create market data with a clear downtrend."""
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    
    # Create downtrending price data
    trend = np.linspace(150, 100, 200)
    noise = np.random.normal(0, 2, 200)
    prices = trend + noise
    
    volumes = np.linspace(1000000, 1500000, 200) + np.random.normal(0, 100000, 200)
    
    prices_df = pd.DataFrame({
        'AAPL': prices
    }, index=dates)
    
    volumes_df = pd.DataFrame({
        'AAPL': volumes
    }, index=dates)
    
    return MarketData(
        prices=prices_df,
        volumes=volumes_df,
        timestamp=datetime.now()
    )


class TestTechnicalAgent:
    """Test cases for Technical Agent."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test agent initialization."""
        agent = TechnicalAgent()
        assert agent.agent_id == "technical_agent"
        assert agent.indicator_weights['trend'] == 0.3
        
        # Test custom config
        custom_config = {"trend_weight": 0.5}
        agent = TechnicalAgent(custom_config)
        assert agent.indicator_weights['trend'] == 0.5
    
    @pytest.mark.asyncio
    async def test_generate_signals_with_valid_data(self, technical_agent, sample_market_data):
        """Test signal generation with valid market data."""
        await technical_agent.initialize({})
        signals = await technical_agent.generate_signals(sample_market_data)
        
        assert isinstance(signals, list)
        # Should generate signals for both symbols
        assert len(signals) <= 2
        
        for signal in signals:
            assert isinstance(signal, TradeSignal)
            assert signal.agent_id == "technical_agent"
            assert signal.symbol in ['AAPL', 'GOOGL']
            assert signal.direction in SignalDirection
            assert 0 <= signal.confidence <= 1
            assert signal.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_generate_signals_with_empty_data(self, technical_agent):
        """Test signal generation with empty market data."""
        empty_data = MarketData(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            timestamp=datetime.now()
        )
        
        signals = await technical_agent.generate_signals(empty_data)
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_generate_signals_insufficient_data(self, technical_agent):
        """Test signal generation with insufficient historical data."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        prices_df = pd.DataFrame({
            'AAPL': np.random.uniform(100, 110, 50)
        }, index=dates)
        
        market_data = MarketData(
            prices=prices_df,
            volumes=None,
            timestamp=datetime.now()
        )
        
        signals = await technical_agent.generate_signals(market_data)
        # Should not generate signals due to insufficient data
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_uptrend_detection(self, technical_agent, trending_market_data):
        """Test detection of uptrending market."""
        await technical_agent.initialize({})
        signals = await technical_agent.generate_signals(trending_market_data)
        
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_downtrend_detection(self, technical_agent, downtrending_market_data):
        """Test detection of downtrending market."""
        await technical_agent.initialize({})
        signals = await technical_agent.generate_signals(downtrending_market_data)
        
        assert len(signals) > 0
        signal = signals[0]
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self, technical_agent, sample_market_data):
        """Test trend analysis method."""
        prices = sample_market_data.prices['AAPL']
        score = await technical_agent._analyze_trends(prices)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_momentum(self, technical_agent, sample_market_data):
        """Test momentum analysis method."""
        prices = sample_market_data.prices['AAPL']
        score = await technical_agent._analyze_momentum(prices)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_volatility(self, technical_agent, sample_market_data):
        """Test volatility analysis method."""
        prices = sample_market_data.prices['AAPL']
        score = await technical_agent._analyze_volatility(prices)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_volume(self, technical_agent, sample_market_data):
        """Test volume analysis method."""
        prices = sample_market_data.prices['AAPL']
        volumes = sample_market_data.volumes['AAPL']
        score = await technical_agent._analyze_volume(prices, volumes)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_patterns(self, technical_agent, sample_market_data):
        """Test pattern analysis method."""
        prices = sample_market_data.prices['AAPL']
        score = await technical_agent._analyze_patterns(prices)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_find_support_resistance(self, technical_agent, sample_market_data):
        """Test support and resistance level detection."""
        prices = sample_market_data.prices['AAPL']
        levels = await technical_agent._find_support_resistance(prices)
        
        assert isinstance(levels, list)
        for level in levels:
            assert isinstance(level, float)
            assert level > 0
    
    @pytest.mark.asyncio
    async def test_signal_metadata(self, technical_agent, sample_market_data):
        """Test that signals contain proper metadata."""
        await technical_agent.initialize({})
        signals = await technical_agent.generate_signals(sample_market_data)
        
        if signals:
            signal = signals[0]
            assert 'strategy' in signal.metadata
            assert signal.metadata['strategy'] == 'technical'
            assert 'technical_score' in signal.metadata
            assert 'indicators' in signal.metadata
            assert isinstance(signal.metadata['indicators'], dict)
    
    @pytest.mark.asyncio
    async def test_validate_signal(self, technical_agent):
        """Test signal validation."""
        valid_signal = TradeSignal(
            agent_id="technical_agent",
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        assert await technical_agent.validate_signal(valid_signal) is True
        
        # Test invalid signals
        invalid_signal = TradeSignal(
            agent_id="technical_agent",
            symbol="",
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now()
        )
        assert await technical_agent.validate_signal(invalid_signal) is False
        
        invalid_signal2 = TradeSignal(
            agent_id="technical_agent",
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=1.5,  # Invalid confidence
            timestamp=datetime.now()
        )
        assert await technical_agent.validate_signal(invalid_signal2) is False
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, technical_agent):
        """Test metrics update functionality."""
        performance_data = pd.DataFrame({
            'profit': [100, -50, 200, -30, 150]
        })
        
        metrics = await technical_agent.update_metrics(performance_data)
        
        assert isinstance(metrics, AgentMetrics)
        assert metrics.win_rate == 0.6  # 3 wins out of 5
        assert metrics.total_signals == 5
        assert metrics.avg_profit_per_signal == 74.0
    
    @pytest.mark.asyncio
    async def test_adapt_parameters(self, technical_agent):
        """Test parameter adaptation based on metrics."""
        metrics = AgentMetrics(
            signal_accuracy=0.7,
            profit_factor=2.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.7,
            avg_profit_per_signal=100,
            total_signals=100,
            timestamp=datetime.now()
        )
        
        await technical_agent.adapt_parameters(metrics)
        assert technical_agent.metrics == metrics
    
    @pytest.mark.asyncio
    async def test_get_confidence_level(self, technical_agent):
        """Test confidence level calculation."""
        # Without metrics
        confidence = await technical_agent.get_confidence_level()
        assert confidence == 0.5
        
        # With metrics
        technical_agent.metrics = AgentMetrics(
            signal_accuracy=0.8,
            profit_factor=2.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.8,
            avg_profit_per_signal=100,
            total_signals=100,
            timestamp=datetime.now()
        )
        
        confidence = await technical_agent.get_confidence_level()
        assert confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling(self, technical_agent):
        """Test error handling in signal generation."""
        # Create market data that will cause an error
        corrupted_data = MarketData(
            prices=pd.DataFrame({
                'AAPL': [np.nan] * 200
            }),
            volumes=None,
            timestamp=datetime.now()
        )
        
        signals = await technical_agent.generate_signals(corrupted_data)
        assert signals == []  # Should handle error gracefully