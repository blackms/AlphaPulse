"""
Unit tests for Fundamental Analysis Agent.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from alpha_pulse.agents.fundamental_agent import FundamentalAgent
from alpha_pulse.agents.interfaces import (
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


@pytest.fixture
def fundamental_agent():
    """Create a fundamental agent instance for testing."""
    config = {
        "min_revenue_growth": 0.1,
        "min_gross_margin": 0.3,
        "min_ebitda_margin": 0.15,
        "min_net_margin": 0.1,
        "min_current_ratio": 1.5,
        "min_quick_ratio": 1.0,
        "min_asset_turnover": 0.5,
        "zscore_threshold": 2.0
    }
    return FundamentalAgent(config)


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamental data."""
    return {
        "AAPL": {
            "revenue": 300000000,
            "revenue_growth": 0.15,
            "gross_margin": 0.35,
            "ebitda_margin": 0.20,
            "net_margin": 0.12,
            "current_ratio": 1.8,
            "quick_ratio": 1.2,
            "asset_turnover": 0.7,
            "debt_to_equity": 0.5,
            "roe": 0.25,
            "fcf_yield": 0.04,
            "pe_ratio": 25,
            "peg_ratio": 1.2,
            "price_to_book": 3.5,
            "enterprise_value": 1000000000,
            "sector": "Technology",
            "industry": "Consumer Electronics"
        },
        "GOOGL": {
            "revenue": 250000000,
            "revenue_growth": 0.20,
            "gross_margin": 0.55,
            "ebitda_margin": 0.30,
            "net_margin": 0.22,
            "current_ratio": 2.5,
            "quick_ratio": 2.3,
            "asset_turnover": 0.6,
            "debt_to_equity": 0.1,
            "roe": 0.18,
            "fcf_yield": 0.03,
            "pe_ratio": 30,
            "peg_ratio": 1.5,
            "price_to_book": 5.0,
            "enterprise_value": 1500000000,
            "sector": "Technology",
            "industry": "Internet Services"
        }
    }


@pytest.fixture
def sample_market_data_with_fundamentals(sample_fundamentals):
    """Create market data with fundamentals."""
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    
    prices_df = pd.DataFrame({
        'AAPL': np.random.uniform(140, 160, 200),
        'GOOGL': np.random.uniform(2000, 2200, 200)
    }, index=dates)
    
    volumes_df = pd.DataFrame({
        'AAPL': np.random.uniform(1e6, 2e6, 200),
        'GOOGL': np.random.uniform(8e5, 1.5e6, 200)
    }, index=dates)
    
    return MarketData(
        prices=prices_df,
        volumes=volumes_df,
        fundamentals=sample_fundamentals,
        timestamp=datetime.now()
    )


@pytest.fixture
def weak_fundamentals():
    """Create sample weak fundamental data."""
    return {
        "XYZ": {
            "revenue": 50000000,
            "revenue_growth": -0.05,  # Negative growth
            "gross_margin": 0.20,     # Low margin
            "ebitda_margin": 0.05,    # Low margin
            "net_margin": 0.02,       # Low margin
            "current_ratio": 0.8,     # Below 1
            "quick_ratio": 0.5,       # Below 1
            "asset_turnover": 0.3,    # Low
            "debt_to_equity": 2.5,    # High debt
            "roe": 0.05,              # Low return
            "fcf_yield": -0.02,       # Negative FCF
            "pe_ratio": 50,           # High P/E
            "peg_ratio": 3.0,         # High PEG
            "price_to_book": 8.0,     # High P/B
            "enterprise_value": 100000000,
            "sector": "Technology",
            "industry": "Software"
        }
    }


class TestFundamentalAgent:
    """Test cases for Fundamental Agent."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test agent initialization."""
        agent = FundamentalAgent()
        assert agent.agent_id == "fundamental_agent"
        assert agent.financial_metrics['revenue_growth'] == 0.1
        assert agent.macro_weights['gdp_growth'] == 0.2
        
        # Test custom config
        custom_config = {"min_revenue_growth": 0.2}
        agent = FundamentalAgent(custom_config)
        assert agent.financial_metrics['revenue_growth'] == 0.2
    
    @pytest.mark.asyncio
    async def test_initialize_method(self, fundamental_agent):
        """Test initialize method."""
        config = {
            "analysis_timeframes": {
                "short_term": 60,
                "medium_term": 120,
                "long_term": 240
            },
            "zscore_threshold": 1.5
        }
        
        await fundamental_agent.initialize(config)
        assert fundamental_agent.analysis_timeframes['short_term'] == 60
        assert fundamental_agent.historical_zscore_threshold == 1.5
    
    @pytest.mark.asyncio
    async def test_generate_signals_with_valid_data(self, fundamental_agent, sample_market_data_with_fundamentals):
        """Test signal generation with valid fundamental data."""
        await fundamental_agent.initialize({})
        signals = await fundamental_agent.generate_signals(sample_market_data_with_fundamentals)
        
        assert isinstance(signals, list)
        # Should generate signals for companies with good fundamentals
        assert len(signals) > 0
        
        for signal in signals:
            assert isinstance(signal, TradeSignal)
            assert signal.agent_id == "fundamental_agent"
            assert signal.symbol in ['AAPL', 'GOOGL']
            assert signal.direction in SignalDirection
            assert 0 <= signal.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_generate_signals_without_fundamentals(self, fundamental_agent):
        """Test signal generation without fundamental data."""
        market_data = MarketData(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=None,
            timestamp=datetime.now()
        )
        
        signals = await fundamental_agent.generate_signals(market_data)
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_analyze_financials(self, fundamental_agent, sample_fundamentals):
        """Test financial analysis method."""
        await fundamental_agent.initialize({})
        
        # Test with good fundamentals
        score = await fundamental_agent._analyze_financials(sample_fundamentals["AAPL"])
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be positive for good fundamentals
        
        # Test with weak fundamentals
        weak_fund = {
            "revenue_growth": -0.1,
            "gross_margin": 0.1,
            "ebitda_margin": 0.05,
            "net_margin": -0.02,
            "current_ratio": 0.5,
            "quick_ratio": 0.3,
            "asset_turnover": 0.2
        }
        score = await fundamental_agent._analyze_financials(weak_fund)
        assert score < 0.5  # Should be low for weak fundamentals
    
    @pytest.mark.asyncio
    async def test_calculate_price_target(self, fundamental_agent, sample_fundamentals):
        """Test price target calculation."""
        target = await fundamental_agent._calculate_price_target(sample_fundamentals["AAPL"])
        
        assert isinstance(target, float)
        assert target > 0
    
    @pytest.mark.asyncio
    async def test_weak_fundamentals_signal(self, fundamental_agent, weak_fundamentals):
        """Test signal generation with weak fundamentals."""
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        
        market_data = MarketData(
            prices=pd.DataFrame({'XYZ': np.random.uniform(10, 15, 200)}, index=dates),
            volumes=pd.DataFrame({'XYZ': np.random.uniform(1e5, 2e5, 200)}, index=dates),
            fundamentals=weak_fundamentals,
            timestamp=datetime.now()
        )
        
        await fundamental_agent.initialize({})
        signals = await fundamental_agent.generate_signals(market_data)
        
        # Should generate sell signal or no signal for weak fundamentals
        if signals:
            assert signals[0].direction in [SignalDirection.SELL, SignalDirection.SHORT]
    
    @pytest.mark.asyncio
    async def test_sector_correlation_update(self, fundamental_agent, sample_market_data_with_fundamentals):
        """Test sector correlation updates."""
        await fundamental_agent.initialize({})
        await fundamental_agent._update_sector_correlations(sample_market_data_with_fundamentals)
        
        assert hasattr(fundamental_agent, 'sector_correlations')
        assert isinstance(fundamental_agent.sector_correlations, dict)
    
    @pytest.mark.asyncio
    async def test_macro_environment_analysis(self, fundamental_agent, sample_market_data_with_fundamentals):
        """Test macro environment analysis."""
        score = await fundamental_agent._analyze_macro_environment(sample_market_data_with_fundamentals)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_historical_trends_analysis(self, fundamental_agent, sample_market_data_with_fundamentals):
        """Test historical trends analysis."""
        symbol = "AAPL"
        score = await fundamental_agent._analyze_historical_trends(symbol, sample_market_data_with_fundamentals)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_validate_signal(self, fundamental_agent):
        """Test signal validation."""
        valid_signal = TradeSignal(
            agent_id="fundamental_agent",
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            target_price=150.0,
            stop_loss=140.0
        )
        
        assert await fundamental_agent.validate_signal(valid_signal) is True
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, fundamental_agent):
        """Test metrics update functionality."""
        performance_data = pd.DataFrame({
            'profit': [200, -100, 300, -50, 400, 150, -75, 250]
        })
        
        metrics = await fundamental_agent.update_metrics(performance_data)
        
        assert isinstance(metrics, AgentMetrics)
        assert metrics.win_rate == 0.625  # 5 wins out of 8
        assert metrics.total_signals == 8
        assert metrics.avg_profit_per_signal > 0
    
    @pytest.mark.asyncio
    async def test_get_confidence_level(self, fundamental_agent):
        """Test confidence level calculation."""
        # Without metrics
        confidence = await fundamental_agent.get_confidence_level()
        assert confidence == 0.5
        
        # With good metrics
        fundamental_agent.metrics = AgentMetrics(
            signal_accuracy=0.75,
            profit_factor=2.5,
            sharpe_ratio=1.8,
            max_drawdown=0.08,
            win_rate=0.75,
            avg_profit_per_signal=150,
            total_signals=50,
            timestamp=datetime.now()
        )
        
        confidence = await fundamental_agent.get_confidence_level()
        assert confidence == 0.75
    
    @pytest.mark.asyncio
    async def test_signal_metadata(self, fundamental_agent, sample_market_data_with_fundamentals):
        """Test that signals contain proper metadata."""
        await fundamental_agent.initialize({})
        signals = await fundamental_agent.generate_signals(sample_market_data_with_fundamentals)
        
        if signals:
            signal = signals[0]
            assert 'strategy' in signal.metadata
            assert signal.metadata['strategy'] == 'fundamental'
            assert 'financial_score' in signal.metadata
            assert 'fundamentals' in signal.metadata
    
    @pytest.mark.asyncio
    async def test_error_handling(self, fundamental_agent):
        """Test error handling in signal generation."""
        # Create market data with corrupted fundamentals
        corrupted_fundamentals = {
            "AAPL": {
                "revenue_growth": "invalid",  # Invalid type
                "gross_margin": None,
                "ebitda_margin": np.nan
            }
        }
        
        market_data = MarketData(
            prices=pd.DataFrame({'AAPL': [150] * 200}),
            volumes=pd.DataFrame({'AAPL': [1e6] * 200}),
            fundamentals=corrupted_fundamentals,
            timestamp=datetime.now()
        )
        
        # Should handle error gracefully
        signals = await fundamental_agent.generate_signals(market_data)
        assert isinstance(signals, list)  # Should return empty list or valid signals