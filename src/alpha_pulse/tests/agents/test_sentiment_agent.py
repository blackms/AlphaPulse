"""
Unit tests for Sentiment Analysis Agent.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from collections import defaultdict

from alpha_pulse.agents.sentiment_agent import SentimentAgent
from alpha_pulse.agents.interfaces import (
    MarketData,
    TradeSignal,
    SignalDirection,
    AgentMetrics
)


@pytest.fixture
def sentiment_agent():
    """Create a sentiment agent instance for testing."""
    config = {
        "news_weight": 0.3,
        "social_media_weight": 0.25,
        "market_data_weight": 0.25,
        "analyst_weight": 0.2,
        "momentum_window": 14,
        "volume_window": 5
    }
    return SentimentAgent(config)


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data."""
    return {
        "AAPL": {
            "news": [
                {"text": "Apple announces record breaking quarterly earnings", "score": 0.9},
                {"text": "iPhone sales exceed expectations", "score": 0.8},
                {"text": "Apple invests heavily in AI research", "score": 0.7}
            ],
            "social_media": {
                "twitter_sentiment": 0.6,
                "reddit_sentiment": 0.7,
                "stocktwits_sentiment": 0.65,
                "volume": 10000,
                "mentions": 5000
            },
            "analyst_ratings": {
                "buy": 20,
                "hold": 5,
                "sell": 2,
                "average_price_target": 180,
                "consensus": "Strong Buy"
            }
        },
        "GOOGL": {
            "news": [
                {"text": "Google faces regulatory challenges", "score": -0.4},
                {"text": "Alphabet reports mixed quarterly results", "score": 0.0},
                {"text": "Google Cloud continues strong growth", "score": 0.6}
            ],
            "social_media": {
                "twitter_sentiment": 0.2,
                "reddit_sentiment": 0.1,
                "stocktwits_sentiment": 0.15,
                "volume": 8000,
                "mentions": 3000
            },
            "analyst_ratings": {
                "buy": 15,
                "hold": 10,
                "sell": 3,
                "average_price_target": 2500,
                "consensus": "Buy"
            }
        }
    }


@pytest.fixture
def negative_sentiment_data():
    """Create sample negative sentiment data."""
    return {
        "XYZ": {
            "news": [
                {"text": "Company announces massive layoffs", "score": -0.8},
                {"text": "CEO resigns amid scandal", "score": -0.9},
                {"text": "Revenue misses expectations by wide margin", "score": -0.7}
            ],
            "social_media": {
                "twitter_sentiment": -0.7,
                "reddit_sentiment": -0.8,
                "stocktwits_sentiment": -0.75,
                "volume": 20000,
                "mentions": 15000
            },
            "analyst_ratings": {
                "buy": 2,
                "hold": 5,
                "sell": 15,
                "average_price_target": 5,
                "consensus": "Strong Sell"
            }
        }
    }


@pytest.fixture
def sample_market_data_with_sentiment(sample_sentiment_data):
    """Create market data with sentiment."""
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
        sentiment=sample_sentiment_data,
        timestamp=datetime.now()
    )


class TestSentimentAgent:
    """Test cases for Sentiment Agent."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test agent initialization."""
        agent = SentimentAgent()
        assert agent.agent_id == "sentiment_agent"
        assert agent.sentiment_sources['news'] == 0.3
        assert agent.sentiment_thresholds['strong_buy'] == 0.7
        
        # Test custom config
        custom_config = {"news_weight": 0.4, "momentum_window": 20}
        agent = SentimentAgent(custom_config)
        assert agent.sentiment_sources['news'] == 0.4
        assert agent.momentum_window == 20
    
    @pytest.mark.asyncio
    async def test_initialize_method(self, sentiment_agent):
        """Test initialize method."""
        config = {"volume_window": 10}
        await sentiment_agent.initialize(config)
        
        assert hasattr(sentiment_agent, 'sentiment_history')
        assert isinstance(sentiment_agent.sentiment_history, defaultdict)
        assert hasattr(sentiment_agent, 'volume_history')
        assert hasattr(sentiment_agent, 'momentum_history')
    
    @pytest.mark.asyncio
    async def test_generate_signals_with_valid_data(self, sentiment_agent, sample_market_data_with_sentiment):
        """Test signal generation with valid sentiment data."""
        await sentiment_agent.initialize({})
        signals = await sentiment_agent.generate_signals(sample_market_data_with_sentiment)
        
        assert isinstance(signals, list)
        # Should generate signals based on sentiment
        assert len(signals) > 0
        
        for signal in signals:
            assert isinstance(signal, TradeSignal)
            assert signal.agent_id == "sentiment_agent"
            assert signal.symbol in ['AAPL', 'GOOGL']
            assert signal.direction in SignalDirection
            assert 0 <= signal.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_generate_signals_without_sentiment(self, sentiment_agent):
        """Test signal generation without sentiment data."""
        market_data = MarketData(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            sentiment=None,
            timestamp=datetime.now()
        )
        
        signals = await sentiment_agent.generate_signals(market_data)
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_positive_sentiment_signal(self, sentiment_agent, sample_market_data_with_sentiment):
        """Test signal generation with positive sentiment."""
        await sentiment_agent.initialize({})
        signals = await sentiment_agent.generate_signals(sample_market_data_with_sentiment)
        
        # AAPL has positive sentiment, should generate buy signal
        aapl_signals = [s for s in signals if s.symbol == 'AAPL']
        if aapl_signals:
            assert aapl_signals[0].direction == SignalDirection.BUY
            assert aapl_signals[0].confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_negative_sentiment_signal(self, sentiment_agent, negative_sentiment_data):
        """Test signal generation with negative sentiment."""
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        
        market_data = MarketData(
            prices=pd.DataFrame({'XYZ': np.random.uniform(10, 15, 200)}, index=dates),
            volumes=pd.DataFrame({'XYZ': np.random.uniform(1e5, 2e5, 200)}, index=dates),
            sentiment=negative_sentiment_data,
            timestamp=datetime.now()
        )
        
        await sentiment_agent.initialize({})
        signals = await sentiment_agent.generate_signals(market_data)
        
        # Should generate sell signal for negative sentiment
        if signals:
            assert signals[0].direction in [SignalDirection.SELL, SignalDirection.SHORT]
    
    @pytest.mark.asyncio
    async def test_analyze_news_sentiment(self, sentiment_agent, sample_sentiment_data):
        """Test news sentiment analysis."""
        await sentiment_agent.initialize({})
        
        # Test with positive news
        score = await sentiment_agent._analyze_news_sentiment(sample_sentiment_data["AAPL"])
        assert isinstance(score, float)
        assert score > 0  # Should be positive
        
        # Test with no news
        empty_sentiment = {"news": []}
        score = await sentiment_agent._analyze_news_sentiment(empty_sentiment)
        assert score == 0
    
    @pytest.mark.asyncio
    async def test_analyze_social_sentiment(self, sentiment_agent, sample_sentiment_data):
        """Test social media sentiment analysis."""
        score = await sentiment_agent._analyze_social_sentiment(sample_sentiment_data["AAPL"])
        assert isinstance(score, float)
        assert -1 <= score <= 1
        assert score > 0  # Should be positive for AAPL
        
        # Test with mixed sentiment
        score = await sentiment_agent._analyze_social_sentiment(sample_sentiment_data["GOOGL"])
        assert score > 0  # Still positive but lower
    
    @pytest.mark.asyncio
    async def test_analyze_market_sentiment(self, sentiment_agent, sample_market_data_with_sentiment):
        """Test market sentiment analysis."""
        await sentiment_agent.initialize({})
        
        score = await sentiment_agent._analyze_market_sentiment("AAPL", sample_market_data_with_sentiment)
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_analyst_sentiment(self, sentiment_agent, sample_sentiment_data):
        """Test analyst sentiment analysis."""
        score = await sentiment_agent._analyze_analyst_sentiment(sample_sentiment_data["AAPL"])
        assert isinstance(score, float)
        assert score > 0  # Should be positive with more buy ratings
        
        # Test with no analyst data
        empty_sentiment = {"analyst_ratings": None}
        score = await sentiment_agent._analyze_analyst_sentiment(empty_sentiment)
        assert score == 0
    
    @pytest.mark.asyncio
    async def test_sentiment_history_tracking(self, sentiment_agent, sample_market_data_with_sentiment):
        """Test that sentiment history is tracked."""
        await sentiment_agent.initialize({})
        
        # Generate signals multiple times
        for _ in range(3):
            await sentiment_agent.generate_signals(sample_market_data_with_sentiment)
        
        # Check sentiment history
        assert len(sentiment_agent.sentiment_history["AAPL"]) == 3
        assert len(sentiment_agent.sentiment_history["GOOGL"]) == 3
    
    @pytest.mark.asyncio
    async def test_validate_signal(self, sentiment_agent):
        """Test signal validation."""
        valid_signal = TradeSignal(
            agent_id="sentiment_agent",
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        assert await sentiment_agent.validate_signal(valid_signal) is True
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, sentiment_agent):
        """Test metrics update functionality."""
        performance_data = pd.DataFrame({
            'profit': [150, -75, 200, 100, -50, 300]
        })
        
        metrics = await sentiment_agent.update_metrics(performance_data)
        
        assert isinstance(metrics, AgentMetrics)
        assert metrics.win_rate == 0.667  # 4 wins out of 6
        assert metrics.total_signals == 6
    
    @pytest.mark.asyncio
    async def test_signal_metadata(self, sentiment_agent, sample_market_data_with_sentiment):
        """Test that signals contain proper metadata."""
        await sentiment_agent.initialize({})
        signals = await sentiment_agent.generate_signals(sample_market_data_with_sentiment)
        
        if signals:
            signal = signals[0]
            assert 'sentiment_score' in signal.metadata
            assert 'sentiment_components' in signal.metadata
            assert isinstance(signal.metadata['sentiment_components'], dict)
    
    @pytest.mark.asyncio
    async def test_textblob_integration(self, sentiment_agent):
        """Test TextBlob sentiment analysis."""
        with patch('textblob.TextBlob') as mock_textblob:
            mock_blob = Mock()
            mock_blob.sentiment.polarity = 0.5
            mock_textblob.return_value = mock_blob
            
            sentiment_data = {
                "news": [{"text": "Test news", "score": None}]
            }
            
            # Should use TextBlob if score is not provided
            score = await sentiment_agent._analyze_news_sentiment(sentiment_data)
            assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sentiment_agent):
        """Test error handling in signal generation."""
        # Create market data with corrupted sentiment
        corrupted_sentiment = {
            "AAPL": {
                "news": "invalid",  # Should be a list
                "social_media": None,
                "analyst_ratings": {}
            }
        }
        
        market_data = MarketData(
            prices=pd.DataFrame({'AAPL': [150] * 200}),
            volumes=pd.DataFrame({'AAPL': [1e6] * 200}),
            sentiment=corrupted_sentiment,
            timestamp=datetime.now()
        )
        
        # Should handle error gracefully
        signals = await sentiment_agent.generate_signals(market_data)
        assert isinstance(signals, list)  # Should return empty list or valid signals
    
    @pytest.mark.asyncio
    async def test_mixed_sentiment_confidence(self, sentiment_agent):
        """Test confidence calculation with mixed sentiment."""
        mixed_sentiment = {
            "TEST": {
                "news": [
                    {"text": "Good news", "score": 0.8},
                    {"text": "Bad news", "score": -0.7}
                ],
                "social_media": {
                    "twitter_sentiment": 0.1,
                    "reddit_sentiment": -0.1,
                    "stocktwits_sentiment": 0.0,
                    "volume": 5000,
                    "mentions": 2000
                },
                "analyst_ratings": {
                    "buy": 10,
                    "hold": 10,
                    "sell": 10,
                    "average_price_target": 50,
                    "consensus": "Hold"
                }
            }
        }
        
        market_data = MarketData(
            prices=pd.DataFrame({'TEST': [50] * 200}),
            volumes=pd.DataFrame({'TEST': [1e5] * 200}),
            sentiment=mixed_sentiment,
            timestamp=datetime.now()
        )
        
        await sentiment_agent.initialize({})
        signals = await sentiment_agent.generate_signals(market_data)
        
        # With mixed sentiment, confidence should be lower
        if signals:
            assert signals[0].confidence < 0.5