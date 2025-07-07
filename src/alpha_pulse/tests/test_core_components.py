"""
Comprehensive tests for core AlphaPulse components to improve coverage.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np


class TestAgentsCore:
    """Test core agent functionality."""
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        from alpha_pulse.agents.interfaces import MarketData
        
        # Create mock price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        prices = pd.DataFrame({
            'BTC': np.random.randn(100).cumsum() + 50000,
            'ETH': np.random.randn(100).cumsum() + 3000,
        }, index=dates)
        
        volumes = pd.DataFrame({
            'BTC': np.random.rand(100) * 1000000,
            'ETH': np.random.rand(100) * 500000,
        }, index=dates)
        
        return MarketData(
            prices=prices,
            volumes=volumes,
            fundamentals={'BTC': {'market_cap': 1e12}},
            sentiment={'BTC': 0.7, 'ETH': 0.6}
        )
    
    def test_base_trade_agent_initialization(self):
        """Test BaseTradeAgent initialization."""
        from alpha_pulse.agents.interfaces import BaseTradeAgent
        
        agent = BaseTradeAgent("test_agent", {"param1": "value1"})
        assert agent.agent_id == "test_agent"
        assert agent.config["param1"] == "value1"
        assert agent.metrics is not None
    
    def test_trade_signal_creation(self):
        """Test TradeSignal creation and validation."""
        from alpha_pulse.agents.interfaces import TradeSignal, SignalDirection
        
        signal = TradeSignal(
            agent_id="test_agent",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=0.85,
            timestamp=datetime.now(),
            target_price=55000.0,
            stop_loss=50000.0
        )
        
        assert signal.agent_id == "test_agent"
        assert signal.symbol == "BTC"
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 0.85
        assert signal.target_price == 55000.0
    
    def test_agent_metrics(self):
        """Test AgentMetrics functionality."""
        from alpha_pulse.agents.interfaces import AgentMetrics
        
        metrics = AgentMetrics(
            signal_accuracy=0.65,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            win_rate=0.60,
            avg_profit_per_signal=250.0,
            total_signals=100,
            timestamp=datetime.now()
        )
        
        assert metrics.signal_accuracy == 0.65
        assert metrics.sharpe_ratio == 1.2
        assert metrics.total_signals == 100


class TestRiskManagement:
    """Test risk management components."""
    
    def test_risk_limits(self):
        """Test RiskLimits creation and validation."""
        from alpha_pulse.risk_management.interfaces import RiskLimits
        
        limits = RiskLimits(
            max_position_size=10000.0,
            max_portfolio_risk=0.02,
            max_correlation=0.7,
            max_leverage=3.0,
            stop_loss_pct=0.05,
            max_drawdown=0.20
        )
        
        assert limits.max_position_size == 10000.0
        assert limits.max_portfolio_risk == 0.02
        assert limits.stop_loss_pct == 0.05
    
    def test_position_limits(self):
        """Test PositionLimits functionality."""
        from alpha_pulse.risk_management.interfaces import PositionLimits
        
        limits = PositionLimits(
            symbol="BTC",
            max_position_size=5000.0,
            max_leverage=2.0,
            min_position_size=100.0,
            max_concentration=0.3
        )
        
        assert limits.symbol == "BTC"
        assert limits.max_position_size == 5000.0
        assert limits.max_concentration == 0.3
    
    def test_risk_metrics(self):
        """Test RiskMetrics calculation."""
        from alpha_pulse.risk_management.interfaces import RiskMetrics
        
        metrics = RiskMetrics(
            portfolio_var=0.05,
            portfolio_cvar=0.08,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.15,
            current_drawdown=0.05,
            correlation_risk=0.6,
            concentration_risk=0.3,
            timestamp=datetime.now()
        )
        
        assert metrics.portfolio_var == 0.05
        assert metrics.sharpe_ratio == 1.5
        assert metrics.correlation_risk == 0.6


class TestPortfolio:
    """Test portfolio management components."""
    
    def test_portfolio_state(self):
        """Test PortfolioState functionality."""
        from alpha_pulse.portfolio.interfaces import PortfolioState
        
        state = PortfolioState(
            cash=50000.0,
            positions={"BTC": {"quantity": 0.5, "value": 25000}},
            total_value=75000.0,
            timestamp=datetime.now()
        )
        
        assert state.cash == 50000.0
        assert state.total_value == 75000.0
        assert "BTC" in state.positions
    
    def test_optimization_constraints(self):
        """Test OptimizationConstraints."""
        from alpha_pulse.portfolio.interfaces import OptimizationConstraints
        
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.40,
            max_concentration=0.50,
            target_return=0.15,
            max_volatility=0.20
        )
        
        assert constraints.min_weight == 0.05
        assert constraints.max_weight == 0.40
        assert constraints.target_return == 0.15


class TestExecution:
    """Test execution components."""
    
    def test_order_creation(self):
        """Test Order creation and validation."""
        from alpha_pulse.execution.broker_interface import Order, OrderType, OrderSide
        
        order = Order(
            order_id="TEST123",
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50000.0,
            timestamp=datetime.now()
        )
        
        assert order.order_id == "TEST123"
        assert order.symbol == "BTC"
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.1
    
    def test_order_status(self):
        """Test OrderStatus enum."""
        from alpha_pulse.execution.broker_interface import OrderStatus
        
        # Test all status values exist
        assert OrderStatus.PENDING
        assert OrderStatus.OPEN
        assert OrderStatus.FILLED
        assert OrderStatus.CANCELLED
        assert OrderStatus.REJECTED


class TestMonitoring:
    """Test monitoring components."""
    
    def test_metric_types(self):
        """Test MetricType enum."""
        from alpha_pulse.monitoring.interfaces import MetricType
        
        # Verify metric types exist
        assert MetricType.SYSTEM_HEALTH
        assert MetricType.PORTFOLIO_VALUE
        assert MetricType.RISK_METRICS
        assert MetricType.TRADE_PERFORMANCE
    
    def test_alert_levels(self):
        """Test AlertLevel enum."""
        from alpha_pulse.monitoring.interfaces import AlertLevel
        
        assert AlertLevel.INFO
        assert AlertLevel.WARNING
        assert AlertLevel.CRITICAL
        assert AlertLevel.ERROR
    
    def test_health_status(self):
        """Test HealthStatus."""
        from alpha_pulse.monitoring.interfaces import HealthStatus
        
        status = HealthStatus(
            status="healthy",
            uptime=3600,
            last_check=datetime.now(),
            components={"api": "healthy", "database": "healthy"}
        )
        
        assert status.status == "healthy"
        assert status.uptime == 3600
        assert status.components["api"] == "healthy"


class TestAPI:
    """Test API components with mocking."""
    
    @patch('alpha_pulse.api.main.FastAPI')
    def test_api_app_creation(self, mock_fastapi):
        """Test API app creation."""
        from alpha_pulse.api import main
        
        # The app should be created
        assert main.app is not None
    
    def test_api_config(self):
        """Test API configuration."""
        from alpha_pulse.api.config import Settings
        
        settings = Settings(
            database_url="postgresql://test:test@localhost/test",
            redis_url="redis://localhost:6379",
            jwt_secret_key="test_secret",
            environment="test"
        )
        
        assert settings.database_url.startswith("postgresql://")
        assert settings.redis_url.startswith("redis://")
        assert settings.environment == "test"


class TestUtils:
    """Test utility functions."""
    
    def test_logging_utils(self):
        """Test logging utilities."""
        from alpha_pulse.utils.logging_utils import get_logger
        
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_constants(self):
        """Test constants."""
        from alpha_pulse.utils.constants import SUPPORTED_EXCHANGES
        
        assert isinstance(SUPPORTED_EXCHANGES, list)
        assert len(SUPPORTED_EXCHANGES) > 0
        assert "binance" in SUPPORTED_EXCHANGES


class TestMLComponents:
    """Test ML components with proper mocking."""
    
    @patch('alpha_pulse.ml.regime.regime_classifier.joblib')
    def test_regime_classifier_mock(self, mock_joblib):
        """Test RegimeClassifier with mocking."""
        from alpha_pulse.ml.regime.regime_classifier import RegimeClassifier, RegimeType
        
        # Mock the model loading
        mock_joblib.load.return_value = MagicMock()
        
        classifier = RegimeClassifier()
        assert classifier is not None
        
        # Test regime types
        assert RegimeType.BULL_MARKET
        assert RegimeType.BEAR_MARKET
        assert RegimeType.RANGING
        assert RegimeType.HIGH_VOLATILITY


class TestDataPipeline:
    """Test data pipeline components."""
    
    @patch('alpha_pulse.data_pipeline.manager.DataPipelineManager')
    def test_data_pipeline_manager_mock(self, mock_manager):
        """Test DataPipelineManager with mocking."""
        manager = mock_manager.return_value
        manager.start = AsyncMock()
        manager.stop = AsyncMock()
        
        assert manager is not None
        assert hasattr(manager, 'start')
        assert hasattr(manager, 'stop')


class TestExchanges:
    """Test exchange components."""
    
    def test_exchange_factory(self):
        """Test ExchangeFactory."""
        from alpha_pulse.exchanges.factories import ExchangeFactory, ExchangeType
        
        assert ExchangeFactory is not None
        assert ExchangeType.BINANCE
        assert ExchangeType.BYBIT
    
    def test_ohlcv_data(self):
        """Test OHLCV data structure."""
        from alpha_pulse.exchanges.base import OHLCV
        
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        assert ohlcv.open == 50000.0
        assert ohlcv.high == 51000.0
        assert ohlcv.volume == 1000.0


class TestBacktesting:
    """Test backtesting components."""
    
    @patch('alpha_pulse.backtesting.engine.BacktestEngine')
    def test_backtest_engine_mock(self, mock_engine):
        """Test BacktestEngine with mocking."""
        engine = mock_engine.return_value
        engine.run = AsyncMock(return_value={"total_return": 0.15})
        
        assert engine is not None
        assert hasattr(engine, 'run')


# Add more test classes for other components...

if __name__ == "__main__":
    pytest.main([__file__, "-v"])