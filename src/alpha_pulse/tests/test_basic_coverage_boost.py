"""
Basic tests to boost coverage to meet minimum 15% requirement.
Tests simple utility functions and basic module imports.
"""
import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBasicImports:
    """Test that basic modules can be imported."""
    
    def test_import_config_modules(self):
        """Test importing configuration modules."""
        from alpha_pulse.config import api_config
        from alpha_pulse.config import database
        from alpha_pulse.config import monitoring
        assert api_config is not None
        assert database is not None
        assert monitoring is not None
    
    def test_import_model_modules(self):
        """Test importing model modules."""
        from alpha_pulse.models import market_regime
        from alpha_pulse.models import portfolio_allocation
        from alpha_pulse.models import risk_metrics
        assert market_regime is not None
        assert portfolio_allocation is not None
        assert risk_metrics is not None
    
    def test_import_utils(self):
        """Test importing utility modules."""
        from alpha_pulse.utils import json_utils
        from alpha_pulse.utils import date_utils
        assert json_utils is not None
        assert date_utils is not None


class TestConfigClasses:
    """Test configuration classes and their defaults."""
    
    def test_api_config_defaults(self):
        """Test API configuration defaults."""
        from alpha_pulse.config.api_config import APIConfig
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 18001
        assert config.debug is False
        assert config.reload is False
        assert config.cors_enabled is True
    
    def test_database_config(self):
        """Test database configuration."""
        from alpha_pulse.config.database import DatabaseConfig
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "alphapulse"
        assert config.pool_size == 20
        assert config.max_overflow == 40
    
    def test_monitoring_config(self):
        """Test monitoring configuration."""
        from alpha_pulse.config.monitoring import MonitoringConfig
        config = MonitoringConfig()
        assert config.prometheus_enabled is True
        assert config.prometheus_port == 8000
        assert config.metrics_interval == 60
        assert config.log_level == "INFO"


class TestModels:
    """Test data model classes."""
    
    def test_market_regime_enum(self):
        """Test MarketRegime enum."""
        from alpha_pulse.models.market_regime import MarketRegime
        
        assert MarketRegime.BULL_MARKET.value == "bull_market"
        assert MarketRegime.BEAR_MARKET.value == "bear_market"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.CRISIS.value == "crisis"
        assert MarketRegime.RECOVERY.value == "recovery"
        
        # Test all values are accessible
        all_regimes = list(MarketRegime)
        assert len(all_regimes) == 5
    
    def test_portfolio_allocation_model(self):
        """Test PortfolioAllocation model."""
        from alpha_pulse.models.portfolio_allocation import PortfolioAllocation
        
        allocation = PortfolioAllocation(
            timestamp=datetime.now(),
            allocations={"BTC": 0.5, "ETH": 0.3, "USDT": 0.2},
            total_value=100000.0,
            strategy="balanced"
        )
        
        assert allocation.total_value == 100000.0
        assert allocation.strategy == "balanced"
        assert sum(allocation.allocations.values()) == 1.0
        assert allocation.get_allocation("BTC") == 0.5
        assert allocation.get_allocation("INVALID") == 0.0
    
    def test_risk_metrics_model(self):
        """Test RiskMetrics model."""
        from alpha_pulse.models.risk_metrics import RiskMetrics
        
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=100000.0,
            var_95=5000.0,
            var_99=7500.0,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            volatility=0.20
        )
        
        assert metrics.portfolio_value == 100000.0
        assert metrics.var_95 == 5000.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.get_risk_score() > 0
        
        # Test risk level categorization
        risk_level = metrics.get_risk_level()
        assert risk_level in ["low", "medium", "high", "critical"]


class TestUtilities:
    """Test utility functions."""
    
    def test_json_utils(self):
        """Test JSON utility functions."""
        from alpha_pulse.utils.json_utils import (
            safe_json_dumps,
            safe_json_loads,
            convert_numpy_types
        )
        
        # Test safe dumps
        data = {"key": "value", "number": 42}
        json_str = safe_json_dumps(data)
        assert isinstance(json_str, str)
        assert "key" in json_str
        
        # Test safe loads
        loaded = safe_json_loads(json_str)
        assert loaded["key"] == "value"
        assert loaded["number"] == 42
        
        # Test numpy conversion
        np_data = {
            "float32": np.float32(1.5),
            "int64": np.int64(100),
            "array": np.array([1, 2, 3])
        }
        converted = convert_numpy_types(np_data)
        assert isinstance(converted["float32"], float)
        assert isinstance(converted["int64"], int)
        assert isinstance(converted["array"], list)
    
    def test_date_utils(self):
        """Test date utility functions."""
        from alpha_pulse.utils.date_utils import (
            get_market_open_close,
            is_market_open,
            get_trading_days,
            to_utc,
            format_timestamp
        )
        
        # Test market hours
        open_time, close_time = get_market_open_close("NYSE")
        assert open_time.hour == 9
        assert open_time.minute == 30
        assert close_time.hour == 16
        assert close_time.minute == 0
        
        # Test trading days
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        trading_days = get_trading_days(start, end, "NYSE")
        assert len(trading_days) > 15  # At least 15 trading days in January
        assert len(trading_days) < 25  # Less than 25 (excludes weekends)
        
        # Test timestamp formatting
        now = datetime.now()
        formatted = format_timestamp(now)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_statistical_utils(self):
        """Test statistical utility functions."""
        from alpha_pulse.utils.statistical_analysis import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_max_drawdown,
            calculate_var,
            calculate_correlation_matrix
        )
        
        # Generate sample returns
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # Daily returns
        
        # Test Sharpe ratio
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert -10 < sharpe < 10  # Reasonable range
        
        # Test Sortino ratio
        sortino = calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        
        # Test max drawdown
        prices = (1 + returns).cumprod() * 100
        max_dd = calculate_max_drawdown(prices)
        assert 0 <= max_dd <= 1  # Drawdown is between 0 and 100%
        
        # Test VaR
        var_95 = calculate_var(returns, confidence=0.95)
        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR is typically negative
        
        # Test correlation matrix
        data = pd.DataFrame({
            'asset1': returns,
            'asset2': returns * 0.8 + np.random.randn(252) * 0.005,
            'asset3': np.random.randn(252) * 0.01
        })
        corr_matrix = calculate_correlation_matrix(data)
        assert corr_matrix.shape == (3, 3)
        assert np.all(np.diag(corr_matrix) == 1.0)  # Diagonal should be 1


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_data_validator(self):
        """Test data validation."""
        from alpha_pulse.data_pipeline.validation import DataValidator
        
        validator = DataValidator()
        
        # Test valid data
        valid_data = {
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "volume": 1000000.0,
            "timestamp": datetime.now()
        }
        assert validator.validate_market_data(valid_data) is True
        
        # Test invalid data
        invalid_data = {
            "symbol": "",  # Empty symbol
            "price": -100,  # Negative price
            "volume": 0,  # Zero volume
            "timestamp": None  # Missing timestamp
        }
        assert validator.validate_market_data(invalid_data) is False
    
    def test_data_transformer(self):
        """Test data transformation."""
        from alpha_pulse.data_pipeline.transformers import DataTransformer
        
        transformer = DataTransformer()
        
        # Test OHLCV transformation
        raw_data = {
            "o": 100.0,
            "h": 105.0,
            "l": 99.0,
            "c": 103.0,
            "v": 50000.0
        }
        
        transformed = transformer.transform_ohlcv(raw_data)
        assert transformed["open"] == 100.0
        assert transformed["high"] == 105.0
        assert transformed["low"] == 99.0
        assert transformed["close"] == 103.0
        assert transformed["volume"] == 50000.0
        
        # Test aggregation
        data_points = [
            {"price": 100, "volume": 1000},
            {"price": 102, "volume": 1500},
            {"price": 101, "volume": 1200}
        ]
        
        aggregated = transformer.aggregate_ticks(data_points)
        assert aggregated["avg_price"] == 101.0
        assert aggregated["total_volume"] == 3700
        assert aggregated["vwap"] > 0


class TestExchangeAdapters:
    """Test exchange adapter base functionality."""
    
    def test_exchange_factory(self):
        """Test exchange factory pattern."""
        from alpha_pulse.exchanges.factory import ExchangeFactory
        from alpha_pulse.exchanges.base import BaseExchange
        
        factory = ExchangeFactory()
        
        # Test factory can create exchanges
        supported_exchanges = factory.get_supported_exchanges()
        assert len(supported_exchanges) > 0
        assert "binance" in supported_exchanges
        assert "bybit" in supported_exchanges
    
    def test_order_types(self):
        """Test order type definitions."""
        from alpha_pulse.exchanges.order_types import OrderType, OrderSide, OrderStatus
        
        # Test order types
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        
        # Test order sides
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        
        # Test order statuses
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"


class TestMonitoring:
    """Test monitoring components."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        from alpha_pulse.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test counter increment
        collector.increment_counter("api_requests")
        collector.increment_counter("api_requests")
        assert collector.get_counter("api_requests") == 2
        
        # Test gauge setting
        collector.set_gauge("active_connections", 10)
        assert collector.get_gauge("active_connections") == 10
        
        # Test histogram observation
        collector.observe_histogram("request_duration", 0.125)
        collector.observe_histogram("request_duration", 0.250)
        stats = collector.get_histogram_stats("request_duration")
        assert stats["count"] == 2
        assert stats["mean"] == 0.1875
    
    def test_alert_levels(self):
        """Test alert level definitions."""
        from alpha_pulse.monitoring.alerts import AlertLevel, Alert
        
        # Test alert levels
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
        
        # Test alert creation
        alert = Alert(
            level=AlertLevel.WARNING,
            message="High memory usage",
            source="system_monitor",
            timestamp=datetime.now()
        )
        
        assert alert.level == AlertLevel.WARNING
        assert "memory" in alert.message.lower()
        assert alert.is_critical() is False
        
        # Test critical alert
        critical_alert = Alert(
            level=AlertLevel.CRITICAL,
            message="System failure",
            source="health_check",
            timestamp=datetime.now()
        )
        assert critical_alert.is_critical() is True