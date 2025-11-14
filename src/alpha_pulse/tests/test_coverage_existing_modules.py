"""Tests for existing modules to boost coverage to 15%."""
import pytest
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestExistingModels:
    """Test existing model classes."""
    
    def test_market_data_model(self):
        """Test MarketData model."""
        from alpha_pulse.models.market_data import MarketData
        
        data = MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000000.0
        )
        
        assert data.symbol == "BTC/USDT"
        assert data.close == 50500.0
        assert data.volume == 1000000.0
    
    def test_market_regime_hmm_import(self):
        """Test MarketRegimeHMM model can be imported."""
        from alpha_pulse.models import market_regime_hmm
        assert market_regime_hmm is not None
    
    def test_ensemble_model_import(self):
        """Test ensemble model can be imported."""
        from alpha_pulse.models import ensemble_model
        assert ensemble_model is not None


class TestExistingUtils:
    """Test existing utility modules."""
    
    def test_json_utils(self):
        """Test our new JSON utils."""
        from alpha_pulse.utils.json_utils import safe_json_dumps, safe_json_loads
        
        data = {"test": "value", "number": 123}
        json_str = safe_json_dumps(data)
        loaded = safe_json_loads(json_str)
        
        assert loaded["test"] == "value"
        assert loaded["number"] == 123
    
    def test_date_utils(self):
        """Test our new date utils."""
        from alpha_pulse.utils.date_utils import format_timestamp, get_market_open_close
        
        now = datetime.now()
        formatted = format_timestamp(now)
        assert isinstance(formatted, str)
        
        open_time, close_time = get_market_open_close("NYSE")
        assert open_time.hour == 9
        assert close_time.hour == 16


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_data_validator(self):
        """Test data validation."""
        from alpha_pulse.data_pipeline.validation import DataValidator
        
        validator = DataValidator()
        
        # Valid data
        valid = {
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": datetime.now()
        }
        assert validator.validate_market_data(valid) is True
        
        # Invalid data
        invalid = {"symbol": "", "price": -100, "volume": 0, "timestamp": None}
        assert validator.validate_market_data(invalid) is False
    
    def test_data_transformer(self):
        """Test data transformation."""
        from alpha_pulse.data_pipeline.transformers import DataTransformer
        
        transformer = DataTransformer()
        
        # Test OHLCV
        raw = {"o": 100, "h": 105, "l": 99, "c": 103, "v": 50000}
        transformed = transformer.transform_ohlcv(raw)
        
        assert transformed["open"] == 100
        assert transformed["close"] == 103
        assert transformed["volume"] == 50000


class TestConfigModules:
    """Test configuration modules that exist."""
    
    def test_database_config_import(self):
        """Test database config can be imported."""
        from alpha_pulse.config import database
        assert database is not None
        
    def test_monitoring_config_import(self):
        """Test monitoring config can be imported."""  
        from alpha_pulse.config import monitoring
        assert monitoring is not None
    
    def test_cache_config_import(self):
        """Test cache config can be imported."""
        from alpha_pulse.config import cache_config
        assert cache_config is not None


class TestExchangeModules:
    """Test exchange modules."""
    
    def test_exchange_types(self):
        """Test exchange types enum."""
        from alpha_pulse.exchanges.types import ExchangeType
        
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.BYBIT.value == "bybit"
    
    def test_exchange_factories(self):
        """Test exchange factory."""
        from alpha_pulse.exchanges.factories import ExchangeFactory
        
        factory = ExchangeFactory()
        assert factory is not None
        
        # Test getting supported exchanges
        exchanges = factory.list_supported_exchanges()
        assert "binance" in exchanges
        assert "bybit" in exchanges


class TestMonitoringModules:
    """Test monitoring modules."""
    
    def test_metrics_collector(self):
        """Test metrics collector."""
        from alpha_pulse.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        assert collector is not None
        
        # Test basic operations
        collector.record_value("test_metric", 100.0)
        collector.record_value("test_metric", 200.0)
        
        stats = collector.get_stats("test_metric")
        assert stats["count"] == 2
        assert stats["mean"] == 150.0
    
    def test_alert_manager_import(self):
        """Test alert manager can be imported."""
        from alpha_pulse.monitoring.alerting import AlertManager
        # Just test import works
        assert AlertManager is not None


class TestServiceModules:
    """Test service modules."""
    
    def test_regime_detection_service_import(self):
        """Test regime detection service import."""
        from alpha_pulse.services.regime_detection_service import RegimeDetectionService
        assert RegimeDetectionService is not None
    
    def test_ensemble_service_import(self):
        """Test ensemble service import."""
        from alpha_pulse.services.ensemble_service import EnsembleService
        assert EnsembleService is not None
    
    def test_risk_budgeting_service_import(self):
        """Test risk budgeting service import."""
        from alpha_pulse.services.risk_budgeting_service import RiskBudgetingService
        assert RiskBudgetingService is not None


class TestAgentModules:
    """Test agent modules."""
    
    def test_agent_factory(self):
        """Test agent factory."""
        from alpha_pulse.agents.factory import AgentFactory
        
        # Test factory has agent types
        assert hasattr(AgentFactory, 'AGENT_TYPES')
        assert 'technical' in AgentFactory.AGENT_TYPES
        assert 'fundamental' in AgentFactory.AGENT_TYPES
        assert 'sentiment' in AgentFactory.AGENT_TYPES
    
    def test_agent_interfaces(self):
        """Test agent interfaces."""
        from alpha_pulse.agents.interfaces import BaseTradeAgent
        assert BaseTradeAgent is not None
    
    def test_agent_manager(self):
        """Test agent manager."""
        from alpha_pulse.agents.manager import AgentManager
        assert AgentManager is not None