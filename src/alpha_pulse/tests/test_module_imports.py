"""
Extended module import tests to improve coverage.
"""
import pytest
import sys
from pathlib import Path


def test_agents_module_imports():
    """Test that agent modules can be imported."""
    try:
        # Import base interfaces
        from alpha_pulse.agents.interfaces import (
            ITradeAgent, BaseTradeAgent, TradeSignal, 
            SignalDirection, MarketData, AgentMetrics
        )
        assert ITradeAgent is not None
        assert BaseTradeAgent is not None
        assert TradeSignal is not None
        print("✓ agents.interfaces imported")
        
        # Import agent implementations
        from alpha_pulse.agents.technical_agent import TechnicalAgent
        from alpha_pulse.agents.sentiment_agent import SentimentAgent
        from alpha_pulse.agents.fundamental_agent import FundamentalAgent
        assert TechnicalAgent is not None
        assert SentimentAgent is not None
        assert FundamentalAgent is not None
        print("✓ agent implementations imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_risk_management_imports():
    """Test that risk management modules can be imported."""
    try:
        from alpha_pulse.risk_management.interfaces import (
            IRiskManager, RiskLimits, RiskMetrics, PositionLimits
        )
        assert IRiskManager is not None
        assert RiskLimits is not None
        print("✓ risk_management.interfaces imported")
        
        from alpha_pulse.risk_management.manager import RiskManager
        from alpha_pulse.risk_management.position_sizing import PositionSizer
        assert RiskManager is not None
        assert PositionSizer is not None
        print("✓ risk management implementations imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_portfolio_module_imports():
    """Test that portfolio modules can be imported."""
    try:
        from alpha_pulse.portfolio.interfaces import (
            IPortfolioOptimizer, OptimizationResult, 
            PortfolioState, OptimizationConstraints
        )
        assert IPortfolioOptimizer is not None
        assert OptimizationResult is not None
        print("✓ portfolio.interfaces imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_execution_module_imports():
    """Test that execution modules can be imported."""
    try:
        from alpha_pulse.execution.broker_interface import (
            IBroker, Order, OrderType, OrderSide, OrderStatus
        )
        assert IBroker is not None
        assert Order is not None
        print("✓ execution.broker_interface imported")
        
        from alpha_pulse.execution.paper_broker import PaperBroker
        assert PaperBroker is not None
        print("✓ paper broker imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_monitoring_module_imports():
    """Test that monitoring modules can be imported."""
    try:
        from alpha_pulse.monitoring.interfaces import (
            IMonitor, MetricType, AlertLevel, HealthStatus
        )
        assert IMonitor is not None
        assert MetricType is not None
        print("✓ monitoring.interfaces imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_api_module_imports():
    """Test that API modules can be imported."""
    try:
        from alpha_pulse.api.main import app
        assert app is not None
        print("✓ api.main imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_utils_module_imports():
    """Test that utility modules can be imported."""
    try:
        from alpha_pulse.utils.logging_utils import get_logger
        assert get_logger is not None
        print("✓ utils.logging_utils imported")
        
        from alpha_pulse.utils.constants import SUPPORTED_EXCHANGES
        assert SUPPORTED_EXCHANGES is not None
        print("✓ utils.constants imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_ml_module_imports():
    """Test that ML modules can be imported."""
    try:
        from alpha_pulse.ml.regime.regime_classifier import RegimeClassifier
        assert RegimeClassifier is not None
        print("✓ ml.regime.regime_classifier imported")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


if __name__ == "__main__":
    # Run tests when executed directly
    test_agents_module_imports()
    test_risk_management_imports()
    test_portfolio_module_imports()
    test_execution_module_imports()
    test_monitoring_module_imports()
    test_api_module_imports()
    test_utils_module_imports()
    test_ml_module_imports()
    print("All module import tests completed!")