"""
Tests for the monitoring system.
"""
import asyncio
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from alpha_pulse.monitoring.collector import EnhancedMetricsCollector
from alpha_pulse.monitoring.config import MonitoringConfig, StorageConfig
from alpha_pulse.portfolio.data_models import PortfolioData, Position, PortfolioPosition
from alpha_pulse.monitoring.metrics_calculations import (
    calculate_performance_metrics,
    calculate_risk_metrics,
    calculate_trade_metrics
)


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio_positions = [
        PortfolioPosition(
            asset_id="BTC",
            quantity=Decimal("1.5"),
            current_price=Decimal("50000.0"),
            market_value=Decimal("75000.0"),
            profit_loss=Decimal("7500.0")
        ),
        PortfolioPosition(
            asset_id="ETH",
            quantity=Decimal("10.0"),
            current_price=Decimal("3000.0"),
            market_value=Decimal("30000.0"),
            profit_loss=Decimal("2000.0")
        )
    ]
    
    return PortfolioData(
        total_value=Decimal("125000.0"),
        cash_balance=Decimal("20000.0"),
        positions=portfolio_positions,
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_trade_data():
    """Create sample trade data for testing."""
    return {
        "symbol": "BTC",
        "side": "buy",
        "quantity": 0.5,
        "requested_quantity": 0.5,
        "price": 50000.0,
        "expected_price": 49800.0,
        "order_type": "market",
        "execution_time": 150.0,  # ms
        "commission": 25.0
    }


@pytest.fixture
def sample_agent_data():
    """Create sample agent data for testing."""
    return {
        "technical": {
            "direction": 1,
            "confidence": 0.8,
            "prediction": True,
            "actual_outcome": True
        },
        "fundamental": {
            "direction": 1,
            "confidence": 0.7,
            "prediction": True,
            "actual_outcome": True
        },
        "sentiment": {
            "direction": 0,
            "confidence": 0.3,
            "prediction": False,
            "actual_outcome": True
        }
    }


@pytest.fixture
def config():
    """Create a test configuration."""
    return MonitoringConfig(
        storage=StorageConfig(
            type="memory",
            memory_max_points=1000
        ),
        collection_interval=1,
        enable_realtime=False
    )


@pytest.mark.asyncio
async def test_collector_initialization(config):
    """Test collector initialization."""
    collector = EnhancedMetricsCollector(config=config)
    assert collector is not None
    assert collector.config == config
    assert collector.storage is not None


@pytest.mark.asyncio
async def test_collector_start_stop(config):
    """Test collector start and stop."""
    collector = EnhancedMetricsCollector(config=config)
    
    # Start collector
    await collector.start()
    assert collector._running == False  # realtime is disabled
    
    # Stop collector
    await collector.stop()
    assert collector._running == False


@pytest.mark.asyncio
async def test_collect_and_store(config, sample_portfolio, sample_trade_data, sample_agent_data):
    """Test collecting and storing metrics."""
    collector = EnhancedMetricsCollector(config=config)
    await collector.start()
    
    try:
        # Collect metrics
        metrics = await collector.collect_and_store(
            portfolio_data=sample_portfolio,
            trade_data=sample_trade_data,
            agent_data=sample_agent_data,
            system_data=True
        )
        
        # Check that metrics were collected
        assert "performance" in metrics
        assert "risk" in metrics
        assert "trade" in metrics
        assert "agent" in metrics
        assert "system" in metrics
        
        # Check performance metrics
        assert "sharpe_ratio" in metrics["performance"]
        assert "max_drawdown" in metrics["performance"]
        
        # Check risk metrics
        assert "position_count" in metrics["risk"]
        assert metrics["risk"]["position_count"] == 2
        
        # Check trade metrics
        assert "symbol" in metrics["trade"]
        assert metrics["trade"]["symbol"] == "BTC"
        
        # Check agent metrics
        # Check for overall agent metrics which are always present
        assert "avg_confidence" in metrics["agent"]
        assert "signal_agreement" in metrics["agent"]
        assert "signal_direction" in metrics["agent"]
        
        # Check system metrics
        assert "cpu_usage_percent" in metrics["system"]
        assert "memory_usage_percent" in metrics["system"]
    finally:
        await collector.stop()


@pytest.mark.asyncio
async def test_get_metrics_history(config, sample_portfolio):
    """Test retrieving metrics history."""
    collector = EnhancedMetricsCollector(config=config)
    await collector.start()
    
    try:
        # Store some metrics
        await collector.collect_and_store(portfolio_data=sample_portfolio)
        
        # Get metrics history
        metrics = await collector.get_metrics_history("performance")
        
        # Check that metrics were retrieved
        assert len(metrics) > 0
        assert "sharpe_ratio" in metrics[0]
    finally:
        await collector.stop()


@pytest.mark.asyncio
async def test_get_latest_metrics(config, sample_portfolio):
    """Test retrieving latest metrics."""
    collector = EnhancedMetricsCollector(config=config)
    await collector.start()
    
    try:
        # Store some metrics
        await collector.collect_and_store(portfolio_data=sample_portfolio)
        
        # Get latest metrics
        metrics = await collector.get_latest_metrics("performance")
        
        # Check that metrics were retrieved
        assert len(metrics) == 1
        assert "sharpe_ratio" in metrics[0]
    finally:
        await collector.stop()


def test_calculate_performance_metrics(sample_portfolio):
    """Test performance metrics calculation."""
    metrics = calculate_performance_metrics(sample_portfolio)
    
    # Check that metrics were calculated
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "total_return" in metrics


def test_calculate_risk_metrics(sample_portfolio):
    """Test risk metrics calculation."""
    metrics = calculate_risk_metrics(sample_portfolio)
    
    # Check that metrics were calculated
    assert "position_count" in metrics
    assert metrics["position_count"] == 2
    assert "concentration_hhi" in metrics
    assert "leverage" in metrics
    assert "portfolio_value" in metrics


def test_calculate_trade_metrics(sample_trade_data):
    """Test trade metrics calculation."""
    metrics = calculate_trade_metrics(sample_trade_data)
    
    # Check that metrics were calculated
    assert "symbol" in metrics
    assert metrics["symbol"] == "BTC"
    assert "slippage" in metrics
    assert "fill_rate" in metrics
    assert metrics["fill_rate"] == 1.0  # Fully filled
    assert "commission_pct" in metrics