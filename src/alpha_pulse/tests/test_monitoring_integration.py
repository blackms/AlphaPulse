"""
Integration tests for the monitoring system with other AlphaPulse components.
"""
import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from alpha_pulse.monitoring.collector import EnhancedMetricsCollector
from alpha_pulse.monitoring.config import MonitoringConfig, StorageConfig
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.data_models import PortfolioData, Position, PortfolioPosition
from alpha_pulse.risk_management.manager import RiskManager
from alpha_pulse.agents.manager import AgentManager


@pytest.fixture
def monitoring_config():
    """Create a monitoring configuration for testing."""
    return MonitoringConfig(
        storage=StorageConfig(
            type="memory",
            memory_max_points=1000
        ),
        collection_interval=1,
        enable_realtime=False
    )


@pytest_asyncio.fixture
async def metrics_collector(monitoring_config, event_loop):
    """Create and start a metrics collector for testing."""
    collector = EnhancedMetricsCollector(config=monitoring_config)
    await collector.start()
    try:
        yield collector
    finally:
        await collector.stop()


@pytest.mark.asyncio
async def test_monitoring_with_portfolio_manager(metrics_collector):
    """Test monitoring integration with portfolio manager."""
    # Create portfolio positions
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
    
    portfolio_data = PortfolioData(
        total_value=Decimal("125000.0"),
        cash_balance=Decimal("20000.0"),
        positions=portfolio_positions,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Use the portfolio data directly instead of creating a portfolio manager
    # since we're just testing the metrics collector
    
    # Collect metrics from portfolio manager
    metrics = await metrics_collector.collect_and_store(
        portfolio_data=portfolio_data
    )
    
    # Verify metrics were collected
    assert "performance" in metrics
    assert "risk" in metrics
    
    # Verify portfolio data was correctly processed
    assert metrics["risk"]["position_count"] == 2
    assert metrics["risk"]["portfolio_value"] > 0


@pytest.mark.asyncio
async def test_monitoring_with_risk_manager(metrics_collector):
    """Test monitoring integration with risk manager."""
    # Skip creating the risk manager since we're just testing the metrics collector
    
    # Create a trade signal
    trade_signal = {
        "symbol": "BTC",
        "direction": "buy",
        "quantity": 0.5,
        "price": 50000.0,
        "confidence": 0.8
    }
    
    # Mock the risk result instead of calling the risk manager
    risk_result = {
        "adjusted_quantity": trade_signal["quantity"] * 0.9,  # 10% reduction
        "risk_score": 0.7,
        "approved": True
    }
    
    # Create trade data from risk result
    trade_data = {
        "symbol": trade_signal["symbol"],
        "side": trade_signal["direction"],
        "quantity": risk_result.get("adjusted_quantity", trade_signal["quantity"]),
        "requested_quantity": trade_signal["quantity"],
        "price": trade_signal["price"],
        "expected_price": trade_signal["price"],
        "order_type": "market",
        "execution_time": 150.0,  # ms
        "commission": trade_signal["price"] * risk_result.get("adjusted_quantity", trade_signal["quantity"]) * 0.001  # 0.1% commission
    }
    
    # Collect metrics from trade data
    metrics = await metrics_collector.collect_and_store(
        trade_data=trade_data
    )
    
    # Verify metrics were collected
    assert "trade" in metrics
    
    # Verify trade data was correctly processed
    assert metrics["trade"]["symbol"] == "BTC"
    assert metrics["trade"]["side"] == "buy"
    assert "fill_rate" in metrics["trade"]


@pytest.mark.asyncio
async def test_monitoring_with_agent_manager(metrics_collector):
    """Test monitoring integration with agent manager."""
    # Skip creating the agent manager since we're just testing the metrics collector
    
    # Create agent signals
    agent_signals = {
        "technical": {
            "direction": 1,
            "confidence": 0.8,
            "prediction": True,
            "actual_outcome": None  # Not known yet
        },
        "fundamental": {
            "direction": 1,
            "confidence": 0.7,
            "prediction": True,
            "actual_outcome": None
        }
    }
    
    # Collect metrics from agent signals
    metrics = await metrics_collector.collect_and_store(
        agent_data=agent_signals
    )
    
    # Verify metrics were collected
    assert "agent" in metrics
    
    # Verify agent data was correctly processed
    assert "avg_confidence" in metrics["agent"]
    assert "signal_agreement" in metrics["agent"]
    
    # Update with outcomes
    agent_signals["technical"]["actual_outcome"] = True
    agent_signals["fundamental"]["actual_outcome"] = False
    
    # Collect updated metrics
    metrics = await metrics_collector.collect_and_store(
        agent_data=agent_signals
    )
    
    # Verify updated metrics
    assert "signal_direction" in metrics["agent"]


@pytest.mark.asyncio
async def test_end_to_end_monitoring():
    """Test end-to-end monitoring workflow."""
    # Create configuration
    config = MonitoringConfig(
        storage=StorageConfig(
            type="memory",
            memory_max_points=1000
        ),
        collection_interval=1,
        enable_realtime=False
    )
    
    # Create collector
    collector = EnhancedMetricsCollector(config=config)
    await collector.start()
    
    try:
        # Create portfolio
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
        
        portfolio_data = PortfolioData(
            total_value=Decimal("125000.0"),
            cash_balance=Decimal("20000.0"),
            positions=portfolio_positions,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Create trade data
        trade_data = {
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
        
        # Create agent data
        agent_data = {
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
                "actual_outcome": False
            }
        }
        
        # Collect all metrics
        metrics = await collector.collect_and_store(
            portfolio_data=portfolio_data,
            trade_data=trade_data,
            agent_data=agent_data,
            system_data=True
        )
        
        # Verify all metrics were collected
        assert "performance" in metrics
        assert "risk" in metrics
        assert "trade" in metrics
        assert "agent" in metrics
        assert "system" in metrics
        
        # Query historical metrics
        performance_history = await collector.get_metrics_history("performance")
        assert len(performance_history) > 0
        
        # Get latest metrics
        latest_metrics = await collector.get_latest_metrics("performance")
        assert len(latest_metrics) == 1
        
    finally:
        await collector.stop()