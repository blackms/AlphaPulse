"""
Integration tests for the monitoring system with other AlphaPulse components.
"""
import asyncio
import pytest
from datetime import datetime, timezone

from alpha_pulse.monitoring.collector import EnhancedMetricsCollector
from alpha_pulse.monitoring.config import MonitoringConfig
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.data_models import PortfolioData, Position
from alpha_pulse.risk_management.manager import RiskManager
from alpha_pulse.agents.manager import AgentManager


@pytest.fixture
def monitoring_config():
    """Create a monitoring configuration for testing."""
    return MonitoringConfig(
        storage=MonitoringConfig.StorageConfig(
            type="memory",
            memory_max_points=1000
        ),
        collection_interval=1,
        enable_realtime=False
    )


@pytest.fixture
async def metrics_collector(monitoring_config):
    """Create and start a metrics collector for testing."""
    collector = EnhancedMetricsCollector(config=monitoring_config)
    await collector.start()
    yield collector
    await collector.stop()


@pytest.mark.asyncio
async def test_monitoring_with_portfolio_manager(metrics_collector):
    """Test monitoring integration with portfolio manager."""
    # Create a simple portfolio
    positions = [
        Position(
            symbol="BTC",
            quantity=1.5,
            current_price=50000.0,
            cost_basis=45000.0
        ),
        Position(
            symbol="ETH",
            quantity=10.0,
            current_price=3000.0,
            cost_basis=2800.0
        )
    ]
    
    portfolio_data = PortfolioData(
        positions=positions,
        cash=20000.0,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Create a portfolio manager
    portfolio_manager = PortfolioManager()
    portfolio_manager.update_portfolio(portfolio_data)
    
    # Collect metrics from portfolio manager
    metrics = await metrics_collector.collect_and_store(
        portfolio_data=portfolio_manager.get_portfolio_data()
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
    # Create a risk manager
    risk_manager = RiskManager()
    
    # Create a trade signal
    trade_signal = {
        "symbol": "BTC",
        "direction": "buy",
        "quantity": 0.5,
        "price": 50000.0,
        "confidence": 0.8
    }
    
    # Process the trade through risk manager
    risk_result = risk_manager.evaluate_trade(trade_signal)
    
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
    # Create an agent manager
    agent_manager = AgentManager()
    
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
    assert "technical_confidence" in metrics["agent"]
    assert metrics["agent"]["technical_confidence"] == 0.8
    assert "fundamental_confidence" in metrics["agent"]
    assert metrics["agent"]["fundamental_confidence"] == 0.7
    
    # Update with outcomes
    agent_signals["technical"]["actual_outcome"] = True
    agent_signals["fundamental"]["actual_outcome"] = False
    
    # Collect updated metrics
    metrics = await metrics_collector.collect_and_store(
        agent_data=agent_signals
    )
    
    # Verify updated metrics
    assert "technical_correct" in metrics["agent"]
    assert metrics["agent"]["technical_correct"] == True
    assert "fundamental_correct" in metrics["agent"]
    assert metrics["agent"]["fundamental_correct"] == False


@pytest.mark.asyncio
async def test_end_to_end_monitoring():
    """Test end-to-end monitoring workflow."""
    # Create configuration
    config = MonitoringConfig(
        storage=MonitoringConfig.StorageConfig(
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
        positions = [
            Position(
                symbol="BTC",
                quantity=1.5,
                current_price=50000.0,
                cost_basis=45000.0
            ),
            Position(
                symbol="ETH",
                quantity=10.0,
                current_price=3000.0,
                cost_basis=2800.0
            )
        ]
        
        portfolio_data = PortfolioData(
            positions=positions,
            cash=20000.0,
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