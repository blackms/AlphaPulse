"""
Multi-tenant tests for AgentManager.

Tests tenant_id parameter validation and data isolation.
Following TDD approach: RED -> GREEN -> REFACTOR
"""
import sys
from types import ModuleType
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock

# Mock all optional visualization/explainability dependencies before importing AgentManager
plotly_modules = ["plotly", "plotly.graph_objects", "plotly.express", "plotly.subplots"]
for missing_module in plotly_modules:
    if missing_module not in sys.modules:
        mock_mod = ModuleType(missing_module)
        mock_mod.__dict__.update({
            'Figure': MagicMock(),
            'Scatter': MagicMock(),
            'Layout': MagicMock(),
            'make_subplots': MagicMock(),
        })
        sys.modules[missing_module] = mock_mod

from alpha_pulse.agents.manager import AgentManager
from alpha_pulse.agents.interfaces import MarketData, TradeSignal, SignalDirection
import pandas as pd
from decimal import Decimal


class TestAgentManagerMultiTenant:
    """Test suite for AgentManager multi-tenant functionality."""

    # RED PHASE: Write failing tests first
    # These tests will fail until AgentManager is refactored

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing."""
        return MarketData(
            prices=pd.DataFrame({
                'BTC/USDT': [50000.0, 50100.0, 50200.0],
                'ETH/USDT': [2000.0, 2010.0, 2020.0]
            }),
            volumes=pd.DataFrame({
                'BTC/USDT': [1000.0, 1100.0, 1200.0],
                'ETH/USDT': [5000.0, 5100.0, 5200.0]
            }),
            metadata={}
        )

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "agent_weights": {
                "technical": 0.5,
                "fundamental": 0.5
            },
            "use_ensemble": False,
            "use_gpu_acceleration": False,
            "enable_explanations": False,
            "enable_quality_validation": False
        }

    @pytest.mark.asyncio
    async def test_generate_signals_requires_tenant_id(
        self, mock_market_data, mock_config, default_tenant_id
    ):
        """Test that generate_signals raises ValueError when tenant_id is missing."""
        # Arrange
        manager = AgentManager(config=mock_config)
        await manager.initialize()

        # Act & Assert
        with pytest.raises(ValueError, match="requires 'tenant_id' parameter"):
            # Call without tenant_id - should raise ValueError
            await manager.generate_signals(mock_market_data)

    @pytest.mark.asyncio
    async def test_generate_signals_with_tenant_id(
        self, mock_market_data, mock_config, default_tenant_id
    ):
        """Test that generate_signals succeeds when tenant_id is provided."""
        # Arrange
        manager = AgentManager(config=mock_config)
        await manager.initialize()

        # Mock agent to return a signal
        mock_agent = AsyncMock()
        mock_signal = TradeSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=Decimal("0.85"),
            quantity=Decimal("1.0"),
            metadata={}
        )
        mock_agent.generate_signals.return_value = [mock_signal]
        manager.agents["technical"] = mock_agent

        # Act
        signals = await manager.generate_signals(
            mock_market_data,
            tenant_id=default_tenant_id
        )

        # Assert
        assert signals is not None
        assert len(signals) > 0
        # Verify tenant_id was passed to agent
        mock_agent.generate_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_signals_isolated_by_tenant(
        self, mock_market_data, mock_config, tenant_1_id, tenant_2_id
    ):
        """Test that signals are properly isolated between tenants."""
        # Arrange
        manager = AgentManager(config=mock_config)
        await manager.initialize()

        # Mock agent to return different signals for different tenants
        mock_agent = AsyncMock()

        def generate_tenant_signals(market_data, tenant_id=None):
            """Return different signals based on tenant_id."""
            if tenant_id == tenant_1_id:
                return [TradeSignal(
                    symbol="BTC/USDT",
                    direction=SignalDirection.LONG,
                    confidence=Decimal("0.85"),
                    quantity=Decimal("1.0"),
                    metadata={"tenant_id": tenant_id}
                )]
            else:
                return [TradeSignal(
                    symbol="ETH/USDT",
                    direction=SignalDirection.SHORT,
                    confidence=Decimal("0.75"),
                    quantity=Decimal("2.0"),
                    metadata={"tenant_id": tenant_id}
                )]

        mock_agent.generate_signals.side_effect = generate_tenant_signals
        manager.agents["technical"] = mock_agent

        # Act
        signals_tenant1 = await manager.generate_signals(
            mock_market_data,
            tenant_id=tenant_1_id
        )
        signals_tenant2 = await manager.generate_signals(
            mock_market_data,
            tenant_id=tenant_2_id
        )

        # Assert
        assert signals_tenant1[0].symbol == "BTC/USDT"
        assert signals_tenant2[0].symbol == "ETH/USDT"
        # Verify signals are isolated
        assert signals_tenant1[0].metadata["tenant_id"] == tenant_1_id
        assert signals_tenant2[0].metadata["tenant_id"] == tenant_2_id

    @pytest.mark.asyncio
    async def test_register_agent_with_tenant_id(
        self, mock_config, default_tenant_id
    ):
        """Test that agent registration includes tenant context."""
        # Arrange
        manager = AgentManager(
            config=mock_config,
            ensemble_service=MagicMock()
        )
        manager.ensemble_service.register_agent = AsyncMock(
            return_value="agent-uuid-123"
        )
        manager.use_ensemble = True

        # Act
        await manager.register_agent(
            agent_type="technical",
            tenant_id=default_tenant_id
        )

        # Assert
        manager.ensemble_service.register_agent.assert_called_once()
        call_kwargs = manager.ensemble_service.register_agent.call_args[1]
        assert call_kwargs.get("tenant_id") == default_tenant_id

    @pytest.mark.asyncio
    async def test_ensemble_aggregation_with_tenant_id(
        self, mock_market_data, mock_config, default_tenant_id
    ):
        """Test that ensemble aggregation includes tenant_id in logs."""
        # Arrange
        config_with_ensemble = {**mock_config, "use_ensemble": True}
        mock_ensemble_service = MagicMock()
        mock_ensemble_service.predict = AsyncMock(return_value=[
            {"symbol": "BTC/USDT", "direction": "long", "confidence": 0.9}
        ])

        manager = AgentManager(
            config=config_with_ensemble,
            ensemble_service=mock_ensemble_service
        )
        manager.ensemble_id = "ensemble-123"
        await manager.initialize()

        # Mock agent
        mock_agent = AsyncMock()
        mock_signal = TradeSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=Decimal("0.85"),
            quantity=Decimal("1.0"),
            metadata={}
        )
        mock_agent.generate_signals.return_value = [mock_signal]
        manager.agents["technical"] = mock_agent

        # Act
        with patch('loguru.logger.debug') as mock_logger:
            await manager.generate_signals(
                mock_market_data,
                tenant_id=default_tenant_id
            )

            # Assert - verify tenant_id appears in debug logs
            # Check if any log call contains the tenant_id
            log_calls = [str(call) for call in mock_logger.call_args_list]
            assert any(default_tenant_id in log for log in log_calls)
