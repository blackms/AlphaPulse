"""
Tests for tail risk hedging service dependency wiring.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from alpha_pulse.portfolio.data_models import PortfolioData, PortfolioPosition
from alpha_pulse.services.exceptions import ServiceConfigurationError
from alpha_pulse.services.tail_risk_hedging_service import TailRiskHedgingService


@pytest.fixture
def tail_risk_config() -> Dict[str, float]:
    """Default configuration for the tail risk hedging service."""
    return {
        "enabled": True,
        "threshold": 0.05,
        "check_interval_minutes": 1,
        "max_hedge_cost": 0.01,
    }


@pytest.fixture
def sample_portfolio_data() -> PortfolioData:
    """Create deterministic portfolio data for testing."""
    positions = [
        PortfolioPosition(
            asset_id="AAPL",
            quantity=Decimal("10"),
            current_price=Decimal("180"),
            market_value=Decimal("1800"),
            profit_loss=Decimal("200"),
        ),
        PortfolioPosition(
            asset_id="MSFT",
            quantity=Decimal("5"),
            current_price=Decimal("330"),
            market_value=Decimal("1650"),
            profit_loss=Decimal("150"),
        ),
    ]
    return PortfolioData(
        total_value=Decimal("4450"),
        cash_balance=Decimal("500"),
        positions=positions,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def hedge_manager() -> Mock:
    """Provide a hedge manager with mocked analysis and recommendation logic."""
    manager = Mock()
    manager.risk_analyzer = Mock()
    manager.risk_analyzer.analyze_tail_risk.return_value = {
        "tail_risk_score": 0.08,
        "confidence": 0.75,
        "risk_factors": {"VIX": 0.4},
    }
    manager.recommend_hedges.return_value = {
        "hedges": [{"instrument": "SPY Put", "notional": 10000}]
    }
    return manager


@pytest.fixture
def alert_manager() -> AsyncMock:
    """Provide an async alert manager."""
    manager = AsyncMock()
    manager.send_alert = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_analyze_portfolio_uses_provider_when_data_missing(
    hedge_manager: Mock,
    alert_manager: AsyncMock,
    tail_risk_config: Dict[str, float],
    sample_portfolio_data: PortfolioData,
):
    """Service should pull portfolio data from provider when direct input is absent."""
    portfolio_provider = AsyncMock(return_value=sample_portfolio_data)
    service = TailRiskHedgingService(
        hedge_manager=hedge_manager,
        alert_manager=alert_manager,
        config=tail_risk_config,
        portfolio_provider=portfolio_provider,
    )

    analysis = await service.analyze_portfolio_tail_risk()

    portfolio_provider.assert_awaited_once()
    assert analysis is not None
    assert analysis.tail_risk_score == pytest.approx(0.08)


@pytest.mark.asyncio
async def test_missing_portfolio_inputs_raise_configuration_error(
    hedge_manager: Mock,
    alert_manager: AsyncMock,
    tail_risk_config: Dict[str, float],
):
    """Fail fast when neither direct portfolio data nor provider is available."""
    service = TailRiskHedgingService(
        hedge_manager=hedge_manager,
        alert_manager=alert_manager,
        config=tail_risk_config,
    )

    with pytest.raises(ServiceConfigurationError):
        await service.analyze_portfolio_tail_risk()
