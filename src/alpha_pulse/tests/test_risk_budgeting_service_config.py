"""
Additional tests covering risk budgeting service configuration safeguards.
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, Callable, List

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, Mock

from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.services.exceptions import ServiceConfigurationError
from alpha_pulse.services.risk_budgeting_service import (
    RiskBudgetingConfig,
    RiskBudgetingService,
)


@pytest.fixture
def risk_config() -> RiskBudgetingConfig:
    """Provide a baseline risk budgeting configuration."""
    return RiskBudgetingConfig(
        base_volatility_target=0.12,
        max_leverage=1.8,
        rebalancing_frequency="daily",
        regime_lookback_days=90,
        regime_update_frequency="daily",
        max_position_size=0.2,
        min_positions=3,
        max_sector_concentration=0.5,
        enable_alerts=True,
        auto_rebalance=False,
        track_performance=False,
        snapshot_frequency="daily",
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create a sample portfolio with basic positions."""
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=10,
            current_price=180.0,
            average_cost=150.0,
            position_type="long",
            sector="technology",
        ),
        "MSFT": Position(
            symbol="MSFT",
            quantity=5,
            current_price=330.0,
            average_cost=310.0,
            position_type="long",
            sector="technology",
        ),
    }
    return Portfolio(
        portfolio_id="test",
        name="Test Portfolio",
        total_value=positions["AAPL"].quantity * positions["AAPL"].current_price
        + positions["MSFT"].quantity * positions["MSFT"].current_price
        + 1_000.0,
        cash_balance=1_000.0,
        positions=positions,
    )


@pytest.fixture
def market_data_frame(sample_portfolio: Portfolio) -> pd.DataFrame:
    """Create deterministic market data for the sample portfolio."""
    dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
    rng = np.random.default_rng(7)
    data = {
        symbol: 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
        for symbol in list(sample_portfolio.positions.keys()) + ["SPY", "VIX"]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def portfolio_provider(sample_portfolio: Portfolio) -> AsyncMock:
    """Async provider returning the sample portfolio."""
    provider = AsyncMock(return_value=[sample_portfolio])
    return provider


@pytest.fixture
def stub_budget_manager() -> Mock:
    """Provide a lightweight budget manager implementation for tests."""
    budget = Mock()
    budget.regime_type = "bull"
    budget.target_volatility = 0.15
    budget.total_budget = 1_000_000.0
    manager = Mock()
    manager.create_regime_based_budget.return_value = budget
    manager.check_rebalancing_triggers.return_value = None
    return manager


@pytest.mark.asyncio
async def test_start_without_dependencies_raises_configuration_error(
    risk_config: RiskBudgetingConfig,
    stub_budget_manager: Mock,
):
    """Risk budgeting service should fail fast when required dependencies are missing."""
    service = RiskBudgetingService(config=risk_config, budget_manager=stub_budget_manager)

    with pytest.raises(ServiceConfigurationError):
        await service.start()


@pytest.mark.asyncio
async def test_initialize_portfolio_budgets_uses_live_data_fetcher(
    risk_config: RiskBudgetingConfig,
    sample_portfolio: Portfolio,
    market_data_frame: pd.DataFrame,
    portfolio_provider: AsyncMock,
    stub_budget_manager: Mock,
):
    """Verify that live data fetcher is invoked with portfolio symbols and market benchmarks."""
    data_fetcher = AsyncMock()
    data_fetcher.fetch_historical_data = AsyncMock(return_value=market_data_frame)
    alerting_system = AsyncMock()

    service = RiskBudgetingService(
        config=risk_config,
        data_fetcher=data_fetcher,
        alerting_system=alerting_system,
        portfolio_provider=portfolio_provider,
        budget_manager=stub_budget_manager,
    )

    regime_result = Mock()
    regime_result.current_regime = Mock(regime_type="bull")
    service.regime_detector.detect_regime = Mock(return_value=regime_result)

    budget = await service.initialize_portfolio_budgets(sample_portfolio)

    assert budget is not None
    data_fetcher.fetch_historical_data.assert_awaited()

    symbols_arg = data_fetcher.fetch_historical_data.await_args.args[0]
    assert "AAPL" in symbols_arg
    assert "MSFT" in symbols_arg
    assert "SPY" in symbols_arg
    assert "VIX" in symbols_arg
