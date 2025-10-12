"""
Live portfolio provider bridging database accessors with service models.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable, Dict, List, Optional, Sequence

from alpha_pulse.api.data import PortfolioDataAccessor
from alpha_pulse.models.portfolio import Portfolio, Position
from alpha_pulse.portfolio.data_models import PortfolioData, PortfolioPosition


class LivePortfolioProvider:
    """Utility to expose live portfolio data for services."""

    def __init__(self, accessor: PortfolioDataAccessor):
        self._accessor = accessor

    async def get_active_portfolios(self) -> Sequence[Portfolio]:
        """Return the primary portfolio as a Portfolio model."""
        raw = await self._accessor.get_portfolio(include_history=False)
        positions = raw.get("positions") if raw else None
        if not raw or not positions:
            return []

        portfolio_positions: Dict[str, Position] = {}
        for record in positions:
            symbol = record.get("symbol")
            current_price = record.get("current_price")
            quantity = record.get("quantity")
            average_cost = record.get("entry_price")
            if not symbol or current_price is None or quantity is None or average_cost is None:
                continue

            portfolio_positions[symbol] = Position(
                symbol=symbol,
                quantity=float(quantity),
                current_price=float(current_price),
                average_cost=float(average_cost),
                position_type=record.get("position_type", "long"),
                sector=record.get("sector"),
                strategy=record.get("strategy"),
                metadata={
                    "pnl": record.get("pnl"),
                    "pnl_percentage": record.get("pnl_percentage"),
                },
            )

        if not portfolio_positions:
            return []

        portfolio = Portfolio(
            portfolio_id=str(raw.get("portfolio_id", "primary")),
            name=str(raw.get("name", "Primary Portfolio")),
            total_value=float(raw.get("total_value", 0.0)),
            cash_balance=float(raw.get("cash", raw.get("cash_balance", 0.0))),
            positions=portfolio_positions,
            metadata={"metrics": raw.get("metrics", {})},
        )
        return [portfolio]

    async def get_portfolio_snapshot(self) -> Optional[PortfolioData]:
        """Return the primary portfolio as PortfolioData for analytics services."""
        raw = await self._accessor.get_portfolio(include_history=False)
        positions = raw.get("positions") if raw else None
        if not raw or not positions:
            return None

        portfolio_positions: List[PortfolioPosition] = []
        for record in positions:
            symbol = record.get("symbol")
            quantity = record.get("quantity")
            current_price = record.get("current_price")
            value = record.get("value")
            pnl = record.get("pnl")
            if not symbol or quantity is None or current_price is None:
                continue

            portfolio_positions.append(
                PortfolioPosition(
                    asset_id=symbol,
                    quantity=Decimal(str(quantity)),
                    current_price=Decimal(str(current_price)),
                    market_value=Decimal(str(value or quantity * current_price)),
                    profit_loss=Decimal(str(pnl or 0)),
                )
            )

        if not portfolio_positions:
            return None

        return PortfolioData(
            total_value=Decimal(str(raw.get("total_value", 0.0))),
            cash_balance=Decimal(str(raw.get("cash", raw.get("cash_balance", 0.0)))),
            positions=portfolio_positions,
            timestamp=None,
            risk_metrics=raw.get("metrics"),
        )
