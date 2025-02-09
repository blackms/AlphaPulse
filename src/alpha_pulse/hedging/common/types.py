"""
Common type definitions for hedging module.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class GridState(Enum):
    """Grid strategy states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class SpotPosition:
    """Immutable spot position state."""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None

    @property
    def pnl(self) -> Optional[Decimal]:
        """Calculate unrealized PnL."""
        if self.current_price is None:
            return None
        return (self.current_price - self.avg_price) * self.quantity


@dataclass(frozen=True)
class FuturesPosition:
    """Immutable futures position state."""
    symbol: str
    quantity: Decimal
    side: str
    entry_price: Decimal
    leverage: Decimal
    margin_used: Decimal
    current_price: Optional[Decimal] = None
    pnl: Optional[Decimal] = None

    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value."""
        if self.current_price is None:
            return None
        return self.quantity * self.current_price


@dataclass(frozen=True)
class PositionState:
    """Immutable position state."""
    spot_quantity: Decimal
    futures_quantity: Decimal
    avg_entry_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    funding_paid: Decimal
    last_update: datetime

    @classmethod
    def create_empty(cls) -> 'PositionState':
        """Create empty position state."""
        return cls(
            spot_quantity=Decimal('0'),
            futures_quantity=Decimal('0'),
            avg_entry_price=Decimal('0'),
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0'),
            funding_paid=Decimal('0'),
            last_update=datetime.now()
        )

    def update(self, **kwargs) -> 'PositionState':
        """Create new instance with updated values."""
        return PositionState(
            spot_quantity=kwargs.get('spot_quantity', self.spot_quantity),
            futures_quantity=kwargs.get('futures_quantity', self.futures_quantity),
            avg_entry_price=kwargs.get('avg_entry_price', self.avg_entry_price),
            unrealized_pnl=kwargs.get('unrealized_pnl', self.unrealized_pnl),
            realized_pnl=kwargs.get('realized_pnl', self.realized_pnl),
            funding_paid=kwargs.get('funding_paid', self.funding_paid),
            last_update=datetime.now()
        )


@dataclass(frozen=True)
class GridMetrics:
    """Immutable grid performance metrics."""
    total_trades: int
    successful_trades: int
    failed_trades: int
    avg_profit_per_trade: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float

    @classmethod
    def create_empty(cls) -> 'GridMetrics':
        """Create empty metrics."""
        return cls(
            total_trades=0,
            successful_trades=0,
            failed_trades=0,
            avg_profit_per_trade=Decimal('0'),
            max_drawdown=Decimal('0'),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0
        )

    def update(self, **kwargs) -> 'GridMetrics':
        """Create new instance with updated values."""
        return GridMetrics(
            total_trades=kwargs.get('total_trades', self.total_trades),
            successful_trades=kwargs.get('successful_trades', self.successful_trades),
            failed_trades=kwargs.get('failed_trades', self.failed_trades),
            avg_profit_per_trade=kwargs.get('avg_profit_per_trade', self.avg_profit_per_trade),
            max_drawdown=kwargs.get('max_drawdown', self.max_drawdown),
            sharpe_ratio=kwargs.get('sharpe_ratio', self.sharpe_ratio),
            sortino_ratio=kwargs.get('sortino_ratio', self.sortino_ratio),
            profit_factor=kwargs.get('profit_factor', self.profit_factor)
        )


@dataclass(frozen=True)
class GridLevel:
    """Immutable grid level."""
    price: Decimal
    quantity: Decimal
    is_long: bool
    order_id: Optional[str] = None

    def update(self, **kwargs) -> 'GridLevel':
        """Create new instance with updated values."""
        return GridLevel(
            price=kwargs.get('price', self.price),
            quantity=kwargs.get('quantity', self.quantity),
            is_long=kwargs.get('is_long', self.is_long),
            order_id=kwargs.get('order_id', self.order_id)
        )


@dataclass(frozen=True)
class MarketState:
    """Immutable market state."""
    current_price: Decimal
    funding_rate: Optional[Decimal]
    volatility: Decimal
    volume: Decimal
    timestamp: datetime

    @classmethod
    def from_raw(cls, price: float, **kwargs) -> 'MarketState':
        """Create from raw values."""
        return cls(
            current_price=Decimal(str(price)),
            funding_rate=Decimal(str(kwargs['funding_rate'])) if 'funding_rate' in kwargs else None,
            volatility=Decimal(str(kwargs.get('volatility', '0'))),
            volume=Decimal(str(kwargs.get('volume', '0'))),
            timestamp=datetime.now()
        )