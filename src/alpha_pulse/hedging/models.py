from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal

@dataclass
class SpotPosition:
    """Represents a spot position in the portfolio."""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Optional[Decimal] = None
    
    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate current market value if price is available."""
        if self.current_price is not None:
            return self.quantity * self.current_price
        return None
    
    @property
    def cost_basis(self) -> Decimal:
        """Calculate total cost basis of the position."""
        return self.quantity * self.avg_price

@dataclass
class GridBotParams:
    """Parameters for grid trading bot configuration."""
    num_grids: int
    price_interval: Decimal
    min_price: Decimal
    max_price: Decimal
    quantity_per_grid: Decimal

@dataclass
class FuturesPosition:
    """Represents a futures position in the portfolio."""
    symbol: str
    quantity: Decimal
    side: str  # "LONG" or "SHORT"
    entry_price: Decimal
    leverage: Decimal
    margin_used: Decimal
    current_price: Optional[Decimal] = None
    grid_bot_params: Optional[GridBotParams] = None
    
    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value if current price is available."""
        if self.current_price is not None:
            return self.quantity * self.current_price
        return None
    
    @property
    def pnl(self) -> Optional[Decimal]:
        """Calculate unrealized PnL if current price is available."""
        if self.current_price is not None:
            price_diff = (self.current_price - self.entry_price)
            if self.side == "SHORT":
                price_diff = -price_diff
            return self.quantity * price_diff
        return None

@dataclass
class HedgeAdjustment:
    """Represents a recommended adjustment to a hedge position."""
    symbol: str
    desired_delta: Decimal
    side: str
    recommendation: str
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH

@dataclass
class HedgeRecommendation:
    """Contains analysis results and recommended hedge adjustments."""
    adjustments: List[HedgeAdjustment]
    current_net_exposure: Decimal
    target_net_exposure: Decimal
    commentary: str
    grid_adjustments: Optional[Dict[str, GridBotParams]] = None
    risk_metrics: Optional[Dict[str, Decimal]] = None