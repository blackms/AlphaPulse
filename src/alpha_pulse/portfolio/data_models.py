from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class Position:
    """Represents a single position in the portfolio."""
    asset_id: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    timestamp: datetime
    metadata: Optional[Dict] = None

    @property
    def market_value(self) -> Decimal:
        """Calculate the current market value of the position."""
        return self.quantity * self.current_price

    @property
    def profit_loss(self) -> Decimal:
        """Calculate the unrealized profit/loss."""
        return self.market_value - (self.quantity * self.entry_price)

@dataclass
class PortfolioData:
    """Represents the complete portfolio state for analysis."""
    positions: List[Position]
    total_value: Decimal
    cash_balance: Decimal
    timestamp: datetime
    risk_metrics: Optional[Dict] = None
    
    @property
    def asset_allocation(self) -> Dict[str, float]:
        """Calculate the percentage allocation for each asset."""
        return {
            pos.asset_id: float(pos.market_value / self.total_value)
            for pos in self.positions
        }

@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics."""
    total_value: Decimal
    pnl_24h: Decimal
    pnl_7d: Decimal
    risk_level: str
    sharpe_ratio: Decimal
    volatility: Decimal
    max_drawdown: Decimal
    timestamp: datetime

@dataclass
class PortfolioAnalysis:
    """Comprehensive portfolio analysis results."""
    allocation: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class LLMAnalysisResult:
    """Represents the analysis results from the LLM."""
    recommendations: List[str]
    risk_assessment: str
    confidence_score: float
    reasoning: str
    timestamp: datetime
    rebalancing_suggestions: Optional[List[Dict[str, float]]] = None
    raw_response: Optional[str] = None