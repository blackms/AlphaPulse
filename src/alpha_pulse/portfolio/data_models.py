"""
Data models for portfolio analysis.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel


@dataclass
class Position:
    """Portfolio position data."""
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: Optional[float] = None


@dataclass
class PortfolioPosition:
    """Portfolio position data for analysis."""
    asset_id: str
    quantity: Decimal
    current_price: Decimal
    market_value: Decimal
    profit_loss: Decimal


@dataclass
class PortfolioData:
    """Portfolio data for analysis."""
    total_value: Decimal
    cash_balance: Decimal
    positions: List[PortfolioPosition]
    risk_metrics: Optional[Dict[str, str]] = None


@dataclass
class LLMAnalysisResult:
    """Result of LLM portfolio analysis."""
    recommendations: List[str]
    risk_assessment: str
    confidence_score: float
    reasoning: str
    timestamp: datetime
    rebalancing_suggestions: Optional[List[dict]] = None
    raw_response: Optional[str] = None


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: Decimal
    total_pnl: Decimal
    daily_return: Decimal
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    var_99: Optional[float] = None


@dataclass
class PortfolioAllocation:
    """Portfolio allocation data."""
    asset_id: str
    weight: Decimal
    target_weight: Optional[Decimal] = None
    rebalance_amount: Optional[Decimal] = None


@dataclass
class PortfolioRebalanceResult:
    """Result of portfolio rebalancing."""
    success: bool
    trades: List[dict]
    new_allocations: List[PortfolioAllocation]
    rebalance_cost: Decimal
    error: Optional[str] = None