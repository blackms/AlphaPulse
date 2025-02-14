"""
Data models for portfolio analysis.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel


@dataclass
class PortfolioPosition:
    """Portfolio position data."""
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


class RebalancingSuggestion(BaseModel):
    """Rebalancing suggestion from LLM."""
    asset: str
    target_allocation: float


@dataclass
class LLMAnalysisResult:
    """Result of LLM portfolio analysis."""
    recommendations: List[str]
    risk_assessment: str
    confidence_score: float
    reasoning: str
    timestamp: datetime
    rebalancing_suggestions: Optional[List[RebalancingSuggestion]] = None
    raw_response: Optional[str] = None