"""
Slippage estimation data models.

Defines data structures for slippage prediction, market impact modeling,
and execution cost analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class SlippageModel(Enum):
    """Types of slippage models."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    ALMGREN_CHRISS = "almgren_chriss"
    KISSELL_GLANTZ = "kissell_glantz"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class ImpactType(Enum):
    """Types of market impact."""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    TOTAL = "total"
    SPREAD = "spread"
    TIMING_RISK = "timing_risk"


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    IS = "implementation_shortfall"
    POV = "percentage_of_volume"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


@dataclass
class SlippageParameters:
    """Parameters for slippage models."""
    # Market impact parameters
    temporary_impact_coefficient: float = 0.1  # Alpha
    permanent_impact_coefficient: float = 0.05  # Beta
    volatility_coefficient: float = 0.01  # Sigma adjustment
    
    # Model-specific parameters
    liquidity_factor: float = 0.5  # For square-root model
    participation_rate_power: float = 0.5  # Exponent for participation rate
    
    # Execution parameters
    risk_aversion: float = 1.0  # Lambda for optimization
    urgency_premium: float = 0.0  # Additional cost for urgent orders
    
    # Calibration info
    calibration_date: Optional[datetime] = None
    calibration_symbol: Optional[str] = None
    calibration_r_squared: Optional[float] = None
    
    def scale_by_volatility(self, volatility: float) -> 'SlippageParameters':
        """Scale parameters by volatility."""
        scaled = SlippageParameters(
            temporary_impact_coefficient=self.temporary_impact_coefficient * volatility,
            permanent_impact_coefficient=self.permanent_impact_coefficient,
            volatility_coefficient=self.volatility_coefficient,
            liquidity_factor=self.liquidity_factor,
            participation_rate_power=self.participation_rate_power,
            risk_aversion=self.risk_aversion,
            urgency_premium=self.urgency_premium
        )
        return scaled


@dataclass
class MarketImpactEstimate:
    """Market impact estimation for a trade."""
    symbol: str
    order_size: float
    side: str  # "buy" or "sell"
    
    # Impact components (in basis points)
    spread_cost: float
    temporary_impact: float
    permanent_impact: float
    timing_risk: float
    
    # Total costs
    total_impact_bps: float
    total_impact_dollars: float
    
    # Execution details
    execution_strategy: ExecutionStrategy
    execution_duration: float  # In minutes
    participation_rate: float
    
    # Confidence intervals
    impact_std_dev: Optional[float] = None
    confidence_interval_95: Optional[Tuple[float, float]] = None
    
    # Model info
    model_used: SlippageModel = SlippageModel.SQUARE_ROOT
    model_confidence: float = 0.8
    
    def get_impact_breakdown(self) -> Dict[str, float]:
        """Get breakdown of impact components."""
        return {
            "spread": self.spread_cost,
            "temporary": self.temporary_impact,
            "permanent": self.permanent_impact,
            "timing_risk": self.timing_risk,
            "total": self.total_impact_bps
        }
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution-related metrics."""
        return {
            "strategy": self.execution_strategy.value,
            "duration_minutes": self.execution_duration,
            "participation_rate": self.participation_rate,
            "avg_rate_per_minute": self.participation_rate / self.execution_duration
        }


@dataclass
class SlippageEstimate:
    """Comprehensive slippage estimate for an order."""
    order_id: str
    symbol: str
    timestamp: datetime
    
    # Order details
    order_size: float
    order_value: float
    side: str  # "buy" or "sell"
    urgency: str  # "low", "medium", "high", "critical"
    
    # Market conditions
    current_spread: float  # In basis points
    current_volatility: float  # Annualized
    average_daily_volume: float
    current_volume: float
    
    # Slippage estimates by model
    model_estimates: Dict[SlippageModel, float]  # Model -> Slippage in bps
    ensemble_estimate: float  # Weighted average of models
    
    # Recommended execution
    recommended_strategy: ExecutionStrategy
    recommended_duration: float  # Minutes
    recommended_participation: float  # As % of volume
    
    # Expected costs
    expected_slippage_bps: float
    expected_slippage_dollars: float
    worst_case_slippage_bps: float  # 95th percentile
    
    # Model metadata
    models_used: List[SlippageModel]
    model_weights: Dict[SlippageModel, float]
    estimation_confidence: float
    
    def get_size_normalized_impact(self) -> float:
        """Get impact normalized by order size."""
        if self.average_daily_volume > 0:
            return self.expected_slippage_bps * (self.order_size / self.average_daily_volume)
        return self.expected_slippage_bps
    
    def adjust_for_urgency(self) -> float:
        """Adjust slippage estimate based on urgency."""
        urgency_multipliers = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.5,
            "critical": 2.0
        }
        multiplier = urgency_multipliers.get(self.urgency, 1.0)
        return self.expected_slippage_bps * multiplier


@dataclass
class ExecutionPlan:
    """Detailed execution plan for an order."""
    order_id: str
    symbol: str
    total_size: float
    
    # Execution schedule
    time_slices: List[Tuple[datetime, datetime]]  # Start, end times
    slice_sizes: List[float]  # Size for each time slice
    slice_participation_rates: List[float]  # Participation rate for each slice
    
    # Strategy details
    primary_strategy: ExecutionStrategy
    adaptive_parameters: Dict[str, Any]  # For adaptive strategies
    
    # Cost estimates
    estimated_costs: List[float]  # Cost for each slice
    total_estimated_cost: float
    
    # Risk metrics
    execution_risk: float  # Volatility of execution cost
    completion_risk: float  # Risk of not completing
    
    # Benchmarks
    arrival_price: float
    expected_vwap: float
    expected_twap: float
    
    def get_slice_schedule(self) -> List[Dict[str, Any]]:
        """Get execution schedule by time slice."""
        schedule = []
        for i, (start, end) in enumerate(self.time_slices):
            schedule.append({
                "slice": i + 1,
                "start": start,
                "end": end,
                "size": self.slice_sizes[i],
                "participation": self.slice_participation_rates[i],
                "estimated_cost": self.estimated_costs[i]
            })
        return schedule
    
    def get_completion_probability(self, by_time: datetime) -> float:
        """Calculate probability of completion by given time."""
        completed_size = sum(
            size for (start, end), size in zip(self.time_slices, self.slice_sizes)
            if end <= by_time
        )
        return completed_size / self.total_size if self.total_size > 0 else 0.0


@dataclass
class RealizedSlippage:
    """Actual slippage after execution."""
    order_id: str
    symbol: str
    execution_start: datetime
    execution_end: datetime
    
    # Order details
    order_size: float
    side: str
    
    # Price metrics
    arrival_price: float
    average_execution_price: float
    closing_price: float
    
    # Slippage measurements
    implementation_shortfall: float  # vs arrival price
    realized_spread: float  # vs mid-quote
    effective_spread: float  # vs quoted spread
    price_improvement: float  # Positive if better than expected
    
    # Market impact
    temporary_impact_realized: float
    permanent_impact_realized: float
    total_impact_bps: float
    
    # Execution quality
    participation_rate_achieved: float
    fill_rate: float  # Percentage filled
    reversion_10min: float  # Price reversion after 10 minutes
    reversion_30min: float  # Price reversion after 30 minutes
    
    # Comparison to estimate
    estimated_slippage: float
    slippage_variance: float  # Actual - Estimated
    model_accuracy: float  # 1 - abs(variance/estimated)
    
    def calculate_total_cost(self) -> float:
        """Calculate total execution cost in dollars."""
        return abs(self.order_size * self.average_execution_price * 
                  self.total_impact_bps / 10000)
    
    def get_execution_metrics(self) -> Dict[str, float]:
        """Get key execution quality metrics."""
        return {
            "implementation_shortfall": self.implementation_shortfall,
            "total_impact_bps": self.total_impact_bps,
            "participation_rate": self.participation_rate_achieved,
            "fill_rate": self.fill_rate,
            "price_improvement": self.price_improvement,
            "model_accuracy": self.model_accuracy
        }


@dataclass
class SlippageAnalysis:
    """Aggregated slippage analysis."""
    period_start: datetime
    period_end: datetime
    
    # Summary statistics
    total_orders: int
    total_volume: float
    total_value: float
    
    # Average slippage by various dimensions
    avg_slippage_bps: float
    avg_slippage_by_size: Dict[str, float]  # Size bucket -> Average
    avg_slippage_by_symbol: Dict[str, float]  # Symbol -> Average
    avg_slippage_by_strategy: Dict[ExecutionStrategy, float]
    
    # Model performance
    model_accuracy_by_model: Dict[SlippageModel, float]
    model_bias_by_model: Dict[SlippageModel, float]  # Systematic over/under estimation
    
    # Cost analysis
    total_slippage_cost: float
    cost_by_impact_type: Dict[ImpactType, float]
    cost_savings_vs_baseline: float  # vs simple aggressive execution
    
    # Outliers and exceptions
    high_slippage_orders: List[str]  # Order IDs with high slippage
    model_failure_orders: List[str]  # Orders where model was very wrong
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get execution performance summary."""
        return {
            "total_orders": self.total_orders,
            "avg_slippage_bps": self.avg_slippage_bps,
            "total_cost": self.total_slippage_cost,
            "cost_per_order": self.total_slippage_cost / self.total_orders if self.total_orders > 0 else 0,
            "best_strategy": min(self.avg_slippage_by_strategy.items(), key=lambda x: x[1])[0].value if self.avg_slippage_by_strategy else None,
            "cost_savings": self.cost_savings_vs_baseline
        }


@dataclass
class OptimalExecutionParams:
    """Parameters for optimal execution algorithms."""
    # Required fields (no defaults) - must come first
    symbol: str
    order_size: float
    risk_aversion: float

    # Market parameters (required) - must come before optional fields
    daily_volatility: float
    daily_volume: float
    bid_ask_spread: float

    # Optional fields with defaults - must come after required fields
    # Almgren-Chriss parameters
    temporary_impact_power: float = 1.0
    permanent_impact_power: float = 0.5

    # Constraints
    min_participation_rate: float = 0.01
    max_participation_rate: float = 0.30
    max_duration_minutes: float = 390  # Full trading day

    # Optimization results
    optimal_duration: Optional[float] = None
    optimal_trajectory: Optional[np.ndarray] = None
    expected_cost: Optional[float] = None
    cost_variance: Optional[float] = None
    
    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of execution strategy."""
        if self.expected_cost and self.cost_variance and self.cost_variance > 0:
            return -self.expected_cost / np.sqrt(self.cost_variance)
        return 0.0