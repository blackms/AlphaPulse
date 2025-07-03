"""
Data models for dynamic risk budgeting.

Defines structures for risk budgets, allocations, and monitoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class RiskBudgetType(Enum):
    """Types of risk budgets."""
    MARKET = "market"
    SPECIFIC = "specific"
    SECTOR = "sector"
    GEOGRAPHIC = "geographic"
    FACTOR = "factor"
    TAIL = "tail"
    TOTAL = "total"


class AllocationMethod(Enum):
    """Risk allocation methodologies."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    REGIME_BASED = "regime_based"
    OPTIMIZATION_BASED = "optimization_based"
    HIERARCHICAL = "hierarchical"


@dataclass
class RiskAllocation:
    """Individual risk allocation within a budget."""
    asset_or_category: str
    allocated_risk: float  # Percentage of total risk budget
    current_utilization: float  # Current usage percentage
    risk_contribution: float  # Actual risk contribution
    
    # Limits and targets
    min_allocation: float = 0.0
    max_allocation: float = 1.0
    target_allocation: float = 0.0
    
    # Volatility metrics
    expected_volatility: float = 0.0
    realized_volatility: float = 0.0
    volatility_target: float = 0.0
    
    # Additional metrics
    sharpe_contribution: float = 0.0
    marginal_risk_contribution: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_within_limits(self) -> bool:
        """Check if allocation is within specified limits."""
        return self.min_allocation <= self.allocated_risk <= self.max_allocation
    
    @property
    def utilization_ratio(self) -> float:
        """Get utilization as ratio of allocated risk."""
        if self.allocated_risk > 0:
            return self.current_utilization / self.allocated_risk
        return 0.0
    
    def get_rebalancing_need(self) -> float:
        """Calculate rebalancing need (positive = need to increase)."""
        return self.target_allocation - self.allocated_risk


@dataclass
class RiskBudget:
    """Complete risk budget with allocations and constraints."""
    budget_id: str
    budget_type: RiskBudgetType
    total_risk_limit: float  # Total risk budget (e.g., 15% volatility)
    
    # Current state
    current_utilization: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Allocations
    allocations: Dict[str, RiskAllocation] = field(default_factory=dict)
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY
    
    # Constraints
    concentration_limit: float = 0.40  # Max 40% to any single allocation
    min_diversification: int = 3  # Minimum number of allocations
    correlation_adjustment: bool = True
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% annualized
    volatility_buffer: float = 0.02  # 2% buffer
    current_volatility: float = 0.0
    
    # Regime adjustments
    regime_multiplier: float = 1.0
    regime_type: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_allocated(self) -> float:
        """Get total allocated risk across all allocations."""
        return sum(alloc.allocated_risk for alloc in self.allocations.values())
    
    def get_total_utilized(self) -> float:
        """Get total utilized risk."""
        return sum(alloc.risk_contribution for alloc in self.allocations.values())
    
    def get_available_budget(self) -> float:
        """Get remaining available risk budget."""
        return max(0, self.total_risk_limit - self.current_utilization)
    
    def is_within_volatility_target(self) -> bool:
        """Check if current volatility is within target range."""
        lower_bound = self.target_volatility - self.volatility_buffer
        upper_bound = self.target_volatility + self.volatility_buffer
        return lower_bound <= self.current_volatility <= upper_bound
    
    def get_concentration_ratio(self) -> float:
        """Calculate concentration ratio (Herfindahl index)."""
        if not self.allocations:
            return 0.0
        
        total = self.get_total_allocated()
        if total == 0:
            return 0.0
        
        shares = [alloc.allocated_risk / total for alloc in self.allocations.values()]
        return sum(share ** 2 for share in shares)
    
    def get_effective_allocations(self) -> int:
        """Get effective number of allocations (inverse HHI)."""
        hhi = self.get_concentration_ratio()
        return 1 / hhi if hhi > 0 else 0


@dataclass
class RiskBudgetSnapshot:
    """Point-in-time snapshot of risk budget state."""
    timestamp: datetime
    budget_id: str
    
    # Risk metrics
    total_risk_utilization: float
    volatility_level: float
    var_95: float
    cvar_95: float
    
    # Allocation summary
    n_allocations: int
    concentration_ratio: float
    largest_allocation: Tuple[str, float]
    
    # Regime context
    market_regime: str
    regime_confidence: float
    
    # Performance metrics
    period_return: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Rebalancing info
    rebalancing_needed: bool
    rebalancing_urgency: str  # low, medium, high
    suggested_changes: List[Tuple[str, float]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskBudgetRebalancing:
    """Rebalancing recommendation for risk budget."""
    rebalancing_id: str
    budget_id: str
    timestamp: datetime
    
    # Trigger information
    trigger_type: str  # regime_change, breach, scheduled, volatility
    trigger_details: Dict[str, Any] = field(default_factory=dict)
    
    # Current vs target state
    current_allocations: Dict[str, float] = field(default_factory=dict)
    target_allocations: Dict[str, float] = field(default_factory=dict)
    allocation_changes: Dict[str, float] = field(default_factory=dict)
    
    # Expected impact
    expected_volatility_change: float = 0.0
    expected_return_impact: float = 0.0
    transaction_cost_estimate: float = 0.0
    
    # Execution plan
    priority_order: List[str] = field(default_factory=list)
    estimated_completion_time: int = 0  # minutes
    
    # Risk considerations
    execution_risk: str = "low"  # low, medium, high
    market_impact_estimate: float = 0.0
    
    approved: bool = False
    executed: bool = False
    execution_timestamp: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_turnover(self) -> float:
        """Calculate total portfolio turnover from rebalancing."""
        return sum(abs(change) for change in self.allocation_changes.values()) / 2
    
    def get_largest_change(self) -> Tuple[str, float]:
        """Get the largest allocation change."""
        if not self.allocation_changes:
            return ("", 0.0)
        
        max_change_key = max(self.allocation_changes, 
                            key=lambda k: abs(self.allocation_changes[k]))
        return (max_change_key, self.allocation_changes[max_change_key])


@dataclass
class VolatilityTarget:
    """Volatility targeting configuration and state."""
    target_id: str
    base_target: float  # Base volatility target (e.g., 12%)
    current_target: float  # Regime-adjusted target
    
    # Scaling factors
    regime_scaling: Dict[str, float] = field(default_factory=lambda: {
        "bull": 1.2,
        "bear": 0.8,
        "sideways": 1.0,
        "crisis": 0.5,
        "recovery": 0.9
    })
    
    # Volatility estimates
    realized_volatility: float = 0.0
    forecast_volatility: float = 0.0
    volatility_of_volatility: float = 0.0
    
    # Leverage calculations
    current_leverage: float = 1.0
    max_leverage: float = 2.0
    min_leverage: float = 0.2
    
    # Smoothing parameters
    volatility_lookback: int = 20  # days
    leverage_smoothing: int = 5  # days
    use_exponential_weighting: bool = True
    
    # Safety mechanisms
    volatility_cap: float = 0.30  # 30% max volatility
    leverage_step_limit: float = 0.20  # Max 20% leverage change per day
    
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_target_leverage(self) -> float:
        """Calculate target leverage based on volatility."""
        if self.forecast_volatility <= 0:
            return 1.0
        
        raw_leverage = self.current_target / self.forecast_volatility
        
        # Apply constraints
        leverage = np.clip(raw_leverage, self.min_leverage, self.max_leverage)
        
        # Apply step limit
        max_change = self.current_leverage * (1 + self.leverage_step_limit)
        min_change = self.current_leverage * (1 - self.leverage_step_limit)
        leverage = np.clip(leverage, min_change, max_change)
        
        return leverage
    
    def get_regime_adjusted_target(self, regime: str) -> float:
        """Get volatility target adjusted for market regime."""
        scaling = self.regime_scaling.get(regime, 1.0)
        return min(self.base_target * scaling, self.volatility_cap)


@dataclass
class RiskBudgetMonitoring:
    """Real-time monitoring of risk budget utilization."""
    budget_id: str
    monitoring_frequency: str = "real-time"  # real-time, minute, hourly
    
    # Thresholds for alerts
    utilization_warning: float = 0.80  # 80% utilization warning
    utilization_critical: float = 0.95  # 95% utilization critical
    volatility_breach_threshold: float = 0.02  # 2% above target
    
    # Current alerts
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    alert_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics tracking
    utilization_history: List[Tuple[datetime, float]] = field(default_factory=list)
    volatility_history: List[Tuple[datetime, float]] = field(default_factory=list)
    breach_count: int = 0
    
    # Auto-rebalancing
    auto_rebalance_enabled: bool = False
    min_rebalance_interval: int = 3600  # seconds
    last_rebalance_time: Optional[datetime] = None
    
    def add_alert(self, alert_type: str, severity: str, message: str):
        """Add a new alert."""
        alert = {
            "timestamp": datetime.utcnow(),
            "type": alert_type,
            "severity": severity,
            "message": message,
            "acknowledged": False
        }
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
    
    def get_active_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get all active critical alerts."""
        return [a for a in self.active_alerts 
                if a["severity"] == "critical" and not a["acknowledged"]]
    
    def should_trigger_rebalance(self) -> bool:
        """Check if auto-rebalancing should be triggered."""
        if not self.auto_rebalance_enabled:
            return False
        
        if self.last_rebalance_time:
            time_since_last = (datetime.utcnow() - self.last_rebalance_time).seconds
            if time_since_last < self.min_rebalance_interval:
                return False
        
        # Check for critical alerts
        return len(self.get_active_critical_alerts()) > 0