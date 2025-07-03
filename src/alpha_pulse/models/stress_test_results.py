"""
Data models for stress test results.

Defines the structure of stress test outputs including scenario results,
position impacts, and risk metric changes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class StressTestStatus(Enum):
    """Status of stress test execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PositionImpact:
    """Impact of stress scenario on individual position."""
    position_id: str
    symbol: str
    initial_value: float
    stressed_value: float
    pnl: float
    price_change_pct: float
    var_contribution: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetricImpact:
    """Impact on risk metrics under stress scenario."""
    metric_name: str
    initial_value: float
    stressed_value: float
    change_pct: float
    breach_threshold: bool = False
    threshold_value: Optional[float] = None


@dataclass
class ScenarioResult:
    """Results for a single stress scenario."""
    scenario_name: str
    scenario_type: str  # historical, hypothetical, monte_carlo, etc.
    probability: float
    total_pnl: float
    pnl_percentage: float
    position_impacts: List[PositionImpact]
    risk_metric_impacts: List[RiskMetricImpact]
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_worst_position(self) -> Optional[PositionImpact]:
        """Get position with worst P&L impact."""
        if not self.position_impacts:
            return None
        return min(self.position_impacts, key=lambda x: x.pnl)
    
    def get_best_position(self) -> Optional[PositionImpact]:
        """Get position with best P&L impact."""
        if not self.position_impacts:
            return None
        return max(self.position_impacts, key=lambda x: x.pnl)
    
    def get_breached_metrics(self) -> List[RiskMetricImpact]:
        """Get risk metrics that breached thresholds."""
        return [m for m in self.risk_metric_impacts if m.breach_threshold]


@dataclass
class StressTestSummary:
    """Summary statistics across all stress scenarios."""
    worst_case_scenario: str
    worst_case_pnl: float
    worst_case_pnl_pct: float
    expected_shortfall: float
    scenarios_passed: int
    scenarios_failed: int
    average_pnl: float
    pnl_volatility: float
    risk_metrics_summary: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """Calculate scenario pass rate."""
        total = self.scenarios_passed + self.scenarios_failed
        return self.scenarios_passed / total if total > 0 else 0.0


@dataclass
class StressTestResult:
    """Complete stress test result."""
    test_id: str
    portfolio_id: str
    test_date: datetime
    scenarios: List[ScenarioResult]
    summary: Optional[StressTestSummary] = None
    status: StressTestStatus = StressTestStatus.COMPLETED
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_scenario_by_name(self, name: str) -> Optional[ScenarioResult]:
        """Get specific scenario result by name."""
        for scenario in self.scenarios:
            if scenario.scenario_name == name:
                return scenario
        return None
    
    def get_scenarios_by_type(self, scenario_type: str) -> List[ScenarioResult]:
        """Get all scenarios of specific type."""
        return [s for s in self.scenarios if s.scenario_type == scenario_type]
    
    def get_failed_scenarios(self, threshold: float = 0.0) -> List[ScenarioResult]:
        """Get scenarios with negative P&L below threshold."""
        return [s for s in self.scenarios if s.total_pnl < threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "portfolio_id": self.portfolio_id,
            "test_date": self.test_date.isoformat(),
            "status": self.status.value,
            "scenarios": [
                {
                    "name": s.scenario_name,
                    "type": s.scenario_type,
                    "probability": s.probability,
                    "pnl": s.total_pnl,
                    "pnl_pct": s.pnl_percentage,
                    "n_positions": len(s.position_impacts),
                    "n_metrics": len(s.risk_metric_impacts)
                }
                for s in self.scenarios
            ],
            "summary": {
                "worst_case_scenario": self.summary.worst_case_scenario,
                "worst_case_pnl": self.summary.worst_case_pnl,
                "worst_case_pnl_pct": self.summary.worst_case_pnl_pct,
                "pass_rate": self.summary.pass_rate,
                "average_pnl": self.summary.average_pnl
            } if self.summary else None,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata
        }


@dataclass
class StressTestReport:
    """Formatted stress test report for presentation."""
    test_result: StressTestResult
    executive_summary: str
    detailed_findings: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    charts: Dict[str, Any] = field(default_factory=dict)
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary text."""
        if not self.test_result.summary:
            return "No summary available."
        
        summary = f"""
Stress Test Executive Summary
============================

Test ID: {self.test_result.test_id}
Date: {self.test_result.test_date.strftime('%Y-%m-%d %H:%M')}
Portfolio: {self.test_result.portfolio_id}

Key Findings:
- Worst Case Scenario: {self.test_result.summary.worst_case_scenario}
- Maximum Loss: {self.test_result.summary.worst_case_pnl:,.2f} ({self.test_result.summary.worst_case_pnl_pct:.1f}%)
- Expected Shortfall: {self.test_result.summary.expected_shortfall:,.2f}
- Scenario Pass Rate: {self.test_result.summary.pass_rate:.1%}

Risk Assessment: {'HIGH' if self.test_result.summary.worst_case_pnl_pct < -20 else 'MODERATE' if self.test_result.summary.worst_case_pnl_pct < -10 else 'LOW'}
"""
        return summary.strip()