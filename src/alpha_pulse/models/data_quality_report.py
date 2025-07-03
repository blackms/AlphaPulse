"""
Data quality report models for comprehensive quality tracking and reporting.

Provides:
- Quality report data structures
- Quality score tracking models
- Alert and notification models
- Historical quality tracking
- Quality trend analysis models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from alpha_pulse.models.base import BaseModel


class ReportType(Enum):
    """Types of quality reports."""
    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class QualityStatus(Enum):
    """Overall quality status levels."""
    EXCELLENT = "excellent"  # > 95%
    GOOD = "good"           # 85-95%
    FAIR = "fair"           # 70-85%
    POOR = "poor"           # 50-70%
    CRITICAL = "critical"   # < 50%


@dataclass
class QualityDimensionScore:
    """Score for a single quality dimension."""
    dimension: str
    score: float  # 0.0 to 1.0
    weight: float
    weighted_score: float
    check_count: int
    pass_count: int
    fail_count: int
    warning_count: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheckSummary:
    """Summary of quality check results."""
    check_name: str
    total_executions: int
    pass_count: int
    fail_count: int
    warning_count: int
    skip_count: int
    average_score: float
    last_execution: datetime
    common_issues: List[str] = field(default_factory=list)


@dataclass
class AnomalyDetectionSummary:
    """Summary of anomaly detection results."""
    method: str
    total_detections: int
    anomaly_count: int
    anomaly_rate: float
    severity_distribution: Dict[str, int]
    average_score: float
    affected_fields: List[str]
    last_detection: datetime


@dataclass
class QualityTrend:
    """Quality trend analysis."""
    metric: str
    direction: str  # 'improving', 'stable', 'degrading'
    current_value: float
    previous_value: float
    change_percent: float
    trend_period: str  # e.g., '24h', '7d', '30d'
    data_points: int
    confidence: float


@dataclass
class DataSourceQuality:
    """Quality metrics for a specific data source."""
    source_name: str
    total_data_points: int
    valid_data_points: int
    quality_score: float
    availability_percent: float
    average_latency_ms: float
    error_rate: float
    last_update: datetime
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityAlert:
    """Quality alert information."""
    alert_id: str
    timestamp: datetime
    severity: str
    metric_type: str
    symbol: str
    current_value: float
    threshold_value: float
    description: str
    suggested_action: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class QualityRecommendation:
    """Quality improvement recommendation."""
    priority: int  # 1 = highest
    category: str
    recommendation: str
    impact: str  # 'high', 'medium', 'low'
    effort: str  # 'high', 'medium', 'low'
    metric_impact: List[str]  # Metrics that would improve
    implementation_notes: Optional[str] = None


@dataclass
class DataQualityReport(BaseModel):
    """Comprehensive data quality report."""
    report_id: str
    report_type: ReportType
    symbol: str
    report_period: Tuple[datetime, datetime]
    generated_at: datetime
    
    # Overall metrics
    overall_score: float
    overall_status: QualityStatus
    total_data_points: int
    valid_data_points: int
    quarantined_data_points: int
    
    # Dimension scores
    dimension_scores: List[QualityDimensionScore]
    
    # Check summaries
    quality_checks: List[QualityCheckSummary]
    
    # Anomaly detection
    anomaly_detection: List[AnomalyDetectionSummary]
    
    # Trends
    quality_trends: List[QualityTrend]
    
    # Data sources
    data_sources: List[DataSourceQuality]
    
    # Alerts
    active_alerts: List[QualityAlert]
    resolved_alerts: List[QualityAlert]
    
    # Recommendations
    recommendations: List[QualityRecommendation]
    
    # SLA compliance
    sla_compliance: Dict[str, bool]
    sla_violations: List[Dict[str, Any]]
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "symbol": self.symbol,
            "report_period": {
                "start": self.report_period[0].isoformat(),
                "end": self.report_period[1].isoformat()
            },
            "generated_at": self.generated_at.isoformat(),
            "overall_score": self.overall_score,
            "overall_status": self.overall_status.value,
            "total_data_points": self.total_data_points,
            "valid_data_points": self.valid_data_points,
            "quarantined_data_points": self.quarantined_data_points,
            "dimension_scores": [
                {
                    "dimension": ds.dimension,
                    "score": ds.score,
                    "weight": ds.weight,
                    "weighted_score": ds.weighted_score,
                    "check_count": ds.check_count,
                    "pass_count": ds.pass_count,
                    "fail_count": ds.fail_count,
                    "warning_count": ds.warning_count,
                    "details": ds.details
                }
                for ds in self.dimension_scores
            ],
            "quality_checks": [
                {
                    "check_name": qc.check_name,
                    "total_executions": qc.total_executions,
                    "pass_count": qc.pass_count,
                    "fail_count": qc.fail_count,
                    "warning_count": qc.warning_count,
                    "skip_count": qc.skip_count,
                    "average_score": qc.average_score,
                    "last_execution": qc.last_execution.isoformat(),
                    "common_issues": qc.common_issues
                }
                for qc in self.quality_checks
            ],
            "anomaly_detection": [
                {
                    "method": ad.method,
                    "total_detections": ad.total_detections,
                    "anomaly_count": ad.anomaly_count,
                    "anomaly_rate": ad.anomaly_rate,
                    "severity_distribution": ad.severity_distribution,
                    "average_score": ad.average_score,
                    "affected_fields": ad.affected_fields,
                    "last_detection": ad.last_detection.isoformat()
                }
                for ad in self.anomaly_detection
            ],
            "quality_trends": [
                {
                    "metric": qt.metric,
                    "direction": qt.direction,
                    "current_value": qt.current_value,
                    "previous_value": qt.previous_value,
                    "change_percent": qt.change_percent,
                    "trend_period": qt.trend_period,
                    "data_points": qt.data_points,
                    "confidence": qt.confidence
                }
                for qt in self.quality_trends
            ],
            "data_sources": [
                {
                    "source_name": ds.source_name,
                    "total_data_points": ds.total_data_points,
                    "valid_data_points": ds.valid_data_points,
                    "quality_score": ds.quality_score,
                    "availability_percent": ds.availability_percent,
                    "average_latency_ms": ds.average_latency_ms,
                    "error_rate": ds.error_rate,
                    "last_update": ds.last_update.isoformat(),
                    "issues": ds.issues
                }
                for ds in self.data_sources
            ],
            "active_alerts": [self._alert_to_dict(alert) for alert in self.active_alerts],
            "resolved_alerts": [self._alert_to_dict(alert) for alert in self.resolved_alerts],
            "recommendations": [
                {
                    "priority": rec.priority,
                    "category": rec.category,
                    "recommendation": rec.recommendation,
                    "impact": rec.impact,
                    "effort": rec.effort,
                    "metric_impact": rec.metric_impact,
                    "implementation_notes": rec.implementation_notes
                }
                for rec in self.recommendations
            ],
            "sla_compliance": self.sla_compliance,
            "sla_violations": self.sla_violations,
            "statistics": self.statistics,
            "metadata": self.metadata
        }
    
    def _alert_to_dict(self, alert: QualityAlert) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
            "severity": alert.severity,
            "metric_type": alert.metric_type,
            "symbol": alert.symbol,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "description": alert.description,
            "suggested_action": alert.suggested_action,
            "resolved": alert.resolved,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "resolution_notes": alert.resolution_notes
        }
    
    def get_quality_summary(self) -> str:
        """Get a text summary of the quality report."""
        summary_lines = [
            f"Data Quality Report for {self.symbol}",
            f"Report Type: {self.report_type.value}",
            f"Period: {self.report_period[0].strftime('%Y-%m-%d %H:%M')} to {self.report_period[1].strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"Overall Quality Score: {self.overall_score:.2%} ({self.overall_status.value.upper()})",
            f"Total Data Points: {self.total_data_points:,}",
            f"Valid Data Points: {self.valid_data_points:,} ({self.valid_data_points/self.total_data_points:.1%})",
            f"Quarantined Data Points: {self.quarantined_data_points:,}",
            f"",
            f"Quality Dimensions:"
        ]
        
        for dim_score in sorted(self.dimension_scores, key=lambda x: x.weighted_score, reverse=True):
            summary_lines.append(
                f"  - {dim_score.dimension}: {dim_score.score:.2%} "
                f"(Pass: {dim_score.pass_count}, Fail: {dim_score.fail_count}, Warning: {dim_score.warning_count})"
            )
        
        if self.active_alerts:
            summary_lines.extend([
                f"",
                f"Active Alerts: {len(self.active_alerts)}"
            ])
            for alert in self.active_alerts[:5]:  # Show top 5
                summary_lines.append(f"  - [{alert.severity}] {alert.description}")
        
        if self.recommendations:
            summary_lines.extend([
                f"",
                f"Top Recommendations:"
            ])
            for rec in self.recommendations[:3]:  # Show top 3
                summary_lines.append(f"  {rec.priority}. {rec.recommendation} (Impact: {rec.impact})")
        
        return "\n".join(summary_lines)
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score from dimension scores."""
        if not self.dimension_scores:
            return 0.0
        
        total_weighted_score = sum(ds.weighted_score for ds in self.dimension_scores)
        total_weight = sum(ds.weight for ds in self.dimension_scores)
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def determine_quality_status(self) -> QualityStatus:
        """Determine quality status based on overall score."""
        if self.overall_score >= 0.95:
            return QualityStatus.EXCELLENT
        elif self.overall_score >= 0.85:
            return QualityStatus.GOOD
        elif self.overall_score >= 0.70:
            return QualityStatus.FAIR
        elif self.overall_score >= 0.50:
            return QualityStatus.POOR
        else:
            return QualityStatus.CRITICAL


@dataclass
class QualityReportSummary:
    """Summary of multiple quality reports."""
    summary_period: Tuple[datetime, datetime]
    total_reports: int
    symbols_covered: List[str]
    average_quality_score: float
    quality_distribution: Dict[str, int]  # Status -> count
    top_issues: List[Tuple[str, int]]  # Issue -> occurrence count
    improvement_areas: List[str]
    overall_trends: List[QualityTrend]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "summary_period": {
                "start": self.summary_period[0].isoformat(),
                "end": self.summary_period[1].isoformat()
            },
            "total_reports": self.total_reports,
            "symbols_covered": self.symbols_covered,
            "average_quality_score": self.average_quality_score,
            "quality_distribution": self.quality_distribution,
            "top_issues": [
                {"issue": issue, "count": count} 
                for issue, count in self.top_issues
            ],
            "improvement_areas": self.improvement_areas,
            "overall_trends": [
                {
                    "metric": trend.metric,
                    "direction": trend.direction,
                    "current_value": trend.current_value,
                    "previous_value": trend.previous_value,
                    "change_percent": trend.change_percent,
                    "trend_period": trend.trend_period,
                    "data_points": trend.data_points,
                    "confidence": trend.confidence
                }
                for trend in self.overall_trends
            ]
        }


@dataclass
class QualityMetricHistory:
    """Historical tracking of quality metrics."""
    symbol: str
    metric_type: str
    time_series: List[Tuple[datetime, float]]  # (timestamp, value)
    aggregation_period: str  # '1m', '5m', '1h', '1d'
    
    def get_latest_value(self) -> Optional[float]:
        """Get the most recent metric value."""
        return self.time_series[-1][1] if self.time_series else None
    
    def get_average(self, hours: int = 24) -> Optional[float]:
        """Get average value over specified hours."""
        if not self.time_series:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_values = [
            value for timestamp, value in self.time_series 
            if timestamp > cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else None
    
    def detect_trend(self, window_size: int = 10) -> str:
        """Detect trend in recent data points."""
        if len(self.time_series) < window_size * 2:
            return "insufficient_data"
        
        recent_values = [value for _, value in self.time_series[-window_size:]]
        older_values = [value for _, value in self.time_series[-2*window_size:-window_size]]
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        if older_avg == 0:
            return "stable"
        
        change_ratio = (recent_avg - older_avg) / older_avg
        
        if change_ratio > 0.05:
            return "improving"
        elif change_ratio < -0.05:
            return "degrading"
        else:
            return "stable"