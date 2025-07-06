"""
Data quality metrics and reporting system for market data.

Provides:
- Comprehensive quality scoring framework
- Real-time quality monitoring and alerting
- Quality trend analysis and reporting
- Quality dashboards and visualization data
- Quality SLA monitoring and compliance tracking
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque
import json
from loguru import logger

from alpha_pulse.models.market_data import MarketDataPoint, TimeSeriesData, DataQuality
from alpha_pulse.data.quality.data_validator import ValidationResult, QualityScore, QualityDimension
from alpha_pulse.data.quality.anomaly_detector import AnomalyResult, AnomalySeverity
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class QualityMetricType(Enum):
    """Types of quality metrics."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    RELIABILITY = "reliability"
    ANOMALY_RATE = "anomaly_rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""
    metric_type: QualityMetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class QualityAlert:
    """Quality alert notification."""
    timestamp: datetime
    symbol: str
    metric_type: QualityMetricType
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    description: str
    suggested_action: str
    alert_id: str = field(default_factory=lambda: f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")


@dataclass
class QualityMetric:
    """Individual quality metric."""
    timestamp: datetime
    symbol: str
    metric_type: QualityMetricType
    value: float
    target_value: float
    variance: float
    trend: str  # 'improving', 'stable', 'degrading'
    confidence: float
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualitySLA:
    """Service Level Agreement for data quality."""
    metric_type: QualityMetricType
    target_value: float
    measurement_period: str  # '1h', '1d', '1w', '1m'
    tolerance: float = 0.05
    breach_threshold: int = 3  # Number of consecutive breaches before alert


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    symbol: str
    report_period: Tuple[datetime, datetime]
    overall_score: float
    dimension_scores: Dict[str, float]
    metrics: List[QualityMetric]
    alerts: List[QualityAlert]
    sla_compliance: Dict[str, bool]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class QualityMetricsCalculator:
    """Calculator for various data quality metrics."""
    
    def __init__(self):
        self.quality_history: Dict[str, List[QualityScore]] = {}
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        self.anomaly_history: Dict[str, List[AnomalyResult]] = {}
    
    def calculate_completeness_metric(
        self, 
        symbol: str, 
        validation_results: List[ValidationResult],
        time_period: timedelta
    ) -> QualityMetric:
        """Calculate data completeness metric."""
        if not validation_results:
            return self._create_empty_metric(symbol, QualityMetricType.COMPLETENESS)
        
        # Calculate completeness from validation results
        completeness_scores = [r.quality_score.completeness for r in validation_results]
        avg_completeness = statistics.mean(completeness_scores)
        variance = statistics.variance(completeness_scores) if len(completeness_scores) > 1 else 0.0
        
        # Determine trend
        trend = self._calculate_trend(completeness_scores)
        
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=QualityMetricType.COMPLETENESS,
            value=avg_completeness,
            target_value=0.95,  # 95% target
            variance=variance,
            trend=trend,
            confidence=min(1.0, len(completeness_scores) / 100),
            sample_size=len(completeness_scores),
            metadata={
                'time_period': str(time_period),
                'min_score': min(completeness_scores),
                'max_score': max(completeness_scores)
            }
        )
    
    def calculate_accuracy_metric(
        self, 
        symbol: str, 
        validation_results: List[ValidationResult],
        time_period: timedelta
    ) -> QualityMetric:
        """Calculate data accuracy metric."""
        if not validation_results:
            return self._create_empty_metric(symbol, QualityMetricType.ACCURACY)
        
        accuracy_scores = [r.quality_score.accuracy for r in validation_results]
        avg_accuracy = statistics.mean(accuracy_scores)
        variance = statistics.variance(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
        trend = self._calculate_trend(accuracy_scores)
        
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=QualityMetricType.ACCURACY,
            value=avg_accuracy,
            target_value=0.90,  # 90% target
            variance=variance,
            trend=trend,
            confidence=min(1.0, len(accuracy_scores) / 100),
            sample_size=len(accuracy_scores),
            metadata={
                'time_period': str(time_period),
                'error_rate': 1.0 - avg_accuracy,
                'validation_failures': sum(1 for r in validation_results if not r.is_valid)
            }
        )
    
    def calculate_timeliness_metric(
        self, 
        symbol: str, 
        validation_results: List[ValidationResult],
        time_period: timedelta
    ) -> QualityMetric:
        """Calculate data timeliness metric."""
        if not validation_results:
            return self._create_empty_metric(symbol, QualityMetricType.TIMELINESS)
        
        timeliness_scores = [r.quality_score.timeliness for r in validation_results]
        avg_timeliness = statistics.mean(timeliness_scores)
        variance = statistics.variance(timeliness_scores) if len(timeliness_scores) > 1 else 0.0
        trend = self._calculate_trend(timeliness_scores)
        
        # Calculate latency statistics
        latencies = [r.validation_duration_ms for r in validation_results]
        avg_latency = statistics.mean(latencies)
        
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=QualityMetricType.TIMELINESS,
            value=avg_timeliness,
            target_value=0.90,  # 90% target
            variance=variance,
            trend=trend,
            confidence=min(1.0, len(timeliness_scores) / 100),
            sample_size=len(timeliness_scores),
            metadata={
                'time_period': str(time_period),
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max(latencies),
                'min_latency_ms': min(latencies)
            }
        )
    
    def calculate_consistency_metric(
        self, 
        symbol: str, 
        validation_results: List[ValidationResult],
        time_period: timedelta
    ) -> QualityMetric:
        """Calculate data consistency metric."""
        if not validation_results:
            return self._create_empty_metric(symbol, QualityMetricType.CONSISTENCY)
        
        consistency_scores = [r.quality_score.consistency for r in validation_results]
        avg_consistency = statistics.mean(consistency_scores)
        variance = statistics.variance(consistency_scores) if len(consistency_scores) > 1 else 0.0
        trend = self._calculate_trend(consistency_scores)
        
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=QualityMetricType.CONSISTENCY,
            value=avg_consistency,
            target_value=0.85,  # 85% target
            variance=variance,
            trend=trend,
            confidence=min(1.0, len(consistency_scores) / 100),
            sample_size=len(consistency_scores),
            metadata={
                'time_period': str(time_period),
                'consistency_variance': variance
            }
        )
    
    def calculate_anomaly_rate_metric(
        self, 
        symbol: str, 
        anomaly_results: List[AnomalyResult],
        time_period: timedelta
    ) -> QualityMetric:
        """Calculate anomaly detection rate metric."""
        if not anomaly_results:
            return self._create_empty_metric(symbol, QualityMetricType.ANOMALY_RATE)
        
        # Calculate anomaly rate
        total_detections = len(anomaly_results)
        anomaly_count = sum(1 for r in anomaly_results if r.is_anomaly)
        anomaly_rate = anomaly_count / total_detections if total_detections > 0 else 0.0
        
        # Calculate severity distribution
        severity_counts = defaultdict(int)
        for result in anomaly_results:
            if result.is_anomaly:
                severity_counts[result.severity.value] += 1
        
        # Target is low anomaly rate (inverted metric)
        quality_score = max(0.0, 1.0 - anomaly_rate)
        
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=QualityMetricType.ANOMALY_RATE,
            value=quality_score,
            target_value=0.95,  # Target: 95% non-anomalous data
            variance=0.0,  # Calculated separately for rates
            trend='stable',  # Would need historical data for trend
            confidence=min(1.0, total_detections / 100),
            sample_size=total_detections,
            metadata={
                'time_period': str(time_period),
                'anomaly_rate': anomaly_rate,
                'anomaly_count': anomaly_count,
                'total_detections': total_detections,
                'severity_distribution': dict(severity_counts)
            }
        )
    
    def calculate_reliability_metric(
        self, 
        symbol: str, 
        validation_results: List[ValidationResult],
        time_period: timedelta
    ) -> QualityMetric:
        """Calculate overall data reliability metric."""
        if not validation_results:
            return self._create_empty_metric(symbol, QualityMetricType.RELIABILITY)
        
        # Reliability combines multiple factors
        overall_scores = [r.quality_score.overall_score for r in validation_results]
        valid_count = sum(1 for r in validation_results if r.is_valid)
        
        # Calculate reliability as combination of quality and validity
        avg_quality = statistics.mean(overall_scores)
        validity_rate = valid_count / len(validation_results)
        reliability_score = (avg_quality + validity_rate) / 2
        
        variance = statistics.variance(overall_scores) if len(overall_scores) > 1 else 0.0
        trend = self._calculate_trend(overall_scores)
        
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=QualityMetricType.RELIABILITY,
            value=reliability_score,
            target_value=0.90,  # 90% target
            variance=variance,
            trend=trend,
            confidence=min(1.0, len(validation_results) / 100),
            sample_size=len(validation_results),
            metadata={
                'time_period': str(time_period),
                'avg_quality_score': avg_quality,
                'validity_rate': validity_rate,
                'valid_count': valid_count,
                'total_validations': len(validation_results)
            }
        )
    
    def _calculate_trend(self, values: List[float], window_size: int = 10) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < window_size:
            return 'stable'
        
        recent_values = values[-window_size:]
        older_values = values[-2*window_size:-window_size] if len(values) >= 2*window_size else values[:-window_size]
        
        if not older_values:
            return 'stable'
        
        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)
        
        change_ratio = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
        
        if change_ratio > 0.05:  # 5% improvement
            return 'improving'
        elif change_ratio < -0.05:  # 5% degradation
            return 'degrading'
        else:
            return 'stable'
    
    def _create_empty_metric(self, symbol: str, metric_type: QualityMetricType) -> QualityMetric:
        """Create an empty metric when no data is available."""
        return QualityMetric(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            metric_type=metric_type,
            value=0.0,
            target_value=0.0,
            variance=0.0,
            trend='unknown',
            confidence=0.0,
            sample_size=0,
            metadata={'status': 'no_data'}
        )


class QualityAlertManager:
    """Manager for quality alerts and notifications."""
    
    def __init__(self, thresholds: Optional[List[QualityThreshold]] = None, main_alert_manager=None):
        self.thresholds = thresholds or self._get_default_thresholds()
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_history: List[QualityAlert] = []
        self.audit_logger = get_audit_logger()
        self.main_alert_manager = main_alert_manager  # Integration with main AlertManager
    
    def _get_default_thresholds(self) -> List[QualityThreshold]:
        """Get default quality thresholds."""
        return [
            QualityThreshold(QualityMetricType.COMPLETENESS, 0.90, 0.80, 0.70),
            QualityThreshold(QualityMetricType.ACCURACY, 0.85, 0.75, 0.65),
            QualityThreshold(QualityMetricType.TIMELINESS, 0.85, 0.75, 0.65),
            QualityThreshold(QualityMetricType.CONSISTENCY, 0.80, 0.70, 0.60),
            QualityThreshold(QualityMetricType.RELIABILITY, 0.85, 0.75, 0.65),
            QualityThreshold(QualityMetricType.ANOMALY_RATE, 0.90, 0.80, 0.70)
        ]
    
    async def check_thresholds(self, metrics: List[QualityMetric]) -> List[QualityAlert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        
        for metric in metrics:
            threshold = self._get_threshold(metric.metric_type)
            if not threshold or not threshold.enabled:
                continue
            
            alert = self._create_alert_if_needed(metric, threshold)
            if alert:
                new_alerts.append(alert)
                await self._process_alert(alert)
        
        return new_alerts
    
    def _get_threshold(self, metric_type: QualityMetricType) -> Optional[QualityThreshold]:
        """Get threshold configuration for a metric type."""
        for threshold in self.thresholds:
            if threshold.metric_type == metric_type:
                return threshold
        return None
    
    def _create_alert_if_needed(self, metric: QualityMetric, threshold: QualityThreshold) -> Optional[QualityAlert]:
        """Create alert if metric breaches threshold."""
        severity = None
        threshold_value = None
        
        # Determine alert severity based on thresholds
        if threshold.emergency_threshold and metric.value <= threshold.emergency_threshold:
            severity = AlertSeverity.EMERGENCY
            threshold_value = threshold.emergency_threshold
        elif metric.value <= threshold.critical_threshold:
            severity = AlertSeverity.CRITICAL
            threshold_value = threshold.critical_threshold
        elif metric.value <= threshold.warning_threshold:
            severity = AlertSeverity.WARNING
            threshold_value = threshold.warning_threshold
        
        if severity:
            alert_key = f"{metric.symbol}_{metric.metric_type.value}"
            
            # Check if this alert is already active
            if alert_key in self.active_alerts:
                existing_alert = self.active_alerts[alert_key]
                # Update if severity increased
                if self._severity_level(severity) > self._severity_level(existing_alert.severity):
                    existing_alert.severity = severity
                    existing_alert.current_value = metric.value
                    existing_alert.threshold_value = threshold_value
                    existing_alert.timestamp = datetime.utcnow()
                return None
            
            # Create new alert
            alert = QualityAlert(
                timestamp=datetime.utcnow(),
                symbol=metric.symbol,
                metric_type=metric.metric_type,
                severity=severity,
                current_value=metric.value,
                threshold_value=threshold_value,
                description=self._generate_alert_description(metric, severity),
                suggested_action=self._generate_suggested_action(metric, severity)
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            return alert
        
        return None
    
    def _severity_level(self, severity: AlertSeverity) -> int:
        """Get numeric level for severity comparison."""
        levels = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.EMERGENCY: 4
        }
        return levels.get(severity, 0)
    
    def _generate_alert_description(self, metric: QualityMetric, severity: AlertSeverity) -> str:
        """Generate alert description."""
        return (
            f"{severity.value.upper()}: {metric.metric_type.value} for {metric.symbol} "
            f"dropped to {metric.value:.3f} (target: {metric.target_value:.3f}). "
            f"Trend: {metric.trend}, Confidence: {metric.confidence:.2f}"
        )
    
    def _generate_suggested_action(self, metric: QualityMetric, severity: AlertSeverity) -> str:
        """Generate suggested action for alert."""
        actions = {
            QualityMetricType.COMPLETENESS: "Check data source connectivity and collection processes",
            QualityMetricType.ACCURACY: "Review data validation rules and source data quality",
            QualityMetricType.TIMELINESS: "Investigate data processing latency and provider delays",
            QualityMetricType.CONSISTENCY: "Check for data source changes or processing errors",
            QualityMetricType.RELIABILITY: "Comprehensive system health check required",
            QualityMetricType.ANOMALY_RATE: "Investigate cause of increased anomaly detections"
        }
        
        base_action = actions.get(metric.metric_type, "Investigate data quality issue")
        
        if severity == AlertSeverity.EMERGENCY:
            return f"IMMEDIATE ACTION REQUIRED: {base_action}"
        elif severity == AlertSeverity.CRITICAL:
            return f"URGENT: {base_action}"
        else:
            return base_action
    
    async def _process_alert(self, alert: QualityAlert) -> None:
        """Process and log alert."""
        severity_map = {
            AlertSeverity.INFO: AuditSeverity.INFO,
            AlertSeverity.WARNING: AuditSeverity.WARNING,
            AlertSeverity.CRITICAL: AuditSeverity.ERROR,
            AlertSeverity.EMERGENCY: AuditSeverity.ERROR
        }
        
        # Log to audit system
        self.audit_logger.log(
            event_type=AuditEventType.QUALITY_ALERT,
            event_data={
                "alert_id": alert.alert_id,
                "symbol": alert.symbol,
                "metric_type": alert.metric_type.value,
                "severity": alert.severity.value,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "description": alert.description,
                "suggested_action": alert.suggested_action
            },
            severity=severity_map.get(alert.severity, AuditSeverity.WARNING)
        )
        
        # Send through main alert system if available
        if self.main_alert_manager:
            try:
                await self.main_alert_manager.send_quality_alert(
                    symbol=alert.symbol,
                    metric_type=alert.metric_type.value,
                    severity=alert.severity.value,
                    current_value=alert.current_value,
                    threshold_value=alert.threshold_value,
                    description=alert.description,
                    suggested_action=alert.suggested_action,
                    channels=["web", "email"]  # Default notification channels
                )
                logger.info(f"Quality alert sent through main alert system: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to send quality alert through main system: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        for key, alert in list(self.active_alerts.items()):
            if alert.alert_id == alert_id:
                del self.active_alerts[key]
                logger.info(f"Resolved alert {alert_id}")
                return True
        return False
    
    def get_active_alerts(self, symbol: Optional[str] = None) -> List[QualityAlert]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        return sorted(alerts, key=lambda x: self._severity_level(x.severity), reverse=True)


class QualityMetricsService:
    """Main service for quality metrics calculation and monitoring."""
    
    def __init__(self, main_alert_manager=None):
        self.calculator = QualityMetricsCalculator()
        self.alert_manager = QualityAlertManager(main_alert_manager=main_alert_manager)
        self.metrics_history: Dict[str, List[QualityMetric]] = defaultdict(list)
        self.sla_configs: Dict[str, List[QualitySLA]] = {}
        self.audit_logger = get_audit_logger()
    
    async def calculate_quality_metrics(
        self, 
        symbol: str,
        validation_results: List[ValidationResult],
        anomaly_results: List[AnomalyResult],
        time_period: timedelta = timedelta(hours=1)
    ) -> List[QualityMetric]:
        """Calculate all quality metrics for a symbol."""
        metrics = []
        
        # Calculate individual metrics
        completeness = self.calculator.calculate_completeness_metric(symbol, validation_results, time_period)
        accuracy = self.calculator.calculate_accuracy_metric(symbol, validation_results, time_period)
        timeliness = self.calculator.calculate_timeliness_metric(symbol, validation_results, time_period)
        consistency = self.calculator.calculate_consistency_metric(symbol, validation_results, time_period)
        reliability = self.calculator.calculate_reliability_metric(symbol, validation_results, time_period)
        anomaly_rate = self.calculator.calculate_anomaly_rate_metric(symbol, anomaly_results, time_period)
        
        metrics.extend([completeness, accuracy, timeliness, consistency, reliability, anomaly_rate])
        
        # Store metrics history
        self.metrics_history[symbol].extend(metrics)
        
        # Keep only recent history (last 30 days)
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.metrics_history[symbol] = [
            m for m in self.metrics_history[symbol] 
            if m.timestamp > cutoff_time
        ]
        
        # Check for alerts
        alerts = await self.alert_manager.check_thresholds(metrics)
        
        return metrics
    
    async def generate_quality_report(
        self, 
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> QualityReport:
        """Generate comprehensive quality report."""
        # Get metrics for the period
        symbol_metrics = self.metrics_history.get(symbol, [])
        period_metrics = [
            m for m in symbol_metrics 
            if start_time <= m.timestamp <= end_time
        ]
        
        if not period_metrics:
            # Return empty report
            return QualityReport(
                symbol=symbol,
                report_period=(start_time, end_time),
                overall_score=0.0,
                dimension_scores={},
                metrics=[],
                alerts=[],
                sla_compliance={},
                recommendations=["No data available for the specified period"],
                trend_analysis={}
            )
        
        # Calculate overall score
        metric_scores = {}
        for metric_type in QualityMetricType:
            type_metrics = [m for m in period_metrics if m.metric_type == metric_type]
            if type_metrics:
                avg_score = statistics.mean(m.value for m in type_metrics)
                metric_scores[metric_type.value] = avg_score
        
        overall_score = statistics.mean(metric_scores.values()) if metric_scores else 0.0
        
        # Get alerts for the period
        period_alerts = [
            a for a in self.alert_manager.alert_history 
            if a.symbol == symbol and start_time <= a.timestamp <= end_time
        ]
        
        # Check SLA compliance
        sla_compliance = self._check_sla_compliance(symbol, period_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(symbol, period_metrics, period_alerts)
        
        # Analyze trends
        trend_analysis = self._analyze_trends(period_metrics)
        
        return QualityReport(
            symbol=symbol,
            report_period=(start_time, end_time),
            overall_score=overall_score,
            dimension_scores=metric_scores,
            metrics=period_metrics,
            alerts=period_alerts,
            sla_compliance=sla_compliance,
            recommendations=recommendations,
            trend_analysis=trend_analysis
        )
    
    def _check_sla_compliance(self, symbol: str, metrics: List[QualityMetric]) -> Dict[str, bool]:
        """Check SLA compliance for quality metrics."""
        compliance = {}
        slas = self.sla_configs.get(symbol, [])
        
        for sla in slas:
            relevant_metrics = [m for m in metrics if m.metric_type == sla.metric_type]
            if relevant_metrics:
                # Check if average meets SLA target
                avg_value = statistics.mean(m.value for m in relevant_metrics)
                compliance[sla.metric_type.value] = abs(avg_value - sla.target_value) <= sla.tolerance
            else:
                compliance[sla.metric_type.value] = False
        
        return compliance
    
    def _generate_recommendations(
        self, 
        symbol: str, 
        metrics: List[QualityMetric], 
        alerts: List[QualityAlert]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyze metric performance
        metric_scores = {}
        for metric_type in QualityMetricType:
            type_metrics = [m for m in metrics if m.metric_type == metric_type]
            if type_metrics:
                metric_scores[metric_type] = statistics.mean(m.value for m in type_metrics)
        
        # Identify lowest performing metrics
        if metric_scores:
            worst_metric = min(metric_scores.items(), key=lambda x: x[1])
            if worst_metric[1] < 0.8:
                recommendations.append(f"Priority: Improve {worst_metric[0].value} (current: {worst_metric[1]:.2f})")
        
        # Analyze alert patterns
        if alerts:
            critical_alerts = [a for a in alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]]
            if critical_alerts:
                recommendations.append("Immediate: Address critical quality alerts")
            
            # Check for recurring alert types
            alert_types = defaultdict(int)
            for alert in alerts:
                alert_types[alert.metric_type] += 1
            
            if alert_types:
                most_frequent = max(alert_types.items(), key=lambda x: x[1])
                if most_frequent[1] > 2:
                    recommendations.append(f"Review: {most_frequent[0].value} has recurring issues")
        
        # Trend-based recommendations
        degrading_metrics = [m for m in metrics if m.trend == 'degrading']
        if degrading_metrics:
            recommendations.append("Monitor: Some quality metrics show degrading trends")
        
        if not recommendations:
            recommendations.append("Quality levels are within acceptable ranges")
        
        return recommendations
    
    def _analyze_trends(self, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Analyze quality trends."""
        trend_analysis = {}
        
        for metric_type in QualityMetricType:
            type_metrics = [m for m in metrics if m.metric_type == metric_type]
            if len(type_metrics) >= 2:
                values = [m.value for m in sorted(type_metrics, key=lambda x: x.timestamp)]
                
                # Calculate trend statistics
                trend_analysis[metric_type.value] = {
                    'direction': type_metrics[-1].trend,
                    'start_value': values[0],
                    'end_value': values[-1],
                    'change': values[-1] - values[0],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                    'variance': statistics.variance(values) if len(values) > 1 else 0
                }
        
        return trend_analysis
    
    def configure_sla(self, symbol: str, slas: List[QualitySLA]) -> None:
        """Configure SLA thresholds for a symbol."""
        self.sla_configs[symbol] = slas
        logger.info(f"Configured {len(slas)} SLAs for {symbol}")
    
    def get_dashboard_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get data for quality dashboard."""
        dashboard_data = {
            'overall_status': {},
            'recent_metrics': {},
            'active_alerts': {},
            'trend_summary': {}
        }
        
        for symbol in symbols:
            # Get recent metrics (last hour)
            recent_metrics = [
                m for m in self.metrics_history.get(symbol, [])
                if m.timestamp > datetime.utcnow() - timedelta(hours=1)
            ]
            
            if recent_metrics:
                # Calculate overall status
                avg_scores = {}
                for metric_type in QualityMetricType:
                    type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
                    if type_metrics:
                        avg_scores[metric_type.value] = statistics.mean(m.value for m in type_metrics)
                
                overall_score = statistics.mean(avg_scores.values()) if avg_scores else 0.0
                
                dashboard_data['overall_status'][symbol] = {
                    'score': overall_score,
                    'status': 'good' if overall_score >= 0.8 else 'warning' if overall_score >= 0.6 else 'critical'
                }
                dashboard_data['recent_metrics'][symbol] = avg_scores
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts(symbol)
            dashboard_data['active_alerts'][symbol] = len(active_alerts)
            
            # Trend summary
            if symbol in self.metrics_history:
                improving = sum(1 for m in recent_metrics if m.trend == 'improving')
                degrading = sum(1 for m in recent_metrics if m.trend == 'degrading')
                dashboard_data['trend_summary'][symbol] = {
                    'improving': improving,
                    'degrading': degrading,
                    'stable': len(recent_metrics) - improving - degrading
                }
        
        return dashboard_data


# Global service instance
_quality_metrics_service: Optional[QualityMetricsService] = None


def get_quality_metrics_service(main_alert_manager=None) -> QualityMetricsService:
    """Get the global quality metrics service instance."""
    global _quality_metrics_service
    
    if _quality_metrics_service is None:
        _quality_metrics_service = QualityMetricsService(main_alert_manager=main_alert_manager)
    
    return _quality_metrics_service