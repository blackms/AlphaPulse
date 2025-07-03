"""
Data validation utilities for market data quality assurance.

Provides:
- Real-time data validation
- Historical data consistency checks
- Cross-provider data verification
- Anomaly detection
- Data quality scoring
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

from alpha_pulse.models.market_data import (
    MarketDataPoint, TimeSeriesData, DataQuality, OHLCV, DataSource
)
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class ValidationLevel(Enum):
    """Data validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of data anomalies."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    MISSING_DATA = "missing_data"
    INCONSISTENT_OHLC = "inconsistent_ohlc"
    STALE_DATA = "stale_data"
    DUPLICATE_DATA = "duplicate_data"
    OUTLIER_PRICE = "outlier_price"
    ZERO_VOLUME = "zero_volume"
    NEGATIVE_PRICE = "negative_price"
    CROSS_PROVIDER_MISMATCH = "cross_provider_mismatch"


@dataclass
class ValidationRule:
    """Data validation rule definition."""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    enabled: bool = True
    parameters: Dict[str, Any] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    anomalies: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    validation_timestamp: datetime
    processing_time_ms: float
    rules_applied: List[str]


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics."""
    completeness: float  # Percentage of expected data points
    accuracy: float  # Accuracy score based on validation rules
    consistency: float  # Consistency across time and providers
    timeliness: float  # Data freshness score
    validity: float  # Format and range validity
    overall_score: float  # Weighted average of all metrics


class MarketDataValidator:
    """Comprehensive market data validator."""

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_cross_validation: bool = True,
        enable_anomaly_detection: bool = True
    ):
        """
        Initialize market data validator.

        Args:
            validation_level: Strictness level for validation
            enable_cross_validation: Enable cross-provider validation
            enable_anomaly_detection: Enable statistical anomaly detection
        """
        self.validation_level = validation_level
        self.enable_cross_validation = enable_cross_validation
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Validation rules by level
        self.validation_rules = self._setup_validation_rules()
        
        # Historical data for anomaly detection
        self._price_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[float]] = {}
        
        # Audit logging
        self.audit_logger = get_audit_logger()

    def _setup_validation_rules(self) -> Dict[ValidationLevel, List[ValidationRule]]:
        """Setup validation rules for different levels."""
        basic_rules = [
            ValidationRule(
                name="price_non_negative",
                description="Prices must be non-negative",
                severity="error"
            ),
            ValidationRule(
                name="volume_non_negative",
                description="Volume must be non-negative",
                severity="error"
            ),
            ValidationRule(
                name="ohlc_consistency",
                description="OHLC values must be consistent",
                severity="error"
            ),
            ValidationRule(
                name="timestamp_validity",
                description="Timestamp must be valid and recent",
                severity="error",
                parameters={"max_age_hours": 24}
            )
        ]
        
        standard_rules = basic_rules + [
            ValidationRule(
                name="price_reasonableness",
                description="Prices must be within reasonable ranges",
                severity="warning",
                parameters={"max_change_percent": 50.0}
            ),
            ValidationRule(
                name="volume_reasonableness",
                description="Volume must be within normal ranges",
                severity="warning",
                parameters={"volume_spike_threshold": 10.0}
            ),
            ValidationRule(
                name="data_freshness",
                description="Data must be reasonably fresh",
                severity="warning",
                parameters={"max_staleness_minutes": 15}
            ),
            ValidationRule(
                name="trading_hours_check",
                description="Data should align with trading hours",
                severity="info"
            )
        ]
        
        strict_rules = standard_rules + [
            ValidationRule(
                name="statistical_outlier",
                description="Detect statistical outliers in price/volume",
                severity="warning",
                parameters={"z_score_threshold": 3.0}
            ),
            ValidationRule(
                name="sequence_continuity",
                description="Time series data should be continuous",
                severity="warning",
                parameters={"max_gap_minutes": 60}
            ),
            ValidationRule(
                name="cross_provider_consistency",
                description="Data should be consistent across providers",
                severity="warning",
                parameters={"max_price_deviation_percent": 2.0}
            )
        ]
        
        critical_rules = strict_rules + [
            ValidationRule(
                name="sub_penny_validation",
                description="Validate sub-penny pricing rules",
                severity="error"
            ),
            ValidationRule(
                name="market_microstructure",
                description="Validate market microstructure rules",
                severity="warning"
            ),
            ValidationRule(
                name="regulatory_compliance",
                description="Ensure regulatory compliance",
                severity="error"
            )
        ]
        
        return {
            ValidationLevel.BASIC: basic_rules,
            ValidationLevel.STANDARD: standard_rules,
            ValidationLevel.STRICT: strict_rules,
            ValidationLevel.CRITICAL: critical_rules
        }

    async def validate_data_point(
        self, 
        data_point: MarketDataPoint,
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> ValidationResult:
        """
        Validate a single market data point.

        Args:
            data_point: Market data point to validate
            historical_context: Historical data for context-aware validation

        Returns:
            ValidationResult with validation details
        """
        start_time = datetime.utcnow()
        anomalies = []
        warnings = []
        errors = []
        quality_scores = []
        rules_applied = []

        # Get validation rules for current level
        rules = self.validation_rules[self.validation_level]

        for rule in rules:
            if not rule.enabled:
                continue

            rules_applied.append(rule.name)
            
            try:
                rule_result = await self._apply_validation_rule(
                    rule, data_point, historical_context
                )
                
                if rule_result:
                    score, issues = rule_result
                    quality_scores.append(score)
                    
                    for issue in issues:
                        if rule.severity == "error":
                            errors.append(f"{rule.name}: {issue}")
                        elif rule.severity == "warning":
                            warnings.append(f"{rule.name}: {issue}")
                        else:
                            # Info level
                            logger.info(f"Data validation info - {rule.name}: {issue}")
                
            except Exception as e:
                logger.error(f"Error applying validation rule {rule.name}: {e}")
                errors.append(f"Validation rule error: {rule.name}")

        # Calculate overall quality score
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Determine if data is valid
        is_valid = len(errors) == 0
        
        # Processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Log validation result
        self.audit_logger.log(
            event_type=AuditEventType.DATA_VALIDATION,
            event_data={
                "symbol": data_point.symbol,
                "validation_level": self.validation_level.value,
                "is_valid": is_valid,
                "quality_score": overall_quality,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "processing_time_ms": processing_time
            },
            severity=AuditSeverity.WARNING if not is_valid else AuditSeverity.INFO
        )
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=overall_quality,
            anomalies=anomalies,
            warnings=warnings,
            errors=errors,
            validation_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time,
            rules_applied=rules_applied
        )

    async def _apply_validation_rule(
        self,
        rule: ValidationRule,
        data_point: MarketDataPoint,
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> Optional[Tuple[float, List[str]]]:
        """Apply a specific validation rule to a data point."""
        issues = []
        score = 1.0  # Perfect score by default

        if rule.name == "price_non_negative":
            if data_point.ohlcv:
                prices = [data_point.ohlcv.open, data_point.ohlcv.high, 
                         data_point.ohlcv.low, data_point.ohlcv.close]
                for price in prices:
                    if price < 0:
                        issues.append(f"Negative price detected: {price}")
                        score = 0.0

        elif rule.name == "volume_non_negative":
            if data_point.ohlcv and data_point.ohlcv.volume < 0:
                issues.append(f"Negative volume detected: {data_point.ohlcv.volume}")
                score = 0.0

        elif rule.name == "ohlc_consistency":
            if data_point.ohlcv:
                ohlc = data_point.ohlcv
                if ohlc.high < ohlc.low:
                    issues.append("High price is less than low price")
                    score = 0.0
                if not (ohlc.low <= ohlc.open <= ohlc.high):
                    issues.append("Open price outside high-low range")
                    score = 0.5
                if not (ohlc.low <= ohlc.close <= ohlc.high):
                    issues.append("Close price outside high-low range")
                    score = 0.5

        elif rule.name == "timestamp_validity":
            max_age = timedelta(hours=rule.parameters.get("max_age_hours", 24))
            age = datetime.utcnow() - data_point.timestamp.replace(tzinfo=None)
            if age > max_age:
                issues.append(f"Data is too old: {age.total_seconds()/3600:.1f} hours")
                score = max(0.0, 1.0 - (age.total_seconds() / (max_age.total_seconds() * 2)))

        elif rule.name == "price_reasonableness":
            if data_point.ohlcv and data_point.previous_close:
                max_change = rule.parameters.get("max_change_percent", 50.0)
                change_pct = abs((data_point.ohlcv.close - data_point.previous_close) / data_point.previous_close * 100)
                if change_pct > max_change:
                    issues.append(f"Extreme price change: {change_pct:.2f}%")
                    score = max(0.1, 1.0 - (change_pct - max_change) / max_change)

        elif rule.name == "volume_reasonableness":
            if data_point.ohlcv and historical_context:
                recent_volumes = [p.ohlcv.volume for p in historical_context[-20:] if p.ohlcv]
                if recent_volumes:
                    avg_volume = statistics.mean(recent_volumes)
                    threshold = rule.parameters.get("volume_spike_threshold", 10.0)
                    if data_point.ohlcv.volume > avg_volume * threshold:
                        issues.append(f"Volume spike detected: {data_point.ohlcv.volume} vs avg {avg_volume}")
                        score = 0.7

        elif rule.name == "data_freshness":
            if data_point.source and data_point.source.timestamp_received:
                staleness = datetime.utcnow() - data_point.source.timestamp_received
                max_staleness = timedelta(minutes=rule.parameters.get("max_staleness_minutes", 15))
                if staleness > max_staleness:
                    issues.append(f"Stale data: {staleness.total_seconds()/60:.1f} minutes old")
                    score = max(0.3, 1.0 - staleness.total_seconds() / (max_staleness.total_seconds() * 2))

        elif rule.name == "statistical_outlier" and self.enable_anomaly_detection:
            if data_point.ohlcv and historical_context:
                score, outlier_issues = self._detect_statistical_outliers(
                    data_point, historical_context, rule.parameters
                )
                issues.extend(outlier_issues)

        # Add more rule implementations as needed...

        return (score, issues) if score is not None else None

    def _detect_statistical_outliers(
        self,
        data_point: MarketDataPoint,
        historical_context: List[MarketDataPoint],
        parameters: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Detect statistical outliers using z-score analysis."""
        issues = []
        score = 1.0
        
        if not data_point.ohlcv or len(historical_context) < 10:
            return score, issues
        
        threshold = parameters.get("z_score_threshold", 3.0)
        
        # Price outlier detection
        recent_closes = [p.ohlcv.close for p in historical_context[-20:] if p.ohlcv]
        if len(recent_closes) >= 5:
            price_mean = statistics.mean(recent_closes)
            price_std = statistics.stdev(recent_closes)
            
            if price_std > 0:
                z_score = abs((float(data_point.ohlcv.close) - price_mean) / price_std)
                if z_score > threshold:
                    issues.append(f"Price outlier detected: z-score {z_score:.2f}")
                    score = max(0.3, 1.0 - (z_score - threshold) / threshold)
        
        # Volume outlier detection
        recent_volumes = [float(p.ohlcv.volume) for p in historical_context[-20:] if p.ohlcv and p.ohlcv.volume > 0]
        if len(recent_volumes) >= 5:
            volume_mean = statistics.mean(recent_volumes)
            volume_std = statistics.stdev(recent_volumes)
            
            if volume_std > 0:
                z_score = abs((float(data_point.ohlcv.volume) - volume_mean) / volume_std)
                if z_score > threshold:
                    issues.append(f"Volume outlier detected: z-score {z_score:.2f}")
                    score = min(score, max(0.5, 1.0 - (z_score - threshold) / (threshold * 2)))
        
        return score, issues

    async def validate_time_series(
        self, 
        time_series: TimeSeriesData,
        cross_validation_data: Optional[List[TimeSeriesData]] = None
    ) -> ValidationResult:
        """
        Validate time series data for consistency and quality.

        Args:
            time_series: Time series data to validate
            cross_validation_data: Data from other providers for cross-validation

        Returns:
            ValidationResult with comprehensive analysis
        """
        start_time = datetime.utcnow()
        anomalies = []
        warnings = []
        errors = []
        quality_scores = []

        # Validate individual data points
        for i, data_point in enumerate(time_series.data_points):
            # Get historical context (previous 20 points)
            historical_context = time_series.data_points[max(0, i-20):i]
            
            point_result = await self.validate_data_point(data_point, historical_context)
            quality_scores.append(point_result.quality_score)
            
            # Aggregate issues
            errors.extend(point_result.errors)
            warnings.extend(point_result.warnings)
            anomalies.extend(point_result.anomalies)

        # Time series specific validations
        ts_score, ts_issues = self._validate_time_series_structure(time_series)
        quality_scores.append(ts_score)
        warnings.extend(ts_issues)

        # Cross-provider validation if enabled
        if self.enable_cross_validation and cross_validation_data:
            cross_score, cross_issues = await self._cross_validate_time_series(
                time_series, cross_validation_data
            )
            quality_scores.append(cross_score)
            warnings.extend(cross_issues)

        # Calculate overall metrics
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        is_valid = len(errors) == 0
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ValidationResult(
            is_valid=is_valid,
            quality_score=overall_quality,
            anomalies=anomalies,
            warnings=warnings,
            errors=errors,
            validation_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time,
            rules_applied=["time_series_validation"]
        )

    def _validate_time_series_structure(self, time_series: TimeSeriesData) -> Tuple[float, List[str]]:
        """Validate time series structure and continuity."""
        issues = []
        score = 1.0

        if len(time_series.data_points) < 2:
            return score, issues

        # Check for time gaps
        expected_interval_seconds = self._parse_interval_seconds(time_series.interval)
        tolerance = expected_interval_seconds * 1.5  # 50% tolerance

        for i in range(1, len(time_series.data_points)):
            prev_point = time_series.data_points[i-1]
            curr_point = time_series.data_points[i]
            
            actual_gap = (curr_point.timestamp - prev_point.timestamp).total_seconds()
            
            if actual_gap > tolerance:
                issues.append(f"Time gap detected: {actual_gap/60:.1f} minutes")
                score = min(score, 0.8)

        # Check for duplicate timestamps
        timestamps = [p.timestamp for p in time_series.data_points]
        if len(timestamps) != len(set(timestamps)):
            issues.append("Duplicate timestamps in time series")
            score = min(score, 0.6)

        # Check for proper ordering
        if timestamps != sorted(timestamps):
            issues.append("Time series not properly ordered")
            score = min(score, 0.7)

        return score, issues

    def _parse_interval_seconds(self, interval: str) -> int:
        """Parse interval string to seconds."""
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800
        }
        return interval_map.get(interval, 86400)

    async def _cross_validate_time_series(
        self,
        primary_series: TimeSeriesData,
        comparison_series: List[TimeSeriesData]
    ) -> Tuple[float, List[str]]:
        """Cross-validate time series against other providers."""
        issues = []
        score = 1.0

        # For each data point, find corresponding points in other series
        for primary_point in primary_series.data_points:
            if not primary_point.ohlcv:
                continue

            comparison_points = []
            for series in comparison_series:
                # Find closest timestamp (within 5 minutes)
                closest_point = None
                min_diff = timedelta(minutes=5)
                
                for comp_point in series.data_points:
                    if comp_point.ohlcv:
                        time_diff = abs(comp_point.timestamp - primary_point.timestamp)
                        if time_diff < min_diff:
                            min_diff = time_diff
                            closest_point = comp_point
                
                if closest_point:
                    comparison_points.append(closest_point)

            # Compare prices if we have comparison data
            if comparison_points:
                primary_close = float(primary_point.ohlcv.close)
                comparison_closes = [float(cp.ohlcv.close) for cp in comparison_points]
                
                # Check for significant deviations
                for comp_close in comparison_closes:
                    deviation_pct = abs(primary_close - comp_close) / primary_close * 100
                    if deviation_pct > 2.0:  # 2% threshold
                        issues.append(
                            f"Cross-provider price deviation: {deviation_pct:.2f}% "
                            f"at {primary_point.timestamp}"
                        )
                        score = min(score, 0.8)

        return score, issues

    def calculate_quality_metrics(self, validation_results: List[ValidationResult]) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        if not validation_results:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0)

        # Completeness: percentage of valid results
        valid_count = sum(1 for r in validation_results if r.is_valid)
        completeness = valid_count / len(validation_results)

        # Accuracy: average quality score
        accuracy = statistics.mean(r.quality_score for r in validation_results)

        # Consistency: standard deviation of quality scores (inverted)
        quality_scores = [r.quality_score for r in validation_results]
        consistency = 1.0 - (statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0)

        # Timeliness: based on processing times
        avg_processing_time = statistics.mean(r.processing_time_ms for r in validation_results)
        timeliness = max(0, 1.0 - avg_processing_time / 1000)  # Normalize to 1 second

        # Validity: percentage without errors
        error_free_count = sum(1 for r in validation_results if not r.errors)
        validity = error_free_count / len(validation_results)

        # Overall score: weighted average
        overall_score = (
            completeness * 0.25 +
            accuracy * 0.30 +
            consistency * 0.15 +
            timeliness * 0.10 +
            validity * 0.20
        )

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            overall_score=overall_score
        )

    async def generate_quality_report(
        self, 
        symbol: str, 
        time_range: Tuple[datetime, datetime],
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        metrics = self.calculate_quality_metrics(validation_results)
        
        # Aggregate issues
        all_errors = []
        all_warnings = []
        all_anomalies = []
        
        for result in validation_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_anomalies.extend(result.anomalies)

        # Issue frequency analysis
        error_frequency = {}
        warning_frequency = {}
        
        for error in all_errors:
            error_type = error.split(':')[0]
            error_frequency[error_type] = error_frequency.get(error_type, 0) + 1
            
        for warning in all_warnings:
            warning_type = warning.split(':')[0]
            warning_frequency[warning_type] = warning_frequency.get(warning_type, 0) + 1

        return {
            "symbol": symbol,
            "time_range": {
                "start": time_range[0].isoformat(),
                "end": time_range[1].isoformat()
            },
            "validation_summary": {
                "total_validations": len(validation_results),
                "valid_count": sum(1 for r in validation_results if r.is_valid),
                "error_count": len(all_errors),
                "warning_count": len(all_warnings),
                "anomaly_count": len(all_anomalies)
            },
            "quality_metrics": {
                "completeness": metrics.completeness,
                "accuracy": metrics.accuracy,
                "consistency": metrics.consistency,
                "timeliness": metrics.timeliness,
                "validity": metrics.validity,
                "overall_score": metrics.overall_score
            },
            "issue_analysis": {
                "top_errors": sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
                "top_warnings": sorted(warning_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "recommendations": self._generate_recommendations(metrics, error_frequency, warning_frequency),
            "report_timestamp": datetime.utcnow().isoformat()
        }

    def _generate_recommendations(
        self,
        metrics: DataQualityMetrics,
        error_frequency: Dict[str, int],
        warning_frequency: Dict[str, int]
    ) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []

        if metrics.completeness < 0.95:
            recommendations.append("Improve data completeness by addressing missing data points")

        if metrics.accuracy < 0.8:
            recommendations.append("Review data provider configurations to improve accuracy")

        if metrics.consistency < 0.7:
            recommendations.append("Investigate inconsistencies across data sources")

        if "price_reasonableness" in warning_frequency:
            recommendations.append("Review price validation thresholds for extreme movements")

        if "volume_reasonableness" in warning_frequency:
            recommendations.append("Analyze volume spikes for potential data quality issues")

        if metrics.timeliness < 0.8:
            recommendations.append("Optimize data processing pipeline for better timeliness")

        return recommendations


# Global validator instance
_data_validator: Optional[MarketDataValidator] = None


def get_data_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> MarketDataValidator:
    """Get the global data validator instance."""
    global _data_validator
    
    if _data_validator is None or _data_validator.validation_level != validation_level:
        _data_validator = MarketDataValidator(validation_level)
    
    return _data_validator