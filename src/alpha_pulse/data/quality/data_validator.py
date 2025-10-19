"""
Comprehensive data quality validation framework for market data.

Provides enterprise-grade data validation with:
- Multi-dimensional quality checks (completeness, accuracy, consistency, timeliness)
- Real-time validation for streaming data
- Statistical anomaly detection
- Quality scoring and trend analysis
- Automated quarantine for bad data
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np
from decimal import Decimal
from loguru import logger

from alpha_pulse.models.market_data import MarketDataPoint, TimeSeriesData, DataQuality
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class QualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QuarantineReason(Enum):
    """Reasons for data quarantine."""
    QUALITY_THRESHOLD = "quality_threshold"
    ANOMALY_DETECTED = "anomaly_detected"
    SOURCE_FAILURE = "source_failure"
    MANUAL_QUARANTINE = "manual_quarantine"
    CONSISTENCY_FAILURE = "consistency_failure"
    TIMELINESS_FAILURE = "timeliness_failure"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    dimension: QualityDimension
    severity: ValidationSeverity
    description: str
    field_name: Optional[str] = None
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityScore:
    """Quality score breakdown by dimension."""
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    timeliness: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.30,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.08,
            'uniqueness': 0.02
        }
        
        return (
            self.completeness * weights['completeness'] +
            self.accuracy * weights['accuracy'] +
            self.consistency * weights['consistency'] +
            self.timeliness * weights['timeliness'] +
            self.validity * weights['validity'] +
            self.uniqueness * weights['uniqueness']
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'timeliness': self.timeliness,
            'validity': self.validity,
            'uniqueness': self.uniqueness,
            'overall_score': self.overall_score
        }


@dataclass
class ValidationResult:
    """Result of data quality validation."""
    symbol: str
    timestamp: datetime
    quality_score: QualityScore
    issues: List[ValidationIssue]
    is_quarantined: bool = False
    quarantine_reason: Optional[QuarantineReason] = None
    validation_duration_ms: float = 0.0
    data_source: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if data passes validation."""
        return not any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return any(issue.severity in [ValidationSeverity.HIGH, ValidationSeverity.MEDIUM] for issue in self.issues)


class DataQualityValidator:
    """Comprehensive data quality validator."""

    def __init__(
        self,
        quality_thresholds: Optional[Dict[str, float]] = None,
        enable_anomaly_detection: bool = True,
        enable_quarantine: bool = True,
        historical_window_size: int = 100
    ):
        """
        Initialize data quality validator.

        Args:
            quality_thresholds: Quality score thresholds for each dimension
            enable_anomaly_detection: Enable statistical anomaly detection
            enable_quarantine: Enable automatic quarantine of bad data
            historical_window_size: Size of historical data window for comparison
        """
        self.quality_thresholds = quality_thresholds or {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.85,
            'timeliness': 0.90,
            'validity': 0.95,
            'uniqueness': 0.98,
            'overall_score': 0.80
        }
        
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_quarantine = enable_quarantine
        self.historical_window_size = historical_window_size
        
        # Historical data for validation context
        self._historical_data: Dict[str, List[MarketDataPoint]] = {}
        self._quality_history: Dict[str, List[QualityScore]] = {}
        
        # Quarantine storage
        self._quarantined_data: Dict[str, List[MarketDataPoint]] = {}
        
        # Performance metrics
        self._validation_stats = {
            'total_validations': 0,
            'quarantined_count': 0,
            'avg_validation_time_ms': 0.0,
            'critical_issues_count': 0
        }
        
        # Audit logging
        self._audit_logger = get_audit_logger()

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
            ValidationResult with quality scores and issues
        """
        start_time = datetime.utcnow()
        issues = []
        
        # Get or use provided historical context
        if historical_context is None:
            historical_context = self._get_historical_context(data_point.symbol)
        
        # Update historical data
        self._update_historical_data(data_point)
        
        # Perform quality checks
        quality_score = QualityScore()
        
        # Completeness checks
        completeness_score, completeness_issues = await self._check_completeness(data_point)
        quality_score.completeness = completeness_score
        issues.extend(completeness_issues)
        
        # Accuracy checks
        accuracy_score, accuracy_issues = await self._check_accuracy(data_point, historical_context)
        quality_score.accuracy = accuracy_score
        issues.extend(accuracy_issues)
        
        # Consistency checks
        consistency_score, consistency_issues = await self._check_consistency(data_point, historical_context)
        quality_score.consistency = consistency_score
        issues.extend(consistency_issues)
        
        # Timeliness checks
        timeliness_score, timeliness_issues = await self._check_timeliness(data_point)
        quality_score.timeliness = timeliness_score
        issues.extend(timeliness_issues)
        
        # Validity checks
        validity_score, validity_issues = await self._check_validity(data_point)
        quality_score.validity = validity_score
        issues.extend(validity_issues)
        
        # Uniqueness checks
        uniqueness_score, uniqueness_issues = await self._check_uniqueness(data_point, historical_context)
        quality_score.uniqueness = uniqueness_score
        issues.extend(uniqueness_issues)
        
        # Calculate validation duration
        validation_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine quarantine status
        is_quarantined, quarantine_reason = self._should_quarantine(quality_score, issues)
        
        # Create validation result
        result = ValidationResult(
            symbol=data_point.symbol,
            timestamp=data_point.timestamp,
            quality_score=quality_score,
            issues=issues,
            is_quarantined=is_quarantined,
            quarantine_reason=quarantine_reason,
            validation_duration_ms=validation_duration,
            data_source=data_point.source.provider if data_point.source else None
        )
        
        # Handle quarantine
        if is_quarantined and self.enable_quarantine:
            await self._quarantine_data(data_point, result)
        
        # Update statistics
        self._update_validation_stats(result)
        
        # Update quality history
        self._update_quality_history(data_point.symbol, quality_score)
        
        # Log validation result
        await self._log_validation_result(result)
        
        return result

    async def _check_completeness(self, data_point: MarketDataPoint) -> Tuple[float, List[ValidationIssue]]:
        """Check data completeness."""
        issues = []
        score = 1.0
        
        # Required fields check
        required_fields = ['symbol', 'timestamp', 'ohlcv']
        for field in required_fields:
            if not hasattr(data_point, field) or getattr(data_point, field) is None:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=ValidationSeverity.CRITICAL,
                    description=f"Missing required field: {field}",
                    field_name=field
                ))
                score -= 0.3
        
        # OHLCV completeness
        if data_point.ohlcv:
            ohlcv_fields = ['open', 'high', 'low', 'close', 'volume']
            missing_ohlcv = [field for field in ohlcv_fields 
                           if getattr(data_point.ohlcv, field, None) is None]
            
            if missing_ohlcv:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=ValidationSeverity.HIGH,
                    description=f"Missing OHLCV fields: {missing_ohlcv}",
                    field_name='ohlcv'
                ))
                score -= 0.1 * len(missing_ohlcv)
        
        # Optional but important fields
        optional_fields = ['previous_close', 'change', 'change_percent']
        missing_optional = [field for field in optional_fields 
                          if getattr(data_point, field, None) is None]
        
        if missing_optional:
            issues.append(ValidationIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=ValidationSeverity.LOW,
                description=f"Missing optional fields: {missing_optional}",
                field_name='optional_fields'
            ))
            score -= 0.05 * len(missing_optional)
        
        return max(0.0, score), issues

    async def _check_accuracy(
        self, 
        data_point: MarketDataPoint, 
        historical_context: List[MarketDataPoint]
    ) -> Tuple[float, List[ValidationIssue]]:
        """Check data accuracy."""
        issues = []
        score = 1.0
        
        if not data_point.ohlcv:
            return 0.0, [ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.CRITICAL,
                description="No OHLCV data for accuracy validation"
            )]
        
        # Price reasonableness checks
        ohlcv = data_point.ohlcv
        
        # OHLCV relationship validation
        if ohlcv.high < ohlcv.low:
            issues.append(ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.CRITICAL,
                description="High price is less than low price",
                field_name='ohlcv',
                actual_value=f"H:{ohlcv.high}, L:{ohlcv.low}"
            ))
            score = 0.0
        
        # Price within range validation
        if not (ohlcv.low <= ohlcv.open <= ohlcv.high):
            issues.append(ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.HIGH,
                description="Open price outside high-low range",
                field_name='open',
                actual_value=ohlcv.open
            ))
            score -= 0.2
        
        if not (ohlcv.low <= ohlcv.close <= ohlcv.high):
            issues.append(ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.HIGH,
                description="Close price outside high-low range",
                field_name='close',
                actual_value=ohlcv.close
            ))
            score -= 0.2
        
        # Historical price comparison
        if historical_context:
            recent_prices = [p.ohlcv.close for p in historical_context[-10:] if p.ohlcv]
            if recent_prices:
                avg_price = statistics.mean(recent_prices)
                price_deviation = abs(float(ohlcv.close) - avg_price) / avg_price * 100
                
                if price_deviation > 20:  # 20% deviation threshold
                    issues.append(ValidationIssue(
                        dimension=QualityDimension.ACCURACY,
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Price deviation {price_deviation:.2f}% from recent average",
                        field_name='close',
                        actual_value=ohlcv.close,
                        threshold=20.0
                    ))
                    score -= min(0.3, price_deviation / 100)
        
        # Volume validation
        if ohlcv.volume < 0:
            issues.append(ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.CRITICAL,
                description="Negative volume",
                field_name='volume',
                actual_value=ohlcv.volume
            ))
            score = 0.0
        
        # Zero volume check (warning for most symbols)
        if ohlcv.volume == 0 and not data_point.symbol.endswith('.TO'):  # TSX allows zero volume
            issues.append(ValidationIssue(
                dimension=QualityDimension.ACCURACY,
                severity=ValidationSeverity.LOW,
                description="Zero volume detected",
                field_name='volume',
                actual_value=ohlcv.volume
            ))
            score -= 0.1
        
        return max(0.0, score), issues

    async def _check_consistency(
        self, 
        data_point: MarketDataPoint, 
        historical_context: List[MarketDataPoint]
    ) -> Tuple[float, List[ValidationIssue]]:
        """Check data consistency."""
        issues = []
        score = 1.0
        
        if not historical_context or not data_point.ohlcv:
            return score, issues
        
        # Get the most recent historical point
        if historical_context:
            prev_point = historical_context[-1]
            if prev_point.ohlcv:
                # Price continuity check
                price_gap = abs(float(data_point.ohlcv.open) - float(prev_point.ohlcv.close))
                price_gap_percent = price_gap / float(prev_point.ohlcv.close) * 100
                
                if price_gap_percent > 10:  # 10% gap threshold
                    issues.append(ValidationIssue(
                        dimension=QualityDimension.CONSISTENCY,
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Large price gap {price_gap_percent:.2f}% from previous close",
                        field_name='open',
                        actual_value=data_point.ohlcv.open,
                        expected_value=prev_point.ohlcv.close,
                        threshold=10.0
                    ))
                    score -= min(0.3, price_gap_percent / 50)
                
                # Volume consistency
                if len(historical_context) >= 5:
                    recent_volumes = [p.ohlcv.volume for p in historical_context[-5:] if p.ohlcv]
                    if recent_volumes:
                        avg_volume = statistics.mean(recent_volumes)
                        if avg_volume > 0:
                            volume_ratio = float(data_point.ohlcv.volume) / avg_volume
                            
                            if volume_ratio > 20:  # 20x volume spike
                                issues.append(ValidationIssue(
                                    dimension=QualityDimension.CONSISTENCY,
                                    severity=ValidationSeverity.MEDIUM,
                                    description=f"Volume spike {volume_ratio:.1f}x normal",
                                    field_name='volume',
                                    actual_value=data_point.ohlcv.volume,
                                    threshold=20.0
                                ))
                                score -= 0.2
        
        # Timestamp sequence check
        if historical_context:
            last_timestamp = historical_context[-1].timestamp
            if data_point.timestamp <= last_timestamp:
                issues.append(ValidationIssue(
                    dimension=QualityDimension.CONSISTENCY,
                    severity=ValidationSeverity.HIGH,
                    description="Timestamp not in sequence",
                    field_name='timestamp',
                    actual_value=data_point.timestamp,
                    expected_value=f">{last_timestamp}"
                ))
                score -= 0.4
        
        return max(0.0, score), issues

    async def _check_timeliness(self, data_point: MarketDataPoint) -> Tuple[float, List[ValidationIssue]]:
        """Check data timeliness."""
        issues = []
        score = 1.0
        
        current_time = datetime.utcnow()
        data_age = current_time - data_point.timestamp.replace(tzinfo=None)
        
        # Real-time data should be very recent
        if data_age > timedelta(minutes=15):
            severity = ValidationSeverity.HIGH if data_age > timedelta(hours=1) else ValidationSeverity.MEDIUM
            issues.append(ValidationIssue(
                dimension=QualityDimension.TIMELINESS,
                severity=severity,
                description=f"Stale data: {data_age.total_seconds()/60:.1f} minutes old",
                field_name='timestamp',
                actual_value=data_point.timestamp,
                threshold=15.0
            ))
            
            # Score decreases with age
            minutes_old = data_age.total_seconds() / 60
            score = max(0.0, 1.0 - (minutes_old - 15) / 60)  # Linear decay after 15 minutes
        
        # Check if data source has latency information
        if data_point.source and hasattr(data_point.source, 'latency_ms') and data_point.source.latency_ms:
            if data_point.source.latency_ms > 5000:  # 5 second threshold
                issues.append(ValidationIssue(
                    dimension=QualityDimension.TIMELINESS,
                    severity=ValidationSeverity.LOW,
                    description=f"High source latency: {data_point.source.latency_ms}ms",
                    field_name='source_latency',
                    actual_value=data_point.source.latency_ms,
                    threshold=5000.0
                ))
                score -= 0.1
        
        return max(0.0, score), issues

    async def _check_validity(self, data_point: MarketDataPoint) -> Tuple[float, List[ValidationIssue]]:
        """Check data validity."""
        issues = []
        score = 1.0
        
        # Symbol format validation
        if not data_point.symbol or len(data_point.symbol) > 10:
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.HIGH,
                description="Invalid symbol format",
                field_name='symbol',
                actual_value=data_point.symbol
            ))
            score -= 0.3
        
        # Timestamp validity
        if data_point.timestamp > datetime.utcnow() + timedelta(minutes=5):
            issues.append(ValidationIssue(
                dimension=QualityDimension.VALIDITY,
                severity=ValidationSeverity.MEDIUM,
                description="Future timestamp",
                field_name='timestamp',
                actual_value=data_point.timestamp
            ))
            score -= 0.2
        
        # Price validity (must be positive)
        if data_point.ohlcv:
            prices = [data_point.ohlcv.open, data_point.ohlcv.high, 
                     data_point.ohlcv.low, data_point.ohlcv.close]
            
            for price_name, price in zip(['open', 'high', 'low', 'close'], prices):
                if price is not None and price <= 0:
                    issues.append(ValidationIssue(
                        dimension=QualityDimension.VALIDITY,
                        severity=ValidationSeverity.CRITICAL,
                        description=f"Non-positive {price_name} price",
                        field_name=price_name,
                        actual_value=price
                    ))
                    score = 0.0
        
        return max(0.0, score), issues

    async def _check_uniqueness(
        self, 
        data_point: MarketDataPoint, 
        historical_context: List[MarketDataPoint]
    ) -> Tuple[float, List[ValidationIssue]]:
        """Check data uniqueness."""
        issues = []
        score = 1.0
        
        if not historical_context:
            return score, issues
        
        # Check for exact duplicates
        for prev_point in historical_context[-10:]:  # Check last 10 points
            if (prev_point.timestamp == data_point.timestamp and 
                prev_point.ohlcv and data_point.ohlcv and
                prev_point.ohlcv.close == data_point.ohlcv.close):
                
                issues.append(ValidationIssue(
                    dimension=QualityDimension.UNIQUENESS,
                    severity=ValidationSeverity.MEDIUM,
                    description="Duplicate data point detected",
                    field_name='timestamp',
                    actual_value=data_point.timestamp
                ))
                score -= 0.5
                break
        
        return max(0.0, score), issues

    def _should_quarantine(
        self, 
        quality_score: QualityScore, 
        issues: List[ValidationIssue]
    ) -> Tuple[bool, Optional[QuarantineReason]]:
        """Determine if data should be quarantined."""
        # Critical issues always trigger quarantine
        if any(issue.severity == ValidationSeverity.CRITICAL for issue in issues):
            return True, QuarantineReason.ANOMALY_DETECTED
        
        # Overall quality threshold
        if quality_score.overall_score < self.quality_thresholds['overall_score']:
            return True, QuarantineReason.QUALITY_THRESHOLD
        
        # Individual dimension thresholds
        for dimension, score in quality_score.to_dict().items():
            if dimension != 'overall_score' and score < self.quality_thresholds.get(dimension, 0.5):
                if dimension == 'timeliness':
                    return True, QuarantineReason.TIMELINESS_FAILURE
                elif dimension == 'consistency':
                    return True, QuarantineReason.CONSISTENCY_FAILURE
                else:
                    return True, QuarantineReason.QUALITY_THRESHOLD
        
        return False, None

    async def _quarantine_data(
        self, 
        data_point: MarketDataPoint, 
        validation_result: ValidationResult
    ) -> None:
        """Quarantine bad data."""
        symbol = data_point.symbol
        
        if symbol not in self._quarantined_data:
            self._quarantined_data[symbol] = []
        
        self._quarantined_data[symbol].append(data_point)
        
        # Log quarantine action
        self._audit_logger.log(
            event_type=AuditEventType.DATA_QUARANTINE,
            event_data={
                "symbol": symbol,
                "timestamp": data_point.timestamp.isoformat(),
                "reason": validation_result.quarantine_reason.value if validation_result.quarantine_reason else "unknown",
                "quality_score": validation_result.quality_score.overall_score,
                "critical_issues": [issue.description for issue in validation_result.issues 
                                  if issue.severity == ValidationSeverity.CRITICAL]
            },
            severity=AuditSeverity.WARNING
        )

    def _get_historical_context(self, symbol: str) -> List[MarketDataPoint]:
        """Get historical data context for validation."""
        return self._historical_data.get(symbol, [])[-self.historical_window_size:]

    def _update_historical_data(self, data_point: MarketDataPoint) -> None:
        """Update historical data storage."""
        symbol = data_point.symbol
        
        if symbol not in self._historical_data:
            self._historical_data[symbol] = []
        
        self._historical_data[symbol].append(data_point)
        
        # Keep only recent history
        if len(self._historical_data[symbol]) > self.historical_window_size:
            self._historical_data[symbol] = self._historical_data[symbol][-self.historical_window_size:]

    def _update_quality_history(self, symbol: str, quality_score: QualityScore) -> None:
        """Update quality score history."""
        if symbol not in self._quality_history:
            self._quality_history[symbol] = []
        
        self._quality_history[symbol].append(quality_score)
        
        # Keep only recent history
        if len(self._quality_history[symbol]) > self.historical_window_size:
            self._quality_history[symbol] = self._quality_history[symbol][-self.historical_window_size:]

    def _update_validation_stats(self, result: ValidationResult) -> None:
        """Update validation statistics."""
        self._validation_stats['total_validations'] += 1
        
        if result.is_quarantined:
            self._validation_stats['quarantined_count'] += 1
        
        if any(issue.severity == ValidationSeverity.CRITICAL for issue in result.issues):
            self._validation_stats['critical_issues_count'] += 1
        
        # Update average validation time
        total_time = (
            self._validation_stats['avg_validation_time_ms'] * 
            (self._validation_stats['total_validations'] - 1) + 
            result.validation_duration_ms
        )
        self._validation_stats['avg_validation_time_ms'] = total_time / self._validation_stats['total_validations']

    async def _log_validation_result(self, result: ValidationResult) -> None:
        """Log validation result for monitoring."""
        # Only log significant issues to avoid noise
        if result.is_quarantined or result.has_warnings:
            self._audit_logger.log(
                event_type=AuditEventType.DATA_VALIDATION,
                event_data={
                    "symbol": result.symbol,
                    "quality_score": result.quality_score.to_dict(),
                    "is_quarantined": result.is_quarantined,
                    "issue_count": len(result.issues),
                    "validation_duration_ms": result.validation_duration_ms,
                    "data_source": result.data_source
                },
                severity=AuditSeverity.WARNING if result.is_quarantined else AuditSeverity.INFO
            )

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = self._validation_stats['total_validations']
        
        return {
            'total_validations': total_validations,
            'quarantine_rate': (
                self._validation_stats['quarantined_count'] / total_validations 
                if total_validations > 0 else 0
            ),
            'critical_issue_rate': (
                self._validation_stats['critical_issues_count'] / total_validations 
                if total_validations > 0 else 0
            ),
            'avg_validation_time_ms': self._validation_stats['avg_validation_time_ms'],
            'quarantined_symbols': list(self._quarantined_data.keys()),
            'quarantined_data_points': sum(len(points) for points in self._quarantined_data.values())
        }

    def get_quality_trends(self, symbol: str, days: int = 7) -> Dict[str, List[float]]:
        """Get quality trends for a symbol."""
        if symbol not in self._quality_history:
            return {}
        
        # Get recent quality scores
        recent_scores = self._quality_history[symbol][-days * 24:]  # Assuming hourly data
        
        trends = {
            'timestamps': [i for i in range(len(recent_scores))],
            'overall_scores': [score.overall_score for score in recent_scores],
            'completeness': [score.completeness for score in recent_scores],
            'accuracy': [score.accuracy for score in recent_scores],
            'consistency': [score.consistency for score in recent_scores],
            'timeliness': [score.timeliness for score in recent_scores]
        }
        
        return trends

    async def validate_time_series(self, time_series: TimeSeriesData) -> List[ValidationResult]:
        """Validate an entire time series."""
        results = []
        
        for i, data_point in enumerate(time_series.data_points):
            # Use previous points as historical context
            historical_context = time_series.data_points[:i] if i > 0 else []
            
            result = await self.validate_data_point(data_point, historical_context)
            results.append(result)
        
        return results

    def clear_quarantine(self, symbol: Optional[str] = None) -> None:
        """Clear quarantined data."""
        if symbol:
            if symbol in self._quarantined_data:
                del self._quarantined_data[symbol]
        else:
            self._quarantined_data.clear()

    def get_quarantined_data(self, symbol: Optional[str] = None) -> Dict[str, List[MarketDataPoint]]:
        """Get quarantined data."""
        if symbol:
            return {symbol: self._quarantined_data.get(symbol, [])}
        return self._quarantined_data.copy()


# Global validator instance
_data_quality_validator: Optional[DataQualityValidator] = None


def get_data_validator(
    quality_thresholds: Optional[Dict[str, float]] = None,
    enable_anomaly_detection: bool = True,
    enable_quarantine: bool = True,
    historical_window_size: int = 100
) -> DataQualityValidator:
    """
    Get the global data quality validator instance.

    Args:
        quality_thresholds: Quality score thresholds for each dimension
        enable_anomaly_detection: Enable statistical anomaly detection
        enable_quarantine: Enable automatic quarantine of bad data
        historical_window_size: Size of historical data window for comparison

    Returns:
        DataQualityValidator instance
    """
    global _data_quality_validator

    if _data_quality_validator is None:
        _data_quality_validator = DataQualityValidator(
            quality_thresholds=quality_thresholds,
            enable_anomaly_detection=enable_anomaly_detection,
            enable_quarantine=enable_quarantine,
            historical_window_size=historical_window_size
        )

    return _data_quality_validator