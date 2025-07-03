"""
Specific quality check implementations for market data validation.

Provides:
- Field-level validation checks
- Cross-field consistency checks
- Statistical validation checks
- Market-specific validation rules
- Custom validation extensions
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import statistics
from decimal import Decimal, InvalidOperation
from loguru import logger

from alpha_pulse.models.market_data import MarketDataPoint, OHLCV, AssetClass


class CheckResult(Enum):
    """Result of a quality check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    check_name: str
    result: CheckResult
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    timestamp: datetime


class MarketDataQualityChecks:
    """Collection of quality checks for market data."""
    
    def __init__(self):
        self.check_registry: Dict[str, Callable] = self._register_checks()
    
    def _register_checks(self) -> Dict[str, Callable]:
        """Register all available quality checks."""
        return {
            # Completeness checks
            "required_fields_check": self.check_required_fields,
            "ohlcv_completeness": self.check_ohlcv_completeness,
            "metadata_completeness": self.check_metadata_completeness,
            
            # Accuracy checks
            "price_reasonableness": self.check_price_reasonableness,
            "ohlc_consistency": self.check_ohlc_consistency,
            "volume_validation": self.check_volume_validation,
            "price_precision": self.check_price_precision,
            "spread_validation": self.check_spread_validation,
            
            # Consistency checks
            "price_continuity": self.check_price_continuity,
            "timestamp_sequence": self.check_timestamp_sequence,
            "volume_consistency": self.check_volume_consistency,
            
            # Timeliness checks
            "data_freshness": self.check_data_freshness,
            "processing_latency": self.check_processing_latency,
            
            # Validity checks
            "symbol_format": self.check_symbol_format,
            "price_validity": self.check_price_validity,
            "timestamp_validity": self.check_timestamp_validity,
            
            # Uniqueness checks
            "duplicate_detection": self.check_duplicate_detection,
            
            # Market-specific checks
            "market_hours_check": self.check_market_hours,
            "corporate_actions_check": self.check_corporate_actions,
            "trading_halt_check": self.check_trading_halt
        }
    
    def run_check(
        self, 
        check_name: str, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Run a specific quality check."""
        if check_name not in self.check_registry:
            return QualityCheckResult(
                check_name=check_name,
                result=CheckResult.SKIP,
                score=0.0,
                message=f"Unknown check: {check_name}",
                details={},
                timestamp=datetime.utcnow()
            )
        
        try:
            check_func = self.check_registry[check_name]
            return check_func(data_point, parameters, historical_context)
        except Exception as e:
            logger.error(f"Error running check {check_name}: {e}")
            return QualityCheckResult(
                check_name=check_name,
                result=CheckResult.FAIL,
                score=0.0,
                message=f"Check failed with error: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    # Completeness Checks
    
    def check_required_fields(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check if all required fields are present."""
        required_fields = parameters.get("required_fields", ["symbol", "timestamp", "ohlcv"])
        missing_fields = []
        
        for field in required_fields:
            if field == "symbol" and not data_point.symbol:
                missing_fields.append("symbol")
            elif field == "timestamp" and not data_point.timestamp:
                missing_fields.append("timestamp")
            elif field == "ohlcv" and not data_point.ohlcv:
                missing_fields.append("ohlcv")
            elif field == "metadata" and not data_point.metadata:
                missing_fields.append("metadata")
        
        if missing_fields:
            return QualityCheckResult(
                check_name="required_fields_check",
                result=CheckResult.FAIL,
                score=1.0 - (len(missing_fields) / len(required_fields)),
                message=f"Missing required fields: {', '.join(missing_fields)}",
                details={"missing_fields": missing_fields},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="required_fields_check",
            result=CheckResult.PASS,
            score=1.0,
            message="All required fields present",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_ohlcv_completeness(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check OHLCV data completeness."""
        if not data_point.ohlcv:
            return QualityCheckResult(
                check_name="ohlcv_completeness",
                result=CheckResult.FAIL,
                score=0.0,
                message="OHLCV data missing",
                details={},
                timestamp=datetime.utcnow()
            )
        
        required_fields = parameters.get("required_ohlcv_fields", ["open", "high", "low", "close", "volume"])
        missing_fields = []
        ohlcv = data_point.ohlcv
        
        if "open" in required_fields and (ohlcv.open is None or ohlcv.open <= 0):
            missing_fields.append("open")
        if "high" in required_fields and (ohlcv.high is None or ohlcv.high <= 0):
            missing_fields.append("high")
        if "low" in required_fields and (ohlcv.low is None or ohlcv.low <= 0):
            missing_fields.append("low")
        if "close" in required_fields and (ohlcv.close is None or ohlcv.close <= 0):
            missing_fields.append("close")
        if "volume" in required_fields and (ohlcv.volume is None or ohlcv.volume < 0):
            missing_fields.append("volume")
        
        if missing_fields:
            return QualityCheckResult(
                check_name="ohlcv_completeness",
                result=CheckResult.FAIL,
                score=1.0 - (len(missing_fields) / len(required_fields)),
                message=f"Missing or invalid OHLCV fields: {', '.join(missing_fields)}",
                details={"missing_fields": missing_fields},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="ohlcv_completeness",
            result=CheckResult.PASS,
            score=1.0,
            message="OHLCV data complete",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_metadata_completeness(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check metadata completeness."""
        required_metadata = parameters.get("required_metadata", [])
        
        if not required_metadata:
            return QualityCheckResult(
                check_name="metadata_completeness",
                result=CheckResult.PASS,
                score=1.0,
                message="No metadata requirements",
                details={},
                timestamp=datetime.utcnow()
            )
        
        if not data_point.metadata:
            return QualityCheckResult(
                check_name="metadata_completeness",
                result=CheckResult.FAIL,
                score=0.0,
                message="Metadata missing",
                details={"required": required_metadata},
                timestamp=datetime.utcnow()
            )
        
        missing_metadata = [
            field for field in required_metadata 
            if field not in data_point.metadata
        ]
        
        if missing_metadata:
            return QualityCheckResult(
                check_name="metadata_completeness",
                result=CheckResult.WARNING,
                score=1.0 - (len(missing_metadata) / len(required_metadata)),
                message=f"Missing metadata fields: {', '.join(missing_metadata)}",
                details={"missing_fields": missing_metadata},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="metadata_completeness",
            result=CheckResult.PASS,
            score=1.0,
            message="All metadata present",
            details={},
            timestamp=datetime.utcnow()
        )
    
    # Accuracy Checks
    
    def check_price_reasonableness(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check if price changes are reasonable."""
        if not data_point.ohlcv or not historical_context:
            return QualityCheckResult(
                check_name="price_reasonableness",
                result=CheckResult.SKIP,
                score=1.0,
                message="Insufficient data for price reasonableness check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        max_change_percent = parameters.get("max_change_percent", 20.0)
        
        # Get previous close price
        prev_data_points = [dp for dp in historical_context if dp.ohlcv]
        if not prev_data_points:
            return QualityCheckResult(
                check_name="price_reasonableness",
                result=CheckResult.SKIP,
                score=1.0,
                message="No historical prices available",
                details={},
                timestamp=datetime.utcnow()
            )
        
        prev_close = float(prev_data_points[-1].ohlcv.close)
        current_prices = [
            float(data_point.ohlcv.open),
            float(data_point.ohlcv.high),
            float(data_point.ohlcv.low),
            float(data_point.ohlcv.close)
        ]
        
        unreasonable_changes = []
        for price_type, price in zip(["open", "high", "low", "close"], current_prices):
            if prev_close > 0:
                change_percent = abs((price - prev_close) / prev_close * 100)
                if change_percent > max_change_percent:
                    unreasonable_changes.append({
                        "price_type": price_type,
                        "price": price,
                        "change_percent": change_percent
                    })
        
        if unreasonable_changes:
            return QualityCheckResult(
                check_name="price_reasonableness",
                result=CheckResult.WARNING,
                score=1.0 - (len(unreasonable_changes) / 4),
                message=f"Unreasonable price changes detected",
                details={
                    "unreasonable_changes": unreasonable_changes,
                    "prev_close": prev_close,
                    "max_change_percent": max_change_percent
                },
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="price_reasonableness",
            result=CheckResult.PASS,
            score=1.0,
            message="Price changes within reasonable bounds",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_ohlc_consistency(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check OHLC price relationships."""
        if not data_point.ohlcv:
            return QualityCheckResult(
                check_name="ohlc_consistency",
                result=CheckResult.SKIP,
                score=1.0,
                message="No OHLCV data to check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        ohlcv = data_point.ohlcv
        violations = []
        
        # Check: Low <= Open <= High
        if not (ohlcv.low <= ohlcv.open <= ohlcv.high):
            violations.append(f"Open {ohlcv.open} not between Low {ohlcv.low} and High {ohlcv.high}")
        
        # Check: Low <= Close <= High
        if not (ohlcv.low <= ohlcv.close <= ohlcv.high):
            violations.append(f"Close {ohlcv.close} not between Low {ohlcv.low} and High {ohlcv.high}")
        
        # Check: Low <= High
        if ohlcv.low > ohlcv.high:
            violations.append(f"Low {ohlcv.low} greater than High {ohlcv.high}")
        
        if violations:
            return QualityCheckResult(
                check_name="ohlc_consistency",
                result=CheckResult.FAIL,
                score=1.0 - (len(violations) / 3),
                message="OHLC consistency violations detected",
                details={"violations": violations},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="ohlc_consistency",
            result=CheckResult.PASS,
            score=1.0,
            message="OHLC relationships valid",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_volume_validation(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Validate volume data."""
        if not data_point.ohlcv:
            return QualityCheckResult(
                check_name="volume_validation",
                result=CheckResult.SKIP,
                score=1.0,
                message="No OHLCV data to check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        min_volume = parameters.get("min_volume", 0)
        volume_spike_threshold = parameters.get("volume_spike_threshold", 20.0)  # 20x average
        
        volume = float(data_point.ohlcv.volume)
        
        # Check minimum volume
        if volume < min_volume:
            return QualityCheckResult(
                check_name="volume_validation",
                result=CheckResult.WARNING,
                score=0.5,
                message=f"Volume {volume} below minimum {min_volume}",
                details={"volume": volume, "min_volume": min_volume},
                timestamp=datetime.utcnow()
            )
        
        # Check for volume spikes if historical data available
        if historical_context and len(historical_context) >= 10:
            hist_volumes = [
                float(dp.ohlcv.volume) 
                for dp in historical_context[-20:] 
                if dp.ohlcv and dp.ohlcv.volume > 0
            ]
            
            if hist_volumes:
                avg_volume = statistics.mean(hist_volumes)
                if avg_volume > 0 and volume > avg_volume * volume_spike_threshold:
                    return QualityCheckResult(
                        check_name="volume_validation",
                        result=CheckResult.WARNING,
                        score=0.7,
                        message=f"Volume spike detected: {volume / avg_volume:.1f}x average",
                        details={
                            "volume": volume,
                            "avg_volume": avg_volume,
                            "spike_ratio": volume / avg_volume
                        },
                        timestamp=datetime.utcnow()
                    )
        
        return QualityCheckResult(
            check_name="volume_validation",
            result=CheckResult.PASS,
            score=1.0,
            message="Volume validation passed",
            details={"volume": volume},
            timestamp=datetime.utcnow()
        )
    
    def check_price_precision(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check price precision based on asset class."""
        if not data_point.ohlcv:
            return QualityCheckResult(
                check_name="price_precision",
                result=CheckResult.SKIP,
                score=1.0,
                message="No OHLCV data to check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        # Get precision requirements
        asset_class = data_point.metadata.get("asset_class") if data_point.metadata else None
        precision_map = parameters.get("precision_map", {
            AssetClass.EQUITY.value: 2,
            AssetClass.OPTION.value: 2,
            AssetClass.CRYPTOCURRENCY.value: 8,
            AssetClass.FOREX.value: 5
        })
        
        expected_precision = precision_map.get(asset_class, 2)
        
        # Check precision of all price fields
        prices = {
            "open": data_point.ohlcv.open,
            "high": data_point.ohlcv.high,
            "low": data_point.ohlcv.low,
            "close": data_point.ohlcv.close
        }
        
        precision_issues = []
        for price_type, price in prices.items():
            try:
                decimal_price = Decimal(str(price))
                # Get number of decimal places
                decimal_places = abs(decimal_price.as_tuple().exponent)
                if decimal_places > expected_precision:
                    precision_issues.append({
                        "price_type": price_type,
                        "price": float(price),
                        "decimal_places": decimal_places,
                        "expected": expected_precision
                    })
            except (InvalidOperation, ValueError):
                precision_issues.append({
                    "price_type": price_type,
                    "price": price,
                    "error": "Invalid decimal format"
                })
        
        if precision_issues:
            return QualityCheckResult(
                check_name="price_precision",
                result=CheckResult.WARNING,
                score=1.0 - (len(precision_issues) / 4),
                message="Price precision issues detected",
                details={
                    "issues": precision_issues,
                    "asset_class": asset_class,
                    "expected_precision": expected_precision
                },
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="price_precision",
            result=CheckResult.PASS,
            score=1.0,
            message="Price precision correct",
            details={"asset_class": asset_class, "precision": expected_precision},
            timestamp=datetime.utcnow()
        )
    
    def check_spread_validation(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Validate bid-ask spread if available."""
        bid = data_point.bid if hasattr(data_point, 'bid') else None
        ask = data_point.ask if hasattr(data_point, 'ask') else None
        
        if bid is None or ask is None:
            return QualityCheckResult(
                check_name="spread_validation",
                result=CheckResult.SKIP,
                score=1.0,
                message="No bid/ask data available",
                details={},
                timestamp=datetime.utcnow()
            )
        
        max_spread_percent = parameters.get("max_spread_percent", 5.0)
        
        if bid <= 0 or ask <= 0:
            return QualityCheckResult(
                check_name="spread_validation",
                result=CheckResult.FAIL,
                score=0.0,
                message="Invalid bid or ask price",
                details={"bid": bid, "ask": ask},
                timestamp=datetime.utcnow()
            )
        
        if bid >= ask:
            return QualityCheckResult(
                check_name="spread_validation",
                result=CheckResult.FAIL,
                score=0.0,
                message="Bid price greater than or equal to ask price",
                details={"bid": bid, "ask": ask},
                timestamp=datetime.utcnow()
            )
        
        spread_percent = (ask - bid) / bid * 100
        
        if spread_percent > max_spread_percent:
            return QualityCheckResult(
                check_name="spread_validation",
                result=CheckResult.WARNING,
                score=max(0.5, 1.0 - (spread_percent / max_spread_percent / 2)),
                message=f"Wide spread detected: {spread_percent:.2f}%",
                details={
                    "bid": bid,
                    "ask": ask,
                    "spread_percent": spread_percent,
                    "max_spread_percent": max_spread_percent
                },
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="spread_validation",
            result=CheckResult.PASS,
            score=1.0,
            message="Spread within acceptable range",
            details={"bid": bid, "ask": ask, "spread_percent": spread_percent},
            timestamp=datetime.utcnow()
        )
    
    # Consistency Checks
    
    def check_price_continuity(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check price continuity between data points."""
        if not data_point.ohlcv or not historical_context:
            return QualityCheckResult(
                check_name="price_continuity",
                result=CheckResult.SKIP,
                score=1.0,
                message="Insufficient data for continuity check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        max_gap_percent = parameters.get("max_gap_percent", 10.0)
        
        # Get previous data point with OHLCV
        prev_points = [dp for dp in historical_context if dp.ohlcv]
        if not prev_points:
            return QualityCheckResult(
                check_name="price_continuity",
                result=CheckResult.SKIP,
                score=1.0,
                message="No previous OHLCV data",
                details={},
                timestamp=datetime.utcnow()
            )
        
        prev_close = float(prev_points[-1].ohlcv.close)
        current_open = float(data_point.ohlcv.open)
        
        if prev_close > 0:
            gap_percent = abs((current_open - prev_close) / prev_close * 100)
            
            if gap_percent > max_gap_percent:
                return QualityCheckResult(
                    check_name="price_continuity",
                    result=CheckResult.WARNING,
                    score=max(0.5, 1.0 - (gap_percent / max_gap_percent / 2)),
                    message=f"Price gap detected: {gap_percent:.2f}%",
                    details={
                        "prev_close": prev_close,
                        "current_open": current_open,
                        "gap_percent": gap_percent,
                        "max_gap_percent": max_gap_percent
                    },
                    timestamp=datetime.utcnow()
                )
        
        return QualityCheckResult(
            check_name="price_continuity",
            result=CheckResult.PASS,
            score=1.0,
            message="Price continuity maintained",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_timestamp_sequence(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check if timestamps are in proper sequence."""
        if not historical_context:
            return QualityCheckResult(
                check_name="timestamp_sequence",
                result=CheckResult.SKIP,
                score=1.0,
                message="No historical data for sequence check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        # Check if current timestamp is after previous
        prev_timestamps = [dp.timestamp for dp in historical_context]
        if prev_timestamps and data_point.timestamp <= prev_timestamps[-1]:
            return QualityCheckResult(
                check_name="timestamp_sequence",
                result=CheckResult.FAIL,
                score=0.0,
                message="Timestamp out of sequence",
                details={
                    "current_timestamp": data_point.timestamp.isoformat(),
                    "previous_timestamp": prev_timestamps[-1].isoformat()
                },
                timestamp=datetime.utcnow()
            )
        
        # Check for duplicate timestamps in recent history
        recent_timestamps = prev_timestamps[-10:] + [data_point.timestamp]
        if len(recent_timestamps) != len(set(recent_timestamps)):
            return QualityCheckResult(
                check_name="timestamp_sequence",
                result=CheckResult.WARNING,
                score=0.5,
                message="Duplicate timestamps detected",
                details={},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="timestamp_sequence",
            result=CheckResult.PASS,
            score=1.0,
            message="Timestamp sequence valid",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_volume_consistency(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check volume consistency patterns."""
        if not data_point.ohlcv or not historical_context:
            return QualityCheckResult(
                check_name="volume_consistency",
                result=CheckResult.SKIP,
                score=1.0,
                message="Insufficient data for volume consistency check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        # Check if volume is zero but price changed
        if data_point.ohlcv.volume == 0:
            price_range = float(data_point.ohlcv.high - data_point.ohlcv.low)
            if price_range > 0:
                return QualityCheckResult(
                    check_name="volume_consistency",
                    result=CheckResult.WARNING,
                    score=0.7,
                    message="Price movement with zero volume",
                    details={
                        "volume": 0,
                        "price_range": price_range,
                        "high": float(data_point.ohlcv.high),
                        "low": float(data_point.ohlcv.low)
                    },
                    timestamp=datetime.utcnow()
                )
        
        return QualityCheckResult(
            check_name="volume_consistency",
            result=CheckResult.PASS,
            score=1.0,
            message="Volume consistency check passed",
            details={},
            timestamp=datetime.utcnow()
        )
    
    # Timeliness Checks
    
    def check_data_freshness(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check if data is fresh enough."""
        max_age_minutes = parameters.get("max_age_minutes", 15)
        
        current_time = datetime.utcnow()
        data_age = current_time - data_point.timestamp
        age_minutes = data_age.total_seconds() / 60
        
        if age_minutes > max_age_minutes:
            return QualityCheckResult(
                check_name="data_freshness",
                result=CheckResult.WARNING,
                score=max(0.5, 1.0 - (age_minutes / max_age_minutes / 2)),
                message=f"Data is {age_minutes:.1f} minutes old",
                details={
                    "data_timestamp": data_point.timestamp.isoformat(),
                    "current_time": current_time.isoformat(),
                    "age_minutes": age_minutes,
                    "max_age_minutes": max_age_minutes
                },
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="data_freshness",
            result=CheckResult.PASS,
            score=1.0,
            message="Data is fresh",
            details={"age_minutes": age_minutes},
            timestamp=datetime.utcnow()
        )
    
    def check_processing_latency(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check processing latency if available."""
        max_latency_ms = parameters.get("max_latency_ms", 5000)
        
        # Check if latency metadata is available
        if not data_point.metadata or "processing_latency_ms" not in data_point.metadata:
            return QualityCheckResult(
                check_name="processing_latency",
                result=CheckResult.SKIP,
                score=1.0,
                message="No latency data available",
                details={},
                timestamp=datetime.utcnow()
            )
        
        latency_ms = data_point.metadata["processing_latency_ms"]
        
        if latency_ms > max_latency_ms:
            return QualityCheckResult(
                check_name="processing_latency",
                result=CheckResult.WARNING,
                score=max(0.5, 1.0 - (latency_ms / max_latency_ms / 2)),
                message=f"High processing latency: {latency_ms}ms",
                details={
                    "latency_ms": latency_ms,
                    "max_latency_ms": max_latency_ms
                },
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="processing_latency",
            result=CheckResult.PASS,
            score=1.0,
            message="Processing latency acceptable",
            details={"latency_ms": latency_ms},
            timestamp=datetime.utcnow()
        )
    
    # Validity Checks
    
    def check_symbol_format(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check symbol format validity."""
        max_length = parameters.get("max_length", 10)
        pattern = parameters.get("pattern", r"^[A-Z0-9\-\.]+$")
        
        if not data_point.symbol:
            return QualityCheckResult(
                check_name="symbol_format",
                result=CheckResult.FAIL,
                score=0.0,
                message="Symbol is empty",
                details={},
                timestamp=datetime.utcnow()
            )
        
        if len(data_point.symbol) > max_length:
            return QualityCheckResult(
                check_name="symbol_format",
                result=CheckResult.WARNING,
                score=0.7,
                message=f"Symbol exceeds maximum length of {max_length}",
                details={"symbol": data_point.symbol, "length": len(data_point.symbol)},
                timestamp=datetime.utcnow()
            )
        
        if pattern and not re.match(pattern, data_point.symbol):
            return QualityCheckResult(
                check_name="symbol_format",
                result=CheckResult.WARNING,
                score=0.8,
                message="Symbol format does not match expected pattern",
                details={"symbol": data_point.symbol, "pattern": pattern},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="symbol_format",
            result=CheckResult.PASS,
            score=1.0,
            message="Symbol format valid",
            details={"symbol": data_point.symbol},
            timestamp=datetime.utcnow()
        )
    
    def check_price_validity(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check if prices are within valid ranges."""
        if not data_point.ohlcv:
            return QualityCheckResult(
                check_name="price_validity",
                result=CheckResult.SKIP,
                score=1.0,
                message="No OHLCV data to validate",
                details={},
                timestamp=datetime.utcnow()
            )
        
        min_price = parameters.get("min_price", 0.01)
        max_price = parameters.get("max_price", 100000)
        
        prices = {
            "open": float(data_point.ohlcv.open),
            "high": float(data_point.ohlcv.high),
            "low": float(data_point.ohlcv.low),
            "close": float(data_point.ohlcv.close)
        }
        
        invalid_prices = []
        for price_type, price in prices.items():
            if price < min_price or price > max_price:
                invalid_prices.append({
                    "price_type": price_type,
                    "price": price,
                    "min_price": min_price,
                    "max_price": max_price
                })
        
        if invalid_prices:
            return QualityCheckResult(
                check_name="price_validity",
                result=CheckResult.FAIL,
                score=1.0 - (len(invalid_prices) / 4),
                message="Invalid prices detected",
                details={"invalid_prices": invalid_prices},
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="price_validity",
            result=CheckResult.PASS,
            score=1.0,
            message="All prices within valid range",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_timestamp_validity(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check timestamp validity."""
        # Check if timestamp is not in the future
        current_time = datetime.utcnow()
        if data_point.timestamp > current_time:
            return QualityCheckResult(
                check_name="timestamp_validity",
                result=CheckResult.FAIL,
                score=0.0,
                message="Timestamp is in the future",
                details={
                    "data_timestamp": data_point.timestamp.isoformat(),
                    "current_time": current_time.isoformat()
                },
                timestamp=datetime.utcnow()
            )
        
        # Check if timestamp is not too old
        max_age_days = parameters.get("max_age_days", 365)
        min_timestamp = current_time - timedelta(days=max_age_days)
        
        if data_point.timestamp < min_timestamp:
            return QualityCheckResult(
                check_name="timestamp_validity",
                result=CheckResult.WARNING,
                score=0.7,
                message=f"Timestamp older than {max_age_days} days",
                details={
                    "data_timestamp": data_point.timestamp.isoformat(),
                    "min_timestamp": min_timestamp.isoformat()
                },
                timestamp=datetime.utcnow()
            )
        
        return QualityCheckResult(
            check_name="timestamp_validity",
            result=CheckResult.PASS,
            score=1.0,
            message="Timestamp valid",
            details={},
            timestamp=datetime.utcnow()
        )
    
    # Uniqueness Checks
    
    def check_duplicate_detection(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Detect duplicate data points."""
        if not historical_context:
            return QualityCheckResult(
                check_name="duplicate_detection",
                result=CheckResult.SKIP,
                score=1.0,
                message="No historical data for duplicate check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        check_window = parameters.get("check_window", 10)
        recent_points = historical_context[-check_window:] if len(historical_context) > check_window else historical_context
        
        # Check for exact duplicates (same timestamp and symbol)
        for hist_point in recent_points:
            if (hist_point.timestamp == data_point.timestamp and 
                hist_point.symbol == data_point.symbol):
                return QualityCheckResult(
                    check_name="duplicate_detection",
                    result=CheckResult.FAIL,
                    score=0.0,
                    message="Duplicate data point detected",
                    details={
                        "timestamp": data_point.timestamp.isoformat(),
                        "symbol": data_point.symbol
                    },
                    timestamp=datetime.utcnow()
                )
        
        # Check for near-duplicates (same OHLCV values)
        if data_point.ohlcv:
            for hist_point in recent_points:
                if hist_point.ohlcv and self._are_ohlcv_equal(data_point.ohlcv, hist_point.ohlcv):
                    time_diff = abs((data_point.timestamp - hist_point.timestamp).total_seconds())
                    if time_diff < 60:  # Within 1 minute
                        return QualityCheckResult(
                            check_name="duplicate_detection",
                            result=CheckResult.WARNING,
                            score=0.5,
                            message="Near-duplicate OHLCV values detected",
                            details={
                                "time_diff_seconds": time_diff,
                                "timestamps": [
                                    data_point.timestamp.isoformat(),
                                    hist_point.timestamp.isoformat()
                                ]
                            },
                            timestamp=datetime.utcnow()
                        )
        
        return QualityCheckResult(
            check_name="duplicate_detection",
            result=CheckResult.PASS,
            score=1.0,
            message="No duplicates detected",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def _are_ohlcv_equal(self, ohlcv1: OHLCV, ohlcv2: OHLCV) -> bool:
        """Check if two OHLCV objects have equal values."""
        return (
            ohlcv1.open == ohlcv2.open and
            ohlcv1.high == ohlcv2.high and
            ohlcv1.low == ohlcv2.low and
            ohlcv1.close == ohlcv2.close and
            ohlcv1.volume == ohlcv2.volume
        )
    
    # Market-specific Checks
    
    def check_market_hours(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check if data aligns with market hours."""
        check_trading_hours = parameters.get("check_trading_hours", True)
        
        if not check_trading_hours:
            return QualityCheckResult(
                check_name="market_hours_check",
                result=CheckResult.SKIP,
                score=1.0,
                message="Market hours check disabled",
                details={},
                timestamp=datetime.utcnow()
            )
        
        # Get asset class
        asset_class = data_point.metadata.get("asset_class") if data_point.metadata else None
        
        # Skip for 24/7 markets
        if asset_class == AssetClass.CRYPTOCURRENCY.value:
            return QualityCheckResult(
                check_name="market_hours_check",
                result=CheckResult.PASS,
                score=1.0,
                message="24/7 market, no hours restriction",
                details={},
                timestamp=datetime.utcnow()
            )
        
        # Check for US equity market hours (simplified)
        if asset_class == AssetClass.EQUITY.value:
            hour = data_point.timestamp.hour
            minute = data_point.timestamp.minute
            weekday = data_point.timestamp.weekday()
            
            # Check if weekend
            if weekday >= 5:  # Saturday or Sunday
                return QualityCheckResult(
                    check_name="market_hours_check",
                    result=CheckResult.WARNING,
                    score=0.5,
                    message="Data point on weekend",
                    details={"weekday": weekday, "timestamp": data_point.timestamp.isoformat()},
                    timestamp=datetime.utcnow()
                )
            
            # Check if outside regular trading hours (9:30 AM - 4:00 PM ET)
            # Simplified check without timezone handling
            if hour < 9 or hour >= 16 or (hour == 9 and minute < 30):
                return QualityCheckResult(
                    check_name="market_hours_check",
                    result=CheckResult.WARNING,
                    score=0.7,
                    message="Data point outside regular market hours",
                    details={
                        "hour": hour,
                        "minute": minute,
                        "timestamp": data_point.timestamp.isoformat()
                    },
                    timestamp=datetime.utcnow()
                )
        
        return QualityCheckResult(
            check_name="market_hours_check",
            result=CheckResult.PASS,
            score=1.0,
            message="Market hours check passed",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_corporate_actions(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check for potential corporate actions affecting prices."""
        if not data_point.ohlcv or not historical_context:
            return QualityCheckResult(
                check_name="corporate_actions_check",
                result=CheckResult.SKIP,
                score=1.0,
                message="Insufficient data for corporate actions check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        check_splits = parameters.get("check_splits", True)
        check_dividends = parameters.get("check_dividends", True)
        
        # Get previous close prices
        prev_closes = [
            float(dp.ohlcv.close) 
            for dp in historical_context[-5:] 
            if dp.ohlcv
        ]
        
        if not prev_closes:
            return QualityCheckResult(
                check_name="corporate_actions_check",
                result=CheckResult.SKIP,
                score=1.0,
                message="No previous close prices",
                details={},
                timestamp=datetime.utcnow()
            )
        
        avg_prev_close = statistics.mean(prev_closes)
        current_close = float(data_point.ohlcv.close)
        
        # Check for potential stock split (price drops by ~50%, ~33%, ~25%, etc.)
        if check_splits and avg_prev_close > 0:
            price_ratio = current_close / avg_prev_close
            split_ratios = [0.5, 0.333, 0.25, 0.2]  # 2:1, 3:1, 4:1, 5:1 splits
            
            for split_ratio in split_ratios:
                if abs(price_ratio - split_ratio) < 0.05:  # Within 5% of split ratio
                    return QualityCheckResult(
                        check_name="corporate_actions_check",
                        result=CheckResult.WARNING,
                        score=0.5,
                        message=f"Potential stock split detected (ratio: {1/split_ratio:.0f}:1)",
                        details={
                            "avg_prev_close": avg_prev_close,
                            "current_close": current_close,
                            "price_ratio": price_ratio
                        },
                        timestamp=datetime.utcnow()
                    )
        
        return QualityCheckResult(
            check_name="corporate_actions_check",
            result=CheckResult.PASS,
            score=1.0,
            message="No corporate actions detected",
            details={},
            timestamp=datetime.utcnow()
        )
    
    def check_trading_halt(
        self, 
        data_point: MarketDataPoint,
        parameters: Dict[str, Any],
        historical_context: Optional[List[MarketDataPoint]] = None
    ) -> QualityCheckResult:
        """Check for potential trading halts."""
        if not data_point.ohlcv or not historical_context:
            return QualityCheckResult(
                check_name="trading_halt_check",
                result=CheckResult.SKIP,
                score=1.0,
                message="Insufficient data for trading halt check",
                details={},
                timestamp=datetime.utcnow()
            )
        
        # Check for zero volume with no price movement (potential halt)
        if data_point.ohlcv.volume == 0:
            # Check if prices are all the same (no movement)
            if (data_point.ohlcv.open == data_point.ohlcv.high == 
                data_point.ohlcv.low == data_point.ohlcv.close):
                
                # Check if this pattern continues
                halt_count = 0
                for dp in historical_context[-5:]:
                    if (dp.ohlcv and dp.ohlcv.volume == 0 and
                        dp.ohlcv.open == dp.ohlcv.high == dp.ohlcv.low == dp.ohlcv.close):
                        halt_count += 1
                
                if halt_count >= 3:
                    return QualityCheckResult(
                        check_name="trading_halt_check",
                        result=CheckResult.WARNING,
                        score=0.3,
                        message="Potential trading halt detected",
                        details={
                            "consecutive_halt_periods": halt_count + 1,
                            "price": float(data_point.ohlcv.close)
                        },
                        timestamp=datetime.utcnow()
                    )
        
        return QualityCheckResult(
            check_name="trading_halt_check",
            result=CheckResult.PASS,
            score=1.0,
            message="No trading halt detected",
            details={},
            timestamp=datetime.utcnow()
        )


# Global quality checks instance
_quality_checks: Optional[MarketDataQualityChecks] = None


def get_quality_checks() -> MarketDataQualityChecks:
    """Get the global quality checks instance."""
    global _quality_checks
    
    if _quality_checks is None:
        _quality_checks = MarketDataQualityChecks()
    
    return _quality_checks