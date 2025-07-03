"""
Stream analytics for complex event processing and pattern detection.

Provides:
- Complex event processing (CEP)
- Pattern detection in streams
- Anomaly detection
- Real-time analytics
- Stream joins and correlations
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re
import logging

import numpy as np
import pandas as pd
from scipy import stats

from alpha_pulse.models.streaming_message import (
    StreamingMessage, MarketDataMessage, TradingSignalMessage,
    RiskAlertMessage, MessageType
)
from alpha_pulse.data.quality.anomaly_detector import AnomalyDetector, AnomalyType


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns to detect in streams."""
    SEQUENCE = "sequence"          # Sequential pattern (A -> B -> C)
    CONJUNCTION = "conjunction"    # All events occur (A AND B AND C)
    DISJUNCTION = "disjunction"   # Any event occurs (A OR B OR C)
    ABSENCE = "absence"           # Event does NOT occur
    THRESHOLD = "threshold"       # Value crosses threshold
    TREND = "trend"              # Trending pattern (up/down/sideways)
    CORRELATION = "correlation"   # Correlated events
    CUSTOM = "custom"            # Custom pattern


class TrendDirection(Enum):
    """Trend directions."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class EventPattern:
    """Defines a pattern to detect in event streams."""
    name: str
    pattern_type: PatternType
    conditions: List[Dict[str, Any]]
    time_window: timedelta
    min_occurrences: int = 1
    max_occurrences: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self):
        """Validate pattern configuration."""
        if not self.conditions:
            raise ValueError("Pattern must have at least one condition")
        if self.min_occurrences < 1:
            raise ValueError("Minimum occurrences must be at least 1")
        if self.max_occurrences and self.max_occurrences < self.min_occurrences:
            raise ValueError("Max occurrences cannot be less than min occurrences")


@dataclass
class PatternMatch:
    """Represents a detected pattern match."""
    pattern: EventPattern
    matched_events: List[StreamingMessage]
    match_time: datetime
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventMatcher:
    """Matches individual events against conditions."""
    
    def __init__(self):
        """Initialize event matcher."""
        self.condition_matchers = {
            "message_type": self._match_message_type,
            "field_equals": self._match_field_equals,
            "field_contains": self._match_field_contains,
            "field_regex": self._match_field_regex,
            "field_range": self._match_field_range,
            "custom": self._match_custom
        }
    
    def match(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Check if event matches condition."""
        condition_type = condition.get("type")
        matcher = self.condition_matchers.get(condition_type)
        
        if not matcher:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False
        
        try:
            return matcher(event, condition)
        except Exception as e:
            logger.error(f"Error matching condition: {e}")
            return False
    
    def _match_message_type(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Match message type."""
        expected_type = condition.get("value")
        if isinstance(expected_type, str):
            expected_type = MessageType(expected_type)
        return event.header.message_type == expected_type
    
    def _match_field_equals(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Match field equals value."""
        field_path = condition.get("field", "").split(".")
        expected_value = condition.get("value")
        
        # Navigate field path
        current = event
        for field in field_path:
            if hasattr(current, field):
                current = getattr(current, field)
            elif isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return False
        
        return current == expected_value
    
    def _match_field_contains(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Match field contains value."""
        field_path = condition.get("field", "").split(".")
        search_value = condition.get("value")
        
        # Navigate field path
        current = event
        for field in field_path:
            if hasattr(current, field):
                current = getattr(current, field)
            elif isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return False
        
        if isinstance(current, str):
            return search_value in current
        elif isinstance(current, (list, set, tuple)):
            return search_value in current
        
        return False
    
    def _match_field_regex(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Match field against regex pattern."""
        field_path = condition.get("field", "").split(".")
        pattern = condition.get("pattern")
        
        # Navigate field path
        current = event
        for field in field_path:
            if hasattr(current, field):
                current = getattr(current, field)
            elif isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return False
        
        if isinstance(current, str):
            return bool(re.match(pattern, current))
        
        return False
    
    def _match_field_range(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Match field within range."""
        field_path = condition.get("field", "").split(".")
        min_value = condition.get("min")
        max_value = condition.get("max")
        
        # Navigate field path
        current = event
        for field in field_path:
            if hasattr(current, field):
                current = getattr(current, field)
            elif isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return False
        
        try:
            value = float(current)
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
            return True
        except (TypeError, ValueError):
            return False
    
    def _match_custom(self, event: StreamingMessage, condition: Dict[str, Any]) -> bool:
        """Match using custom function."""
        custom_func = condition.get("function")
        if callable(custom_func):
            return custom_func(event)
        return False


class PatternDetector:
    """Detects complex patterns in event streams."""
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize pattern detector."""
        self.patterns: Dict[str, EventPattern] = {}
        self.event_buffer: deque = deque(maxlen=buffer_size)
        self.pattern_states: Dict[str, List[StreamingMessage]] = defaultdict(list)
        self.event_matcher = EventMatcher()
        self.detected_patterns: List[PatternMatch] = []
    
    def register_pattern(self, pattern: EventPattern):
        """Register a pattern to detect."""
        pattern.validate()
        self.patterns[pattern.name] = pattern
        logger.info(f"Registered pattern: {pattern.name}")
    
    async def process_event(self, event: StreamingMessage) -> List[PatternMatch]:
        """Process event and detect patterns."""
        self.event_buffer.append(event)
        detected = []
        
        for pattern in self.patterns.values():
            matches = await self._detect_pattern(event, pattern)
            detected.extend(matches)
        
        self.detected_patterns.extend(detected)
        return detected
    
    async def _detect_pattern(
        self, 
        event: StreamingMessage, 
        pattern: EventPattern
    ) -> List[PatternMatch]:
        """Detect specific pattern."""
        if pattern.pattern_type == PatternType.SEQUENCE:
            return await self._detect_sequence_pattern(event, pattern)
        elif pattern.pattern_type == PatternType.CONJUNCTION:
            return await self._detect_conjunction_pattern(event, pattern)
        elif pattern.pattern_type == PatternType.DISJUNCTION:
            return await self._detect_disjunction_pattern(event, pattern)
        elif pattern.pattern_type == PatternType.ABSENCE:
            return await self._detect_absence_pattern(event, pattern)
        elif pattern.pattern_type == PatternType.THRESHOLD:
            return await self._detect_threshold_pattern(event, pattern)
        else:
            return []
    
    async def _detect_sequence_pattern(
        self, 
        event: StreamingMessage, 
        pattern: EventPattern
    ) -> List[PatternMatch]:
        """Detect sequential pattern (A -> B -> C)."""
        matches = []
        state_key = f"{pattern.name}_state"
        current_state = self.pattern_states.get(state_key, [])
        
        # Check if event matches next condition in sequence
        next_condition_idx = len(current_state)
        if next_condition_idx < len(pattern.conditions):
            condition = pattern.conditions[next_condition_idx]
            if self.event_matcher.match(event, condition):
                current_state.append(event)
                
                # Check if sequence is complete
                if len(current_state) == len(pattern.conditions):
                    # Verify time window
                    time_diff = event.header.timestamp - current_state[0].header.timestamp
                    if time_diff <= pattern.time_window:
                        match = PatternMatch(
                            pattern=pattern,
                            matched_events=current_state.copy(),
                            match_time=event.header.timestamp,
                            confidence=1.0
                        )
                        matches.append(match)
                    
                    # Reset state
                    self.pattern_states[state_key] = []
                else:
                    self.pattern_states[state_key] = current_state
        
        # Clean expired partial sequences
        cutoff_time = event.header.timestamp - pattern.time_window
        if current_state and current_state[0].header.timestamp < cutoff_time:
            self.pattern_states[state_key] = []
        
        return matches
    
    async def _detect_conjunction_pattern(
        self, 
        event: StreamingMessage, 
        pattern: EventPattern
    ) -> List[PatternMatch]:
        """Detect conjunction pattern (A AND B AND C)."""
        matches = []
        
        # Get events within time window
        cutoff_time = event.header.timestamp - pattern.time_window
        window_events = [
            e for e in self.event_buffer
            if e.header.timestamp >= cutoff_time
        ]
        
        # Check if all conditions are satisfied
        matched_events = []
        for condition in pattern.conditions:
            condition_matches = [
                e for e in window_events
                if self.event_matcher.match(e, condition)
            ]
            if not condition_matches:
                return []  # All conditions must be met
            matched_events.extend(condition_matches)
        
        # Check occurrence constraints
        if (len(matched_events) >= pattern.min_occurrences and
            (not pattern.max_occurrences or len(matched_events) <= pattern.max_occurrences)):
            match = PatternMatch(
                pattern=pattern,
                matched_events=matched_events,
                match_time=event.header.timestamp,
                confidence=1.0
            )
            matches.append(match)
        
        return matches
    
    async def _detect_disjunction_pattern(
        self, 
        event: StreamingMessage, 
        pattern: EventPattern
    ) -> List[PatternMatch]:
        """Detect disjunction pattern (A OR B OR C)."""
        matches = []
        
        # Check if event matches any condition
        for condition in pattern.conditions:
            if self.event_matcher.match(event, condition):
                match = PatternMatch(
                    pattern=pattern,
                    matched_events=[event],
                    match_time=event.header.timestamp,
                    confidence=1.0
                )
                matches.append(match)
                break
        
        return matches
    
    async def _detect_absence_pattern(
        self, 
        event: StreamingMessage, 
        pattern: EventPattern
    ) -> List[PatternMatch]:
        """Detect absence pattern (event NOT occurring)."""
        matches = []
        
        # Check if expected event is absent within time window
        cutoff_time = event.header.timestamp - pattern.time_window
        window_events = [
            e for e in self.event_buffer
            if e.header.timestamp >= cutoff_time
        ]
        
        # Check if any condition is NOT satisfied
        for condition in pattern.conditions:
            condition_matches = [
                e for e in window_events
                if self.event_matcher.match(e, condition)
            ]
            if not condition_matches:
                # Event is absent
                match = PatternMatch(
                    pattern=pattern,
                    matched_events=[],  # No events matched (that's the point)
                    match_time=event.header.timestamp,
                    confidence=1.0,
                    metadata={"absent_condition": condition}
                )
                matches.append(match)
        
        return matches
    
    async def _detect_threshold_pattern(
        self, 
        event: StreamingMessage, 
        pattern: EventPattern
    ) -> List[PatternMatch]:
        """Detect threshold crossing pattern."""
        matches = []
        
        # Threshold patterns should have specific threshold conditions
        for condition in pattern.conditions:
            if condition.get("type") == "field_range" and self.event_matcher.match(event, condition):
                match = PatternMatch(
                    pattern=pattern,
                    matched_events=[event],
                    match_time=event.header.timestamp,
                    confidence=1.0,
                    metadata={"threshold_condition": condition}
                )
                matches.append(match)
        
        return matches


class StreamCorrelator:
    """Correlates events across multiple streams."""
    
    def __init__(self, correlation_window: timedelta = timedelta(minutes=5)):
        """Initialize stream correlator."""
        self.correlation_window = correlation_window
        self.event_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_results: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    async def add_event(self, stream_name: str, event: StreamingMessage):
        """Add event to stream for correlation analysis."""
        self.event_streams[stream_name].append(event)
        
        # Clean old events
        cutoff_time = datetime.utcnow() - self.correlation_window
        while (self.event_streams[stream_name] and 
               self.event_streams[stream_name][0].header.timestamp < cutoff_time):
            self.event_streams[stream_name].popleft()
    
    async def calculate_correlations(
        self, 
        stream1: str, 
        stream2: str,
        value_extractor1: Callable[[StreamingMessage], float],
        value_extractor2: Callable[[StreamingMessage], float]
    ) -> float:
        """Calculate correlation between two streams."""
        events1 = list(self.event_streams.get(stream1, []))
        events2 = list(self.event_streams.get(stream2, []))
        
        if not events1 or not events2:
            return 0.0
        
        # Extract values and align by timestamp
        values1 = []
        values2 = []
        
        for e1 in events1:
            timestamp1 = e1.header.timestamp
            # Find closest event in stream2
            closest_e2 = min(
                events2,
                key=lambda e: abs((e.header.timestamp - timestamp1).total_seconds())
            )
            
            # Only include if within reasonable time window (1 second)
            if abs((closest_e2.header.timestamp - timestamp1).total_seconds()) <= 1:
                try:
                    v1 = value_extractor1(e1)
                    v2 = value_extractor2(closest_e2)
                    if v1 is not None and v2 is not None:
                        values1.append(float(v1))
                        values2.append(float(v2))
                except Exception:
                    continue
        
        # Calculate correlation
        if len(values1) >= 2:
            correlation = np.corrcoef(values1, values2)[0, 1]
            self.correlation_results[stream1][stream2] = correlation
            return correlation
        
        return 0.0


class TrendAnalyzer:
    """Analyzes trends in streaming data."""
    
    def __init__(self, window_size: int = 20):
        """Initialize trend analyzer."""
        self.window_size = window_size
        self.value_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_value(self, series_name: str, timestamp: datetime, value: float):
        """Add value to series."""
        self.value_buffers[series_name].append((timestamp, value))
    
    def detect_trend(self, series_name: str) -> Tuple[TrendDirection, float]:
        """Detect trend in series."""
        values = self.value_buffers.get(series_name, [])
        if len(values) < 3:
            return TrendDirection.SIDEWAYS, 0.0
        
        # Extract values
        timestamps = [v[0] for v in values]
        prices = [v[1] for v in values]
        
        # Convert timestamps to numeric values
        time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_values, prices)
        
        # Determine trend direction
        if abs(slope) < std_err:  # Not significant
            return TrendDirection.SIDEWAYS, abs(r_value)
        elif slope > 0:
            return TrendDirection.UP, abs(r_value)
        else:
            return TrendDirection.DOWN, abs(r_value)
    
    def get_trend_strength(self, series_name: str) -> float:
        """Get trend strength (0-1)."""
        _, strength = self.detect_trend(series_name)
        return strength


class StreamAnomalyDetector:
    """Detects anomalies in streaming data."""
    
    def __init__(
        self,
        window_size: int = 100,
        anomaly_threshold: float = 3.0
    ):
        """Initialize anomaly detector."""
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_detector = AnomalyDetector()
        self.value_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
    
    async def process_value(
        self, 
        series_name: str, 
        timestamp: datetime, 
        value: float
    ) -> Optional[Dict[str, Any]]:
        """Process value and detect anomalies."""
        self.value_windows[series_name].append(value)
        
        if len(self.value_windows[series_name]) < 10:
            return None  # Not enough data
        
        # Update baseline statistics
        values = list(self.value_windows[series_name])
        self.baseline_stats[series_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values)
        }
        
        # Check for anomaly
        z_score = abs((value - self.baseline_stats[series_name]["mean"]) / 
                     (self.baseline_stats[series_name]["std"] + 1e-10))
        
        if z_score > self.anomaly_threshold:
            return {
                "series": series_name,
                "timestamp": timestamp,
                "value": value,
                "z_score": z_score,
                "baseline_mean": self.baseline_stats[series_name]["mean"],
                "baseline_std": self.baseline_stats[series_name]["std"],
                "anomaly_type": "statistical_outlier"
            }
        
        return None


class ComplexEventProcessor:
    """Main complex event processing engine."""
    
    def __init__(self):
        """Initialize CEP engine."""
        self.pattern_detector = PatternDetector()
        self.stream_correlator = StreamCorrelator()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = StreamAnomalyDetector()
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.metrics = {
            "events_processed": 0,
            "patterns_detected": 0,
            "anomalies_detected": 0,
            "processing_errors": 0
        }
    
    def register_pattern(self, pattern: EventPattern):
        """Register pattern for detection."""
        self.pattern_detector.register_pattern(pattern)
    
    def register_event_handler(self, pattern_name: str, handler: Callable):
        """Register handler for pattern matches."""
        self.event_handlers[pattern_name].append(handler)
    
    async def process_event(self, event: StreamingMessage) -> Dict[str, Any]:
        """Process event through CEP engine."""
        results = {
            "patterns": [],
            "anomalies": [],
            "trends": {},
            "correlations": {}
        }
        
        try:
            self.metrics["events_processed"] += 1
            
            # Pattern detection
            pattern_matches = await self.pattern_detector.process_event(event)
            results["patterns"] = pattern_matches
            self.metrics["patterns_detected"] += len(pattern_matches)
            
            # Trigger handlers for matched patterns
            for match in pattern_matches:
                handlers = self.event_handlers.get(match.pattern.name, [])
                for handler in handlers:
                    try:
                        await handler(match)
                    except Exception as e:
                        logger.error(f"Error in pattern handler: {e}")
            
            # Extract numeric values for analysis
            if isinstance(event, MarketDataMessage) and event.last_trade:
                series_name = f"{event.symbol}_{event.exchange}"
                value = float(event.last_trade)
                
                # Trend analysis
                self.trend_analyzer.add_value(series_name, event.header.timestamp, value)
                trend_direction, trend_strength = self.trend_analyzer.detect_trend(series_name)
                results["trends"][series_name] = {
                    "direction": trend_direction.value,
                    "strength": trend_strength
                }
                
                # Anomaly detection
                anomaly = await self.anomaly_detector.process_value(
                    series_name, event.header.timestamp, value
                )
                if anomaly:
                    results["anomalies"].append(anomaly)
                    self.metrics["anomalies_detected"] += 1
            
            # Stream correlation
            if hasattr(event, "symbol"):
                await self.stream_correlator.add_event(event.symbol, event)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing event in CEP: {e}")
            self.metrics["processing_errors"] += 1
            return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CEP metrics."""
        return self.metrics.copy()