"""
Streaming data processor for real-time market data and events.

Provides:
- Real-time data processing pipelines
- Stream aggregation and windowing
- Complex event processing
- Stream transformations
- State management
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable, AsyncIterator, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import logging

import numpy as np
import pandas as pd

from alpha_pulse.models.streaming_message import (
    StreamingMessage, MarketDataMessage, TradingSignalMessage,
    MessageType, EventSeverity
)
from alpha_pulse.models.market_data import OHLCV
from alpha_pulse.streaming.kafka_consumer import AlphaPulseKafkaConsumer
from alpha_pulse.streaming.kafka_producer import AlphaPulseKafkaProducer


logger = logging.getLogger(__name__)

T = TypeVar('T', bound=StreamingMessage)


class WindowType(Enum):
    """Types of time windows for stream processing."""
    TUMBLING = "tumbling"      # Non-overlapping fixed windows
    SLIDING = "sliding"        # Overlapping sliding windows
    SESSION = "session"        # Gap-based session windows
    HOPPING = "hopping"        # Fixed size, fixed slide windows


class AggregationType(Enum):
    """Types of aggregations for windowed data."""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    PERCENTILE = "percentile"
    CUSTOM = "custom"


@dataclass
class WindowConfig:
    """Configuration for time windows."""
    window_type: WindowType
    size: timedelta
    slide: Optional[timedelta] = None  # For sliding windows
    gap: Optional[timedelta] = None    # For session windows
    late_arrival_delay: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    
    def validate(self):
        """Validate window configuration."""
        if self.window_type == WindowType.SLIDING and not self.slide:
            raise ValueError("Sliding window requires slide parameter")
        if self.window_type == WindowType.SESSION and not self.gap:
            raise ValueError("Session window requires gap parameter")
        if self.slide and self.slide > self.size:
            raise ValueError("Slide cannot be larger than window size")


@dataclass
class StreamState:
    """Maintains state for stream processing."""
    windows: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    aggregates: Dict[str, Any] = field(default_factory=dict)
    watermarks: Dict[str, datetime] = field(default_factory=dict)
    pending_events: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeWindow(Generic[T]):
    """Represents a time window for stream processing."""
    
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        window_type: WindowType
    ):
        """Initialize time window."""
        self.start_time = start_time
        self.end_time = end_time
        self.window_type = window_type
        self.messages: List[T] = []
        self.is_closed = False
    
    def add_message(self, message: T) -> bool:
        """Add message to window if it fits."""
        msg_time = message.header.timestamp
        
        if self.start_time <= msg_time < self.end_time:
            self.messages.append(message)
            return True
        return False
    
    def close(self):
        """Close the window."""
        self.is_closed = True
    
    def size(self) -> int:
        """Get number of messages in window."""
        return len(self.messages)
    
    def duration(self) -> timedelta:
        """Get window duration."""
        return self.end_time - self.start_time


class WindowManager(Generic[T]):
    """Manages time windows for stream processing."""
    
    def __init__(self, config: WindowConfig):
        """Initialize window manager."""
        self.config = config
        self.config.validate()
        self.active_windows: Dict[str, TimeWindow[T]] = {}
        self.watermark = datetime.min
    
    def assign_windows(self, message: T) -> List[str]:
        """Assign message to appropriate windows."""
        msg_time = message.header.timestamp
        window_keys = []
        
        if self.config.window_type == WindowType.TUMBLING:
            window_keys = self._assign_tumbling_window(msg_time)
        elif self.config.window_type == WindowType.SLIDING:
            window_keys = self._assign_sliding_windows(msg_time)
        elif self.config.window_type == WindowType.HOPPING:
            window_keys = self._assign_hopping_windows(msg_time)
        elif self.config.window_type == WindowType.SESSION:
            window_keys = self._assign_session_window(message)
        
        # Add message to assigned windows
        for key in window_keys:
            if key not in self.active_windows:
                start_time = self._parse_window_key(key)[0]
                end_time = start_time + self.config.size
                self.active_windows[key] = TimeWindow(
                    start_time, end_time, self.config.window_type
                )
            
            self.active_windows[key].add_message(message)
        
        return window_keys
    
    def _assign_tumbling_window(self, timestamp: datetime) -> List[str]:
        """Assign to tumbling window."""
        window_start = self._get_window_start(timestamp, self.config.size)
        return [self._create_window_key(window_start, window_start + self.config.size)]
    
    def _assign_sliding_windows(self, timestamp: datetime) -> List[str]:
        """Assign to sliding windows."""
        windows = []
        slide = self.config.slide or self.config.size
        
        # Find all windows that contain this timestamp
        current_window_start = self._get_window_start(timestamp, slide)
        
        # Check previous windows that might still contain this timestamp
        window_start = current_window_start - self.config.size + slide
        while window_start <= timestamp < window_start + self.config.size:
            if window_start + self.config.size > timestamp:
                windows.append(
                    self._create_window_key(window_start, window_start + self.config.size)
                )
            window_start += slide
        
        return windows
    
    def _assign_hopping_windows(self, timestamp: datetime) -> List[str]:
        """Assign to hopping windows."""
        # Similar to sliding but with fixed hop size
        return self._assign_sliding_windows(timestamp)
    
    def _assign_session_window(self, message: T) -> List[str]:
        """Assign to session window based on gaps."""
        # Session windows are created based on activity gaps
        # This is a simplified implementation
        msg_time = message.header.timestamp
        session_key = None
        
        # Find existing session or create new one
        for key, window in self.active_windows.items():
            if not window.is_closed and window.messages:
                last_msg_time = window.messages[-1].header.timestamp
                if msg_time - last_msg_time <= self.config.gap:
                    session_key = key
                    # Extend session window
                    window.end_time = msg_time + self.config.gap
                    break
        
        if not session_key:
            # Create new session window
            window_start = msg_time
            window_end = msg_time + self.config.gap
            session_key = self._create_window_key(window_start, window_end)
        
        return [session_key]
    
    def _get_window_start(self, timestamp: datetime, window_size: timedelta) -> datetime:
        """Calculate window start time."""
        epoch = datetime(1970, 1, 1)
        window_ms = int(window_size.total_seconds() * 1000)
        timestamp_ms = int((timestamp - epoch).total_seconds() * 1000)
        window_start_ms = (timestamp_ms // window_ms) * window_ms
        return epoch + timedelta(milliseconds=window_start_ms)
    
    def _create_window_key(self, start: datetime, end: datetime) -> str:
        """Create window key."""
        return f"{start.isoformat()}_{end.isoformat()}"
    
    def _parse_window_key(self, key: str) -> tuple[datetime, datetime]:
        """Parse window key to get start and end times."""
        parts = key.split('_')
        return datetime.fromisoformat(parts[0]), datetime.fromisoformat(parts[1])
    
    def advance_watermark(self, timestamp: datetime):
        """Advance watermark and close expired windows."""
        self.watermark = max(self.watermark, timestamp - self.config.late_arrival_delay)
        
        # Close windows that are before watermark
        closed_windows = []
        for key, window in list(self.active_windows.items()):
            if window.end_time <= self.watermark:
                window.close()
                closed_windows.append(key)
        
        return closed_windows
    
    def get_window(self, key: str) -> Optional[TimeWindow[T]]:
        """Get window by key."""
        return self.active_windows.get(key)
    
    def remove_window(self, key: str):
        """Remove window."""
        if key in self.active_windows:
            del self.active_windows[key]


class StreamAggregator:
    """Performs aggregations on windowed stream data."""
    
    def __init__(self):
        """Initialize aggregator."""
        self.aggregation_functions = {
            AggregationType.COUNT: self._count,
            AggregationType.SUM: self._sum,
            AggregationType.AVG: self._avg,
            AggregationType.MIN: self._min,
            AggregationType.MAX: self._max,
            AggregationType.STDDEV: self._stddev,
            AggregationType.PERCENTILE: self._percentile
        }
    
    def aggregate(
        self,
        messages: List[StreamingMessage],
        aggregation_type: AggregationType,
        field_extractor: Callable[[StreamingMessage], Any],
        **kwargs
    ) -> Any:
        """Perform aggregation on messages."""
        if aggregation_type == AggregationType.CUSTOM:
            custom_func = kwargs.get('custom_function')
            if not custom_func:
                raise ValueError("Custom aggregation requires custom_function parameter")
            return custom_func(messages)
        
        aggregator = self.aggregation_functions.get(aggregation_type)
        if not aggregator:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        
        values = [field_extractor(msg) for msg in messages]
        return aggregator(values, **kwargs)
    
    def _count(self, values: List[Any], **kwargs) -> int:
        """Count aggregation."""
        return len(values)
    
    def _sum(self, values: List[Any], **kwargs) -> float:
        """Sum aggregation."""
        return sum(float(v) for v in values if v is not None)
    
    def _avg(self, values: List[Any], **kwargs) -> float:
        """Average aggregation."""
        numeric_values = [float(v) for v in values if v is not None]
        return statistics.mean(numeric_values) if numeric_values else 0
    
    def _min(self, values: List[Any], **kwargs) -> Any:
        """Minimum aggregation."""
        return min(values) if values else None
    
    def _max(self, values: List[Any], **kwargs) -> Any:
        """Maximum aggregation."""
        return max(values) if values else None
    
    def _stddev(self, values: List[Any], **kwargs) -> float:
        """Standard deviation aggregation."""
        numeric_values = [float(v) for v in values if v is not None]
        return statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
    
    def _percentile(self, values: List[Any], **kwargs) -> float:
        """Percentile aggregation."""
        percentile = kwargs.get('percentile', 50)
        numeric_values = sorted(float(v) for v in values if v is not None)
        if not numeric_values:
            return 0
        
        return np.percentile(numeric_values, percentile)


class StreamProcessor:
    """Base class for stream processors."""
    
    def __init__(self, name: str):
        """Initialize processor."""
        self.name = name
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "processing_time_ms": []
        }
    
    async def process(self, message: StreamingMessage) -> Optional[StreamingMessage]:
        """Process a single message."""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        return self.metrics.copy()


class MarketDataProcessor(StreamProcessor):
    """Processes market data streams."""
    
    def __init__(
        self,
        window_config: WindowConfig,
        aggregations: List[tuple[AggregationType, str]]  # (type, field_name)
    ):
        """Initialize market data processor."""
        super().__init__("MarketDataProcessor")
        self.window_manager = WindowManager[MarketDataMessage](window_config)
        self.aggregator = StreamAggregator()
        self.aggregations = aggregations
        self.state = StreamState()
    
    async def process(self, message: StreamingMessage) -> Optional[StreamingMessage]:
        """Process market data message."""
        if not isinstance(message, MarketDataMessage):
            logger.warning(f"Expected MarketDataMessage, got {type(message)}")
            return None
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Assign to windows
            window_keys = self.window_manager.assign_windows(message)
            
            # Update watermark
            closed_windows = self.window_manager.advance_watermark(message.header.timestamp)
            
            # Process closed windows
            for window_key in closed_windows:
                window = self.window_manager.get_window(window_key)
                if window:
                    await self._process_window(window_key, window)
                    self.window_manager.remove_window(window_key)
            
            # Update metrics
            self.metrics["messages_processed"] += 1
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.metrics["processing_time_ms"].append(processing_time)
            
            # Keep only last 1000 processing times
            if len(self.metrics["processing_time_ms"]) > 1000:
                self.metrics["processing_time_ms"] = self.metrics["processing_time_ms"][-1000:]
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            self.metrics["messages_failed"] += 1
            return None
    
    async def _process_window(self, window_key: str, window: TimeWindow[MarketDataMessage]):
        """Process completed window."""
        if not window.messages:
            return
        
        results = {}
        
        # Perform aggregations
        for agg_type, field_name in self.aggregations:
            if field_name == "price":
                extractor = lambda msg: msg.last_trade or msg.ohlcv.close if msg.ohlcv else None
            elif field_name == "volume":
                extractor = lambda msg: msg.ohlcv.volume if msg.ohlcv else None
            elif field_name == "spread":
                extractor = lambda msg: float(msg.ask - msg.bid) if msg.ask and msg.bid else None
            else:
                continue
            
            result = self.aggregator.aggregate(
                window.messages,
                agg_type,
                extractor
            )
            results[f"{field_name}_{agg_type.value}"] = result
        
        # Store results in state
        self.state.aggregates[window_key] = {
            "window_start": window.start_time,
            "window_end": window.end_time,
            "message_count": window.size(),
            "aggregates": results
        }
        
        logger.info(
            f"Processed window {window_key}: {window.size()} messages, "
            f"aggregates: {results}"
        )


class TradingSignalProcessor(StreamProcessor):
    """Processes trading signal streams."""
    
    def __init__(
        self,
        signal_buffer_size: int = 100,
        correlation_window: timedelta = timedelta(minutes=5)
    ):
        """Initialize trading signal processor."""
        super().__init__("TradingSignalProcessor")
        self.signal_buffer_size = signal_buffer_size
        self.correlation_window = correlation_window
        self.signal_buffer: deque[TradingSignalMessage] = deque(maxlen=signal_buffer_size)
        self.signal_correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    async def process(self, message: StreamingMessage) -> Optional[StreamingMessage]:
        """Process trading signal message."""
        if not isinstance(message, TradingSignalMessage):
            logger.warning(f"Expected TradingSignalMessage, got {type(message)}")
            return None
        
        try:
            # Add to buffer
            self.signal_buffer.append(message)
            
            # Calculate signal correlations
            await self._update_signal_correlations(message)
            
            # Check for conflicting signals
            conflicts = await self._check_signal_conflicts(message)
            if conflicts:
                logger.warning(
                    f"Signal conflicts detected for {message.symbol}: {conflicts}"
                )
                message.header.metadata["conflicts"] = conflicts
            
            # Update metrics
            self.metrics["messages_processed"] += 1
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing trading signal: {e}")
            self.metrics["messages_failed"] += 1
            return None
    
    async def _update_signal_correlations(self, new_signal: TradingSignalMessage):
        """Update correlations between signals."""
        cutoff_time = new_signal.header.timestamp - self.correlation_window
        
        # Get recent signals for the same symbol
        recent_signals = [
            sig for sig in self.signal_buffer
            if sig.symbol == new_signal.symbol and sig.header.timestamp >= cutoff_time
        ]
        
        # Calculate correlation between strategies
        strategy_signals = defaultdict(list)
        for sig in recent_signals:
            strategy_signals[sig.strategy_name].append(sig.signal_type.value)
        
        # Update correlation matrix
        for strategy, signals in strategy_signals.items():
            if strategy != new_signal.strategy_name:
                # Simple agreement ratio
                agreement = sum(
                    1 for s in signals if s == new_signal.signal_type.value
                ) / len(signals) if signals else 0
                
                self.signal_correlations[new_signal.strategy_name][strategy] = agreement
    
    async def _check_signal_conflicts(self, signal: TradingSignalMessage) -> List[Dict[str, Any]]:
        """Check for conflicting signals."""
        conflicts = []
        cutoff_time = signal.header.timestamp - timedelta(seconds=30)  # 30-second window
        
        # Check recent signals for the same symbol
        for recent_signal in reversed(self.signal_buffer):
            if (recent_signal.symbol == signal.symbol and
                recent_signal.signal_id != signal.signal_id and
                recent_signal.header.timestamp >= cutoff_time):
                
                # Check for opposite signals
                if (signal.signal_type.value == "buy" and recent_signal.signal_type.value == "sell") or \
                   (signal.signal_type.value == "sell" and recent_signal.signal_type.value == "buy"):
                    conflicts.append({
                        "signal_id": recent_signal.signal_id,
                        "strategy": recent_signal.strategy_name,
                        "signal_type": recent_signal.signal_type.value,
                        "confidence": recent_signal.confidence,
                        "timestamp": recent_signal.header.timestamp
                    })
        
        return conflicts


class StreamingPipeline:
    """Orchestrates streaming data processing pipeline."""
    
    def __init__(
        self,
        consumer: AlphaPulseKafkaConsumer,
        producer: AlphaPulseKafkaProducer,
        processors: List[StreamProcessor]
    ):
        """Initialize streaming pipeline."""
        self.consumer = consumer
        self.producer = producer
        self.processors = processors
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
    
    async def start(self):
        """Start the streaming pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        logger.info("Starting streaming pipeline...")
        
        # Register message handlers
        await self._register_handlers()
        
        # Start consuming
        try:
            await self.consumer.consume()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop(self):
        """Stop the streaming pipeline."""
        self.is_running = False
        logger.info("Stopping streaming pipeline...")
    
    async def _register_handlers(self):
        """Register handlers for different message types."""
        # Market data handler
        async def handle_market_data(message: MarketDataMessage):
            for processor in self.processors:
                if isinstance(processor, MarketDataProcessor):
                    result = await processor.process(message)
                    if result:
                        self.processed_count += 1
                    else:
                        self.error_count += 1
        
        # Trading signal handler
        async def handle_trading_signal(message: TradingSignalMessage):
            for processor in self.processors:
                if isinstance(processor, TradingSignalProcessor):
                    result = await processor.process(message)
                    if result:
                        self.processed_count += 1
                    else:
                        self.error_count += 1
        
        # Register with consumer
        self.consumer.register_handler(MessageType.MARKET_DATA, handle_market_data)
        self.consumer.register_handler(MessageType.TRADING_SIGNAL, handle_trading_signal)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        metrics = {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "is_running": self.is_running,
            "processors": {}
        }
        
        # Add processor metrics
        for processor in self.processors:
            metrics["processors"][processor.name] = processor.get_metrics()
        
        return metrics