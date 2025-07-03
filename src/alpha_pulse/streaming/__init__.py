"""
Streaming data processing module for AlphaPulse.

Provides real-time data streaming capabilities using Apache Kafka/Pulsar
for high-throughput, low-latency market data processing.
"""

from alpha_pulse.streaming.kafka_producer import (
    AlphaPulseKafkaProducer,
    ProducerConfig,
    SerializationFormat,
    PartitionStrategy,
    TopicManager
)

from alpha_pulse.streaming.kafka_consumer import (
    AlphaPulseKafkaConsumer,
    ConsumerConfig,
    ConsumerMode,
    OffsetResetPolicy
)

from alpha_pulse.streaming.streaming_processor import (
    StreamProcessor,
    MarketDataProcessor,
    TradingSignalProcessor,
    StreamingPipeline,
    WindowType,
    WindowConfig,
    AggregationType
)

from alpha_pulse.streaming.stream_analytics import (
    ComplexEventProcessor,
    PatternDetector,
    EventPattern,
    PatternType,
    TrendAnalyzer,
    StreamCorrelator,
    StreamAnomalyDetector
)

__all__ = [
    # Producer
    "AlphaPulseKafkaProducer",
    "ProducerConfig",
    "SerializationFormat",
    "PartitionStrategy",
    "TopicManager",
    
    # Consumer
    "AlphaPulseKafkaConsumer",
    "ConsumerConfig",
    "ConsumerMode",
    "OffsetResetPolicy",
    
    # Processing
    "StreamProcessor",
    "MarketDataProcessor",
    "TradingSignalProcessor", 
    "StreamingPipeline",
    "WindowType",
    "WindowConfig",
    "AggregationType",
    
    # Analytics
    "ComplexEventProcessor",
    "PatternDetector",
    "EventPattern",
    "PatternType",
    "TrendAnalyzer",
    "StreamCorrelator",
    "StreamAnomalyDetector"
]