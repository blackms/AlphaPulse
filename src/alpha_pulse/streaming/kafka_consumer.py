"""
Kafka consumer implementation for streaming data.

Provides:
- Asynchronous message consumption
- Consumer groups and offset management
- Message deserialization
- Error handling and retries
- Dead letter queue support
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable, List, Set, AsyncIterator
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import pickle

from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener
from aiokafka.errors import KafkaError, CommitFailedError
from kafka import KafkaConsumer as SyncKafkaConsumer, TopicPartition
import msgpack

from alpha_pulse.models.streaming_message import (
    StreamingMessage, MessageEnvelope, MessageType,
    MarketDataMessage, TradingSignalMessage, RiskAlertMessage,
    PortfolioUpdateMessage, SystemEventMessage, DataQualityEventMessage
)
from alpha_pulse.streaming.kafka_producer import SerializationFormat, TopicManager


logger = logging.getLogger(__name__)


class ConsumerMode(Enum):
    """Consumer processing modes."""
    SUBSCRIBE = "subscribe"  # Subscribe to topics
    ASSIGN = "assign"      # Manually assign partitions


class OffsetResetPolicy(Enum):
    """Offset reset policies."""
    EARLIEST = "earliest"  # Start from beginning
    LATEST = "latest"      # Start from end
    NONE = "none"         # Throw exception if no offset


@dataclass
class ConsumerConfig:
    """Kafka consumer configuration."""
    bootstrap_servers: List[str]
    group_id: str
    client_id: str = "alphapulse-consumer"
    auto_offset_reset: OffsetResetPolicy = OffsetResetPolicy.LATEST
    enable_auto_commit: bool = False
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000  # 5 minutes
    session_timeout_ms: int = 10000
    heartbeat_interval_ms: int = 3000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500
    consumer_timeout_ms: Optional[int] = None
    serialization_format: SerializationFormat = SerializationFormat.JSON
    isolation_level: str = "read_committed"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConsumerConfig':
        """Create from configuration dictionary."""
        return cls(
            bootstrap_servers=config.get("bootstrap_servers", ["localhost:9092"]),
            group_id=config["group_id"],
            client_id=config.get("client_id", "alphapulse-consumer"),
            auto_offset_reset=OffsetResetPolicy(
                config.get("auto_offset_reset", "latest")
            ),
            enable_auto_commit=config.get("enable_auto_commit", False),
            auto_commit_interval_ms=config.get("auto_commit_interval_ms", 5000),
            max_poll_records=config.get("max_poll_records", 500),
            max_poll_interval_ms=config.get("max_poll_interval_ms", 300000),
            session_timeout_ms=config.get("session_timeout_ms", 10000),
            heartbeat_interval_ms=config.get("heartbeat_interval_ms", 3000),
            fetch_min_bytes=config.get("fetch_min_bytes", 1),
            fetch_max_wait_ms=config.get("fetch_max_wait_ms", 500),
            consumer_timeout_ms=config.get("consumer_timeout_ms"),
            serialization_format=SerializationFormat(
                config.get("serialization_format", "json")
            ),
            isolation_level=config.get("isolation_level", "read_committed")
        )


class MessageDeserializer:
    """Handles message deserialization from different formats."""
    
    def __init__(self, format: SerializationFormat):
        """Initialize deserializer."""
        self.format = format
        self._deserializers = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.MSGPACK: self._deserialize_msgpack,
            SerializationFormat.AVRO: self._deserialize_avro,
            SerializationFormat.PROTOBUF: self._deserialize_protobuf
        }
    
    def deserialize(self, data: bytes, message_type: Optional[MessageType] = None) -> StreamingMessage:
        """Deserialize message from bytes."""
        deserializer = self._deserializers.get(self.format)
        if not deserializer:
            raise ValueError(f"Unsupported deserialization format: {self.format}")
        
        message_dict = deserializer(data)
        return self._reconstruct_message(message_dict, message_type)
    
    def _deserialize_json(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from JSON."""
        return json.loads(data.decode('utf-8'))
    
    def _deserialize_msgpack(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from MessagePack."""
        return msgpack.unpackb(data, raw=False)
    
    def _deserialize_avro(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from Avro format."""
        # TODO: Implement Avro deserialization with schema registry
        raise NotImplementedError("Avro deserialization not yet implemented")
    
    def _deserialize_protobuf(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from Protocol Buffers."""
        # TODO: Implement Protobuf deserialization
        raise NotImplementedError("Protobuf deserialization not yet implemented")
    
    def _reconstruct_message(
        self, 
        data: Dict[str, Any], 
        message_type: Optional[MessageType]
    ) -> StreamingMessage:
        """Reconstruct typed message from dictionary."""
        # Get message type from header if not provided
        if not message_type:
            header_data = data.get("header", {})
            message_type = MessageType(header_data.get("message_type"))
        
        # Map to specific message class
        message_classes = {
            MessageType.MARKET_DATA: MarketDataMessage,
            MessageType.TRADING_SIGNAL: TradingSignalMessage,
            MessageType.RISK_ALERT: RiskAlertMessage,
            MessageType.PORTFOLIO_UPDATE: PortfolioUpdateMessage,
            MessageType.SYSTEM_EVENT: SystemEventMessage,
            MessageType.DATA_QUALITY_EVENT: DataQualityEventMessage
        }
        
        message_class = message_classes.get(message_type, StreamingMessage)
        return message_class.from_dict(data)


class ConsumerMetrics:
    """Tracks consumer metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.messages_consumed = 0
        self.messages_failed = 0
        self.bytes_consumed = 0
        self.last_message_time = None
        self.processing_times = []
        self.lag_by_partition = defaultdict(int)
        self.errors_by_type = defaultdict(int)
    
    def record_message(self, size: int, processing_time: float):
        """Record successful message processing."""
        self.messages_consumed += 1
        self.bytes_consumed += size
        self.last_message_time = datetime.utcnow()
        self.processing_times.append(processing_time)
        
        # Keep only last 1000 processing times
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def record_error(self, error_type: str):
        """Record processing error."""
        self.messages_failed += 1
        self.errors_by_type[error_type] += 1
    
    def update_lag(self, partition: int, lag: int):
        """Update partition lag."""
        self.lag_by_partition[partition] = lag
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        return {
            "messages_consumed": self.messages_consumed,
            "messages_failed": self.messages_failed,
            "bytes_consumed": self.bytes_consumed,
            "last_message_time": self.last_message_time,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "total_lag": sum(self.lag_by_partition.values()),
            "lag_by_partition": dict(self.lag_by_partition),
            "errors_by_type": dict(self.errors_by_type)
        }


class MessageProcessor:
    """Handles message processing with error handling."""
    
    def __init__(
        self,
        handler: Callable[[StreamingMessage], AsyncIterator[None]],
        error_handler: Optional[Callable[[Exception, StreamingMessage], AsyncIterator[None]]] = None,
        dead_letter_handler: Optional[Callable[[StreamingMessage, Exception], AsyncIterator[None]]] = None,
        max_retries: int = 3,
        retry_delay_ms: int = 1000
    ):
        """Initialize message processor."""
        self.handler = handler
        self.error_handler = error_handler
        self.dead_letter_handler = dead_letter_handler
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
    
    async def process(self, message: StreamingMessage) -> bool:
        """Process message with retry logic."""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                await self.handler(message)
                return True
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                logger.warning(
                    f"Error processing message {message.header.message_id}: {e}. "
                    f"Retry {retry_count}/{self.max_retries}"
                )
                
                if retry_count <= self.max_retries:
                    # Wait before retry
                    await asyncio.sleep(self.retry_delay_ms / 1000)
                
                # Call error handler if provided
                if self.error_handler:
                    try:
                        await self.error_handler(e, message)
                    except Exception as handler_error:
                        logger.error(f"Error handler failed: {handler_error}")
        
        # Max retries exceeded - send to dead letter queue
        if self.dead_letter_handler and last_error:
            try:
                await self.dead_letter_handler(message, last_error)
            except Exception as dlq_error:
                logger.error(f"Dead letter handler failed: {dlq_error}")
        
        return False


class AlphaPulseKafkaConsumer:
    """Kafka consumer for AlphaPulse streaming data."""
    
    def __init__(self, config: ConsumerConfig):
        """Initialize Kafka consumer."""
        self.config = config
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._sync_consumer: Optional[SyncKafkaConsumer] = None
        self._deserializer = MessageDeserializer(config.serialization_format)
        self._metrics = ConsumerMetrics()
        self._is_started = False
        self._subscribed_topics: Set[str] = set()
        self._message_processors: Dict[MessageType, MessageProcessor] = {}
        self._running = False
    
    class RebalanceListener(ConsumerRebalanceListener):
        """Handles partition rebalancing."""
        
        def __init__(self, consumer_name: str):
            self.consumer_name = consumer_name
        
        async def on_partitions_revoked(self, revoked):
            """Called before partitions are revoked."""
            logger.info(
                f"{self.consumer_name}: Partitions revoked: "
                f"{[f'{tp.topic}:{tp.partition}' for tp in revoked]}"
            )
        
        async def on_partitions_assigned(self, assigned):
            """Called after partitions are assigned."""
            logger.info(
                f"{self.consumer_name}: Partitions assigned: "
                f"{[f'{tp.topic}:{tp.partition}' for tp in assigned]}"
            )
    
    async def start(self):
        """Start the async consumer."""
        if self._is_started:
            logger.warning("Consumer already started")
            return
        
        try:
            self._consumer = AIOKafkaConsumer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset.value,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                max_poll_records=self.config.max_poll_records,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                fetch_min_bytes=self.config.fetch_min_bytes,
                fetch_max_wait_ms=self.config.fetch_max_wait_ms,
                consumer_timeout_ms=self.config.consumer_timeout_ms,
                isolation_level=self.config.isolation_level,
                value_deserializer=lambda v: v  # We handle deserialization
            )
            
            await self._consumer.start()
            self._is_started = True
            
            logger.info(
                f"Kafka consumer started: {self.config.bootstrap_servers}, "
                f"group: {self.config.group_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self):
        """Stop the async consumer."""
        self._running = False
        
        if self._consumer and self._is_started:
            await self._consumer.stop()
            self._is_started = False
            logger.info("Kafka consumer stopped")
    
    async def subscribe(
        self, 
        topics: List[str],
        listener: Optional[ConsumerRebalanceListener] = None
    ):
        """Subscribe to topics."""
        if not self._is_started:
            raise RuntimeError("Consumer not started. Call start() first.")
        
        if not listener:
            listener = self.RebalanceListener(self.config.client_id)
        
        self._consumer.subscribe(topics, listener=listener)
        self._subscribed_topics.update(topics)
        
        logger.info(f"Subscribed to topics: {topics}")
    
    async def subscribe_by_message_types(
        self,
        message_types: List[MessageType],
        listener: Optional[ConsumerRebalanceListener] = None
    ):
        """Subscribe to topics by message types."""
        topics = [TopicManager.get_topic(mt) for mt in message_types]
        await self.subscribe(topics, listener)
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[StreamingMessage], AsyncIterator[None]],
        error_handler: Optional[Callable] = None,
        dead_letter_handler: Optional[Callable] = None,
        max_retries: int = 3
    ):
        """Register message handler for specific message type."""
        processor = MessageProcessor(
            handler=handler,
            error_handler=error_handler,
            dead_letter_handler=dead_letter_handler,
            max_retries=max_retries
        )
        self._message_processors[message_type] = processor
        
        logger.info(f"Registered handler for {message_type.value}")
    
    async def consume(self, max_messages: Optional[int] = None):
        """Start consuming messages."""
        if not self._is_started:
            raise RuntimeError("Consumer not started. Call start() first.")
        
        self._running = True
        messages_consumed = 0
        
        logger.info("Starting message consumption...")
        
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                
                if max_messages and messages_consumed >= max_messages:
                    break
                
                try:
                    # Process message
                    await self._process_message(msg)
                    messages_consumed += 1
                    
                    # Commit offset if auto-commit is disabled
                    if not self.config.enable_auto_commit:
                        await self._consumer.commit()
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._metrics.record_error(type(e).__name__)
                
        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            logger.info(f"Consumed {messages_consumed} messages")
    
    async def _process_message(self, kafka_msg):
        """Process a single Kafka message."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract message type from headers
            message_type = None
            if kafka_msg.headers:
                for key, value in kafka_msg.headers:
                    if key == "message_type":
                        message_type = MessageType(value.decode('utf-8'))
                        break
            
            # Deserialize message
            message = self._deserializer.deserialize(kafka_msg.value, message_type)
            
            # Create envelope
            envelope = MessageEnvelope(
                topic=kafka_msg.topic,
                partition=kafka_msg.partition,
                key=kafka_msg.key.decode('utf-8') if kafka_msg.key else None,
                headers={k: v.decode('utf-8') for k, v in kafka_msg.headers or []},
                timestamp=datetime.fromtimestamp(kafka_msg.timestamp / 1000),
                offset=kafka_msg.offset
            )
            
            # Add envelope to message metadata
            message.header.metadata["kafka_envelope"] = envelope.to_dict()
            
            # Find and execute handler
            processor = self._message_processors.get(message.header.message_type)
            if processor:
                success = await processor.process(message)
                if success:
                    processing_time = asyncio.get_event_loop().time() - start_time
                    self._metrics.record_message(len(kafka_msg.value), processing_time)
                else:
                    self._metrics.record_error("processing_failed")
            else:
                logger.warning(
                    f"No handler registered for message type: "
                    f"{message.header.message_type.value}"
                )
            
            # Update lag metrics
            if hasattr(self._consumer, 'highwater'):
                lag = self._consumer.highwater(
                    TopicPartition(kafka_msg.topic, kafka_msg.partition)
                ) - kafka_msg.offset
                self._metrics.update_lag(kafka_msg.partition, lag)
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise
    
    async def seek_to_beginning(self, partitions: Optional[List[TopicPartition]] = None):
        """Seek to the beginning of partitions."""
        if not self._is_started:
            raise RuntimeError("Consumer not started")
        
        if partitions:
            await self._consumer.seek_to_beginning(*partitions)
        else:
            await self._consumer.seek_to_beginning()
    
    async def seek_to_end(self, partitions: Optional[List[TopicPartition]] = None):
        """Seek to the end of partitions."""
        if not self._is_started:
            raise RuntimeError("Consumer not started")
        
        if partitions:
            await self._consumer.seek_to_end(*partitions)
        else:
            await self._consumer.seek_to_end()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return self._metrics.get_metrics()
    
    async def commit_async(self, offsets: Optional[Dict[TopicPartition, int]] = None):
        """Commit offsets asynchronously."""
        if not self._is_started:
            raise RuntimeError("Consumer not started")
        
        try:
            if offsets:
                await self._consumer.commit(offsets)
            else:
                await self._consumer.commit()
        except CommitFailedError as e:
            logger.error(f"Failed to commit offsets: {e}")
            raise