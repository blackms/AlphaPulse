"""
Kafka producer implementation for streaming data.

Provides:
- Asynchronous message production
- Partitioning strategies
- Message serialization
- Delivery guarantees
- Error handling and retries
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from kafka import KafkaProducer as SyncKafkaProducer
import msgpack

from alpha_pulse.models.streaming_message import (
    StreamingMessage, MessageEnvelope, MessageBatch,
    MessageType, MarketDataMessage, TradingSignalMessage
)
from alpha_pulse.config.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    MSGPACK = "msgpack"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class PartitionStrategy(Enum):
    """Partitioning strategies for message routing."""
    RANDOM = "random"
    KEY_BASED = "key_based"
    ROUND_ROBIN = "round_robin"
    CUSTOM = "custom"


@dataclass
class ProducerConfig:
    """Kafka producer configuration."""
    bootstrap_servers: List[str]
    client_id: str = "alphapulse-producer"
    acks: Union[int, str] = "all"  # 0, 1, or 'all'
    retries: int = 3
    max_in_flight_requests: int = 5
    compression_type: str = "gzip"  # none, gzip, snappy, lz4, zstd
    batch_size: int = 16384
    linger_ms: int = 10
    buffer_memory: int = 33554432  # 32MB
    request_timeout_ms: int = 30000
    enable_idempotence: bool = True
    serialization_format: SerializationFormat = SerializationFormat.JSON
    partition_strategy: PartitionStrategy = PartitionStrategy.KEY_BASED
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ProducerConfig':
        """Create from configuration dictionary."""
        return cls(
            bootstrap_servers=config.get("bootstrap_servers", ["localhost:9092"]),
            client_id=config.get("client_id", "alphapulse-producer"),
            acks=config.get("acks", "all"),
            retries=config.get("retries", 3),
            max_in_flight_requests=config.get("max_in_flight_requests", 5),
            compression_type=config.get("compression_type", "gzip"),
            batch_size=config.get("batch_size", 16384),
            linger_ms=config.get("linger_ms", 10),
            buffer_memory=config.get("buffer_memory", 33554432),
            request_timeout_ms=config.get("request_timeout_ms", 30000),
            enable_idempotence=config.get("enable_idempotence", True),
            serialization_format=SerializationFormat(
                config.get("serialization_format", "json")
            ),
            partition_strategy=PartitionStrategy(
                config.get("partition_strategy", "key_based")
            )
        )


class MessageSerializer:
    """Handles message serialization for different formats."""
    
    def __init__(self, format: SerializationFormat):
        """Initialize serializer."""
        self.format = format
        self._serializers = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.MSGPACK: self._serialize_msgpack,
            SerializationFormat.AVRO: self._serialize_avro,
            SerializationFormat.PROTOBUF: self._serialize_protobuf
        }
    
    def serialize(self, message: StreamingMessage) -> bytes:
        """Serialize message to bytes."""
        serializer = self._serializers.get(self.format)
        if not serializer:
            raise ValueError(f"Unsupported serialization format: {self.format}")
        
        return serializer(message)
    
    def _serialize_json(self, message: StreamingMessage) -> bytes:
        """Serialize to JSON."""
        return json.dumps(message.to_dict(), default=str).encode('utf-8')
    
    def _serialize_msgpack(self, message: StreamingMessage) -> bytes:
        """Serialize to MessagePack."""
        return msgpack.packb(message.to_dict(), use_bin_type=True)
    
    def _serialize_avro(self, message: StreamingMessage) -> bytes:
        """Serialize to Avro format."""
        # TODO: Implement Avro serialization with schema registry
        raise NotImplementedError("Avro serialization not yet implemented")
    
    def _serialize_protobuf(self, message: StreamingMessage) -> bytes:
        """Serialize to Protocol Buffers."""
        # TODO: Implement Protobuf serialization
        raise NotImplementedError("Protobuf serialization not yet implemented")


class PartitionSelector:
    """Handles partition selection for messages."""
    
    def __init__(self, strategy: PartitionStrategy, num_partitions: int):
        """Initialize partition selector."""
        self.strategy = strategy
        self.num_partitions = num_partitions
        self._round_robin_counter = 0
        self._selectors = {
            PartitionStrategy.RANDOM: self._select_random,
            PartitionStrategy.KEY_BASED: self._select_key_based,
            PartitionStrategy.ROUND_ROBIN: self._select_round_robin,
            PartitionStrategy.CUSTOM: self._select_custom
        }
    
    def select_partition(
        self, 
        key: Optional[str], 
        message: StreamingMessage,
        custom_selector: Optional[Callable] = None
    ) -> int:
        """Select partition for message."""
        selector = self._selectors.get(self.strategy)
        if not selector:
            raise ValueError(f"Unsupported partition strategy: {self.strategy}")
        
        if self.strategy == PartitionStrategy.CUSTOM and not custom_selector:
            raise ValueError("Custom selector required for CUSTOM strategy")
        
        return selector(key, message, custom_selector)
    
    def _select_random(self, key: Optional[str], message: StreamingMessage, _) -> int:
        """Random partition selection."""
        import random
        return random.randint(0, self.num_partitions - 1)
    
    def _select_key_based(self, key: Optional[str], message: StreamingMessage, _) -> int:
        """Key-based partition selection using consistent hashing."""
        if not key:
            # Fall back to message ID if no key provided
            key = message.header.message_id
        
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_partitions
    
    def _select_round_robin(self, key: Optional[str], message: StreamingMessage, _) -> int:
        """Round-robin partition selection."""
        partition = self._round_robin_counter % self.num_partitions
        self._round_robin_counter += 1
        return partition
    
    def _select_custom(
        self, 
        key: Optional[str], 
        message: StreamingMessage,
        custom_selector: Callable
    ) -> int:
        """Custom partition selection."""
        return custom_selector(key, message, self.num_partitions)


class AlphaPulseKafkaProducer:
    """Kafka producer for AlphaPulse streaming data."""
    
    def __init__(self, config: ProducerConfig):
        """Initialize Kafka producer."""
        self.config = config
        self._producer: Optional[AIOKafkaProducer] = None
        self._sync_producer: Optional[SyncKafkaProducer] = None
        self._serializer = MessageSerializer(config.serialization_format)
        self._partition_selector: Optional[PartitionSelector] = None
        self._is_started = False
        self._metrics = {
            "messages_sent": 0,
            "messages_failed": 0,
            "bytes_sent": 0,
            "last_error": None,
            "last_send_time": None
        }
    
    async def start(self):
        """Start the async producer."""
        if self._is_started:
            logger.warning("Producer already started")
            return
        
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                acks=self.config.acks,
                enable_idempotence=self.config.enable_idempotence,
                max_batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                compression_type=self.config.compression_type,
                max_request_size=1048576,  # 1MB
                value_serializer=lambda v: v  # We handle serialization
            )
            
            await self._producer.start()
            self._is_started = True
            
            # Get partition info for the first topic (will be updated per topic)
            # This is a placeholder - actual implementation would query per topic
            self._partition_selector = PartitionSelector(
                self.config.partition_strategy,
                num_partitions=3  # Default, will be updated
            )
            
            logger.info(f"Kafka producer started: {self.config.bootstrap_servers}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise
    
    async def stop(self):
        """Stop the async producer."""
        if self._producer and self._is_started:
            await self._producer.stop()
            self._is_started = False
            logger.info("Kafka producer stopped")
    
    def start_sync(self):
        """Start synchronous producer for non-async contexts."""
        if self._sync_producer:
            return
        
        self._sync_producer = SyncKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            client_id=f"{self.config.client_id}-sync",
            acks=self.config.acks,
            retries=self.config.retries,
            max_in_flight_requests_per_connection=self.config.max_in_flight_requests,
            compression_type=self.config.compression_type,
            batch_size=self.config.batch_size,
            linger_ms=self.config.linger_ms,
            buffer_memory=self.config.buffer_memory,
            value_serializer=lambda v: v  # We handle serialization
        )
        
        logger.info(f"Sync Kafka producer started: {self.config.bootstrap_servers}")
    
    def stop_sync(self):
        """Stop synchronous producer."""
        if self._sync_producer:
            self._sync_producer.close()
            self._sync_producer = None
            logger.info("Sync Kafka producer stopped")
    
    async def send_message(
        self,
        topic: str,
        message: StreamingMessage,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        custom_partition_selector: Optional[Callable] = None
    ) -> MessageEnvelope:
        """Send a single message asynchronously."""
        if not self._is_started:
            raise RuntimeError("Producer not started. Call start() first.")
        
        try:
            # Serialize message
            value = self._serializer.serialize(message)
            
            # Determine partition if not specified
            if partition is None and self._partition_selector:
                partition = self._partition_selector.select_partition(
                    key, message, custom_partition_selector
                )
            
            # Convert headers to list of tuples
            kafka_headers = []
            if headers:
                kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]
            
            # Add message type header
            kafka_headers.append(
                ("message_type", message.header.message_type.value.encode('utf-8'))
            )
            
            # Send message
            record_metadata = await self._producer.send(
                topic=topic,
                value=value,
                key=key.encode('utf-8') if key else None,
                partition=partition,
                headers=kafka_headers
            )
            
            # Update metrics
            self._metrics["messages_sent"] += 1
            self._metrics["bytes_sent"] += len(value)
            self._metrics["last_send_time"] = datetime.utcnow()
            
            # Create envelope
            envelope = MessageEnvelope(
                topic=topic,
                partition=record_metadata.partition,
                key=key,
                headers=headers or {},
                timestamp=datetime.utcnow(),
                offset=record_metadata.offset
            )
            
            logger.debug(
                f"Message sent to {topic}:{record_metadata.partition} "
                f"at offset {record_metadata.offset}"
            )
            
            return envelope
            
        except Exception as e:
            self._metrics["messages_failed"] += 1
            self._metrics["last_error"] = str(e)
            logger.error(f"Failed to send message: {e}")
            raise
    
    def send_message_sync(
        self,
        topic: str,
        message: StreamingMessage,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> MessageEnvelope:
        """Send a single message synchronously."""
        if not self._sync_producer:
            raise RuntimeError("Sync producer not started. Call start_sync() first.")
        
        try:
            # Serialize message
            value = self._serializer.serialize(message)
            
            # Convert headers to list of tuples
            kafka_headers = []
            if headers:
                kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]
            
            # Add message type header
            kafka_headers.append(
                ("message_type", message.header.message_type.value.encode('utf-8'))
            )
            
            # Send message
            record_metadata = self._sync_producer.send(
                topic=topic,
                value=value,
                key=key.encode('utf-8') if key else None,
                partition=partition,
                headers=kafka_headers
            ).get(timeout=self.config.request_timeout_ms / 1000)
            
            # Create envelope
            envelope = MessageEnvelope(
                topic=topic,
                partition=record_metadata.partition,
                key=key,
                headers=headers or {},
                timestamp=datetime.utcnow(),
                offset=record_metadata.offset
            )
            
            return envelope
            
        except Exception as e:
            logger.error(f"Failed to send sync message: {e}")
            raise
    
    async def send_batch(
        self,
        topic: str,
        batch: MessageBatch,
        key_extractor: Optional[Callable[[StreamingMessage], str]] = None
    ) -> List[MessageEnvelope]:
        """Send a batch of messages."""
        envelopes = []
        
        for message in batch.messages:
            key = key_extractor(message) if key_extractor else None
            envelope = await self.send_message(
                topic=topic,
                message=message,
                key=key,
                headers={"batch_id": batch.batch_id}
            )
            envelopes.append(envelope)
        
        logger.info(f"Sent batch {batch.batch_id} with {batch.size()} messages")
        return envelopes
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        return self._metrics.copy()
    
    async def flush(self):
        """Flush any pending messages."""
        if self._producer and self._is_started:
            await self._producer.flush()
    
    def flush_sync(self):
        """Flush pending messages synchronously."""
        if self._sync_producer:
            self._sync_producer.flush()


class TopicManager:
    """Manages Kafka topics for AlphaPulse."""
    
    # Topic definitions
    TOPICS = {
        MessageType.MARKET_DATA: "alphapulse.market-data",
        MessageType.TRADING_SIGNAL: "alphapulse.trading-signals",
        MessageType.RISK_ALERT: "alphapulse.risk-alerts",
        MessageType.PORTFOLIO_UPDATE: "alphapulse.portfolio-updates",
        MessageType.SYSTEM_EVENT: "alphapulse.system-events",
        MessageType.DATA_QUALITY_EVENT: "alphapulse.data-quality-events",
        MessageType.AUDIT_EVENT: "alphapulse.audit-events"
    }
    
    # High-priority topics
    HIGH_PRIORITY_TOPICS = {
        MessageType.TRADING_SIGNAL,
        MessageType.RISK_ALERT
    }
    
    @classmethod
    def get_topic(cls, message_type: MessageType) -> str:
        """Get topic name for message type."""
        return cls.TOPICS.get(message_type, "alphapulse.default")
    
    @classmethod
    def is_high_priority(cls, message_type: MessageType) -> bool:
        """Check if message type is high priority."""
        return message_type in cls.HIGH_PRIORITY_TOPICS
    
    @classmethod
    def get_all_topics(cls) -> List[str]:
        """Get all topic names."""
        return list(cls.TOPICS.values())