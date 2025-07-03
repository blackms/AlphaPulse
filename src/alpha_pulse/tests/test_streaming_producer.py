"""
Tests for Kafka producer functionality.
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
import uuid
from unittest.mock import Mock, AsyncMock, patch

from alpha_pulse.streaming.kafka_producer import (
    AlphaPulseKafkaProducer, ProducerConfig, SerializationFormat,
    PartitionStrategy, MessageSerializer, PartitionSelector, TopicManager
)
from alpha_pulse.models.streaming_message import (
    StreamingMessage, MarketDataMessage, TradingSignalMessage,
    MessageHeader, MessageType, SignalType
)
from alpha_pulse.models.market_data import OHLCV


class TestProducerConfig:
    """Test producer configuration."""
    
    def test_from_config(self):
        """Test creating config from dictionary."""
        config_dict = {
            "bootstrap_servers": ["localhost:9092", "localhost:9093"],
            "client_id": "test-producer",
            "acks": 1,
            "compression_type": "snappy"
        }
        
        config = ProducerConfig.from_config(config_dict)
        
        assert config.bootstrap_servers == ["localhost:9092", "localhost:9093"]
        assert config.client_id == "test-producer"
        assert config.acks == 1
        assert config.compression_type == "snappy"
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProducerConfig(bootstrap_servers=["localhost:9092"])
        
        assert config.client_id == "alphapulse-producer"
        assert config.acks == "all"
        assert config.retries == 3
        assert config.enable_idempotence is True


class TestMessageSerializer:
    """Test message serialization."""
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        serializer = MessageSerializer(SerializationFormat.JSON)
        
        message = StreamingMessage(
            header=MessageHeader(
                message_id="test-123",
                message_type=MessageType.MARKET_DATA,
                timestamp=datetime.utcnow(),
                source="test"
            ),
            payload={"test": "data"}
        )
        
        serialized = serializer.serialize(message)
        assert isinstance(serialized, bytes)
        
        # Verify it's valid JSON
        import json
        deserialized = json.loads(serialized.decode('utf-8'))
        assert deserialized["header"]["message_id"] == "test-123"
        assert deserialized["payload"]["test"] == "data"
    
    def test_msgpack_serialization(self):
        """Test MessagePack serialization."""
        serializer = MessageSerializer(SerializationFormat.MSGPACK)
        
        message = StreamingMessage(
            header=MessageHeader(
                message_id="test-456",
                message_type=MessageType.TRADING_SIGNAL,
                timestamp=datetime.utcnow(),
                source="test"
            ),
            payload={"signal": "buy"}
        )
        
        serialized = serializer.serialize(message)
        assert isinstance(serialized, bytes)
        
        # Verify it's valid MessagePack
        import msgpack
        deserialized = msgpack.unpackb(serialized, raw=False)
        assert deserialized["header"]["message_id"] == "test-456"


class TestPartitionSelector:
    """Test partition selection strategies."""
    
    def test_key_based_selection(self):
        """Test key-based partition selection."""
        selector = PartitionSelector(PartitionStrategy.KEY_BASED, num_partitions=10)
        
        message = Mock(spec=StreamingMessage)
        message.header.message_id = "test-123"
        
        # Same key should always get same partition
        partition1 = selector.select_partition("symbol1", message)
        partition2 = selector.select_partition("symbol1", message)
        assert partition1 == partition2
        assert 0 <= partition1 < 10
        
        # Different keys should (likely) get different partitions
        partition3 = selector.select_partition("symbol2", message)
        assert 0 <= partition3 < 10
    
    def test_round_robin_selection(self):
        """Test round-robin partition selection."""
        selector = PartitionSelector(PartitionStrategy.ROUND_ROBIN, num_partitions=3)
        
        message = Mock(spec=StreamingMessage)
        
        # Should cycle through partitions
        partitions = []
        for _ in range(6):
            partition = selector.select_partition(None, message)
            partitions.append(partition)
        
        assert partitions == [0, 1, 2, 0, 1, 2]
    
    def test_random_selection(self):
        """Test random partition selection."""
        selector = PartitionSelector(PartitionStrategy.RANDOM, num_partitions=5)
        
        message = Mock(spec=StreamingMessage)
        
        # Should return valid partition numbers
        for _ in range(10):
            partition = selector.select_partition(None, message)
            assert 0 <= partition < 5


@pytest.mark.asyncio
class TestAlphaPulseKafkaProducer:
    """Test Kafka producer."""
    
    async def test_producer_lifecycle(self):
        """Test producer start/stop lifecycle."""
        config = ProducerConfig(bootstrap_servers=["localhost:9092"])
        producer = AlphaPulseKafkaProducer(config)
        
        # Mock the internal producer
        with patch('alpha_pulse.streaming.kafka_producer.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Start producer
            await producer.start()
            assert producer._is_started is True
            mock_producer.start.assert_called_once()
            
            # Stop producer
            await producer.stop()
            assert producer._is_started is False
            mock_producer.stop.assert_called_once()
    
    async def test_send_message(self):
        """Test sending a message."""
        config = ProducerConfig(bootstrap_servers=["localhost:9092"])
        producer = AlphaPulseKafkaProducer(config)
        
        # Mock the internal producer
        with patch('alpha_pulse.streaming.kafka_producer.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Mock send result
            mock_record = Mock()
            mock_record.partition = 0
            mock_record.offset = 123
            mock_producer.send.return_value = mock_record
            
            await producer.start()
            
            # Create and send message
            message = MarketDataMessage(
                header=MessageHeader(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.MARKET_DATA,
                    timestamp=datetime.utcnow(),
                    source="test"
                ),
                payload={},
                symbol="BTC/USD",
                exchange="binance",
                last_trade=Decimal("50000")
            )
            
            envelope = await producer.send_message(
                topic="test-topic",
                message=message,
                key="BTC/USD"
            )
            
            # Verify send was called
            mock_producer.send.assert_called_once()
            
            # Verify envelope
            assert envelope.topic == "test-topic"
            assert envelope.partition == 0
            assert envelope.offset == 123
            assert envelope.key == "BTC/USD"
            
            # Verify metrics updated
            assert producer._metrics["messages_sent"] == 1
            assert producer._metrics["bytes_sent"] > 0
    
    async def test_send_batch(self):
        """Test sending a batch of messages."""
        config = ProducerConfig(bootstrap_servers=["localhost:9092"])
        producer = AlphaPulseKafkaProducer(config)
        
        with patch('alpha_pulse.streaming.kafka_producer.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Mock send results
            mock_record = Mock()
            mock_record.partition = 0
            mock_record.offset = 100
            mock_producer.send.return_value = mock_record
            
            await producer.start()
            
            # Create batch
            from alpha_pulse.models.streaming_message import MessageBatch
            messages = []
            for i in range(5):
                msg = StreamingMessage(
                    header=MessageHeader(
                        message_id=f"msg-{i}",
                        message_type=MessageType.MARKET_DATA,
                        timestamp=datetime.utcnow(),
                        source="test"
                    ),
                    payload={"index": i}
                )
                messages.append(msg)
            
            batch = MessageBatch(
                messages=messages,
                batch_id="batch-123",
                created_at=datetime.utcnow()
            )
            
            # Send batch
            envelopes = await producer.send_batch(
                topic="test-topic",
                batch=batch
            )
            
            # Verify all messages sent
            assert len(envelopes) == 5
            assert mock_producer.send.call_count == 5


class TestTopicManager:
    """Test topic management."""
    
    def test_get_topic(self):
        """Test getting topic for message type."""
        assert TopicManager.get_topic(MessageType.MARKET_DATA) == "alphapulse.market-data"
        assert TopicManager.get_topic(MessageType.TRADING_SIGNAL) == "alphapulse.trading-signals"
        assert TopicManager.get_topic(MessageType.RISK_ALERT) == "alphapulse.risk-alerts"
    
    def test_is_high_priority(self):
        """Test high priority topic identification."""
        assert TopicManager.is_high_priority(MessageType.TRADING_SIGNAL) is True
        assert TopicManager.is_high_priority(MessageType.RISK_ALERT) is True
        assert TopicManager.is_high_priority(MessageType.MARKET_DATA) is False
    
    def test_get_all_topics(self):
        """Test getting all topics."""
        topics = TopicManager.get_all_topics()
        assert len(topics) == 7
        assert "alphapulse.market-data" in topics
        assert "alphapulse.trading-signals" in topics