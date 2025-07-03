"""
Tests for Kafka consumer functionality.
"""

import pytest
import asyncio
from datetime import datetime
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from alpha_pulse.streaming.kafka_consumer import (
    AlphaPulseKafkaConsumer, ConsumerConfig, ConsumerMode,
    OffsetResetPolicy, MessageDeserializer, ConsumerMetrics,
    MessageProcessor
)
from alpha_pulse.models.streaming_message import (
    StreamingMessage, MarketDataMessage, MessageHeader, MessageType
)
from alpha_pulse.streaming.kafka_producer import SerializationFormat


class TestConsumerConfig:
    """Test consumer configuration."""
    
    def test_from_config(self):
        """Test creating config from dictionary."""
        config_dict = {
            "bootstrap_servers": ["localhost:9092"],
            "group_id": "test-group",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": True
        }
        
        config = ConsumerConfig.from_config(config_dict)
        
        assert config.bootstrap_servers == ["localhost:9092"]
        assert config.group_id == "test-group"
        assert config.auto_offset_reset == OffsetResetPolicy.EARLIEST
        assert config.enable_auto_commit is True
    
    def test_required_fields(self):
        """Test required configuration fields."""
        with pytest.raises(KeyError):
            # Missing group_id
            ConsumerConfig.from_config({})


class TestMessageDeserializer:
    """Test message deserialization."""
    
    def test_json_deserialization(self):
        """Test JSON deserialization."""
        deserializer = MessageDeserializer(SerializationFormat.JSON)
        
        # Create test data
        test_data = {
            "header": {
                "message_id": "test-123",
                "message_type": "market_data",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "test",
                "version": "1.0",
                "metadata": {}
            },
            "payload": {"test": "data"},
            "schema_version": "1.0"
        }
        
        serialized = json.dumps(test_data).encode('utf-8')
        message = deserializer.deserialize(serialized)
        
        assert isinstance(message, StreamingMessage)
        assert message.header.message_id == "test-123"
        assert message.header.message_type == MessageType.MARKET_DATA
        assert message.payload["test"] == "data"
    
    def test_typed_message_reconstruction(self):
        """Test reconstruction of typed messages."""
        deserializer = MessageDeserializer(SerializationFormat.JSON)
        
        # Create market data message
        test_data = {
            "header": {
                "message_id": "market-123",
                "message_type": "market_data",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "test",
                "version": "1.0",
                "metadata": {}
            },
            "payload": {
                "symbol": "BTC/USD",
                "exchange": "binance",
                "last_trade": "50000"
            },
            "schema_version": "1.0"
        }
        
        serialized = json.dumps(test_data).encode('utf-8')
        message = deserializer.deserialize(serialized, MessageType.MARKET_DATA)
        
        assert isinstance(message, MarketDataMessage)
        assert message.symbol == "BTC/USD"
        assert message.exchange == "binance"


class TestConsumerMetrics:
    """Test consumer metrics tracking."""
    
    def test_record_message(self):
        """Test recording successful message."""
        metrics = ConsumerMetrics()
        
        metrics.record_message(size=1024, processing_time=0.01)
        metrics.record_message(size=2048, processing_time=0.02)
        
        assert metrics.messages_consumed == 2
        assert metrics.bytes_consumed == 3072
        assert len(metrics.processing_times) == 2
        assert metrics.last_message_time is not None
    
    def test_record_error(self):
        """Test recording errors."""
        metrics = ConsumerMetrics()
        
        metrics.record_error("TimeoutError")
        metrics.record_error("TimeoutError")
        metrics.record_error("ValueError")
        
        assert metrics.messages_failed == 3
        assert metrics.errors_by_type["TimeoutError"] == 2
        assert metrics.errors_by_type["ValueError"] == 1
    
    def test_update_lag(self):
        """Test updating partition lag."""
        metrics = ConsumerMetrics()
        
        metrics.update_lag(partition=0, lag=100)
        metrics.update_lag(partition=1, lag=200)
        metrics.update_lag(partition=0, lag=150)  # Update partition 0
        
        assert metrics.lag_by_partition[0] == 150
        assert metrics.lag_by_partition[1] == 200
    
    def test_get_metrics(self):
        """Test getting metrics summary."""
        metrics = ConsumerMetrics()
        
        metrics.record_message(size=1000, processing_time=0.01)
        metrics.record_error("TestError")
        metrics.update_lag(0, 50)
        
        summary = metrics.get_metrics()
        
        assert summary["messages_consumed"] == 1
        assert summary["messages_failed"] == 1
        assert summary["bytes_consumed"] == 1000
        assert summary["total_lag"] == 50
        assert summary["avg_processing_time_ms"] == 10.0


@pytest.mark.asyncio
class TestMessageProcessor:
    """Test message processor with retry logic."""
    
    async def test_successful_processing(self):
        """Test successful message processing."""
        # Create handler
        handler_called = False
        async def handler(message):
            nonlocal handler_called
            handler_called = True
        
        processor = MessageProcessor(handler=handler)
        
        # Process message
        message = Mock(spec=StreamingMessage)
        message.header.message_id = "test-123"
        
        success = await processor.process(message)
        
        assert success is True
        assert handler_called is True
    
    async def test_retry_on_failure(self):
        """Test retry logic on failure."""
        attempt_count = 0
        
        async def failing_handler(message):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Test error")
        
        processor = MessageProcessor(
            handler=failing_handler,
            max_retries=3,
            retry_delay_ms=10
        )
        
        message = Mock(spec=StreamingMessage)
        message.header.message_id = "test-123"
        
        success = await processor.process(message)
        
        assert success is True
        assert attempt_count == 3
    
    async def test_dead_letter_on_max_retries(self):
        """Test dead letter queue on max retries exceeded."""
        dlq_called = False
        dlq_message = None
        dlq_error = None
        
        async def failing_handler(message):
            raise ValueError("Persistent error")
        
        async def dead_letter_handler(message, error):
            nonlocal dlq_called, dlq_message, dlq_error
            dlq_called = True
            dlq_message = message
            dlq_error = error
        
        processor = MessageProcessor(
            handler=failing_handler,
            dead_letter_handler=dead_letter_handler,
            max_retries=2,
            retry_delay_ms=10
        )
        
        message = Mock(spec=StreamingMessage)
        message.header.message_id = "test-123"
        
        success = await processor.process(message)
        
        assert success is False
        assert dlq_called is True
        assert dlq_message == message
        assert isinstance(dlq_error, ValueError)


@pytest.mark.asyncio
class TestAlphaPulseKafkaConsumer:
    """Test Kafka consumer."""
    
    async def test_consumer_lifecycle(self):
        """Test consumer start/stop lifecycle."""
        config = ConsumerConfig(
            bootstrap_servers=["localhost:9092"],
            group_id="test-group"
        )
        consumer = AlphaPulseKafkaConsumer(config)
        
        with patch('alpha_pulse.streaming.kafka_consumer.AIOKafkaConsumer') as mock_consumer_class:
            mock_consumer = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            # Start consumer
            await consumer.start()
            assert consumer._is_started is True
            mock_consumer.start.assert_called_once()
            
            # Stop consumer
            await consumer.stop()
            assert consumer._is_started is False
            mock_consumer.stop.assert_called_once()
    
    async def test_subscribe_to_topics(self):
        """Test subscribing to topics."""
        config = ConsumerConfig(
            bootstrap_servers=["localhost:9092"],
            group_id="test-group"
        )
        consumer = AlphaPulseKafkaConsumer(config)
        
        with patch('alpha_pulse.streaming.kafka_consumer.AIOKafkaConsumer') as mock_consumer_class:
            mock_consumer = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            await consumer.start()
            
            # Subscribe to topics
            topics = ["topic1", "topic2"]
            await consumer.subscribe(topics)
            
            mock_consumer.subscribe.assert_called_once()
            call_args = mock_consumer.subscribe.call_args[0]
            assert call_args[0] == topics
    
    async def test_subscribe_by_message_types(self):
        """Test subscribing by message types."""
        config = ConsumerConfig(
            bootstrap_servers=["localhost:9092"],
            group_id="test-group"
        )
        consumer = AlphaPulseKafkaConsumer(config)
        
        with patch('alpha_pulse.streaming.kafka_consumer.AIOKafkaConsumer') as mock_consumer_class:
            mock_consumer = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            await consumer.start()
            
            # Subscribe by message types
            await consumer.subscribe_by_message_types([
                MessageType.MARKET_DATA,
                MessageType.TRADING_SIGNAL
            ])
            
            mock_consumer.subscribe.assert_called_once()
            call_args = mock_consumer.subscribe.call_args[0]
            assert "alphapulse.market-data" in call_args[0]
            assert "alphapulse.trading-signals" in call_args[0]
    
    async def test_register_handler(self):
        """Test registering message handlers."""
        config = ConsumerConfig(
            bootstrap_servers=["localhost:9092"],
            group_id="test-group"
        )
        consumer = AlphaPulseKafkaConsumer(config)
        
        handler_called = False
        async def test_handler(message):
            nonlocal handler_called
            handler_called = True
        
        consumer.register_handler(
            MessageType.MARKET_DATA,
            test_handler
        )
        
        assert MessageType.MARKET_DATA in consumer._message_processors
    
    async def test_consume_messages(self):
        """Test consuming messages."""
        config = ConsumerConfig(
            bootstrap_servers=["localhost:9092"],
            group_id="test-group"
        )
        consumer = AlphaPulseKafkaConsumer(config)
        
        with patch('alpha_pulse.streaming.kafka_consumer.AIOKafkaConsumer') as mock_consumer_class:
            mock_consumer = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            # Mock message
            mock_kafka_msg = Mock()
            mock_kafka_msg.topic = "alphapulse.market-data"
            mock_kafka_msg.partition = 0
            mock_kafka_msg.offset = 100
            mock_kafka_msg.key = b"BTC/USD"
            mock_kafka_msg.timestamp = int(datetime.utcnow().timestamp() * 1000)
            mock_kafka_msg.headers = [(b"message_type", b"market_data")]
            
            # Create message data
            message_data = {
                "header": {
                    "message_id": "test-123",
                    "message_type": "market_data",
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "test",
                    "version": "1.0",
                    "metadata": {}
                },
                "payload": {
                    "symbol": "BTC/USD",
                    "exchange": "binance"
                },
                "schema_version": "1.0"
            }
            mock_kafka_msg.value = json.dumps(message_data).encode('utf-8')
            
            # Set up async iterator
            async def mock_iter():
                yield mock_kafka_msg
                raise StopAsyncIteration
            
            mock_consumer.__aiter__.return_value = mock_iter()
            
            await consumer.start()
            
            # Register handler
            handler_called = False
            processed_message = None
            
            async def handler(message):
                nonlocal handler_called, processed_message
                handler_called = True
                processed_message = message
            
            consumer.register_handler(MessageType.MARKET_DATA, handler)
            
            # Consume
            await consumer.consume(max_messages=1)
            
            # Verify handler was called
            assert handler_called is True
            assert processed_message is not None
            assert processed_message.header.message_id == "test-123"