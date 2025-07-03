"""
Demo script for streaming data processing with Kafka.

This demonstrates:
- Setting up Kafka producer and consumer
- Streaming market data
- Real-time pattern detection
- Complex event processing
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import uuid
from typing import List

from alpha_pulse.streaming import (
    AlphaPulseKafkaProducer, ProducerConfig,
    AlphaPulseKafkaConsumer, ConsumerConfig,
    MarketDataProcessor, TradingSignalProcessor,
    StreamingPipeline, WindowConfig, WindowType,
    AggregationType, ComplexEventProcessor,
    EventPattern, PatternType
)
from alpha_pulse.models.streaming_message import (
    MessageFactory, MarketDataMessage, TradingSignalMessage,
    MessageType, SignalType, MessageHeader
)
from alpha_pulse.models.market_data import OHLCV
from alpha_pulse.config.config_loader import ConfigLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDemo:
    """Demonstrates streaming data processing."""
    
    def __init__(self):
        """Initialize demo."""
        self.config = ConfigLoader().load_config("streaming_config.yaml")
        self.producer = None
        self.consumer = None
        self.cep_engine = ComplexEventProcessor()
        self.is_running = False
    
    async def setup(self):
        """Set up streaming components."""
        # Create producer
        producer_config = ProducerConfig.from_config(
            self.config["kafka"]["producer"]
        )
        producer_config.bootstrap_servers = self.config["kafka"]["bootstrap_servers"]
        self.producer = AlphaPulseKafkaProducer(producer_config)
        await self.producer.start()
        
        # Create consumer
        consumer_config = ConsumerConfig.from_config(
            self.config["kafka"]["consumer"]
        )
        consumer_config.bootstrap_servers = self.config["kafka"]["bootstrap_servers"]
        consumer_config.group_id = f"demo-consumer-{uuid.uuid4().hex[:8]}"
        self.consumer = AlphaPulseKafkaConsumer(consumer_config)
        await self.consumer.start()
        
        # Subscribe to topics
        await self.consumer.subscribe_by_message_types([
            MessageType.MARKET_DATA,
            MessageType.TRADING_SIGNAL
        ])
        
        # Register CEP patterns
        self._register_patterns()
        
        logger.info("Streaming components initialized")
    
    def _register_patterns(self):
        """Register complex event patterns."""
        # Rapid price movement pattern
        rapid_price_pattern = EventPattern(
            name="rapid_price_movement",
            pattern_type=PatternType.THRESHOLD,
            conditions=[{
                "type": "field_range",
                "field": "payload.price_change_pct",
                "min": 0.02  # 2% change
            }],
            time_window=timedelta(seconds=60)
        )
        self.cep_engine.register_pattern(rapid_price_pattern)
        
        # Volume spike pattern
        volume_spike_pattern = EventPattern(
            name="volume_spike",
            pattern_type=PatternType.THRESHOLD,
            conditions=[{
                "type": "field_range",
                "field": "payload.volume",
                "min": 100000
            }],
            time_window=timedelta(minutes=5)
        )
        self.cep_engine.register_pattern(volume_spike_pattern)
        
        # Conflicting signals pattern
        conflicting_signals_pattern = EventPattern(
            name="conflicting_signals",
            pattern_type=PatternType.SEQUENCE,
            conditions=[
                {
                    "type": "message_type",
                    "value": "trading_signal"
                },
                {
                    "type": "custom",
                    "function": lambda msg: (
                        hasattr(msg, 'signal_type') and 
                        msg.signal_type in [SignalType.BUY, SignalType.SELL]
                    )
                }
            ],
            time_window=timedelta(seconds=30),
            min_occurrences=2
        )
        self.cep_engine.register_pattern(conflicting_signals_pattern)
        
        # Register pattern handlers
        self.cep_engine.register_event_handler(
            "rapid_price_movement",
            self._handle_rapid_price_movement
        )
        self.cep_engine.register_event_handler(
            "volume_spike",
            self._handle_volume_spike
        )
    
    async def _handle_rapid_price_movement(self, pattern_match):
        """Handle rapid price movement detection."""
        logger.warning(
            f"⚡ Rapid price movement detected: {pattern_match.pattern.name} "
            f"at {pattern_match.match_time}"
        )
    
    async def _handle_volume_spike(self, pattern_match):
        """Handle volume spike detection."""
        logger.warning(
            f"📊 Volume spike detected: {pattern_match.pattern.name} "
            f"at {pattern_match.match_time}"
        )
    
    async def generate_sample_data(self, duration_seconds: int = 60):
        """Generate sample streaming data."""
        logger.info(f"Generating sample data for {duration_seconds} seconds...")
        
        symbols = ["BTC/USD", "ETH/USD", "AAPL", "GOOGL", "MSFT"]
        exchanges = ["binance", "coinbase", "nasdaq"]
        strategies = ["momentum", "mean_reversion", "arbitrage"]
        
        start_time = asyncio.get_event_loop().time()
        message_count = 0
        
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            # Generate market data
            for symbol in symbols:
                # Create market data message
                base_price = 100 + random.uniform(-10, 10)
                market_data = MarketDataMessage(
                    header=MessageHeader(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.MARKET_DATA,
                        timestamp=datetime.utcnow(),
                        source="demo_generator"
                    ),
                    payload={},
                    symbol=symbol,
                    exchange=random.choice(exchanges),
                    ohlcv=OHLCV(
                        open=Decimal(str(base_price)),
                        high=Decimal(str(base_price + random.uniform(0, 2))),
                        low=Decimal(str(base_price - random.uniform(0, 2))),
                        close=Decimal(str(base_price + random.uniform(-1, 1))),
                        volume=Decimal(str(random.randint(1000, 200000)))
                    ),
                    bid=Decimal(str(base_price - 0.1)),
                    ask=Decimal(str(base_price + 0.1)),
                    last_trade=Decimal(str(base_price))
                )
                
                # Add price change for pattern detection
                market_data.payload["price_change_pct"] = random.uniform(-0.05, 0.05)
                market_data.payload["volume"] = float(market_data.ohlcv.volume)
                
                # Send to Kafka
                try:
                    await self.producer.send_message(
                        topic="alphapulse.market-data",
                        message=market_data,
                        key=symbol
                    )
                    message_count += 1
                except Exception as e:
                    logger.error(f"Failed to send market data: {e}")
            
            # Generate trading signals occasionally
            if random.random() < 0.2:  # 20% chance
                signal = TradingSignalMessage(
                    header=MessageHeader(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.TRADING_SIGNAL,
                        timestamp=datetime.utcnow(),
                        source="demo_generator"
                    ),
                    payload={},
                    signal_id=str(uuid.uuid4()),
                    symbol=random.choice(symbols),
                    signal_type=random.choice([SignalType.BUY, SignalType.SELL, SignalType.HOLD]),
                    strategy_name=random.choice(strategies),
                    confidence=random.uniform(0.5, 0.95),
                    quantity=random.uniform(0.1, 10.0),
                    reasoning=f"Demo signal for testing"
                )
                
                # Send to Kafka
                try:
                    await self.producer.send_message(
                        topic="alphapulse.trading-signals",
                        message=signal,
                        key=signal.symbol
                    )
                    message_count += 1
                except Exception as e:
                    logger.error(f"Failed to send trading signal: {e}")
            
            # Small delay
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated {message_count} messages")
    
    async def process_streams(self):
        """Process streaming data."""
        logger.info("Starting stream processing...")
        
        # Create processors
        market_data_processor = MarketDataProcessor(
            window_config=WindowConfig(
                window_type=WindowType.TUMBLING,
                size=timedelta(seconds=60)
            ),
            aggregations=[
                (AggregationType.AVG, "price"),
                (AggregationType.SUM, "volume"),
                (AggregationType.MAX, "price"),
                (AggregationType.MIN, "price")
            ]
        )
        
        signal_processor = TradingSignalProcessor(
            signal_buffer_size=100,
            correlation_window=timedelta(minutes=5)
        )
        
        # Process messages
        processed_count = 0
        
        # Register handlers
        async def handle_market_data(message: MarketDataMessage):
            nonlocal processed_count
            
            # Process through stream processor
            result = await market_data_processor.process(message)
            
            # Process through CEP engine
            cep_results = await self.cep_engine.process_event(message)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.info(f"Processed {processed_count} market data messages")
                
                # Log detected patterns
                if cep_results["patterns"]:
                    logger.info(f"Patterns detected: {len(cep_results['patterns'])}")
                
                # Log trends
                if cep_results["trends"]:
                    for series, trend in cep_results["trends"].items():
                        logger.info(
                            f"Trend for {series}: {trend['direction']} "
                            f"(strength: {trend['strength']:.2f})"
                        )
        
        async def handle_trading_signal(message: TradingSignalMessage):
            # Process signal
            result = await signal_processor.process(message)
            
            # Check for conflicts
            if message.header.metadata.get("conflicts"):
                logger.warning(
                    f"⚠️ Signal conflicts for {message.symbol}: "
                    f"{len(message.header.metadata['conflicts'])} conflicts"
                )
        
        # Register handlers with consumer
        self.consumer.register_handler(MessageType.MARKET_DATA, handle_market_data)
        self.consumer.register_handler(MessageType.TRADING_SIGNAL, handle_trading_signal)
        
        # Start consuming (with timeout for demo)
        try:
            await asyncio.wait_for(
                self.consumer.consume(max_messages=500),
                timeout=65  # Slightly longer than data generation
            )
        except asyncio.TimeoutError:
            logger.info("Stream processing timeout reached")
        
        # Log final metrics
        logger.info("\n=== Stream Processing Metrics ===")
        logger.info(f"Market Data Processor: {market_data_processor.get_metrics()}")
        logger.info(f"Signal Processor: {signal_processor.get_metrics()}")
        logger.info(f"CEP Engine: {self.cep_engine.get_metrics()}")
        logger.info(f"Consumer: {self.consumer.get_metrics()}")
        logger.info(f"Producer: {self.producer.get_metrics()}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        
        logger.info("Cleanup completed")
    
    async def run(self):
        """Run the streaming demo."""
        try:
            # Setup
            await self.setup()
            
            # Start data generation and processing concurrently
            await asyncio.gather(
                self.generate_sample_data(duration_seconds=60),
                self.process_streams()
            )
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    logger.info("🚀 Starting AlphaPulse Streaming Demo")
    logger.info("=" * 50)
    
    # Check if Kafka is available
    try:
        demo = StreamingDemo()
        await demo.run()
    except Exception as e:
        logger.error(f"Failed to run demo: {e}")
        logger.info("\n⚠️ Make sure Kafka is running:")
        logger.info("  docker run -d --name kafka-demo \\")
        logger.info("    -p 9092:9092 \\")
        logger.info("    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \\")
        logger.info("    -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092 \\")
        logger.info("    -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \\")
        logger.info("    confluentinc/cp-kafka:latest")
    
    logger.info("\n✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(main()