"""
Streaming message models for Kafka and Pulsar.

Provides:
- Message wrapper structures
- Schema definitions for different message types
- Serialization/deserialization helpers
- Message metadata tracking
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from decimal import Decimal

from alpha_pulse.models.base import BaseModel
from alpha_pulse.models.market_data import OHLCV, MarketDataPoint


class MessageType(Enum):
    """Types of streaming messages."""
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    SYSTEM_EVENT = "system_event"
    DATA_QUALITY_EVENT = "data_quality_event"
    AUDIT_EVENT = "audit_event"


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"
    CLOSE_POSITION = "close_position"


class EventSeverity(Enum):
    """Severity levels for events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MessageHeader:
    """Common header for all streaming messages."""
    message_id: str
    message_type: MessageType
    timestamp: datetime
    source: str  # System/service that produced the message
    version: str = "1.0"
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "version": self.version,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "metadata": self.metadata
        }


@dataclass
class StreamingMessage(BaseModel):
    """Base class for all streaming messages."""
    header: MessageHeader
    payload: Dict[str, Any]
    schema_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "header": self.header.to_dict(),
            "payload": self.payload,
            "schema_version": self.schema_version
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamingMessage':
        """Create from dictionary."""
        header_data = data.get("header", {})
        header = MessageHeader(
            message_id=header_data["message_id"],
            message_type=MessageType(header_data["message_type"]),
            timestamp=datetime.fromisoformat(header_data["timestamp"]),
            source=header_data["source"],
            version=header_data.get("version", "1.0"),
            correlation_id=header_data.get("correlation_id"),
            trace_id=header_data.get("trace_id"),
            metadata=header_data.get("metadata", {})
        )
        
        return cls(
            header=header,
            payload=data.get("payload", {}),
            schema_version=data.get("schema_version", "1.0")
        )


@dataclass
class MarketDataMessage(StreamingMessage):
    """Market data streaming message."""
    symbol: str
    exchange: str
    ohlcv: Optional[OHLCV] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last_trade: Optional[Decimal] = None
    
    def __post_init__(self):
        """Initialize with proper message type."""
        self.header.message_type = MessageType.MARKET_DATA
        
        # Update payload
        self.payload = {
            "symbol": self.symbol,
            "exchange": self.exchange
        }
        
        if self.ohlcv:
            self.payload["ohlcv"] = {
                "open": str(self.ohlcv.open),
                "high": str(self.ohlcv.high),
                "low": str(self.ohlcv.low),
                "close": str(self.ohlcv.close),
                "volume": str(self.ohlcv.volume),
                "vwap": str(self.ohlcv.vwap) if self.ohlcv.vwap else None
            }
        
        if self.bid is not None:
            self.payload["bid"] = str(self.bid)
        if self.ask is not None:
            self.payload["ask"] = str(self.ask)
        if self.last_trade is not None:
            self.payload["last_trade"] = str(self.last_trade)
    
    @classmethod
    def from_market_data_point(cls, data_point: MarketDataPoint, header: MessageHeader) -> 'MarketDataMessage':
        """Create from MarketDataPoint."""
        return cls(
            header=header,
            payload={},
            symbol=data_point.symbol,
            exchange=data_point.metadata.get("exchange", "unknown") if data_point.metadata else "unknown",
            ohlcv=data_point.ohlcv,
            bid=data_point.bid if hasattr(data_point, 'bid') else None,
            ask=data_point.ask if hasattr(data_point, 'ask') else None
        )


@dataclass
class TradingSignalMessage(StreamingMessage):
    """Trading signal streaming message."""
    signal_id: str
    symbol: str
    signal_type: SignalType
    strategy_name: str
    confidence: float
    quantity: Optional[float] = None
    price_target: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    expiry: Optional[datetime] = None
    reasoning: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with proper message type."""
        self.header.message_type = MessageType.TRADING_SIGNAL
        
        # Update payload
        self.payload = {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strategy_name": self.strategy_name,
            "confidence": self.confidence
        }
        
        if self.quantity is not None:
            self.payload["quantity"] = self.quantity
        if self.price_target is not None:
            self.payload["price_target"] = str(self.price_target)
        if self.stop_loss is not None:
            self.payload["stop_loss"] = str(self.stop_loss)
        if self.take_profit is not None:
            self.payload["take_profit"] = str(self.take_profit)
        if self.expiry is not None:
            self.payload["expiry"] = self.expiry.isoformat()
        if self.reasoning is not None:
            self.payload["reasoning"] = self.reasoning


@dataclass
class RiskAlertMessage(StreamingMessage):
    """Risk alert streaming message."""
    alert_id: str
    alert_type: str  # position_risk, portfolio_risk, market_risk, etc.
    severity: EventSeverity
    affected_positions: List[str]
    risk_metrics: Dict[str, float]
    description: str
    recommended_action: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with proper message type."""
        self.header.message_type = MessageType.RISK_ALERT
        
        # Update payload
        self.payload = {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "affected_positions": self.affected_positions,
            "risk_metrics": self.risk_metrics,
            "description": self.description
        }
        
        if self.recommended_action:
            self.payload["recommended_action"] = self.recommended_action


@dataclass
class PortfolioUpdateMessage(StreamingMessage):
    """Portfolio update streaming message."""
    portfolio_id: str
    update_type: str  # position_opened, position_closed, rebalance_complete, etc.
    positions: List[Dict[str, Any]]
    total_value: Decimal
    cash_balance: Decimal
    metrics: Dict[str, float]  # sharpe_ratio, max_drawdown, etc.
    
    def __post_init__(self):
        """Initialize with proper message type."""
        self.header.message_type = MessageType.PORTFOLIO_UPDATE
        
        # Update payload
        self.payload = {
            "portfolio_id": self.portfolio_id,
            "update_type": self.update_type,
            "positions": self.positions,
            "total_value": str(self.total_value),
            "cash_balance": str(self.cash_balance),
            "metrics": self.metrics
        }


@dataclass
class SystemEventMessage(StreamingMessage):
    """System event streaming message."""
    event_id: str
    event_type: str  # startup, shutdown, health_check, config_change, etc.
    severity: EventSeverity
    component: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with proper message type."""
        self.header.message_type = MessageType.SYSTEM_EVENT
        
        # Update payload
        self.payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "details": self.details
        }


@dataclass
class DataQualityEventMessage(StreamingMessage):
    """Data quality event streaming message."""
    event_id: str
    dataset_id: str
    quality_score: float
    dimension_scores: Dict[str, float]
    issues: List[Dict[str, Any]]
    severity: EventSeverity
    
    def __post_init__(self):
        """Initialize with proper message type."""
        self.header.message_type = MessageType.DATA_QUALITY_EVENT
        
        # Update payload
        self.payload = {
            "event_id": self.event_id,
            "dataset_id": self.dataset_id,
            "quality_score": self.quality_score,
            "dimension_scores": self.dimension_scores,
            "issues": self.issues,
            "severity": self.severity.value
        }


@dataclass
class MessageEnvelope:
    """Envelope for message routing and metadata."""
    topic: str
    partition: Optional[int] = None
    key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    offset: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "partition": self.partition,
            "key": self.key,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "offset": self.offset
        }


@dataclass
class MessageBatch:
    """Batch of messages for efficient processing."""
    messages: List[StreamingMessage]
    batch_id: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def size(self) -> int:
        """Get batch size."""
        return len(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "message_count": self.size(),
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata
        }


class MessageFactory:
    """Factory for creating streaming messages."""
    
    @staticmethod
    def create_message(
        message_type: MessageType,
        source: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> StreamingMessage:
        """Create a streaming message of the specified type."""
        import uuid
        
        header = MessageHeader(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            timestamp=datetime.utcnow(),
            source=source,
            correlation_id=correlation_id,
            trace_id=trace_id
        )
        
        # Create specific message type based on message_type
        if message_type == MessageType.MARKET_DATA:
            return MarketDataMessage(
                header=header,
                payload=payload,
                symbol=payload.get("symbol", ""),
                exchange=payload.get("exchange", "")
            )
        elif message_type == MessageType.TRADING_SIGNAL:
            return TradingSignalMessage(
                header=header,
                payload=payload,
                signal_id=payload.get("signal_id", str(uuid.uuid4())),
                symbol=payload.get("symbol", ""),
                signal_type=SignalType(payload.get("signal_type", "hold")),
                strategy_name=payload.get("strategy_name", ""),
                confidence=payload.get("confidence", 0.0)
            )
        elif message_type == MessageType.RISK_ALERT:
            return RiskAlertMessage(
                header=header,
                payload=payload,
                alert_id=payload.get("alert_id", str(uuid.uuid4())),
                alert_type=payload.get("alert_type", ""),
                severity=EventSeverity(payload.get("severity", "info")),
                affected_positions=payload.get("affected_positions", []),
                risk_metrics=payload.get("risk_metrics", {}),
                description=payload.get("description", "")
            )
        else:
            # Generic streaming message
            return StreamingMessage(
                header=header,
                payload=payload
            )