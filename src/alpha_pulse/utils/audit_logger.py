"""
Comprehensive audit logging system for AlphaPulse.

This module provides structured audit logging for all security-relevant events,
including trading decisions, API access, authentication, and system changes.
"""

import json
import time
import threading
import queue
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import hashlib
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

from sqlalchemy import Column, String, DateTime, JSON, Index, Float, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from alpha_pulse.config.secure_settings import get_secrets_manager
from alpha_pulse.models.encrypted_fields import EncryptedJSON, EncryptedString

Base = declarative_base()


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILED = "auth.failed"
    AUTH_TOKEN_CREATED = "auth.token_created"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"
    
    # Trading events
    TRADE_DECISION = "trade.decision"
    TRADE_EXECUTED = "trade.executed"
    TRADE_CANCELLED = "trade.cancelled"
    TRADE_FAILED = "trade.failed"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_MODIFIED = "position.modified"
    
    # Risk management events
    RISK_LIMIT_TRIGGERED = "risk.limit_triggered"
    RISK_OVERRIDE = "risk.override"
    STOP_LOSS_TRIGGERED = "risk.stop_loss"
    DRAWDOWN_ALERT = "risk.drawdown_alert"
    
    # API access events
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"
    API_RATE_LIMITED = "api.rate_limited"
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGED = "config.changed"
    SECRET_ACCESSED = "secret.accessed"
    
    # Agent events
    AGENT_DECISION = "agent.decision"
    AGENT_SIGNAL = "agent.signal"
    AGENT_ERROR = "agent.error"
    
    # Data events
    DATA_ACCESS = "data.access"
    DATA_MODIFIED = "data.modified"
    DATA_EXPORTED = "data.exported"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """Context information for audit events."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None


class AuditLog(Base):
    """Database model for audit logs."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    
    # Context fields
    user_id = Column(String(100), index=True)
    session_id = Column(String(100), index=True)
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    request_id = Column(String(100), index=True)
    correlation_id = Column(String(100), index=True)
    
    # Event details (encrypted for sensitive data)
    event_data = Column(EncryptedJSON(encryption_context="audit_logs"))
    
    # Performance metrics
    duration_ms = Column(Float)
    
    # Success/failure tracking
    success = Column(Boolean, default=True)
    error_message = Column(EncryptedString(encryption_context="audit_logs"))
    
    # Compliance fields
    data_classification = Column(String(50))  # public, internal, confidential, restricted
    regulatory_flags = Column(JSON)  # GDPR, PCI, SOX flags
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_audit_timestamp_type', 'timestamp', 'event_type'),
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_correlation', 'correlation_id'),
    )


class AuditLogger:
    """
    Thread-safe audit logger with batching and async writes.
    
    Features:
    - Structured logging with consistent format
    - Async batch writes for performance
    - Automatic context propagation
    - Compliance metadata tracking
    - Performance metrics collection
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 10000):
        """
        Initialize the audit logger.
        
        Args:
            batch_size: Number of events to batch before writing
            flush_interval: Maximum seconds between flushes
            max_queue_size: Maximum events to queue before blocking
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._flush_thread = None
        self._context_local = threading.local()
        self._start_flush_thread()
        
    def _start_flush_thread(self):
        """Start the background thread for flushing audit logs."""
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            daemon=True
        )
        self._flush_thread.start()
        
    def _flush_worker(self):
        """Background worker that flushes audit logs to database."""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Wait for events with timeout
                timeout = max(0.1, self.flush_interval - (time.time() - last_flush))
                
                try:
                    event = self._queue.get(timeout=timeout)
                    batch.append(event)
                except queue.Empty:
                    pass
                
                # Flush if batch is full or timeout reached
                should_flush = (
                    len(batch) >= self.batch_size or
                    time.time() - last_flush >= self.flush_interval
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                # Log flush errors but don't stop the worker
                print(f"Audit flush error: {e}")
                traceback.print_exc()
                
        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)
            
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Write a batch of audit events to the database."""
        from alpha_pulse.config.database import get_db_session
        
        session = get_db_session()
        try:
            # Convert to ORM objects
            logs = []
            for event in batch:
                log = AuditLog(**event)
                logs.append(log)
                
            # Bulk insert
            session.bulk_save_objects(logs)
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"Failed to write audit batch: {e}")
            # Could implement fallback to file logging here
        finally:
            session.close()
            
    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for setting audit context.
        
        Example:
            with audit_logger.context(user_id="123", ip_address="1.2.3.4"):
                # All audit logs in this block will have the context
                audit_logger.log_api_request(...)
        """
        old_context = getattr(self._context_local, 'context', None)
        
        # Merge with existing context
        new_context = AuditContext(**(asdict(old_context) if old_context else {}))
        for key, value in kwargs.items():
            if hasattr(new_context, key):
                setattr(new_context, key, value)
                
        self._context_local.context = new_context
        
        try:
            yield new_context
        finally:
            self._context_local.context = old_context
            
    def get_context(self) -> AuditContext:
        """Get the current audit context."""
        return getattr(self._context_local, 'context', AuditContext())
        
    def log(self,
            event_type: Union[AuditEventType, str],
            event_data: Dict[str, Any],
            severity: AuditSeverity = AuditSeverity.INFO,
            success: bool = True,
            error_message: Optional[str] = None,
            duration_ms: Optional[float] = None,
            data_classification: str = "internal",
            regulatory_flags: Optional[Dict[str, bool]] = None):
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            event_data: Event-specific data
            severity: Event severity
            success: Whether the operation succeeded
            error_message: Error message if failed
            duration_ms: Operation duration in milliseconds
            data_classification: Data classification level
            regulatory_flags: Compliance flags (GDPR, PCI, etc.)
        """
        # Get current context
        context = self.get_context()
        
        # Build audit record
        audit_record = {
            'timestamp': datetime.now(timezone.utc),
            'event_type': event_type.value if isinstance(event_type, AuditEventType) else event_type,
            'severity': severity.value,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'ip_address': context.ip_address,
            'user_agent': context.user_agent,
            'request_id': context.request_id,
            'correlation_id': context.correlation_id,
            'event_data': event_data,
            'success': success,
            'error_message': error_message,
            'duration_ms': duration_ms,
            'data_classification': data_classification,
            'regulatory_flags': regulatory_flags or {}
        }
        
        # Add to queue (non-blocking)
        try:
            self._queue.put_nowait(audit_record)
        except queue.Full:
            # Log queue overflow but don't block
            print(f"Audit queue full, dropping event: {event_type}")
            
    # Convenience methods for common events
    
    def log_login(self, user_id: str, success: bool, method: str = "password", 
                  error: Optional[str] = None):
        """Log a login attempt."""
        self.log(
            AuditEventType.AUTH_LOGIN if success else AuditEventType.AUTH_FAILED,
            {
                'user_id': user_id,
                'method': method,
                'error': error
            },
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            success=success,
            error_message=error,
            regulatory_flags={'GDPR': True}
        )
        
    def log_api_request(self, method: str, path: str, 
                       status_code: Optional[int] = None,
                       duration_ms: Optional[float] = None):
        """Log an API request."""
        self.log(
            AuditEventType.API_REQUEST,
            {
                'method': method,
                'path': path,
                'status_code': status_code
            },
            severity=AuditSeverity.INFO,
            duration_ms=duration_ms
        )
        
    def log_trade_decision(self, agent: str, symbol: str, action: str,
                          quantity: float, reasoning: Dict[str, Any],
                          confidence: float):
        """Log a trading decision."""
        self.log(
            AuditEventType.TRADE_DECISION,
            {
                'agent': agent,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'reasoning': reasoning,
                'confidence': confidence
            },
            data_classification="confidential",
            regulatory_flags={'SOX': True}
        )
        
    def log_trade_execution(self, order_id: str, symbol: str, 
                           action: str, quantity: float,
                           price: float, success: bool,
                           error: Optional[str] = None):
        """Log a trade execution."""
        self.log(
            AuditEventType.TRADE_EXECUTED if success else AuditEventType.TRADE_FAILED,
            {
                'order_id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price
            },
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            success=success,
            error_message=error,
            data_classification="restricted",
            regulatory_flags={'SOX': True, 'PCI': True}
        )
        
    def log_risk_event(self, risk_type: str, threshold: float,
                      actual_value: float, action_taken: str):
        """Log a risk management event."""
        self.log(
            AuditEventType.RISK_LIMIT_TRIGGERED,
            {
                'risk_type': risk_type,
                'threshold': threshold,
                'actual_value': actual_value,
                'action_taken': action_taken
            },
            severity=AuditSeverity.WARNING,
            data_classification="confidential"
        )
        
    def log_secret_access(self, secret_name: str, purpose: str):
        """Log access to a secret."""
        # Hash the secret name for security
        hashed_name = hashlib.sha256(secret_name.encode()).hexdigest()[:16]
        
        self.log(
            AuditEventType.SECRET_ACCESSED,
            {
                'secret_hash': hashed_name,
                'purpose': purpose
            },
            data_classification="restricted",
            regulatory_flags={'SOX': True}
        )
        
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the audit logger and flush remaining events."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=timeout)


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Decorator for auditing function calls
def audit_call(event_type: Union[AuditEventType, str],
               extract_data: Optional[callable] = None,
               data_classification: str = "internal"):
    """
    Decorator to audit function calls.
    
    Args:
        event_type: Type of audit event
        extract_data: Function to extract audit data from args/kwargs
        data_classification: Data classification level
        
    Example:
        @audit_call(AuditEventType.TRADE_DECISION, 
                   extract_data=lambda *args, **kwargs: {'symbol': args[1]})
        def make_trade_decision(self, symbol, quantity):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                # Extract audit data
                if extract_data:
                    event_data = extract_data(*args, **kwargs)
                else:
                    event_data = {
                        'function': func.__name__,
                        'module': func.__module__
                    }
                    
                # Execute function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                raise
                
            finally:
                # Log the audit event
                duration_ms = (time.time() - start_time) * 1000
                audit_logger.log(
                    event_type=event_type,
                    event_data=event_data,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    data_classification=data_classification
                )
                
        return wrapper
    return decorator