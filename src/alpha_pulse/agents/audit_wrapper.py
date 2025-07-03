"""
Audit logging wrapper for trading agents.

Provides decorators and utilities to automatically audit all agent decisions
and trading operations.
"""

import time
import functools
from typing import Any, Dict, Optional, Callable
import traceback

from alpha_pulse.utils.audit_logger import (
    get_audit_logger, 
    AuditEventType, 
    AuditSeverity,
    audit_call
)


class AuditableAgent:
    """Mixin class to add audit capabilities to trading agents."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with audit logger."""
        super().__init__(*args, **kwargs)
        self.audit_logger = get_audit_logger()
        
    def audit_decision(self, 
                      decision_type: str,
                      symbol: str,
                      action: str,
                      reasoning: Dict[str, Any],
                      confidence: float,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Audit a trading decision.
        
        Args:
            decision_type: Type of decision (entry, exit, hold, etc.)
            symbol: Trading symbol
            action: Action taken (buy, sell, hold)
            reasoning: Dictionary explaining the decision
            confidence: Confidence score (0-1)
            metadata: Additional metadata
        """
        agent_name = self.__class__.__name__
        
        event_data = {
            'agent': agent_name,
            'decision_type': decision_type,
            'symbol': symbol,
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence
        }
        
        if metadata:
            event_data['metadata'] = metadata
            
        self.audit_logger.log_trade_decision(
            agent=agent_name,
            symbol=symbol,
            action=action,
            quantity=metadata.get('quantity', 0) if metadata else 0,
            reasoning=reasoning,
            confidence=confidence
        )
        
    def audit_signal(self,
                    signal_type: str,
                    symbol: str,
                    strength: float,
                    components: Dict[str, Any]):
        """
        Audit a trading signal generation.
        
        Args:
            signal_type: Type of signal
            symbol: Trading symbol
            strength: Signal strength (-1 to 1)
            components: Signal components
        """
        agent_name = self.__class__.__name__
        
        self.audit_logger.log(
            event_type=AuditEventType.AGENT_SIGNAL,
            event_data={
                'agent': agent_name,
                'signal_type': signal_type,
                'symbol': symbol,
                'strength': strength,
                'components': components
            },
            data_classification="confidential"
        )
        
    def audit_error(self,
                   operation: str,
                   error: Exception,
                   context: Dict[str, Any]):
        """
        Audit an agent error.
        
        Args:
            operation: Operation that failed
            error: Exception that occurred
            context: Context information
        """
        agent_name = self.__class__.__name__
        
        self.audit_logger.log(
            event_type=AuditEventType.AGENT_ERROR,
            event_data={
                'agent': agent_name,
                'operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context
            },
            severity=AuditSeverity.ERROR,
            success=False,
            error_message=str(error)
        )


def audit_agent_method(operation: str, 
                      extract_context: Optional[Callable] = None):
    """
    Decorator to audit agent method calls.
    
    Args:
        operation: Name of the operation being performed
        extract_context: Function to extract context from method args
        
    Example:
        @audit_agent_method("analyze_market")
        def analyze_market(self, symbol, timeframe):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'audit_logger'):
                # If not an AuditableAgent, just call the function
                return func(self, *args, **kwargs)
                
            agent_name = self.__class__.__name__
            start_time = time.time()
            
            # Extract context
            if extract_context:
                context = extract_context(self, *args, **kwargs)
            else:
                context = {
                    'args': str(args)[:200],  # Limit size
                    'kwargs': str(kwargs)[:200]
                }
                
            success = True
            result = None
            error_message = None
            
            try:
                # Execute the method
                result = func(self, *args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                
                # Log the error
                self.audit_error(operation, e, context)
                raise
                
            finally:
                # Log the operation
                duration_ms = (time.time() - start_time) * 1000
                
                event_data = {
                    'agent': agent_name,
                    'operation': operation,
                    'context': context,
                    'duration_ms': duration_ms
                }
                
                # Add result summary if available
                if result is not None and success:
                    if hasattr(result, '__dict__'):
                        event_data['result_type'] = type(result).__name__
                    elif isinstance(result, (list, dict)):
                        event_data['result_size'] = len(result)
                        
                self.audit_logger.log(
                    event_type=AuditEventType.AGENT_DECISION,
                    event_data=event_data,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    data_classification="confidential"
                )
                
        return wrapper
    return decorator


def audit_trading_operation(func):
    """
    Decorator specifically for trading operations.
    
    Automatically extracts symbol, action, and quantity from common
    trading method signatures.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        audit_logger = get_audit_logger()
        
        # Try to extract common trading parameters
        symbol = None
        action = None
        quantity = None
        
        # Common parameter positions
        if len(args) > 1:
            symbol = args[1] if isinstance(args[1], str) else kwargs.get('symbol')
        if len(args) > 2:
            action = args[2] if isinstance(args[2], str) else kwargs.get('action')
        if len(args) > 3:
            quantity = args[3] if isinstance(args[3], (int, float)) else kwargs.get('quantity')
            
        # Extract from kwargs if not in args
        symbol = symbol or kwargs.get('symbol')
        action = action or kwargs.get('action')
        quantity = quantity or kwargs.get('quantity')
        
        start_time = time.time()
        success = True
        error_message = None
        result = None
        
        try:
            result = func(*args, **kwargs)
            
            # If result contains order information, log it
            if isinstance(result, dict) and 'order_id' in result:
                audit_logger.log_trade_execution(
                    order_id=result['order_id'],
                    symbol=symbol or result.get('symbol', 'unknown'),
                    action=action or result.get('action', 'unknown'),
                    quantity=quantity or result.get('quantity', 0),
                    price=result.get('price', 0),
                    success=True
                )
                
            return result
            
        except Exception as e:
            success = False
            error_message = str(e)
            
            # Log failed trade
            if symbol and action:
                audit_logger.log_trade_execution(
                    order_id='failed',
                    symbol=symbol,
                    action=action,
                    quantity=quantity or 0,
                    price=0,
                    success=False,
                    error=error_message
                )
            raise
            
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log the operation details
            event_data = {
                'function': func.__name__,
                'module': func.__module__,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'duration_ms': duration_ms
            }
            
            audit_logger.log(
                event_type=AuditEventType.TRADE_DECISION,
                event_data=event_data,
                success=success,
                error_message=error_message,
                duration_ms=duration_ms,
                data_classification="restricted"
            )
            
    return wrapper


# Specialized decorators for different agent types

def audit_technical_analysis(indicator: str):
    """Decorator for technical analysis operations."""
    def extract_context(self, *args, **kwargs):
        return {
            'indicator': indicator,
            'symbol': args[0] if args else kwargs.get('symbol'),
            'timeframe': args[1] if len(args) > 1 else kwargs.get('timeframe')
        }
    
    return audit_agent_method(
        f"technical_analysis_{indicator}",
        extract_context=extract_context
    )


def audit_fundamental_analysis(metric: str):
    """Decorator for fundamental analysis operations."""
    def extract_context(self, *args, **kwargs):
        return {
            'metric': metric,
            'symbol': args[0] if args else kwargs.get('symbol'),
            'period': kwargs.get('period', 'latest')
        }
    
    return audit_agent_method(
        f"fundamental_analysis_{metric}",
        extract_context=extract_context
    )


def audit_sentiment_analysis(source: str):
    """Decorator for sentiment analysis operations."""
    def extract_context(self, *args, **kwargs):
        return {
            'source': source,
            'symbol': args[0] if args else kwargs.get('symbol'),
            'lookback': kwargs.get('lookback', '24h')
        }
    
    return audit_agent_method(
        f"sentiment_analysis_{source}",
        extract_context=extract_context
    )


def audit_risk_check(risk_type: str):
    """Decorator for risk management checks."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Extract risk parameters
            position = args[0] if args else kwargs.get('position')
            threshold = kwargs.get('threshold')
            
            result = func(self, *args, **kwargs)
            
            # Log risk event if limit triggered
            if isinstance(result, dict) and result.get('limit_triggered', False):
                audit_logger.log_risk_event(
                    risk_type=risk_type,
                    threshold=threshold or result.get('threshold', 0),
                    actual_value=result.get('actual_value', 0),
                    action_taken=result.get('action', 'none')
                )
                
            return result
            
        return wrapper
    return decorator