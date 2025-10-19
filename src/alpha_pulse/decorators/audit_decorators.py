"""
Audit decorators for comprehensive logging of trading operations.

These decorators provide automatic audit logging for critical functions
with proper context extraction and compliance metadata.
"""

import functools
import time
import inspect
import traceback
from typing import Any, Callable, Dict, Optional, Union, Type
from datetime import datetime

from alpha_pulse.utils.audit_logger import (
    get_audit_logger, 
    AuditEventType, 
    AuditSeverity,
    audit_call
)


def audit_trade_decision(
    extract_reasoning: bool = True,
    include_market_data: bool = True,
    compliance_flags: Optional[Dict[str, bool]] = None
):
    """
    Decorator for auditing trading decisions.
    
    Args:
        extract_reasoning: Whether to extract and log decision reasoning
        include_market_data: Whether to include market data in logs
        compliance_flags: Additional compliance flags (default: SOX, MiFID II)
    
    Example:
        @audit_trade_decision()
        def make_trading_decision(self, symbol: str, signals: Dict) -> TradingDecision:
            ...
    """
    if compliance_flags is None:
        compliance_flags = {'SOX': True, 'MiFID_II': True}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            
            # Extract context from function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Build event data
            event_data = {
                'function': func.__name__,
                'module': func.__module__,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            # Extract common trading parameters
            params = bound_args.arguments
            if 'symbol' in params:
                event_data['symbol'] = params['symbol']
            if 'quantity' in params:
                event_data['quantity'] = params['quantity']
            if 'side' in params:
                event_data['side'] = params['side']
            if 'signals' in params and extract_reasoning:
                event_data['signals'] = params['signals']
            if 'market_data' in params and include_market_data:
                event_data['market_data'] = params['market_data']
            
            # Extract self/cls for agent identification
            if args and hasattr(args[0], '__class__'):
                event_data['agent'] = args[0].__class__.__name__
            
            success = True
            error_message = None
            result = None
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Extract result data
                if result is not None:
                    if hasattr(result, 'to_dict'):
                        event_data['decision'] = result.to_dict()
                    elif hasattr(result, '__dict__'):
                        event_data['decision'] = {
                            k: v for k, v in result.__dict__.items()
                            if not k.startswith('_')
                        }
                    else:
                        event_data['decision'] = str(result)
                
                # Extract confidence if available
                if hasattr(result, 'confidence'):
                    event_data['confidence'] = result.confidence
                elif isinstance(result, dict) and 'confidence' in result:
                    event_data['confidence'] = result['confidence']
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                event_data['error'] = {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                event_data['duration_ms'] = duration_ms
                
                # Log the trading decision
                audit_logger.log(
                    event_type=AuditEventType.TRADE_DECISION,
                    event_data=event_data,
                    severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    data_classification='confidential',
                    regulatory_flags=compliance_flags
                )
                
        return wrapper
    return decorator


def audit_risk_check(
    risk_type: str,
    threshold_param: str = 'threshold',
    value_param: str = 'value'
):
    """
    Decorator for auditing risk management checks.
    
    Args:
        risk_type: Type of risk being checked (e.g., 'position_size', 'drawdown')
        threshold_param: Parameter name containing the threshold
        value_param: Parameter name containing the actual value
    
    Example:
        @audit_risk_check(risk_type='position_size')
        def check_position_size_limit(self, symbol: str, quantity: float, threshold: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            
            # Extract parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params = bound_args.arguments
            
            event_data = {
                'function': func.__name__,
                'risk_type': risk_type,
                'threshold': params.get(threshold_param),
                'value': params.get(value_param),
                'symbol': params.get('symbol'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            success = True
            error_message = None
            result = None
            triggered = False
            
            try:
                result = func(*args, **kwargs)
                
                # Check if risk limit was triggered
                if isinstance(result, bool):
                    triggered = not result  # Assuming False means limit exceeded
                elif hasattr(result, 'triggered'):
                    triggered = result.triggered
                elif isinstance(result, dict) and 'triggered' in result:
                    triggered = result['triggered']
                
                event_data['triggered'] = triggered
                event_data['result'] = str(result)
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                event_data['error'] = str(e)
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Determine event type and severity
                if triggered:
                    event_type = AuditEventType.RISK_LIMIT_TRIGGERED
                    severity = AuditSeverity.WARNING
                else:
                    event_type = AuditEventType.RISK_LIMIT_TRIGGERED
                    severity = AuditSeverity.INFO
                
                if not success:
                    severity = AuditSeverity.ERROR
                
                audit_logger.log(
                    event_type=event_type,
                    event_data=event_data,
                    severity=severity,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    data_classification='confidential',
                    regulatory_flags={'SOX': True, 'MiFID_II': True}
                )
                
        return wrapper
    return decorator


def audit_portfolio_action(action_type: str = 'update'):
    """
    Decorator for auditing portfolio management actions.
    
    Args:
        action_type: Type of portfolio action ('update', 'rebalance', 'optimize')
    
    Example:
        @audit_portfolio_action(action_type='rebalance')
        def rebalance_portfolio(self, target_weights: Dict[str, float]):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            
            # Extract parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params = bound_args.arguments
            
            event_data = {
                'function': func.__name__,
                'action_type': action_type,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Extract relevant portfolio parameters
            for param in ['portfolio_id', 'positions', 'target_weights', 'constraints']:
                if param in params:
                    event_data[param] = params[param]
            
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                
                # Extract result metrics
                if hasattr(result, 'to_dict'):
                    event_data['result'] = result.to_dict()
                elif isinstance(result, dict):
                    event_data['result'] = result
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                event_data['error'] = str(e)
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                audit_logger.log(
                    event_type=AuditEventType.TRADE_DECISION,
                    event_data=event_data,
                    severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    data_classification='confidential',
                    regulatory_flags={'SOX': True}
                )
                
        return wrapper
    return decorator


def audit_agent_signal(agent_type: str, include_market_data: bool = False):
    """
    Decorator for auditing trading agent signals.

    Args:
        agent_type: Type of agent (e.g., 'technical', 'fundamental', 'sentiment')
        include_market_data: Whether to include market data in logs (default: False)

    Example:
        @audit_agent_signal(agent_type='technical', include_market_data=True)
        def generate_signal(self, market_data: MarketData) -> Signal:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()

            event_data = {
                'function': func.__name__,
                'agent_type': agent_type,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Extract symbol if available
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            params = bound_args.arguments

            if 'symbol' in params:
                event_data['symbol'] = params['symbol']
            if 'market_data' in params:
                if hasattr(params['market_data'], 'symbol'):
                    event_data['symbol'] = params['market_data'].symbol
                # Include market data if requested
                if include_market_data:
                    # Extract key market data attributes for audit log
                    market_data_obj = params['market_data']
                    event_data['market_data'] = {}
                    if hasattr(market_data_obj, 'prices') and market_data_obj.prices is not None:
                        event_data['market_data']['has_prices'] = True
                    if hasattr(market_data_obj, 'volumes') and market_data_obj.volumes is not None:
                        event_data['market_data']['has_volumes'] = True
                    if hasattr(market_data_obj, 'fundamentals') and market_data_obj.fundamentals:
                        event_data['market_data']['has_fundamentals'] = True
                    if hasattr(market_data_obj, 'sentiment') and market_data_obj.sentiment:
                        event_data['market_data']['has_sentiment'] = True
            
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                
                # Extract signal details
                if result is not None:
                    if hasattr(result, 'to_dict'):
                        event_data['signal'] = result.to_dict()
                    elif hasattr(result, 'signal') and hasattr(result, 'confidence'):
                        event_data['signal'] = {
                            'type': str(result.signal),
                            'confidence': result.confidence
                        }
                    elif isinstance(result, dict):
                        event_data['signal'] = result
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                event_data['error'] = str(e)
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                audit_logger.log(
                    event_type=AuditEventType.AGENT_SIGNAL,
                    event_data=event_data,
                    severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    data_classification='internal'
                )
                
        return wrapper
    return decorator


def audit_data_access(
    data_type: str,
    purpose: str,
    include_query: bool = False
):
    """
    Decorator for auditing data access operations.
    
    Args:
        data_type: Type of data being accessed
        purpose: Purpose of data access
        include_query: Whether to include query details in logs
    
    Example:
        @audit_data_access(data_type='market_data', purpose='analysis')
        def fetch_historical_data(self, symbol: str, timeframe: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            
            # Extract parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            params = bound_args.arguments
            
            event_data = {
                'function': func.__name__,
                'data_type': data_type,
                'purpose': purpose,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add query details if requested
            if include_query:
                event_data['query_params'] = {
                    k: v for k, v in params.items()
                    if k not in ['self', 'cls'] and not k.startswith('_')
                }
            
            success = True
            error_message = None
            record_count = None
            
            try:
                result = func(*args, **kwargs)
                
                # Try to extract record count
                if hasattr(result, '__len__'):
                    record_count = len(result)
                elif hasattr(result, 'count'):
                    record_count = result.count()
                
                if record_count is not None:
                    event_data['record_count'] = record_count
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                event_data['error'] = str(e)
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Determine regulatory flags based on data type
                regulatory_flags = {}
                if data_type in ['user_data', 'personal_data']:
                    regulatory_flags['GDPR'] = True
                if data_type in ['financial_data', 'trading_data']:
                    regulatory_flags['SOX'] = True
                
                audit_logger.log(
                    event_type=AuditEventType.DATA_ACCESS,
                    event_data=event_data,
                    severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
                    success=success,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    regulatory_flags=regulatory_flags
                )
                
        return wrapper
    return decorator


def audit_config_change(config_type: str = 'system'):
    """
    Decorator for auditing configuration changes.
    
    Args:
        config_type: Type of configuration being changed
    
    Example:
        @audit_config_change(config_type='trading_parameters')
        def update_risk_limits(self, new_limits: Dict[str, float]):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Extract parameters to capture before state
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            params = bound_args.arguments
            
            event_data = {
                'function': func.__name__,
                'config_type': config_type,
                'timestamp': datetime.utcnow().isoformat(),
                'parameters': {
                    k: v for k, v in params.items()
                    if k not in ['self', 'cls'] and not k.startswith('_')
                }
            }
            
            # Try to capture before state
            if args and hasattr(args[0], 'get_config'):
                try:
                    event_data['before_state'] = args[0].get_config()
                except:
                    pass
            
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                
                # Try to capture after state
                if args and hasattr(args[0], 'get_config'):
                    try:
                        event_data['after_state'] = args[0].get_config()
                    except:
                        pass
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                event_data['error'] = str(e)
                raise
                
            finally:
                audit_logger.log(
                    event_type=AuditEventType.CONFIG_CHANGED,
                    event_data=event_data,
                    severity=AuditSeverity.WARNING if success else AuditSeverity.ERROR,
                    success=success,
                    error_message=error_message,
                    data_classification='confidential',
                    regulatory_flags={'SOX': True}
                )
                
        return wrapper
    return decorator


def audit_secret_access(purpose: str):
    """
    Decorator for auditing access to secrets and credentials.
    
    Args:
        purpose: Purpose of accessing the secret
    
    Example:
        @audit_secret_access(purpose='api_authentication')
        def get_api_key(self, key_name: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Extract secret name if available
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            params = bound_args.arguments
            
            secret_name = params.get('key_name') or params.get('secret_name') or 'unknown'
            
            # Log the secret access
            audit_logger.log_secret_access(
                secret_name=secret_name,
                purpose=purpose
            )
            
            # Execute the function
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


# Batch decorator for multiple audit types
def audit_multiple(*decorators):
    """
    Apply multiple audit decorators to a single function.
    
    Example:
        @audit_multiple(
            audit_trade_decision(),
            audit_risk_check(risk_type='position_size')
        )
        def complex_trading_operation(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return decorator