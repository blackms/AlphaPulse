"""
Validation decorators for AlphaPulse API endpoints.

Provides decorators for automatic input validation:
- Parameter validation
- Request body validation
- Response validation
- Security validation
"""

import functools
import inspect
from typing import Any, Dict, List, Optional, Union, Callable, Type
from fastapi import HTTPException, Request
from pydantic import BaseModel, ValidationError as PydanticValidationError

from alpha_pulse.utils.input_validator import validator, ValidationError, ValidationResult
from alpha_pulse.utils.sql_injection_prevention import sql_guard
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


def validate_parameters(**validation_rules):
    """
    Decorator for validating function parameters.
    
    Usage:
        @validate_parameters(
            symbol={'type': 'stock_symbol', 'required': True},
            amount={'type': 'decimal', 'min_value': 0.01, 'max_value': 1000000}
        )
        def place_order(symbol: str, amount: float):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            validation_errors = []
            for param_name, validation_rule in validation_rules.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    
                    # Skip validation for None values unless required
                    if param_value is None and not validation_rule.get('required', False):
                        continue
                    
                    result = _validate_single_parameter(param_name, param_value, validation_rule)
                    if not result.is_valid:
                        validation_errors.extend(result.errors)
                    else:
                        # Update parameter with sanitized value
                        bound_args.arguments[param_name] = result.sanitized_value
            
            if validation_errors:
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'Parameter validation failed',
                        'details': validation_errors
                    }
                )
            
            # Call function with validated parameters
            if inspect.iscoroutinefunction(func):
                return await func(*bound_args.args, **bound_args.kwargs)
            else:
                return func(*bound_args.args, **bound_args.kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            validation_errors = []
            for param_name, validation_rule in validation_rules.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    
                    # Skip validation for None values unless required
                    if param_value is None and not validation_rule.get('required', False):
                        continue
                    
                    result = _validate_single_parameter(param_name, param_value, validation_rule)
                    if not result.is_valid:
                        validation_errors.extend(result.errors)
                    else:
                        # Update parameter with sanitized value
                        bound_args.arguments[param_name] = result.sanitized_value
            
            if validation_errors:
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'Parameter validation failed',
                        'details': validation_errors
                    }
                )
            
            # Call function with validated parameters
            return func(*bound_args.args, **bound_args.kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_request_body(model: Type[BaseModel] = None, custom_validation: Dict[str, Any] = None):
    """
    Decorator for validating request body.
    
    Usage:
        @validate_request_body(model=TradeRequest)
        async def create_trade(request: TradeRequest):
            pass
        
        @validate_request_body(custom_validation={
            'required_fields': ['symbol', 'quantity'],
            'fields': {
                'symbol': {'type': 'stock_symbol'},
                'quantity': {'type': 'integer', 'min_value': 1}
            }
        })
        async def create_trade(request_data: dict):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Request object in arguments
            request_obj = None
            for arg in args:
                if isinstance(arg, Request):
                    request_obj = arg
                    break
            
            if not request_obj:
                # If no Request object found, assume first argument is request data
                if args and isinstance(args[0], dict):
                    request_data = args[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="No request data found for validation"
                    )
            else:
                # Extract request body
                try:
                    request_data = await request_obj.json()
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid JSON in request body"
                    )
            
            # Validate using Pydantic model if provided
            if model:
                try:
                    validated_data = model(**request_data)
                    # Replace request data with validated model
                    if request_obj:
                        request_obj.state.validated_data = validated_data
                    else:
                        args = (validated_data,) + args[1:]
                except PydanticValidationError as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            'error': 'Request body validation failed',
                            'details': [f"{err['loc'][-1]}: {err['msg']}" for err in e.errors()]
                        }
                    )
            
            # Validate using custom rules if provided
            elif custom_validation:
                validation_errors = _validate_request_body_custom(request_data, custom_validation)
                if validation_errors:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            'error': 'Request body validation failed',
                            'details': validation_errors
                        }
                    )
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_financial_data(strict_mode: bool = True):
    """
    Decorator for validating financial data with enhanced security.
    
    Usage:
        @validate_financial_data(strict_mode=True)
        def calculate_portfolio_value(positions: dict):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Log financial operation
            audit_logger.log(
                event_type=AuditEventType.TRADING_ACTION,
                event_data={
                    'function': func.__name__,
                    'validation_mode': 'strict' if strict_mode else 'normal',
                    'parameter_count': len(kwargs)
                },
                data_classification="confidential"
            )
            
            # Enhanced validation for financial data
            validation_errors = []
            
            for param_name, param_value in kwargs.items():
                if isinstance(param_value, (int, float, str)):
                    # Validate monetary amounts
                    if 'amount' in param_name.lower() or 'price' in param_name.lower():
                        result = validator.validate_decimal(
                            param_value,
                            field_name=param_name,
                            min_value=0.01 if strict_mode else None,
                            max_value=1000000000 if strict_mode else None
                        )
                        if not result.is_valid:
                            validation_errors.extend(result.errors)
                    
                    # Validate percentages
                    elif 'percentage' in param_name.lower() or 'rate' in param_name.lower():
                        result = validator.validate_percentage(param_value, param_name)
                        if not result.is_valid:
                            validation_errors.extend(result.errors)
                    
                    # Validate symbols
                    elif 'symbol' in param_name.lower():
                        result = validator.validate_stock_symbol(param_value, param_name)
                        if not result.is_valid:
                            validation_errors.extend(result.errors)
            
            if validation_errors:
                audit_logger.log(
                    event_type=AuditEventType.TRADING_ACTION,
                    event_data={
                        'function': func.__name__,
                        'validation_errors': validation_errors,
                        'status': 'validation_failed'
                    },
                    severity=AuditSeverity.WARNING
                )
                
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'Financial data validation failed',
                        'details': validation_errors
                    }
                )
            
            # Call function
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Log financial operation
            audit_logger.log(
                event_type=AuditEventType.TRADING_ACTION,
                event_data={
                    'function': func.__name__,
                    'validation_mode': 'strict' if strict_mode else 'normal',
                    'parameter_count': len(kwargs)
                },
                data_classification="confidential"
            )
            
            # Enhanced validation for financial data
            validation_errors = []
            
            for param_name, param_value in kwargs.items():
                if isinstance(param_value, (int, float, str)):
                    # Validate monetary amounts
                    if 'amount' in param_name.lower() or 'price' in param_name.lower():
                        result = validator.validate_decimal(
                            param_value,
                            field_name=param_name,
                            min_value=0.01 if strict_mode else None,
                            max_value=1000000000 if strict_mode else None
                        )
                        if not result.is_valid:
                            validation_errors.extend(result.errors)
                    
                    # Validate percentages
                    elif 'percentage' in param_name.lower() or 'rate' in param_name.lower():
                        result = validator.validate_percentage(param_value, param_name)
                        if not result.is_valid:
                            validation_errors.extend(result.errors)
                    
                    # Validate symbols
                    elif 'symbol' in param_name.lower():
                        result = validator.validate_stock_symbol(param_value, param_name)
                        if not result.is_valid:
                            validation_errors.extend(result.errors)
            
            if validation_errors:
                audit_logger.log(
                    event_type=AuditEventType.TRADING_ACTION,
                    event_data={
                        'function': func.__name__,
                        'validation_errors': validation_errors,
                        'status': 'validation_failed'
                    },
                    severity=AuditSeverity.WARNING
                )
                
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'Financial data validation failed',
                        'details': validation_errors
                    }
                )
            
            # Call function
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def prevent_sql_injection(allow_raw_sql: bool = False):
    """
    Decorator for SQL injection prevention.
    
    Usage:
        @prevent_sql_injection(allow_raw_sql=False)
        def get_user_data(user_id: int):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Register function for SQL monitoring
            if allow_raw_sql:
                sql_guard.register_safe_function(func.__name__)
            
            # Monitor for SQL injection in string parameters
            for param_name, param_value in kwargs.items():
                if isinstance(param_value, str):
                    try:
                        sql_guard.analyze_query(param_value)
                    except Exception as e:
                        audit_logger = get_audit_logger()
                        audit_logger.log(
                            event_type=AuditEventType.SECURITY_VIOLATION,
                            event_data={
                                'function': func.__name__,
                                'parameter': param_name,
                                'violation_type': 'sql_injection_attempt',
                                'error': str(e)
                            },
                            severity=AuditSeverity.CRITICAL
                        )
                        
                        raise HTTPException(
                            status_code=400,
                            detail="Potential SQL injection detected"
                        )
            
            # Call function
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Register function for SQL monitoring
            if allow_raw_sql:
                sql_guard.register_safe_function(func.__name__)
            
            # Monitor for SQL injection in string parameters
            for param_name, param_value in kwargs.items():
                if isinstance(param_value, str):
                    try:
                        sql_guard.analyze_query(param_value)
                    except Exception as e:
                        audit_logger = get_audit_logger()
                        audit_logger.log(
                            event_type=AuditEventType.SECURITY_VIOLATION,
                            event_data={
                                'function': func.__name__,
                                'parameter': param_name,
                                'violation_type': 'sql_injection_attempt',
                                'error': str(e)
                            },
                            severity=AuditSeverity.CRITICAL
                        )
                        
                        raise HTTPException(
                            status_code=400,
                            detail="Potential SQL injection detected"
                        )
            
            # Call function
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_pagination(max_page_size: int = 100):
    """
    Decorator for pagination parameter validation.
    
    Usage:
        @validate_pagination(max_page_size=50)
        def get_trades(page: int = 1, page_size: int = 20):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate pagination parameters
            page = kwargs.get('page', 1)
            page_size = kwargs.get('page_size', 20)
            
            result = validator.validate_pagination(page, page_size, max_page_size)
            if not result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'Pagination validation failed',
                        'details': result.errors
                    }
                )
            
            # Update with validated values
            kwargs.update(result.sanitized_value)
            
            # Call function
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Validate pagination parameters
            page = kwargs.get('page', 1)
            page_size = kwargs.get('page_size', 20)
            
            result = validator.validate_pagination(page, page_size, max_page_size)
            if not result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'Pagination validation failed',
                        'details': result.errors
                    }
                )
            
            # Update with validated values
            kwargs.update(result.sanitized_value)
            
            # Call function
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _validate_single_parameter(param_name: str, param_value: Any, validation_rule: Dict[str, Any]) -> ValidationResult:
    """Validate a single parameter based on its rule."""
    param_type = validation_rule.get('type', 'string')
    
    # Route to appropriate validator
    if param_type == 'string':
        return validator.validate_string(
            param_value,
            field_name=param_name,
            min_length=validation_rule.get('min_length', 0),
            max_length=validation_rule.get('max_length', 10000),
            pattern=validation_rule.get('pattern'),
            allow_empty=validation_rule.get('allow_empty', False)
        )
    
    elif param_type == 'integer':
        return validator.validate_integer(
            param_value,
            field_name=param_name,
            min_value=validation_rule.get('min_value'),
            max_value=validation_rule.get('max_value')
        )
    
    elif param_type == 'decimal':
        return validator.validate_decimal(
            param_value,
            field_name=param_name,
            min_value=validation_rule.get('min_value'),
            max_value=validation_rule.get('max_value'),
            max_decimal_places=validation_rule.get('max_decimal_places', 8)
        )
    
    elif param_type == 'email':
        return validator.validate_email(param_value, field_name=param_name)
    
    elif param_type == 'phone':
        return validator.validate_phone(param_value, field_name=param_name)
    
    elif param_type == 'datetime':
        return validator.validate_datetime(param_value, field_name=param_name)
    
    elif param_type == 'stock_symbol':
        return validator.validate_stock_symbol(param_value, field_name=param_name)
    
    elif param_type == 'percentage':
        return validator.validate_percentage(param_value, field_name=param_name)
    
    elif param_type == 'password':
        return validator.validate_password(param_value, field_name=param_name)
    
    else:
        # Default to string validation
        return validator.validate_string(param_value, field_name=param_name)


def _validate_request_body_custom(request_data: Dict[str, Any], validation_rules: Dict[str, Any]) -> List[str]:
    """Validate request body using custom validation rules."""
    errors = []
    
    # Check required fields
    required_fields = validation_rules.get('required_fields', [])
    for field in required_fields:
        if field not in request_data:
            errors.append(f"Required field '{field}' is missing")
    
    # Validate each field
    field_rules = validation_rules.get('fields', {})
    for field_name, field_value in request_data.items():
        if field_name in field_rules:
            result = _validate_single_parameter(field_name, field_value, field_rules[field_name])
            if not result.is_valid:
                errors.extend(result.errors)
    
    return errors