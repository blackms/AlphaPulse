"""
Validation middleware for AlphaPulse API.

Provides comprehensive input validation for all API endpoints:
- Request body validation
- Query parameter validation
- Path parameter validation
- File upload validation
- Security-focused validation (XSS, SQL injection prevention)
"""

import json
import time
from typing import Dict, Any, Optional, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp

from alpha_pulse.utils.input_validator import validator, ValidationError, ValidationResult
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity
from alpha_pulse.config.validation_rules import get_endpoint_validation_rules


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive input validation.
    
    Features:
    - Automatic validation based on endpoint configuration
    - Request body and parameter validation
    - Security-focused validation (XSS, SQL injection)
    - Performance monitoring
    - Detailed error reporting
    """
    
    def __init__(
        self,
        app: ASGIApp,
        validate_query_params: bool = True,
        validate_path_params: bool = True,
        validate_request_body: bool = True,
        validate_file_uploads: bool = True,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_performance_monitoring: bool = True
    ):
        """
        Initialize validation middleware.
        
        Args:
            app: ASGI application
            validate_query_params: Enable query parameter validation
            validate_path_params: Enable path parameter validation
            validate_request_body: Enable request body validation
            validate_file_uploads: Enable file upload validation
            max_request_size: Maximum request size in bytes
            enable_performance_monitoring: Enable performance monitoring
        """
        super().__init__(app)
        self.validate_query_params = validate_query_params
        self.validate_path_params = validate_path_params
        self.validate_request_body = validate_request_body
        self.validate_file_uploads = validate_file_uploads
        self.max_request_size = max_request_size
        self.enable_performance_monitoring = enable_performance_monitoring
        
        self.audit_logger = get_audit_logger()
        
        # Performance metrics
        self.metrics = {
            'requests_validated': 0,
            'validation_errors': 0,
            'total_validation_time': 0.0,
            'average_validation_time': 0.0
        }
        
        # Skip validation for certain endpoints
        self.skip_validation_paths = {
            '/health',
            '/metrics',
            '/docs',
            '/openapi.json',
            '/favicon.ico'
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with comprehensive validation."""
        start_time = time.time() if self.enable_performance_monitoring else None
        
        # Skip validation for certain paths
        if request.url.path in self.skip_validation_paths:
            return await call_next(request)
        
        try:
            # Validate request size
            if hasattr(request, 'headers'):
                content_length = request.headers.get('content-length')
                if content_length and int(content_length) > self.max_request_size:
                    return self._create_error_response(
                        "Request too large",
                        400,
                        {"max_size": self.max_request_size}
                    )
            
            # Get validation rules for this endpoint
            validation_rules = get_endpoint_validation_rules(
                request.url.path,
                request.method
            )
            
            # Perform validation
            validation_errors = []
            
            # Validate query parameters
            if self.validate_query_params and validation_rules.get('query_params'):
                query_errors = await self._validate_query_parameters(
                    request, 
                    validation_rules['query_params']
                )
                validation_errors.extend(query_errors)
            
            # Validate path parameters
            if self.validate_path_params and validation_rules.get('path_params'):
                path_errors = await self._validate_path_parameters(
                    request,
                    validation_rules['path_params']
                )
                validation_errors.extend(path_errors)
            
            # Validate request body
            if self.validate_request_body and validation_rules.get('body'):
                body_errors = await self._validate_request_body(
                    request,
                    validation_rules['body']
                )
                validation_errors.extend(body_errors)
            
            # Validate file uploads
            if self.validate_file_uploads and validation_rules.get('files'):
                file_errors = await self._validate_file_uploads(
                    request,
                    validation_rules['files']
                )
                validation_errors.extend(file_errors)
            
            # Check for validation errors
            if validation_errors:
                self.metrics['validation_errors'] += 1
                self._log_validation_failure(request, validation_errors)
                return self._create_validation_error_response(validation_errors)
            
            # Add validation info to request state
            request.state.validation_passed = True
            request.state.validation_time = time.time() - start_time if start_time else 0
            
            # Update metrics
            self.metrics['requests_validated'] += 1
            if start_time:
                validation_time = time.time() - start_time
                self.metrics['total_validation_time'] += validation_time
                self.metrics['average_validation_time'] = (
                    self.metrics['total_validation_time'] / self.metrics['requests_validated']
                )
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Add validation headers to response
            if hasattr(request.state, 'validation_time'):
                response.headers['X-Validation-Time'] = str(request.state.validation_time)
            
            return response
            
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.API_ERROR,
                event_data={
                    'validation_error': str(e),
                    'endpoint': request.url.path,
                    'method': request.method
                },
                severity=AuditSeverity.ERROR
            )
            
            return self._create_error_response(
                "Internal validation error",
                500
            )
    
    async def _validate_query_parameters(
        self, 
        request: Request, 
        rules: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Validate query parameters against rules."""
        errors = []
        
        for param_name, param_rules in rules.items():
            param_value = request.query_params.get(param_name)
            
            # Check required parameters
            if param_rules.get('required', False) and param_value is None:
                errors.append(f"Query parameter '{param_name}' is required")
                continue
            
            # Skip validation if parameter is not provided and not required
            if param_value is None:
                continue
            
            # Validate parameter based on type
            result = self._validate_parameter(param_name, param_value, param_rules)
            if not result.is_valid:
                errors.extend(result.errors)
        
        return errors
    
    async def _validate_path_parameters(
        self, 
        request: Request, 
        rules: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Validate path parameters against rules."""
        errors = []
        
        # Extract path parameters from request
        path_params = getattr(request, 'path_params', {})
        
        for param_name, param_rules in rules.items():
            param_value = path_params.get(param_name)
            
            if param_value is None:
                errors.append(f"Path parameter '{param_name}' is missing")
                continue
            
            # Validate parameter
            result = self._validate_parameter(param_name, param_value, param_rules)
            if not result.is_valid:
                errors.extend(result.errors)
        
        return errors
    
    async def _validate_request_body(
        self, 
        request: Request, 
        rules: Dict[str, Any]
    ) -> List[str]:
        """Validate request body against rules."""
        errors = []
        
        try:
            # Get request body
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    body = await request.json()
                    errors.extend(self._validate_json_body(body, rules))
                elif 'application/x-www-form-urlencoded' in content_type:
                    form_data = await request.form()
                    errors.extend(self._validate_form_data(form_data, rules))
                
        except json.JSONDecodeError:
            errors.append("Invalid JSON in request body")
        except Exception as e:
            errors.append(f"Error processing request body: {str(e)}")
        
        return errors
    
    def _validate_json_body(self, body: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate JSON request body."""
        errors = []
        
        # Check required fields
        required_fields = rules.get('required_fields', [])
        for field in required_fields:
            if field not in body:
                errors.append(f"Required field '{field}' is missing from request body")
        
        # Validate each field
        field_rules = rules.get('fields', {})
        for field_name, field_value in body.items():
            if field_name in field_rules:
                result = self._validate_parameter(field_name, field_value, field_rules[field_name])
                if not result.is_valid:
                    errors.extend(result.errors)
        
        # Validate against JSON schema if provided
        if 'schema' in rules:
            schema_result = validator.validate_json_schema(body, rules['schema'])
            if not schema_result.is_valid:
                errors.extend(schema_result.errors)
        
        return errors
    
    def _validate_form_data(self, form_data, rules: Dict[str, Any]) -> List[str]:
        """Validate form data."""
        errors = []
        
        # Convert form data to dict
        form_dict = dict(form_data)
        
        # Check required fields
        required_fields = rules.get('required_fields', [])
        for field in required_fields:
            if field not in form_dict:
                errors.append(f"Required field '{field}' is missing from form data")
        
        # Validate each field
        field_rules = rules.get('fields', {})
        for field_name, field_value in form_dict.items():
            if field_name in field_rules:
                result = self._validate_parameter(field_name, field_value, field_rules[field_name])
                if not result.is_valid:
                    errors.extend(result.errors)
        
        return errors
    
    async def _validate_file_uploads(
        self, 
        request: Request, 
        rules: Dict[str, Any]
    ) -> List[str]:
        """Validate file uploads."""
        errors = []
        
        content_type = request.headers.get('content-type', '')
        if 'multipart/form-data' in content_type:
            try:
                form = await request.form()
                
                for field_name, file_info in form.items():
                    if hasattr(file_info, 'filename'):  # This is a file upload
                        if field_name in rules:
                            file_rules = rules[field_name]
                            result = validator.validate_file_upload(
                                filename=file_info.filename,
                                content_type=file_info.content_type,
                                file_size=file_info.size if hasattr(file_info, 'size') else 0,
                                allowed_extensions=file_rules.get('allowed_extensions'),
                                max_size=file_rules.get('max_size', 10 * 1024 * 1024)
                            )
                            
                            if not result.is_valid:
                                errors.extend(result.errors)
                                
            except Exception as e:
                errors.append(f"Error processing file upload: {str(e)}")
        
        return errors
    
    def _validate_parameter(
        self, 
        param_name: str, 
        param_value: Any, 
        rules: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a single parameter based on rules."""
        param_type = rules.get('type', 'string')
        
        # Route to appropriate validator based on type
        if param_type == 'string':
            return validator.validate_string(
                param_value,
                field_name=param_name,
                min_length=rules.get('min_length', 0),
                max_length=rules.get('max_length', 10000),
                pattern=rules.get('pattern'),
                allow_empty=rules.get('allow_empty', False)
            )
        
        elif param_type == 'integer':
            return validator.validate_integer(
                param_value,
                field_name=param_name,
                min_value=rules.get('min_value'),
                max_value=rules.get('max_value')
            )
        
        elif param_type == 'decimal':
            return validator.validate_decimal(
                param_value,
                field_name=param_name,
                min_value=rules.get('min_value'),
                max_value=rules.get('max_value'),
                max_decimal_places=rules.get('max_decimal_places', 8)
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
    
    def _create_validation_error_response(self, errors: List[str]) -> JSONResponse:
        """Create standardized validation error response."""
        return JSONResponse(
            status_code=400,
            content={
                'error': 'Validation failed',
                'details': errors,
                'timestamp': time.time(),
                'type': 'validation_error'
            }
        )
    
    def _create_error_response(
        self, 
        message: str, 
        status_code: int, 
        details: Dict[str, Any] = None
    ) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                'error': message,
                'details': details,
                'timestamp': time.time(),
                'type': 'error'
            }
        )
    
    def _log_validation_failure(self, request: Request, errors: List[str]):
        """Log validation failures for monitoring."""
        client_ip = request.client.host if request.client else "unknown"
        
        self.audit_logger.log(
            event_type=AuditEventType.API_ERROR,
            event_data={
                'validation_failure': True,
                'errors': errors,
                'endpoint': request.url.path,
                'method': request.method,
                'ip_address': client_ip,
                'user_agent': request.headers.get('User-Agent', ''),
                'query_params': str(request.query_params)
            },
            severity=AuditSeverity.WARNING,
            data_classification="internal"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation middleware metrics."""
        return self.metrics.copy()


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        secret_key: str,
        exempt_paths: List[str] = None,
        cookie_name: str = 'csrftoken',
        header_name: str = 'X-CSRFToken'
    ):
        """Initialize CSRF protection."""
        super().__init__(app)
        self.secret_key = secret_key
        self.exempt_paths = set(exempt_paths or [])
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.audit_logger = get_audit_logger()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply CSRF protection."""
        # Skip CSRF for safe methods and exempt paths
        if (request.method in ['GET', 'HEAD', 'OPTIONS'] or 
            request.url.path in self.exempt_paths):
            return await call_next(request)
        
        # Check CSRF token
        csrf_token = self._get_csrf_token(request)
        if not csrf_token or not self._validate_csrf_token(csrf_token):
            self._log_csrf_violation(request)
            return JSONResponse(
                status_code=403,
                content={
                    'error': 'CSRF token missing or invalid',
                    'type': 'csrf_error'
                }
            )
        
        return await call_next(request)
    
    def _get_csrf_token(self, request: Request) -> Optional[str]:
        """Get CSRF token from request."""
        # Try header first
        token = request.headers.get(self.header_name)
        if token:
            return token
        
        # Try form data
        if hasattr(request, '_form'):
            form_data = request._form
            return form_data.get('csrfmiddlewaretoken')
        
        return None
    
    def _validate_csrf_token(self, token: str) -> bool:
        """Validate CSRF token."""
        # Simple validation - in production, use proper CSRF token validation
        import hmac
        import hashlib
        
        expected = hmac.new(
            self.secret_key.encode(),
            b'csrf_token',
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(token, expected)
    
    def _log_csrf_violation(self, request: Request):
        """Log CSRF violations."""
        client_ip = request.client.host if request.client else "unknown"
        
        self.audit_logger.log(
            event_type=AuditEventType.SECURITY_VIOLATION,
            event_data={
                'violation_type': 'csrf_violation',
                'endpoint': request.url.path,
                'method': request.method,
                'ip_address': client_ip,
                'user_agent': request.headers.get('User-Agent', '')
            },
            severity=AuditSeverity.WARNING,
            data_classification="restricted"
        )