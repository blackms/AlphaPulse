"""
FastAPI middleware for automatic audit logging of all API requests.
"""

import time
import uuid
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that automatically logs all API requests and responses.
    
    Features:
    - Logs all HTTP requests with timing information
    - Captures request/response metadata
    - Propagates correlation IDs
    - Handles errors gracefully
    """
    
    def __init__(self, app: ASGIApp, 
                 exclude_paths: Optional[list] = None,
                 include_request_body: bool = False,
                 include_response_body: bool = False):
        """
        Initialize the audit middleware.
        
        Args:
            app: The ASGI application
            exclude_paths: List of path prefixes to exclude from logging
            include_request_body: Whether to log request bodies
            include_response_body: Whether to log response bodies
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ['/health', '/metrics', '/docs', '/openapi.json']
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.audit_logger = get_audit_logger()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log audit information."""
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
            
        # Generate request ID if not present
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        correlation_id = request.headers.get('X-Correlation-ID', request_id)
        
        # Extract user information from JWT if present
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = getattr(request.state.user, 'id', None)
        
        # Get client information
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Start timing
        start_time = time.time()
        
        # Set audit context
        with self.audit_logger.context(
            user_id=user_id,
            ip_address=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            correlation_id=correlation_id
        ):
            # Build request data
            request_data = {
                'method': request.method,
                'path': request.url.path,
                'query_params': dict(request.query_params),
                'headers': {
                    k: v for k, v in request.headers.items() 
                    if k.lower() not in ['authorization', 'cookie']
                }
            }
            
            # Optionally include request body
            if self.include_request_body and request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    # Note: This consumes the body, so we need to be careful
                    request_data['body_size'] = int(request.headers.get('content-length', 0))
                except:
                    pass
                    
            # Log the incoming request
            self.audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data=request_data,
                severity=AuditSeverity.INFO
            )
            
            # Process the request
            response = None
            error_message = None
            
            try:
                response = await call_next(request)
                
            except Exception as e:
                error_message = str(e)
                # Log the error
                self.audit_logger.log(
                    event_type=AuditEventType.API_ERROR,
                    event_data={
                        **request_data,
                        'error': error_message,
                        'error_type': type(e).__name__
                    },
                    severity=AuditSeverity.ERROR,
                    success=False,
                    error_message=error_message
                )
                raise
                
            finally:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log the response
                if response:
                    response_data = {
                        **request_data,
                        'status_code': response.status_code,
                        'duration_ms': duration_ms
                    }
                    
                    # Determine severity based on status code
                    if response.status_code >= 500:
                        severity = AuditSeverity.ERROR
                    elif response.status_code >= 400:
                        severity = AuditSeverity.WARNING
                    else:
                        severity = AuditSeverity.INFO
                        
                    # Check for rate limiting
                    if response.status_code == 429:
                        self.audit_logger.log(
                            event_type=AuditEventType.API_RATE_LIMITED,
                            event_data=response_data,
                            severity=AuditSeverity.WARNING,
                            success=False,
                            duration_ms=duration_ms
                        )
                    else:
                        self.audit_logger.log(
                            event_type=AuditEventType.API_RESPONSE,
                            event_data=response_data,
                            severity=severity,
                            success=response.status_code < 400,
                            duration_ms=duration_ms
                        )
                        
                    # Add correlation headers to response
                    response.headers['X-Request-ID'] = request_id
                    response.headers['X-Correlation-ID'] = correlation_id
                    
        return response


class SecurityEventMiddleware(BaseHTTPMiddleware):
    """
    Middleware that detects and logs security-relevant events.
    
    Features:
    - Detects suspicious patterns
    - Logs authentication failures
    - Tracks API abuse attempts
    """
    
    def __init__(self, app: ASGIApp):
        """Initialize the security middleware."""
        super().__init__(app)
        self.audit_logger = get_audit_logger()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and detect security events."""
        # Detect suspicious user agents
        user_agent = request.headers.get('User-Agent', '')
        suspicious_agents = ['sqlmap', 'nikto', 'scanner', 'bot']
        
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            self.audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data={
                    'method': request.method,
                    'path': request.url.path,
                    'user_agent': user_agent,
                    'suspicious': True
                },
                severity=AuditSeverity.WARNING,
                data_classification='restricted'
            )
            
        # Detect SQL injection attempts
        query_string = str(request.url.query)
        sql_patterns = ['union', 'select', 'drop', '--', '/*', '*/', 'xp_', 'sp_']
        
        if any(pattern in query_string.lower() for pattern in sql_patterns):
            self.audit_logger.log(
                event_type=AuditEventType.API_REQUEST,
                event_data={
                    'method': request.method,
                    'path': request.url.path,
                    'query': query_string,
                    'potential_sql_injection': True
                },
                severity=AuditSeverity.CRITICAL,
                data_classification='restricted'
            )
            
        # Process request
        response = await call_next(request)
        
        return response