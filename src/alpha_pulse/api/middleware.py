"""
Middleware components for the AlphaPulse API.
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram

# Prometheus metrics
request_counter = Counter(
    "alphapulse_api_requests_total",
    "Total number of requests received",
    ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "alphapulse_api_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"]
)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        
        # Log request
        logger.info(f"Request: {method} {path}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            request_counter.labels(
                method=method,
                endpoint=path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            # Log response
            logger.info(
                f"Response: {method} {path} - Status: {response.status_code} "
                f"Duration: {duration:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # Log error
            logger.error(
                f"Error processing {method} {path}: {str(e)}"
            )
            
            # Update metrics for failed requests
            request_counter.labels(
                method=method,
                endpoint=path,
                status=500
            ).inc()
            
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    TODO: Implement rate limiting logic based on API key or IP.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # TODO: Implement rate limiting
        return await call_next(request)