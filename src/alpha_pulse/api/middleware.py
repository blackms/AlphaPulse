"""
Middleware components for the AlphaPulse API.
"""
import time
from typing import Callable, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram

from .config import config

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
    """Middleware for rate limiting requests."""
    
    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or config.rate_limit.requests_per_minute
        self.request_counts: Dict[tuple, int] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not config.rate_limit.enabled:
            return await call_next(request)
            
        client_ip = request.client.host
        current_time = int(time.time() / 60)  # Current minute
        
        # Clean up old entries
        for ip_time in list(self.request_counts.keys()):
            ip, minute = ip_time
            if minute < current_time:
                del self.request_counts[(ip, minute)]
        
        # Check and update rate limit
        if self.request_counts.get((client_ip, current_time), 0) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
            
        self.request_counts[(client_ip, current_time)] = self.request_counts.get((client_ip, current_time), 0) + 1
        
        return await call_next(request)


def setup_middleware(app):
    """Set up middleware for the application."""
    # CORS is handled in main.py
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Add rate limiting middleware
    if config.rate_limit.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=config.rate_limit.requests_per_minute
        )