"""
Intelligent rate limiting middleware for AlphaPulse API.

Implements multiple rate limiting algorithms:
- Token bucket for burst handling
- Sliding window for precise rate control
- Fixed window for simple rate limits
- Adaptive limits based on system metrics
"""

import time
import json
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone, timedelta
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp
import redis
import hashlib

from alpha_pulse.config.rate_limits import (
    get_rate_limit,
    get_adaptive_factor,
    UserTier,
    APIKeyType,
    RateLimitConfig,
    PRIORITY_LEVELS
)
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType
from alpha_pulse.models.rate_limit_tracking import RateLimitEvent


class RateLimitAlgorithm:
    """Base class for rate limiting algorithms."""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "rl"):
        """Initialize with Redis client."""
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.audit_logger = get_audit_logger()
        
    def get_key(self, identifier: str, scope: str) -> str:
        """Generate Redis key for rate limit."""
        return f"{self.key_prefix}:{scope}:{identifier}"
        
    async def is_allowed(self, identifier: str, scope: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed. Returns (allowed, info)."""
        raise NotImplementedError
        
    async def record_request(self, identifier: str, scope: str, allowed: bool):
        """Record the request for tracking."""
        pass


class TokenBucketAlgorithm(RateLimitAlgorithm):
    """Token bucket algorithm for burst handling."""
    
    async def is_allowed(self, identifier: str, scope: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed using token bucket."""
        key = self.get_key(identifier, scope)
        now = time.time()
        
        # Get or initialize bucket state
        pipeline = self.redis.pipeline()
        pipeline.hmget(key, 'tokens', 'last_refill')
        pipeline.expire(key, config.window_seconds * 2)
        bucket_data = await pipeline.execute()
        
        tokens = float(bucket_data[0][0] or config.burst_size or config.requests)
        last_refill = float(bucket_data[0][1] or now)
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - last_refill
        tokens_to_add = (time_elapsed / config.window_seconds) * config.requests
        tokens = min(config.burst_size or config.requests, tokens + tokens_to_add)
        
        # Check if request can be served
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False
            
        # Update bucket state
        await self.redis.hmset(key, {
            'tokens': str(tokens),
            'last_refill': str(now)
        })
        await self.redis.expire(key, config.window_seconds * 2)
        
        info = {
            'algorithm': 'token_bucket',
            'tokens_remaining': tokens,
            'refill_rate': config.requests / config.window_seconds,
            'bucket_size': config.burst_size or config.requests
        }
        
        return allowed, info


class SlidingWindowAlgorithm(RateLimitAlgorithm):
    """Sliding window algorithm for precise rate control."""
    
    async def is_allowed(self, identifier: str, scope: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed using sliding window."""
        key = self.get_key(identifier, scope)
        now = time.time()
        window_start = now - config.window_seconds
        
        # Remove old entries and count current requests
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, window_start)
        pipeline.zcard(key)
        pipeline.expire(key, config.window_seconds)
        
        results = await pipeline.execute()
        current_requests = results[1]
        
        # Check if under limit
        if current_requests < config.requests:
            # Add current request
            await self.redis.zadd(key, {str(now): now})
            allowed = True
            remaining = config.requests - current_requests - 1
        else:
            allowed = False
            remaining = 0
            
        # Calculate time until window resets
        oldest_request = await self.redis.zrange(key, 0, 0, withscores=True)
        reset_time = (oldest_request[0][1] + config.window_seconds) if oldest_request else now + config.window_seconds
        
        info = {
            'algorithm': 'sliding_window',
            'requests_remaining': remaining,
            'window_size': config.window_seconds,
            'reset_time': reset_time
        }
        
        return allowed, info


class FixedWindowAlgorithm(RateLimitAlgorithm):
    """Fixed window algorithm for simple rate limits."""
    
    async def is_allowed(self, identifier: str, scope: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed using fixed window."""
        key = self.get_key(identifier, scope)
        now = time.time()
        window_start = int(now // config.window_seconds) * config.window_seconds
        window_key = f"{key}:{window_start}"
        
        # Get current count and increment
        pipeline = self.redis.pipeline()
        pipeline.incr(window_key)
        pipeline.expire(window_key, config.window_seconds)
        
        results = await pipeline.execute()
        current_count = results[0]
        
        allowed = current_count <= config.requests
        remaining = max(0, config.requests - current_count)
        reset_time = window_start + config.window_seconds
        
        info = {
            'algorithm': 'fixed_window',
            'requests_remaining': remaining,
            'window_start': window_start,
            'reset_time': reset_time
        }
        
        return allowed, info


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system metrics."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize adaptive rate limiter."""
        self.redis = redis_client
        self.token_bucket = TokenBucketAlgorithm(redis_client, "tb")
        self.sliding_window = SlidingWindowAlgorithm(redis_client, "sw") 
        self.fixed_window = FixedWindowAlgorithm(redis_client, "fw")
        
    def get_algorithm(self, config: RateLimitConfig) -> RateLimitAlgorithm:
        """Select appropriate algorithm based on configuration."""
        if config.burst_size:
            return self.token_bucket
        elif config.window_seconds <= 60:
            return self.sliding_window
        else:
            return self.fixed_window
            
    async def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics for adaptive limiting."""
        try:
            import psutil
            return {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'avg_response_time': await self._get_avg_response_time(),
                'error_rate': await self._get_error_rate()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'avg_response_time': 0,
                'error_rate': 0
            }
            
    async def _get_avg_response_time(self) -> float:
        """Get average response time from Redis metrics."""
        try:
            response_times = await self.redis.lrange("metrics:response_times", 0, 99)
            if response_times:
                return sum(float(t) for t in response_times) / len(response_times)
        except:
            pass
        return 0
        
    async def _get_error_rate(self) -> float:
        """Get error rate from Redis metrics."""
        try:
            error_count = await self.redis.get("metrics:errors:1min") or 0
            total_count = await self.redis.get("metrics:requests:1min") or 1
            return (int(error_count) / int(total_count)) * 100
        except:
            pass
        return 0
        
    async def check_rate_limit(
        self,
        identifier: str,
        scope: str,
        config: RateLimitConfig,
        adaptive: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit with optional adaptive adjustment."""
        
        # Apply adaptive adjustment if enabled
        if adaptive:
            system_metrics = await self.get_system_metrics()
            adaptive_factor = get_adaptive_factor(system_metrics)
            if adaptive_factor < 1.0:
                config = RateLimitConfig(
                    requests=int(config.requests * adaptive_factor),
                    window_seconds=config.window_seconds,
                    burst_size=int(config.burst_size * adaptive_factor) if config.burst_size else None
                )
                
        # Select and use appropriate algorithm
        algorithm = self.get_algorithm(config)
        allowed, info = await algorithm.is_allowed(identifier, scope, config)
        
        # Add adaptive info
        if adaptive:
            info['adaptive_factor'] = adaptive_factor
            info['system_metrics'] = system_metrics
            
        return allowed, info


class PriorityQueue:
    """Priority queue for handling requests based on user tier."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize priority queue."""
        self.redis = redis_client
        
    async def enqueue_request(
        self,
        request_id: str,
        priority: int,
        user_tier: UserTier,
        timeout_ms: int = 5000
    ) -> bool:
        """Enqueue request with priority."""
        queue_key = f"queue:priority:{priority}"
        
        # Add to priority queue with timestamp
        score = time.time() * 1000  # milliseconds
        await self.redis.zadd(queue_key, {request_id: score})
        
        # Set expiry
        await self.redis.expire(queue_key, timeout_ms // 1000 + 10)
        
        return True
        
    async def dequeue_request(self, max_wait_ms: int = 5000) -> Optional[str]:
        """Dequeue highest priority request."""
        # Check all priority levels
        for priority in sorted(PRIORITY_LEVELS.values()):
            queue_key = f"queue:priority:{priority}"
            
            # Get oldest request from this priority level
            items = await self.redis.zrange(queue_key, 0, 0, withscores=True)
            if items:
                request_id, timestamp = items[0]
                
                # Check if request hasn't expired
                if time.time() * 1000 - timestamp < max_wait_ms:
                    await self.redis.zrem(queue_key, request_id)
                    return request_id.decode() if isinstance(request_id, bytes) else request_id
                else:
                    # Remove expired request
                    await self.redis.zrem(queue_key, request_id)
                    
        return None


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Main rate limiting middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        redis_url: str = "redis://localhost:6379",
        enable_adaptive: bool = True,
        enable_priority_queue: bool = True,
        default_algorithm: str = "token_bucket"
    ):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.rate_limiter = AdaptiveRateLimiter(self.redis)
        self.priority_queue = PriorityQueue(self.redis) if enable_priority_queue else None
        self.enable_adaptive = enable_adaptive
        self.audit_logger = get_audit_logger()
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        start_time = time.time()
        
        # Extract identifiers
        client_ip = self._get_client_ip(request)
        user_info = self._get_user_info(request)
        endpoint = request.url.path
        
        # Determine rate limit configuration
        config = get_rate_limit(
            endpoint=endpoint,
            user_tier=user_info.get('tier'),
            api_key_type=user_info.get('api_key_type'),
            limit_type=self._get_limit_type(endpoint)
        )
        
        # Check multiple rate limits
        rate_limit_checks = [
            ("ip", client_ip, config),
            ("user", user_info.get('user_id', 'anonymous'), config),
            ("endpoint", f"{endpoint}:{client_ip}", config)
        ]
        
        # Process each rate limit check
        for limit_type, identifier, limit_config in rate_limit_checks:
            if identifier:
                allowed, info = await self.rate_limiter.check_rate_limit(
                    identifier=identifier,
                    scope=limit_type,
                    config=limit_config,
                    adaptive=self.enable_adaptive
                )
                
                if not allowed:
                    return await self._handle_rate_limited(
                        request, limit_type, identifier, info, start_time
                    )
                    
        # Handle priority queuing if enabled
        if self.priority_queue and user_info.get('tier'):
            priority = PRIORITY_LEVELS.get(user_info['tier'], 5)
            if priority > 1:  # Only queue non-institutional users
                request_id = f"{client_ip}:{int(time.time() * 1000)}"
                await self.priority_queue.enqueue_request(
                    request_id, priority, user_info['tier']
                )
                
                # Wait for turn
                dequeued = await self.priority_queue.dequeue_request()
                if dequeued != request_id:
                    # Still in queue, apply delay
                    delay_ms = min(1000, priority * 100)
                    await asyncio.sleep(delay_ms / 1000)
                    
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, info)
        
        # Log successful request
        await self._log_rate_limit_event(
            request, "allowed", identifier, info, start_time
        )
        
        return response
        
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
            
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
            
        # Fallback to client IP
        return request.client.host if request.client else "unknown"
        
    def _get_user_info(self, request: Request) -> Dict[str, Any]:
        """Extract user information from request."""
        user_info = {
            'user_id': None,
            'tier': UserTier.ANONYMOUS,
            'api_key_type': None
        }
        
        # Check for JWT user
        if hasattr(request.state, 'user'):
            user = request.state.user
            user_info['user_id'] = getattr(user, 'username', None)
            
            # Determine tier from user role
            role = getattr(user, 'role', 'anonymous')
            tier_mapping = {
                'admin': UserTier.SYSTEM,
                'institutional': UserTier.INSTITUTIONAL,
                'premium': UserTier.PREMIUM,
                'trader': UserTier.BASIC,
                'viewer': UserTier.BASIC
            }
            user_info['tier'] = tier_mapping.get(role, UserTier.ANONYMOUS)
            
        # Check for API key
        api_key = request.headers.get('X-API-Key')
        if api_key:
            # Determine API key type (would be from database lookup)
            user_info['api_key_type'] = APIKeyType.READ_ONLY  # Default
            
        return user_info
        
    def _get_limit_type(self, endpoint: str) -> str:
        """Determine the type of rate limit to apply."""
        if '/auth/' in endpoint or endpoint.endswith('/token'):
            return 'auth'
        elif '/trades/' in endpoint:
            return 'trading'
        elif '/data/' in endpoint or '/metrics/' in endpoint:
            return 'data'
        else:
            return 'global'
            
    async def _handle_rate_limited(
        self,
        request: Request,
        limit_type: str,
        identifier: str,
        info: Dict[str, Any],
        start_time: float
    ) -> Response:
        """Handle rate limited request."""
        
        # Log rate limit event
        await self._log_rate_limit_event(
            request, "blocked", identifier, info, start_time
        )
        
        # Audit log the rate limit hit
        self.audit_logger.log(
            event_type=AuditEventType.API_RATE_LIMITED,
            event_data={
                'limit_type': limit_type,
                'identifier': identifier,
                'algorithm': info.get('algorithm'),
                'endpoint': request.url.path,
                'method': request.method
            }
        )
        
        # Determine retry after time
        retry_after = info.get('reset_time', time.time() + 60) - time.time()
        
        # Create rate limit response
        response_data = {
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Limit: {info.get('algorithm', 'unknown')}",
            "retry_after": int(retry_after),
            "limit_type": limit_type
        }
        
        response = JSONResponse(
            status_code=429,
            content=response_data
        )
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, info)
        response.headers["Retry-After"] = str(int(retry_after))
        
        return response
        
    def _add_rate_limit_headers(self, response: Response, info: Dict[str, Any]):
        """Add rate limiting headers to response."""
        if 'requests_remaining' in info:
            response.headers["X-RateLimit-Remaining"] = str(info['requests_remaining'])
            
        if 'reset_time' in info:
            response.headers["X-RateLimit-Reset"] = str(int(info['reset_time']))
            
        if 'algorithm' in info:
            response.headers["X-RateLimit-Algorithm"] = info['algorithm']
            
        if 'adaptive_factor' in info:
            response.headers["X-RateLimit-Adaptive"] = str(info['adaptive_factor'])
            
    async def _log_rate_limit_event(
        self,
        request: Request,
        action: str,
        identifier: str,
        info: Dict[str, Any],
        start_time: float
    ):
        """Log rate limiting event for monitoring."""
        # This would typically write to database
        # For now, we'll store in Redis for immediate use
        event_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'identifier': identifier,
            'endpoint': request.url.path,
            'method': request.method,
            'user_agent': request.headers.get('User-Agent', ''),
            'processing_time': (time.time() - start_time) * 1000,
            'algorithm': info.get('algorithm'),
            'adaptive_factor': info.get('adaptive_factor', 1.0)
        }
        
        # Store in Redis for real-time monitoring
        await self.redis.lpush(
            "rate_limit_events",
            json.dumps(event_data)
        )
        
        # Keep only last 10000 events
        await self.redis.ltrim("rate_limit_events", 0, 9999)