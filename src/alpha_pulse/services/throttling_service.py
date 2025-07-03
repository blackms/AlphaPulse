"""
Throttling service for AlphaPulse API.

Provides intelligent request throttling and queue management:
- Priority-based request queuing
- Circuit breaker pattern
- Graceful degradation
- Load shedding
- Quality of Service (QoS) controls
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import redis

from alpha_pulse.config.rate_limits import UserTier, PRIORITY_LEVELS, CIRCUIT_BREAKER_CONFIG
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 1    # System-critical operations
    HIGH = 2        # Institutional users
    NORMAL = 3      # Premium users
    LOW = 4         # Basic users
    BACKGROUND = 5  # Background/anonymous operations


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ThrottleRequest:
    """Request to be throttled."""
    request_id: str
    priority: RequestPriority
    user_tier: UserTier
    endpoint: str
    estimated_duration: float = 1.0
    max_wait_time: float = 30.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueMetrics:
    """Queue performance metrics."""
    total_requests: int = 0
    processed_requests: int = 0
    dropped_requests: int = 0
    avg_wait_time: float = 0.0
    avg_processing_time: float = 0.0
    queue_length: int = 0
    throughput_per_second: float = 0.0


class CircuitBreaker:
    """Circuit breaker for service protection."""
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        """Initialize circuit breaker."""
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        
        self.audit_logger = get_audit_logger()
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker open for {self.service_name}")
                
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise
            
    async def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
                
    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self._transition_to_open()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to_open()
            
    def _transition_to_open(self):
        """Transition to open state."""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = time.time()
        self.success_count = 0
        
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_START,
            event_data={
                'circuit_breaker': self.service_name,
                'state_change': 'open',
                'failure_count': self.failure_count
            },
            severity=AuditSeverity.WARNING
        )
        
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = time.time()
        self.success_count = 0
        
    def _transition_to_closed(self):
        """Transition to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = time.time()
        
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_START,
            event_data={
                'circuit_breaker': self.service_name,
                'state_change': 'closed',
                'recovery_time': time.time() - self.last_failure_time
            }
        )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class PriorityQueue:
    """Priority-based request queue with QoS controls."""
    
    def __init__(self, redis_client: redis.Redis, max_size: int = 10000):
        """Initialize priority queue."""
        self.redis = redis_client
        self.max_size = max_size
        self.metrics = QueueMetrics()
        self.audit_logger = get_audit_logger()
        
    async def enqueue(self, request: ThrottleRequest) -> bool:
        """Add request to priority queue."""
        # Check queue capacity
        current_size = await self._get_queue_size()
        if current_size >= self.max_size:
            await self._handle_queue_overflow(request)
            return False
            
        # Serialize request
        request_data = {
            'request_id': request.request_id,
            'priority': request.priority.value,
            'user_tier': request.user_tier.value,
            'endpoint': request.endpoint,
            'estimated_duration': request.estimated_duration,
            'max_wait_time': request.max_wait_time,
            'created_at': request.created_at,
            'metadata': json.dumps(request.metadata)
        }
        
        # Add to priority queue (lower score = higher priority)
        queue_key = f"throttle_queue:{request.priority.value}"
        score = request.created_at  # FIFO within same priority
        
        await self.redis.zadd(queue_key, {json.dumps(request_data): score})
        
        # Update metrics
        self.metrics.total_requests += 1
        
        return True
        
    async def dequeue(self, timeout: float = 1.0) -> Optional[ThrottleRequest]:
        """Get next request from queue."""
        # Check all priority levels in order
        for priority in RequestPriority:
            queue_key = f"throttle_queue:{priority.value}"
            
            # Get oldest request from this priority level
            items = await self.redis.zrange(queue_key, 0, 0, withscores=True)
            
            if items:
                request_json, score = items[0]
                
                # Check if request hasn't expired
                request_data = json.loads(request_json)
                created_at = request_data['created_at']
                max_wait_time = request_data['max_wait_time']
                
                if time.time() - created_at > max_wait_time:
                    # Remove expired request
                    await self.redis.zrem(queue_key, request_json)
                    self.metrics.dropped_requests += 1
                    continue
                    
                # Remove from queue and return
                await self.redis.zrem(queue_key, request_json)
                
                # Reconstruct request object
                request = ThrottleRequest(
                    request_id=request_data['request_id'],
                    priority=RequestPriority(request_data['priority']),
                    user_tier=UserTier(request_data['user_tier']),
                    endpoint=request_data['endpoint'],
                    estimated_duration=request_data['estimated_duration'],
                    max_wait_time=request_data['max_wait_time'],
                    created_at=request_data['created_at'],
                    metadata=json.loads(request_data['metadata'])
                )
                
                # Update metrics
                wait_time = time.time() - request.created_at
                self.metrics.avg_wait_time = (
                    (self.metrics.avg_wait_time * self.metrics.processed_requests + wait_time) /
                    (self.metrics.processed_requests + 1)
                )
                self.metrics.processed_requests += 1
                
                return request
                
        return None
        
    async def _get_queue_size(self) -> int:
        """Get total queue size across all priorities."""
        total_size = 0
        for priority in RequestPriority:
            queue_key = f"throttle_queue:{priority.value}"
            size = await self.redis.zcard(queue_key)
            total_size += size
            
        self.metrics.queue_length = total_size
        return total_size
        
    async def _handle_queue_overflow(self, request: ThrottleRequest):
        """Handle queue overflow by dropping low-priority requests."""
        # Drop oldest low-priority requests to make room
        for priority in reversed(list(RequestPriority)):
            if priority.value >= RequestPriority.LOW.value:
                queue_key = f"throttle_queue:{priority.value}"
                
                # Remove oldest request from this priority
                removed = await self.redis.zpopmin(queue_key, count=1)
                if removed:
                    self.metrics.dropped_requests += 1
                    
                    # Log dropped request
                    self.audit_logger.log(
                        event_type=AuditEventType.API_ERROR,
                        event_data={
                            'action': 'request_dropped',
                            'reason': 'queue_overflow',
                            'dropped_priority': priority.value,
                            'new_request_priority': request.priority.value
                        },
                        severity=AuditSeverity.WARNING
                    )
                    return
                    
    async def get_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        await self._get_queue_size()  # Update queue length
        return self.metrics
        
    async def cleanup_expired(self):
        """Clean up expired requests from all queues."""
        current_time = time.time()
        total_expired = 0
        
        for priority in RequestPriority:
            queue_key = f"throttle_queue:{priority.value}"
            
            # Get all requests in this queue
            items = await self.redis.zrange(queue_key, 0, -1, withscores=True)
            
            for request_json, score in items:
                try:
                    request_data = json.loads(request_json)
                    created_at = request_data['created_at']
                    max_wait_time = request_data['max_wait_time']
                    
                    if current_time - created_at > max_wait_time:
                        await self.redis.zrem(queue_key, request_json)
                        total_expired += 1
                        
                except (json.JSONDecodeError, KeyError):
                    # Remove malformed entries
                    await self.redis.zrem(queue_key, request_json)
                    total_expired += 1
                    
        self.metrics.dropped_requests += total_expired
        return total_expired


class LoadBalancer:
    """Load balancer for distributing requests across workers."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize load balancer."""
        self.redis = redis_client
        self.worker_metrics = {}
        self.audit_logger = get_audit_logger()
        
    async def get_best_worker(self, request: ThrottleRequest) -> Optional[str]:
        """Get the best worker for handling request."""
        # Get available workers
        workers = await self._get_available_workers()
        
        if not workers:
            return None
            
        # Score workers based on current load
        best_worker = None
        best_score = float('inf')
        
        for worker_id in workers:
            score = await self._calculate_worker_score(worker_id, request)
            if score < best_score:
                best_score = score
                best_worker = worker_id
                
        return best_worker
        
    async def _get_available_workers(self) -> List[str]:
        """Get list of available workers."""
        # Workers register themselves with heartbeat
        workers = []
        worker_keys = await self.redis.keys("worker:*:heartbeat")
        
        current_time = time.time()
        for key in worker_keys:
            last_heartbeat = await self.redis.get(key)
            if last_heartbeat and current_time - float(last_heartbeat) < 30:  # 30 second timeout
                worker_id = key.split(':')[1]
                workers.append(worker_id)
                
        return workers
        
    async def _calculate_worker_score(self, worker_id: str, request: ThrottleRequest) -> float:
        """Calculate worker suitability score (lower is better)."""
        score = 0.0
        
        # Current queue length
        queue_length = await self.redis.llen(f"worker:{worker_id}:queue")
        score += queue_length * 10
        
        # CPU usage
        cpu_usage = await self.redis.get(f"worker:{worker_id}:cpu")
        if cpu_usage:
            score += float(cpu_usage)
            
        # Memory usage
        memory_usage = await self.redis.get(f"worker:{worker_id}:memory")
        if memory_usage:
            score += float(memory_usage) * 0.5
            
        # Worker specialization (some workers might be better for certain endpoints)
        if request.endpoint.startswith('/api/v1/trades/'):
            trading_score = await self.redis.get(f"worker:{worker_id}:trading_score")
            if trading_score:
                score *= (1 - float(trading_score))  # Lower score if better at trading
                
        return score
        
    async def assign_request(self, worker_id: str, request: ThrottleRequest):
        """Assign request to worker."""
        # Add to worker queue
        request_data = {
            'request_id': request.request_id,
            'priority': request.priority.value,
            'endpoint': request.endpoint,
            'assigned_at': time.time()
        }
        
        await self.redis.lpush(f"worker:{worker_id}:queue", json.dumps(request_data))
        
        # Update worker metrics
        await self.redis.incr(f"worker:{worker_id}:assigned_count")


class ThrottlingService:
    """Main throttling service orchestrating all components."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize throttling service."""
        self.redis = redis_client
        self.priority_queue = PriorityQueue(redis_client)
        self.load_balancer = LoadBalancer(redis_client)
        self.circuit_breakers = {}
        self.audit_logger = get_audit_logger()
        
        # Background tasks
        self._cleanup_task = None
        self._metrics_task = None
        
    async def start(self):
        """Start the throttling service."""
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        # Start metrics collection task
        self._metrics_task = asyncio.create_task(self._metrics_worker())
        
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_START,
            event_data={
                'service': 'throttling_service',
                'status': 'started'
            }
        )
        
    async def stop(self):
        """Stop the throttling service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        if self._metrics_task:
            self._metrics_task.cancel()
            
        self.audit_logger.log(
            event_type=AuditEventType.SYSTEM_STOP,
            event_data={
                'service': 'throttling_service',
                'status': 'stopped'
            }
        )
        
    async def throttle_request(self, request: ThrottleRequest) -> bool:
        """Throttle incoming request."""
        # Check circuit breaker for endpoint
        circuit_breaker = self._get_circuit_breaker(request.endpoint)
        
        try:
            # Attempt to process through circuit breaker
            return await circuit_breaker.call(self._process_request, request)
            
        except CircuitBreakerOpenError:
            # Circuit breaker is open, reject request
            self.audit_logger.log(
                event_type=AuditEventType.API_ERROR,
                event_data={
                    'action': 'request_rejected',
                    'reason': 'circuit_breaker_open',
                    'endpoint': request.endpoint,
                    'request_id': request.request_id
                },
                severity=AuditSeverity.WARNING
            )
            return False
            
    async def _process_request(self, request: ThrottleRequest) -> bool:
        """Process request through throttling pipeline."""
        # Add to priority queue
        if not await self.priority_queue.enqueue(request):
            return False
            
        # For demonstration, we'll immediately dequeue and process
        # In a real system, this would be handled by separate workers
        queued_request = await self.priority_queue.dequeue()
        if queued_request:
            # Find best worker
            worker_id = await self.load_balancer.get_best_worker(queued_request)
            
            if worker_id:
                await self.load_balancer.assign_request(worker_id, queued_request)
                return True
                
        return False
        
    def _get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for endpoint."""
        if endpoint not in self.circuit_breakers:
            self.circuit_breakers[endpoint] = CircuitBreaker(
                service_name=endpoint,
                failure_threshold=CIRCUIT_BREAKER_CONFIG['failure_threshold'],
                recovery_timeout=CIRCUIT_BREAKER_CONFIG['recovery_timeout']
            )
            
        return self.circuit_breakers[endpoint]
        
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks."""
        while True:
            try:
                # Clean up expired requests
                await self.priority_queue.cleanup_expired()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Sleep for 60 seconds
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.audit_logger.log(
                    event_type=AuditEventType.SYSTEM_START,
                    event_data={
                        'cleanup_error': str(e)
                    },
                    severity=AuditSeverity.ERROR
                )
                await asyncio.sleep(60)
                
    async def _metrics_worker(self):
        """Background worker for metrics collection."""
        while True:
            try:
                # Collect and store metrics
                await self._collect_metrics()
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.audit_logger.log(
                    event_type=AuditEventType.SYSTEM_START,
                    event_data={
                        'metrics_error': str(e)
                    },
                    severity=AuditSeverity.ERROR
                )
                await asyncio.sleep(30)
                
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data."""
        # Remove metrics older than 24 hours
        cutoff_time = time.time() - 86400
        
        metrics_keys = await self.redis.keys("metrics:*")
        for key in metrics_keys:
            # For sorted sets, remove old entries
            await self.redis.zremrangebyscore(key, 0, cutoff_time)
            
    async def _collect_metrics(self):
        """Collect and store current metrics."""
        timestamp = time.time()
        
        # Queue metrics
        queue_metrics = await self.priority_queue.get_metrics()
        
        metrics_data = {
            'timestamp': timestamp,
            'queue_length': queue_metrics.queue_length,
            'total_requests': queue_metrics.total_requests,
            'processed_requests': queue_metrics.processed_requests,
            'dropped_requests': queue_metrics.dropped_requests,
            'avg_wait_time': queue_metrics.avg_wait_time
        }
        
        # Store in time series
        await self.redis.zadd(
            "metrics:throttling",
            {json.dumps(metrics_data): timestamp}
        )
        
    async def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        queue_metrics = await self.priority_queue.get_metrics()
        
        # Circuit breaker states
        circuit_breaker_states = {}
        for endpoint, cb in self.circuit_breakers.items():
            circuit_breaker_states[endpoint] = cb.state.value
            
        return {
            'queue_metrics': {
                'total_requests': queue_metrics.total_requests,
                'processed_requests': queue_metrics.processed_requests,
                'dropped_requests': queue_metrics.dropped_requests,
                'current_queue_length': queue_metrics.queue_length,
                'avg_wait_time': queue_metrics.avg_wait_time
            },
            'circuit_breakers': circuit_breaker_states,
            'active_workers': len(await self.load_balancer._get_available_workers())
        }