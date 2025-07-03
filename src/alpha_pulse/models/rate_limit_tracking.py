"""
Database models for rate limiting and DDoS protection tracking.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RateLimitEvent(Base):
    """Track rate limiting events for analysis and monitoring."""
    __tablename__ = "rate_limit_events"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Request identification
    ip_address = Column(String(50), nullable=False, index=True)
    user_id = Column(String(100), index=True)
    api_key_id = Column(String(100), index=True)
    
    # Request details
    endpoint = Column(String(200), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    user_agent = Column(String(500))
    
    # Rate limiting details
    limit_type = Column(String(50), nullable=False)  # ip, user, api_key, endpoint
    limit_value = Column(Integer, nullable=False)    # requests allowed
    window_seconds = Column(Integer, nullable=False) # time window
    current_count = Column(Integer, nullable=False)  # current request count
    
    # Action taken
    action = Column(String(20), nullable=False)  # allowed, blocked, delayed
    delay_ms = Column(Integer, default=0)        # delay applied in milliseconds
    
    # Additional context
    request_size = Column(Integer)
    response_status = Column(Integer)
    response_time_ms = Column(Float)
    
    # Create indexes for common queries
    __table_args__ = (
        Index('idx_rate_limit_ip_timestamp', 'ip_address', 'timestamp'),
        Index('idx_rate_limit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_rate_limit_endpoint_timestamp', 'endpoint', 'timestamp'),
        Index('idx_rate_limit_blocked', 'action', 'timestamp'),
    )


class IPReputationScore(Base):
    """Track IP reputation scores for DDoS protection."""
    __tablename__ = "ip_reputation_scores"
    
    id = Column(Integer, primary_key=True)
    ip_address = Column(String(50), unique=True, nullable=False, index=True)
    
    # Reputation score (0-100, higher is better)
    score = Column(Integer, default=50, nullable=False)
    
    # Score components
    successful_requests = Column(Integer, default=0)
    rate_limit_violations = Column(Integer, default=0)
    authentication_failures = Column(Integer, default=0)
    malicious_patterns = Column(Integer, default=0)
    
    # Classification
    is_whitelisted = Column(Boolean, default=False)
    is_blacklisted = Column(Boolean, default=False)
    
    # Geographic info
    country_code = Column(String(2))
    is_vpn = Column(Boolean, default=False)
    is_proxy = Column(Boolean, default=False)
    
    # Timestamps
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_seen = Column(DateTime(timezone=True), nullable=False)
    last_updated = Column(DateTime(timezone=True), nullable=False)
    
    # Automatic expiry
    expires_at = Column(DateTime(timezone=True))


class TrafficPattern(Base):
    """Track traffic patterns for DDoS detection."""
    __tablename__ = "traffic_patterns"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Time aggregation (1min, 5min, 15min, 1hour)
    window_minutes = Column(Integer, nullable=False, index=True)
    
    # Traffic metrics
    total_requests = Column(Integer, default=0)
    unique_ips = Column(Integer, default=0)
    new_connections = Column(Integer, default=0)
    
    # Request breakdown
    get_requests = Column(Integer, default=0)
    post_requests = Column(Integer, default=0)
    auth_attempts = Column(Integer, default=0)
    trading_requests = Column(Integer, default=0)
    
    # Response metrics
    success_responses = Column(Integer, default=0)
    error_responses = Column(Integer, default=0)
    rate_limited = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time = Column(Float, default=0)
    cpu_usage = Column(Float, default=0)
    memory_usage = Column(Float, default=0)
    
    # Anomaly detection
    anomaly_score = Column(Float, default=0)
    is_anomaly = Column(Boolean, default=False)
    
    # Geographic distribution
    top_countries = Column(JSON)  # [{country: count}, ...]
    
    # Create indexes for analysis
    __table_args__ = (
        Index('idx_traffic_window_timestamp', 'window_minutes', 'timestamp'),
        Index('idx_traffic_anomaly', 'is_anomaly', 'timestamp'),
    )


class SecurityEvent(Base):
    """Track security events and threats."""
    __tablename__ = "security_events"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Event classification
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    
    # Source information
    ip_address = Column(String(50), nullable=False, index=True)
    user_id = Column(String(100), index=True)
    user_agent = Column(String(500))
    
    # Event details
    description = Column(String(1000), nullable=False)
    details = Column(JSON)  # Additional event-specific data
    
    # Actions taken
    action_taken = Column(String(50), nullable=False)
    blocked_duration = Column(Integer, default=0)  # seconds
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(String(1000))
    
    # False positive tracking
    is_false_positive = Column(Boolean, default=False)
    marked_fp_at = Column(DateTime(timezone=True))
    marked_fp_by = Column(String(100))


class AdaptiveLimit(Base):
    """Track adaptive rate limiting adjustments."""
    __tablename__ = "adaptive_limits"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Limit scope
    limit_scope = Column(String(50), nullable=False)  # global, endpoint, user_tier
    scope_value = Column(String(100), nullable=False) # *, /api/v1/trades, premium
    
    # Original and adjusted limits
    original_limit = Column(Integer, nullable=False)
    adjusted_limit = Column(Integer, nullable=False)
    adjustment_factor = Column(Float, nullable=False)
    
    # System metrics that triggered adjustment
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    avg_response_time = Column(Float)
    error_rate = Column(Float)
    
    # Duration of adjustment
    starts_at = Column(DateTime(timezone=True), nullable=False)
    ends_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Reason for adjustment
    trigger_reason = Column(String(100), nullable=False)
    
    # Create index for active limits
    __table_args__ = (
        Index('idx_adaptive_active', 'is_active', 'limit_scope', 'scope_value'),
    )


class CircuitBreakerState(Base):
    """Track circuit breaker states for different services."""
    __tablename__ = "circuit_breaker_states"
    
    id = Column(Integer, primary_key=True)
    service_name = Column(String(100), unique=True, nullable=False, index=True)
    
    # Circuit breaker state
    state = Column(String(20), nullable=False)  # closed, open, half_open
    failure_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    
    # Timing
    last_failure_at = Column(DateTime(timezone=True))
    last_success_at = Column(DateTime(timezone=True))
    state_changed_at = Column(DateTime(timezone=True))
    next_attempt_at = Column(DateTime(timezone=True))
    
    # Configuration
    failure_threshold = Column(Integer, default=5)
    recovery_timeout = Column(Integer, default=60)
    
    # Statistics
    total_requests = Column(Integer, default=0)
    total_failures = Column(Integer, default=0)
    uptime_percentage = Column(Float, default=100.0)


class RateLimitConfig(Base):
    """Store dynamic rate limit configurations."""
    __tablename__ = "rate_limit_configs"
    
    id = Column(Integer, primary_key=True)
    
    # Configuration scope
    config_type = Column(String(50), nullable=False, index=True)  # ip, user_tier, api_key, endpoint
    config_key = Column(String(100), nullable=False, index=True)  # specific identifier
    
    # Rate limit settings
    requests_per_window = Column(Integer, nullable=False)
    window_seconds = Column(Integer, nullable=False)
    burst_size = Column(Integer)
    
    # Priority and QoS
    priority_level = Column(Integer, default=3)  # 1=highest, 5=lowest
    queue_timeout_ms = Column(Integer, default=5000)
    
    # Adaptive settings
    enable_adaptive = Column(Boolean, default=True)
    min_limit = Column(Integer)  # Minimum limit during adaptive reduction
    max_limit = Column(Integer)  # Maximum limit during adaptive increase
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    created_by = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Expiry
    expires_at = Column(DateTime(timezone=True))
    
    # Create composite index for lookups
    __table_args__ = (
        Index('idx_rate_config_lookup', 'config_type', 'config_key', 'is_active'),
    )