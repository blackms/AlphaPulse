"""
Configuration constants for quota enforcement middleware.

Centralizes all tunable parameters for easy configuration and testing.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class QuotaEnforcementConfig:
    """Configuration for quota enforcement middleware."""

    # Feature flags
    enabled: bool = True
    """Master feature flag to enable/disable quota enforcement globally."""

    # Cache configuration
    cache_ttl_seconds: int = 300
    """Time-to-live for Redis cache entries (default: 5 minutes)."""

    # Performance targets
    target_p99_latency_ms: float = 10.0
    """Target p99 latency for quota checks in milliseconds."""

    target_throughput_rps: int = 1000
    """Target throughput in requests per second per tenant."""

    target_cache_hit_rate: float = 0.90
    """Target cache hit rate (default: 90%)."""

    # Path exclusions
    exclude_paths: List[str] = None
    """Paths to exclude from quota enforcement (e.g., /health, /metrics)."""

    # Retry configuration
    retry_after_seconds: int = 300
    """Retry-After header value for 429 responses (default: 5 minutes)."""

    # Logging configuration
    log_level: str = "INFO"
    """Log level for quota operations."""

    log_cache_hits: bool = False
    """Whether to log cache hits (can be noisy in production)."""

    log_cache_misses: bool = True
    """Whether to log cache misses."""

    def __post_init__(self):
        """Initialize default values."""
        if self.exclude_paths is None:
            self.exclude_paths = ["/health", "/metrics", "/docs", "/openapi.json"]


# Default configuration instance
DEFAULT_CONFIG = QuotaEnforcementConfig()


# Tier-specific quota limits (MB)
class QuotaTiers:
    """Predefined quota tiers for different subscription levels."""

    FREE = 100  # 100 MB
    STARTER = 500  # 500 MB
    PRO = 2000  # 2 GB
    ENTERPRISE = 10000  # 10 GB
    UNLIMITED = 100000  # 100 GB (soft limit)


# Overage limits (MB)
class OverageLimits:
    """Overage allowances per tier."""

    FREE = 10  # 10 MB (10% overage)
    STARTER = 50  # 50 MB (10% overage)
    PRO = 200  # 200 MB (10% overage)
    ENTERPRISE = 1000  # 1 GB (10% overage)
    UNLIMITED = 10000  # 10 GB (10% overage)


# Redis key patterns
class RedisKeyPatterns:
    """Redis key naming patterns."""

    QUOTA = "quota:cache:{tenant_id}:quota_mb"
    USAGE = "quota:cache:{tenant_id}:current_usage_mb"
    OVERAGE_ALLOWED = "quota:cache:{tenant_id}:overage_allowed"
    OVERAGE_LIMIT = "quota:cache:{tenant_id}:overage_limit_mb"

    @staticmethod
    def get_quota_key(tenant_id: str) -> str:
        """Get Redis key for quota value."""
        return f"quota:cache:{tenant_id}:quota_mb"

    @staticmethod
    def get_usage_key(tenant_id: str) -> str:
        """Get Redis key for current usage."""
        return f"quota:cache:{tenant_id}:current_usage_mb"

    @staticmethod
    def get_overage_allowed_key(tenant_id: str) -> str:
        """Get Redis key for overage_allowed flag."""
        return f"quota:cache:{tenant_id}:overage_allowed"

    @staticmethod
    def get_overage_limit_key(tenant_id: str) -> str:
        """Get Redis key for overage limit."""
        return f"quota:cache:{tenant_id}:overage_limit_mb"
