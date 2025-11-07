"""
Data models for quota enforcement system.

Defines data structures for quota configuration, decisions, and status.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from uuid import UUID


class QuotaDecision(str, Enum):
    """Quota enforcement decision."""

    ALLOW = "allow"      # Within quota, request allowed
    WARN = "warn"        # Over quota but within overage, request allowed with warning
    REJECT = "reject"    # Exceeds hard limit, request rejected


class QuotaStatus(str, Enum):
    """Quota status for response headers."""

    OK = "ok"            # Within quota (< 80%)
    WARNING = "warning"  # Approaching limit (80-100%)
    EXCEEDED = "exceeded"  # Over quota


@dataclass
class QuotaConfig:
    """
    Quota configuration for a tenant.

    Attributes:
        tenant_id: Unique tenant identifier
        quota_mb: Base quota in megabytes
        current_usage_mb: Current usage in megabytes
        overage_allowed: Whether overage is permitted
        overage_limit_mb: Additional megabytes allowed over quota
    """

    tenant_id: UUID
    quota_mb: float
    current_usage_mb: float
    overage_allowed: bool
    overage_limit_mb: float

    @property
    def hard_limit_mb(self) -> float:
        """
        Calculate hard limit (quota + overage if allowed).

        Returns:
            Hard limit in megabytes
        """
        if self.overage_allowed:
            return self.quota_mb + self.overage_limit_mb
        return self.quota_mb

    @property
    def remaining_mb(self) -> float:
        """
        Calculate remaining quota (can be negative if over quota).

        Returns:
            Remaining quota in megabytes
        """
        return self.quota_mb - self.current_usage_mb

    @property
    def percent_used(self) -> float:
        """
        Calculate percentage of quota used.

        Returns:
            Percentage (0-100+)
        """
        if self.quota_mb == 0:
            return 100.0
        return (self.current_usage_mb / self.quota_mb) * 100.0

    @property
    def usage_percent(self) -> float:
        """
        Calculate usage as percentage of quota (alias for percent_used).

        Returns:
            Percentage (0-100+)
        """
        return self.percent_used

    @property
    def status(self) -> QuotaStatus:
        """
        Determine quota status based on usage.

        Returns:
            QuotaStatus enum value
        """
        if self.current_usage_mb > self.quota_mb:
            return QuotaStatus.EXCEEDED
        elif self.percent_used >= 80.0:
            return QuotaStatus.WARNING
        else:
            return QuotaStatus.OK

    def can_allocate(self, size_mb: float) -> bool:
        """
        Check if size can be allocated without exceeding hard limit.

        Args:
            size_mb: Size to allocate in megabytes

        Returns:
            True if allocation would not exceed hard limit
        """
        return (self.current_usage_mb + size_mb) <= self.hard_limit_mb

    def would_exceed_quota(self, size_mb: float) -> bool:
        """
        Check if size would exceed base quota (but not necessarily hard limit).

        Args:
            size_mb: Size to check in megabytes

        Returns:
            True if allocation would exceed quota_mb
        """
        return (self.current_usage_mb + size_mb) > self.quota_mb


@dataclass
class QuotaCheckResult:
    """
    Result of a quota check operation.

    Attributes:
        decision: Enforcement decision (ALLOW/WARN/REJECT)
        quota_config: Current quota configuration
        requested_mb: Requested allocation size
        new_usage_mb: Usage after request (if allowed)
        message: Human-readable message
    """

    decision: QuotaDecision
    quota_config: QuotaConfig
    requested_mb: float
    new_usage_mb: Optional[float] = None
    message: Optional[str] = None

    @property
    def allowed(self) -> bool:
        """Check if request was allowed."""
        return self.decision in (QuotaDecision.ALLOW, QuotaDecision.WARN)

    @property
    def rejected(self) -> bool:
        """Check if request was rejected."""
        return self.decision == QuotaDecision.REJECT

    def to_headers(self) -> dict[str, str]:
        """
        Convert to HTTP response headers.

        Returns:
            Dictionary of header names to values
        """
        headers = {
            "X-Cache-Quota-Limit": str(int(self.quota_config.quota_mb)),
            "X-Cache-Quota-Used": f"{self.quota_config.current_usage_mb:.1f}",
            "X-Cache-Quota-Remaining": f"{self.quota_config.remaining_mb:.1f}",
            "X-Cache-Quota-Percent": f"{self.quota_config.percent_used:.1f}",
            "X-Cache-Quota-Status": self.quota_config.status.value,
        }

        if self.rejected:
            # Add Retry-After header for 429 responses (5 minutes)
            headers["Retry-After"] = "300"

        return headers


@dataclass
class CacheMetrics:
    """
    Metrics for cache operations.

    Attributes:
        cache_hits: Number of Redis cache hits
        cache_misses: Number of Redis cache misses
        db_queries: Number of PostgreSQL queries
        redis_errors: Number of Redis errors
        avg_latency_ms: Average operation latency in milliseconds
    """

    cache_hits: int = 0
    cache_misses: int = 0
    db_queries: int = 0
    redis_errors: int = 0
    avg_latency_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0-100)
        """
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100.0
