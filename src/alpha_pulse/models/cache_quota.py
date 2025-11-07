"""
SQLAlchemy ORM models for cache quota and metrics tables.

These models provide object-relational mapping for:
- tenant_cache_quotas: Per-tenant cache quota configuration
- tenant_cache_metrics: Per-tenant cache performance metrics

Usage:
    from alpha_pulse.models.cache_quota import TenantCacheQuota, TenantCacheMetric

    # Query quota for a tenant
    quota = await session.get(TenantCacheQuota, tenant_id=tenant_id)

    # Query metrics for a date range
    metrics = await session.execute(
        select(TenantCacheMetric)
        .where(TenantCacheMetric.tenant_id == tenant_id)
        .where(TenantCacheMetric.metric_date >= start_date)
        .order_by(TenantCacheMetric.metric_date.desc())
    )
"""
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    TIMESTAMP,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Use shared Base (same as user_data.py)
Base = declarative_base()


class TenantCacheQuota(Base):
    """
    ORM model for tenant_cache_quotas table.

    Tracks per-tenant cache quota configuration and current usage.

    Attributes:
        id: Primary key
        tenant_id: Foreign key to tenants table
        quota_mb: Maximum cache size in MB
        current_usage_mb: Current cache usage in MB (synced from Redis)
        quota_reset_at: Next quota reset timestamp (monthly)
        overage_allowed: Allow temporary quota overage
        overage_limit_mb: Maximum overage before hard limit (MB)
        created_at: Creation timestamp
        updated_at: Last update timestamp (auto-updated by trigger)

    Relationships:
        tenant: Many-to-one relationship with Tenant model

    Example:
        >>> quota = TenantCacheQuota(
        ...     tenant_id=UUID('...'),
        ...     quota_mb=500,  # Pro tier
        ...     overage_allowed=True,
        ...     overage_limit_mb=50
        ... )
        >>> session.add(quota)
        >>> await session.commit()
    """

    __tablename__ = 'tenant_cache_quotas'

    id = Column(Integer, primary_key=True)
    tenant_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False,
        unique=True,
        index=True
    )

    quota_mb = Column(Integer, nullable=False, default=100, server_default='100')
    current_usage_mb = Column(Numeric(10, 2), default=0, server_default='0')
    quota_reset_at = Column(TIMESTAMP, server_default='NOW()')
    overage_allowed = Column(Boolean, default=False, server_default='false')
    overage_limit_mb = Column(Integer, default=10, server_default='10')

    created_at = Column(TIMESTAMP, nullable=False, server_default='NOW()')
    updated_at = Column(TIMESTAMP, nullable=False, server_default='NOW()')

    # Relationships
    # tenant = relationship("Tenant", back_populates="cache_quota")

    def __repr__(self):
        return (
            f"<TenantCacheQuota(tenant_id={self.tenant_id}, "
            f"quota_mb={self.quota_mb}, "
            f"usage_mb={self.current_usage_mb}, "
            f"usage_pct={self.usage_percent:.1f}%)>"
        )

    @property
    def usage_percent(self) -> float:
        """Calculate usage as percentage of quota."""
        if self.quota_mb == 0:
            return 0.0
        return float((self.current_usage_mb or 0) / self.quota_mb * 100)

    @property
    def is_over_quota(self) -> bool:
        """Check if current usage exceeds quota."""
        return (self.current_usage_mb or 0) > self.quota_mb

    @property
    def is_over_hard_limit(self) -> bool:
        """Check if usage exceeds quota + overage limit."""
        hard_limit = self.quota_mb + (
            self.overage_limit_mb if self.overage_allowed else 0
        )
        return (self.current_usage_mb or 0) > hard_limit

    @property
    def available_mb(self) -> Decimal:
        """Calculate available quota in MB."""
        return Decimal(self.quota_mb) - (self.current_usage_mb or Decimal(0))


class TenantCacheMetric(Base):
    """
    ORM model for tenant_cache_metrics table.

    Tracks per-tenant cache performance metrics on a daily basis.

    Attributes:
        id: Primary key
        tenant_id: Foreign key to tenants table
        metric_date: Date for these metrics
        total_requests: Total cache requests (hits + misses)
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        hit_rate: Computed column (cache_hits / total_requests * 100)
        avg_response_time_ms: Average cache response time in milliseconds
        total_bytes_served: Total bytes served from cache
        created_at: Creation timestamp

    Relationships:
        tenant: Many-to-one relationship with Tenant model

    Example:
        >>> metric = TenantCacheMetric(
        ...     tenant_id=UUID('...'),
        ...     metric_date=date.today(),
        ...     total_requests=1000,
        ...     cache_hits=850,
        ...     cache_misses=150
        ... )
        >>> # hit_rate is automatically calculated: 85.0
        >>> session.add(metric)
        >>> await session.commit()
    """

    __tablename__ = 'tenant_cache_metrics'

    id = Column(Integer, primary_key=True)
    tenant_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('tenants.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    metric_date = Column(Date, nullable=False)
    total_requests = Column(BigInteger, default=0, server_default='0')
    cache_hits = Column(BigInteger, default=0, server_default='0')
    cache_misses = Column(BigInteger, default=0, server_default='0')

    # Computed column (GENERATED ALWAYS AS ... STORED)
    # Note: SQLAlchemy doesn't natively support computed columns in column definitions,
    # so we rely on the database-level computed column from the migration.
    # This is just for ORM mapping.
    hit_rate = Column(Numeric(5, 2))

    avg_response_time_ms = Column(Numeric(10, 2))
    total_bytes_served = Column(BigInteger, default=0, server_default='0')

    created_at = Column(TIMESTAMP, nullable=False, server_default='NOW()')

    # Relationships
    # tenant = relationship("Tenant", back_populates="cache_metrics")

    __table_args__ = (
        UniqueConstraint(
            'tenant_id',
            'metric_date',
            name='uq_tenant_cache_metrics_tenant_date'
        ),
    )

    def __repr__(self):
        return (
            f"<TenantCacheMetric(tenant_id={self.tenant_id}, "
            f"date={self.metric_date}, "
            f"hits={self.cache_hits}, "
            f"misses={self.cache_misses}, "
            f"hit_rate={self.hit_rate:.1f}%)>"
        )

    @property
    def calculated_hit_rate(self) -> float:
        """
        Calculate hit rate from hits and total requests.

        This is a fallback in case the computed column isn't populated yet.
        """
        if self.total_requests == 0:
            return 0.0
        return float(self.cache_hits / self.total_requests * 100)

    @property
    def total_bytes_served_mb(self) -> float:
        """Get total bytes served in MB."""
        return float((self.total_bytes_served or 0) / 1024 / 1024)
