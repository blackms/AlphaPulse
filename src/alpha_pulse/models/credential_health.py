"""
Database models for credential health check system.

This module contains SQLAlchemy models for tracking credential health
and webhook configuration for credential failure notifications.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID as UUIDType

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func, text

from .encrypted_fields import EncryptedString

Base = declarative_base()


class CredentialHealthCheck(Base):
    """
    Model for tracking credential health check results.

    Stores validation history, consecutive failure tracking, and
    webhook notification status for tenant credentials.
    """

    __tablename__ = "credential_health_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Tenant and credential identification
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="Tenant UUID from tenants table"
    )
    exchange = Column(
        String(50),
        nullable=False,
        comment="Exchange name (e.g., binance, coinbase)"
    )
    credential_type = Column(
        String(50),
        nullable=False,
        server_default="trading",
        comment="Credential type (trading, readonly, withdrawal)"
    )

    # Health check results
    check_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When this health check was performed"
    )
    is_valid = Column(
        Boolean,
        nullable=False,
        comment="Whether credential passed validation"
    )
    validation_error = Column(
        Text,
        nullable=True,
        comment="Error message if validation failed"
    )

    # Failure tracking
    consecutive_failures = Column(
        Integer,
        nullable=False,
        server_default="0",
        comment="Number of consecutive failures (reset to 0 on success)"
    )
    last_success_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of last successful validation"
    )
    first_failure_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when current failure streak started"
    )

    # Webhook notification tracking
    webhook_sent = Column(
        Boolean,
        nullable=False,
        server_default="false",
        comment="Whether webhook notification was sent for this check"
    )
    webhook_sent_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When webhook notification was sent"
    )
    webhook_delivery_status = Column(
        String(20),
        nullable=True,
        comment="Webhook delivery status (success, failed, timeout)"
    )

    # Exchange metadata from validation
    exchange_account_id = Column(
        String(255),
        nullable=True,
        comment="Exchange account ID from validation response"
    )
    exchange_permissions = Column(
        JSONB,
        nullable=True,
        server_default=text("'[]'::jsonb"),
        comment="Exchange permissions returned by validation"
    )

    # Task metadata
    task_id = Column(
        String(255),
        nullable=True,
        comment="Celery task ID that performed this check"
    )
    check_duration_ms = Column(
        Integer,
        nullable=True,
        comment="How long the check took in milliseconds"
    )

    # Audit fields
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Record creation timestamp"
    )

    # Indexes for query performance
    __table_args__ = (
        # Composite unique constraint for latest check per credential
        Index(
            "idx_cred_health_tenant_exchange_type_timestamp",
            "tenant_id", "exchange", "credential_type", "check_timestamp"
        ),
        # Query for active failures
        Index("idx_cred_health_is_valid", "is_valid"),
        # Query for webhook delivery status
        Index("idx_cred_health_webhook_sent", "webhook_sent"),
        # Query for recent checks
        Index("idx_cred_health_check_timestamp", "check_timestamp"),
        # Query for consecutive failures
        Index("idx_cred_health_consecutive_failures", "consecutive_failures"),
        # Query all checks for a tenant
        Index("idx_cred_health_tenant_id", "tenant_id"),
        # Check constraint for valid credential types
        CheckConstraint(
            "credential_type IN ('trading', 'readonly', 'withdrawal')",
            name="ck_cred_health_credential_type"
        ),
        # Check constraint for valid webhook statuses
        CheckConstraint(
            "webhook_delivery_status IS NULL OR webhook_delivery_status IN ('success', 'failed', 'timeout', 'error')",
            name="ck_cred_health_webhook_status"
        ),
        # Check constraint: webhook_sent_at requires webhook_sent = true
        CheckConstraint(
            "(webhook_sent = false AND webhook_sent_at IS NULL) OR (webhook_sent = true)",
            name="ck_cred_health_webhook_consistency"
        ),
    )

    @validates("consecutive_failures")
    def validate_consecutive_failures(self, key, value):
        """Ensure consecutive_failures is non-negative."""
        if value < 0:
            raise ValueError("consecutive_failures must be non-negative")
        return value

    @property
    def is_failing(self) -> bool:
        """Check if credential is currently in failure state."""
        return not self.is_valid

    @property
    def has_consecutive_failures(self) -> bool:
        """Check if credential has any consecutive failures."""
        return self.consecutive_failures > 0

    def __repr__(self):
        return (
            f"<CredentialHealthCheck(id={self.id}, "
            f"tenant_id={self.tenant_id}, "
            f"exchange={self.exchange}, "
            f"is_valid={self.is_valid}, "
            f"consecutive_failures={self.consecutive_failures})>"
        )


class TenantWebhook(Base):
    """
    Model for tenant webhook configuration.

    Stores webhook URLs and secrets for sending credential failure
    notifications and other events to tenants.
    """

    __tablename__ = "tenant_webhooks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Tenant identification
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        comment="Tenant UUID from tenants table"
    )

    # Webhook configuration
    event_type = Column(
        String(100),
        nullable=False,
        comment="Event type (credential.failed, system.alert, etc.)"
    )
    webhook_url = Column(
        String(2048),
        nullable=False,
        comment="HTTPS URL for webhook delivery"
    )
    webhook_secret = Column(
        EncryptedString(encryption_context="webhook_secret"),
        nullable=False,
        comment="Secret for HMAC-SHA256 signature generation"
    )

    # Status and validation
    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true",
        comment="Whether webhook is enabled"
    )
    is_verified = Column(
        Boolean,
        nullable=False,
        server_default="false",
        comment="Whether webhook URL has been verified via test"
    )
    verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When webhook was last verified"
    )

    # Delivery tracking
    last_delivery_attempt_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time webhook delivery was attempted"
    )
    last_successful_delivery_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful webhook delivery"
    )
    consecutive_failures = Column(
        Integer,
        nullable=False,
        server_default="0",
        comment="Consecutive delivery failures (for circuit breaker)"
    )
    total_deliveries = Column(
        Integer,
        nullable=False,
        server_default="0",
        comment="Total number of webhooks sent"
    )
    successful_deliveries = Column(
        Integer,
        nullable=False,
        server_default="0",
        comment="Number of successful deliveries"
    )

    # Configuration
    timeout_seconds = Column(
        Integer,
        nullable=False,
        server_default="10",
        comment="HTTP timeout for webhook delivery"
    )
    retry_attempts = Column(
        Integer,
        nullable=False,
        server_default="3",
        comment="Number of retry attempts on failure"
    )

    # Metadata
    description = Column(
        Text,
        nullable=True,
        comment="Human-readable description of this webhook"
    )
    custom_headers = Column(
        JSONB,
        nullable=True,
        server_default=text("'{}'::jsonb"),
        comment="Optional custom HTTP headers to include"
    )

    # Audit fields
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Record creation timestamp"
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Record last update timestamp"
    )
    created_by = Column(
        String(255),
        nullable=True,
        comment="User who created this webhook"
    )

    # Indexes and constraints
    __table_args__ = (
        # Unique constraint: one webhook per tenant per event type
        UniqueConstraint(
            "tenant_id", "event_type",
            name="uq_tenant_webhooks_tenant_event"
        ),
        # Query active webhooks
        Index("idx_tenant_webhooks_active", "is_active"),
        # Query by tenant
        Index("idx_tenant_webhooks_tenant_id", "tenant_id"),
        # Query by event type
        Index("idx_tenant_webhooks_event_type", "event_type"),
        # Check constraint for valid event types
        CheckConstraint(
            "event_type IN ('credential.failed', 'credential.expiring', 'system.alert', 'trade.executed', 'risk.breach')",
            name="ck_tenant_webhooks_event_type"
        ),
        # Check constraint for HTTPS URLs
        CheckConstraint(
            "webhook_url LIKE 'https://%'",
            name="ck_tenant_webhooks_https_only"
        ),
        # Check constraint for timeout range
        CheckConstraint(
            "timeout_seconds BETWEEN 1 AND 60",
            name="ck_tenant_webhooks_timeout_range"
        ),
        # Check constraint for retry range
        CheckConstraint(
            "retry_attempts BETWEEN 0 AND 10",
            name="ck_tenant_webhooks_retry_range"
        ),
    )

    @validates("webhook_url")
    def validate_webhook_url(self, key, value):
        """Ensure webhook URL uses HTTPS."""
        if not value.startswith("https://"):
            raise ValueError("Webhook URL must use HTTPS protocol")
        return value

    @validates("consecutive_failures")
    def validate_consecutive_failures(self, key, value):
        """Ensure consecutive_failures is non-negative."""
        if value < 0:
            raise ValueError("consecutive_failures must be non-negative")
        return value

    @property
    def success_rate(self) -> float:
        """Calculate webhook delivery success rate."""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100.0

    @property
    def is_healthy(self) -> bool:
        """Check if webhook is healthy (active, verified, low failure rate)."""
        return (
            self.is_active
            and self.is_verified
            and self.consecutive_failures < 5  # Circuit breaker threshold
        )

    def increment_delivery_stats(self, success: bool):
        """Update delivery statistics after webhook attempt."""
        self.total_deliveries += 1
        self.last_delivery_attempt_at = datetime.utcnow()

        if success:
            self.successful_deliveries += 1
            self.consecutive_failures = 0
            self.last_successful_delivery_at = datetime.utcnow()
        else:
            self.consecutive_failures += 1

    def __repr__(self):
        return (
            f"<TenantWebhook(id={self.id}, "
            f"tenant_id={self.tenant_id}, "
            f"event_type={self.event_type}, "
            f"is_active={self.is_active}, "
            f"success_rate={self.success_rate:.1f}%)>"
        )


# Utility functions
def get_latest_health_check(
    session,
    tenant_id: UUIDType,
    exchange: str,
    credential_type: str = "trading"
) -> Optional[CredentialHealthCheck]:
    """
    Get the most recent health check for a credential.

    Args:
        session: SQLAlchemy session
        tenant_id: Tenant UUID
        exchange: Exchange name
        credential_type: Credential type

    Returns:
        Latest CredentialHealthCheck or None
    """
    return (
        session.query(CredentialHealthCheck)
        .filter_by(
            tenant_id=tenant_id,
            exchange=exchange,
            credential_type=credential_type
        )
        .order_by(CredentialHealthCheck.check_timestamp.desc())
        .first()
    )


def get_failing_credentials(
    session,
    min_consecutive_failures: int = 3
):
    """
    Get all credentials with consecutive failures above threshold.

    Args:
        session: SQLAlchemy session
        min_consecutive_failures: Minimum failure count

    Returns:
        Query result with failing credentials
    """
    # Subquery to get latest check per credential
    from sqlalchemy import and_
    from sqlalchemy.sql import exists

    subq = (
        session.query(
            CredentialHealthCheck.tenant_id,
            CredentialHealthCheck.exchange,
            CredentialHealthCheck.credential_type,
            func.max(CredentialHealthCheck.check_timestamp).label("max_timestamp")
        )
        .group_by(
            CredentialHealthCheck.tenant_id,
            CredentialHealthCheck.exchange,
            CredentialHealthCheck.credential_type
        )
        .subquery()
    )

    return (
        session.query(CredentialHealthCheck)
        .join(
            subq,
            and_(
                CredentialHealthCheck.tenant_id == subq.c.tenant_id,
                CredentialHealthCheck.exchange == subq.c.exchange,
                CredentialHealthCheck.credential_type == subq.c.credential_type,
                CredentialHealthCheck.check_timestamp == subq.c.max_timestamp
            )
        )
        .filter(
            CredentialHealthCheck.consecutive_failures >= min_consecutive_failures,
            CredentialHealthCheck.is_valid == False
        )
        .all()
    )


def get_tenant_webhook(
    session,
    tenant_id: UUIDType,
    event_type: str
) -> Optional[TenantWebhook]:
    """
    Get webhook configuration for a tenant and event type.

    Args:
        session: SQLAlchemy session
        tenant_id: Tenant UUID
        event_type: Event type (e.g., "credential.failed")

    Returns:
        TenantWebhook or None
    """
    return (
        session.query(TenantWebhook)
        .filter_by(
            tenant_id=tenant_id,
            event_type=event_type,
            is_active=True
        )
        .first()
    )


# Export models
__all__ = [
    "CredentialHealthCheck",
    "TenantWebhook",
    "get_latest_health_check",
    "get_failing_credentials",
    "get_tenant_webhook",
]
