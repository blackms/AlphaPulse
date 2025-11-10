"""Add credential health check and webhook tables

Revision ID: 010_credential_health
Revises: 009
Create Date: 2025-11-10

EPIC-003: Credential Management
Story 3.3: Create credential health check job (Phase 4)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '010_credential_health'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create credential_health_checks and tenant_webhooks tables.
    """

    # Create credential_health_checks table
    op.create_table(
        'credential_health_checks',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),

        # Tenant and credential identification
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('exchange', sa.String(50), nullable=False),
        sa.Column('credential_type', sa.String(50), nullable=False, server_default='trading'),

        # Health check results
        sa.Column('check_timestamp', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('is_valid', sa.Boolean, nullable=False),
        sa.Column('validation_error', sa.Text, nullable=True),

        # Failure tracking
        sa.Column('consecutive_failures', sa.Integer, nullable=False, server_default='0'),
        sa.Column('last_success_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('first_failure_at', sa.TIMESTAMP(timezone=True), nullable=True),

        # Webhook notification tracking
        sa.Column('webhook_sent', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('webhook_sent_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('webhook_delivery_status', sa.String(20), nullable=True),

        # Exchange metadata
        sa.Column('exchange_account_id', sa.String(255), nullable=True),
        sa.Column('exchange_permissions', postgresql.JSONB, nullable=True, server_default=sa.text("'[]'::jsonb")),

        # Task metadata
        sa.Column('task_id', sa.String(255), nullable=True),
        sa.Column('check_duration_ms', sa.Integer, nullable=True),

        # Audit fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),

        # Foreign key constraint
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),

        # Check constraints
        sa.CheckConstraint(
            "credential_type IN ('trading', 'readonly', 'withdrawal')",
            name='ck_cred_health_credential_type'
        ),
        sa.CheckConstraint(
            "webhook_delivery_status IS NULL OR webhook_delivery_status IN ('success', 'failed', 'timeout', 'error')",
            name='ck_cred_health_webhook_status'
        ),
        sa.CheckConstraint(
            "(webhook_sent = false AND webhook_sent_at IS NULL) OR (webhook_sent = true)",
            name='ck_cred_health_webhook_consistency'
        ),
        sa.CheckConstraint(
            "consecutive_failures >= 0",
            name='ck_cred_health_consecutive_failures_positive'
        ),
    )

    # Create indexes for credential_health_checks
    op.create_index(
        'idx_cred_health_tenant_exchange_type_timestamp',
        'credential_health_checks',
        ['tenant_id', 'exchange', 'credential_type', 'check_timestamp']
    )
    op.create_index('idx_cred_health_is_valid', 'credential_health_checks', ['is_valid'])
    op.create_index('idx_cred_health_webhook_sent', 'credential_health_checks', ['webhook_sent'])
    op.create_index('idx_cred_health_check_timestamp', 'credential_health_checks', ['check_timestamp'])
    op.create_index('idx_cred_health_consecutive_failures', 'credential_health_checks', ['consecutive_failures'])
    op.create_index('idx_cred_health_tenant_id', 'credential_health_checks', ['tenant_id'])

    print("✅ credential_health_checks table created successfully")

    # Create tenant_webhooks table
    op.create_table(
        'tenant_webhooks',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),

        # Tenant identification
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),

        # Webhook configuration
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('webhook_url', sa.String(2048), nullable=False),
        sa.Column('webhook_secret', sa.String(255), nullable=False),  # Will be encrypted at app layer

        # Status and validation
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('verified_at', sa.TIMESTAMP(timezone=True), nullable=True),

        # Delivery tracking
        sa.Column('last_delivery_attempt_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_successful_delivery_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('consecutive_failures', sa.Integer, nullable=False, server_default='0'),
        sa.Column('total_deliveries', sa.Integer, nullable=False, server_default='0'),
        sa.Column('successful_deliveries', sa.Integer, nullable=False, server_default='0'),

        # Configuration
        sa.Column('timeout_seconds', sa.Integer, nullable=False, server_default='10'),
        sa.Column('retry_attempts', sa.Integer, nullable=False, server_default='3'),

        # Metadata
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('custom_headers', postgresql.JSONB, nullable=True, server_default=sa.text("'{}'::jsonb")),

        # Audit fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('created_by', sa.String(255), nullable=True),

        # Foreign key constraint
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),

        # Unique constraint: one webhook per tenant per event type
        sa.UniqueConstraint('tenant_id', 'event_type', name='uq_tenant_webhooks_tenant_event'),

        # Check constraints
        sa.CheckConstraint(
            "event_type IN ('credential.failed', 'credential.expiring', 'system.alert', 'trade.executed', 'risk.breach')",
            name='ck_tenant_webhooks_event_type'
        ),
        sa.CheckConstraint(
            "webhook_url LIKE 'https://%'",
            name='ck_tenant_webhooks_https_only'
        ),
        sa.CheckConstraint(
            "timeout_seconds BETWEEN 1 AND 60",
            name='ck_tenant_webhooks_timeout_range'
        ),
        sa.CheckConstraint(
            "retry_attempts BETWEEN 0 AND 10",
            name='ck_tenant_webhooks_retry_range'
        ),
        sa.CheckConstraint(
            "consecutive_failures >= 0",
            name='ck_tenant_webhooks_consecutive_failures_positive'
        ),
    )

    # Create indexes for tenant_webhooks
    op.create_index('idx_tenant_webhooks_active', 'tenant_webhooks', ['is_active'])
    op.create_index('idx_tenant_webhooks_tenant_id', 'tenant_webhooks', ['tenant_id'])
    op.create_index('idx_tenant_webhooks_event_type', 'tenant_webhooks', ['event_type'])

    print("✅ tenant_webhooks table created successfully")

    # Add comments to tables
    op.execute("""
        COMMENT ON TABLE credential_health_checks IS 'Tracks credential validation results and consecutive failure streaks for webhook alerting';
        COMMENT ON TABLE tenant_webhooks IS 'Stores webhook configuration and delivery statistics for tenant notifications';

        COMMENT ON COLUMN credential_health_checks.tenant_id IS 'Tenant UUID from tenants table';
        COMMENT ON COLUMN credential_health_checks.exchange IS 'Exchange name (e.g., binance, coinbase)';
        COMMENT ON COLUMN credential_health_checks.credential_type IS 'Credential type (trading, readonly, withdrawal)';
        COMMENT ON COLUMN credential_health_checks.check_timestamp IS 'When this health check was performed';
        COMMENT ON COLUMN credential_health_checks.is_valid IS 'Whether credential passed validation';
        COMMENT ON COLUMN credential_health_checks.validation_error IS 'Error message if validation failed';
        COMMENT ON COLUMN credential_health_checks.consecutive_failures IS 'Number of consecutive failures (reset to 0 on success)';
        COMMENT ON COLUMN credential_health_checks.last_success_at IS 'Timestamp of last successful validation';
        COMMENT ON COLUMN credential_health_checks.first_failure_at IS 'Timestamp when current failure streak started';
        COMMENT ON COLUMN credential_health_checks.webhook_sent IS 'Whether webhook notification was sent for this check';
        COMMENT ON COLUMN credential_health_checks.webhook_sent_at IS 'When webhook notification was sent';
        COMMENT ON COLUMN credential_health_checks.webhook_delivery_status IS 'Webhook delivery status (success, failed, timeout)';
        COMMENT ON COLUMN credential_health_checks.exchange_account_id IS 'Exchange account ID from validation response';
        COMMENT ON COLUMN credential_health_checks.exchange_permissions IS 'Exchange permissions returned by validation';
        COMMENT ON COLUMN credential_health_checks.task_id IS 'Celery task ID that performed this check';
        COMMENT ON COLUMN credential_health_checks.check_duration_ms IS 'How long the check took in milliseconds';

        COMMENT ON COLUMN tenant_webhooks.tenant_id IS 'Tenant UUID from tenants table';
        COMMENT ON COLUMN tenant_webhooks.event_type IS 'Event type (credential.failed, system.alert, etc.)';
        COMMENT ON COLUMN tenant_webhooks.webhook_url IS 'HTTPS URL for webhook delivery';
        COMMENT ON COLUMN tenant_webhooks.webhook_secret IS 'Secret for HMAC-SHA256 signature generation';
        COMMENT ON COLUMN tenant_webhooks.is_active IS 'Whether webhook is enabled';
        COMMENT ON COLUMN tenant_webhooks.is_verified IS 'Whether webhook URL has been verified via test';
        COMMENT ON COLUMN tenant_webhooks.verified_at IS 'When webhook was last verified';
        COMMENT ON COLUMN tenant_webhooks.last_delivery_attempt_at IS 'Last time webhook delivery was attempted';
        COMMENT ON COLUMN tenant_webhooks.last_successful_delivery_at IS 'Last successful webhook delivery';
        COMMENT ON COLUMN tenant_webhooks.consecutive_failures IS 'Consecutive delivery failures (for circuit breaker)';
        COMMENT ON COLUMN tenant_webhooks.total_deliveries IS 'Total number of webhooks sent';
        COMMENT ON COLUMN tenant_webhooks.successful_deliveries IS 'Number of successful deliveries';
        COMMENT ON COLUMN tenant_webhooks.timeout_seconds IS 'HTTP timeout for webhook delivery';
        COMMENT ON COLUMN tenant_webhooks.retry_attempts IS 'Number of retry attempts on failure';
        COMMENT ON COLUMN tenant_webhooks.description IS 'Human-readable description of this webhook';
        COMMENT ON COLUMN tenant_webhooks.custom_headers IS 'Optional custom HTTP headers to include';
        COMMENT ON COLUMN tenant_webhooks.created_by IS 'User who created this webhook';
    """)

    print("✅ Column comments added successfully")
    print("✅ Story 3.3 Phase 4 database migration complete!")


def downgrade() -> None:
    """
    Drop credential_health_checks and tenant_webhooks tables.
    """
    # Drop indexes for tenant_webhooks
    op.drop_index('idx_tenant_webhooks_event_type', table_name='tenant_webhooks')
    op.drop_index('idx_tenant_webhooks_tenant_id', table_name='tenant_webhooks')
    op.drop_index('idx_tenant_webhooks_active', table_name='tenant_webhooks')

    # Drop tenant_webhooks table
    op.drop_table('tenant_webhooks')
    print("✅ tenant_webhooks table dropped successfully")

    # Drop indexes for credential_health_checks
    op.drop_index('idx_cred_health_tenant_id', table_name='credential_health_checks')
    op.drop_index('idx_cred_health_consecutive_failures', table_name='credential_health_checks')
    op.drop_index('idx_cred_health_check_timestamp', table_name='credential_health_checks')
    op.drop_index('idx_cred_health_webhook_sent', table_name='credential_health_checks')
    op.drop_index('idx_cred_health_is_valid', table_name='credential_health_checks')
    op.drop_index('idx_cred_health_tenant_exchange_type_timestamp', table_name='credential_health_checks')

    # Drop credential_health_checks table
    op.drop_table('credential_health_checks')
    print("✅ credential_health_checks table dropped successfully")
