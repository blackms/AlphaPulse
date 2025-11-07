"""Add cache quota and metrics tables for multi-tenant caching

Revision ID: 010_cache_quotas
Revises: 07d2e2f23eab
Create Date: 2025-11-07

EPIC-004: Caching Layer
User Story: 4.2 (3 SP) - Create Quota Management Schema

This migration adds:
1. tenant_cache_quotas - Per-tenant cache quota tracking
2. tenant_cache_metrics - Per-tenant cache performance metrics
3. RLS policies for tenant isolation
4. Indexes for performance
5. Computed columns for hit_rate calculation
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '010_cache_quotas'
down_revision = '07d2e2f23eab'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create cache quota and metrics tables with tenant isolation.

    Tables:
    - tenant_cache_quotas: Per-tenant quota configuration and usage tracking
    - tenant_cache_metrics: Per-tenant cache performance metrics (daily)

    Security:
    - RLS enabled on both tables
    - Policies enforce tenant isolation via app.current_tenant_id
    """

    # ========================================================================
    # Table: tenant_cache_quotas
    # ========================================================================
    op.create_table(
        'tenant_cache_quotas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('quota_mb', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('current_usage_mb', sa.Numeric(10, 2), server_default='0'),
        sa.Column('quota_reset_at', sa.TIMESTAMP(), server_default=sa.text('NOW()')),
        sa.Column('overage_allowed', sa.Boolean(), server_default='false'),
        sa.Column('overage_limit_mb', sa.Integer(), server_default='10'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', name='uq_tenant_cache_quotas_tenant_id'),
        sa.CheckConstraint('quota_mb > 0', name='ck_tenant_cache_quotas_quota_positive'),
        sa.CheckConstraint('current_usage_mb >= 0', name='ck_tenant_cache_quotas_usage_non_negative'),
        sa.CheckConstraint('overage_limit_mb >= 0', name='ck_tenant_cache_quotas_overage_non_negative')
    )

    # Note: No explicit tenant_id index needed - unique constraint creates one

    # Partial index for overage detection (WHERE current_usage_mb > quota_mb)
    op.execute("""
        CREATE INDEX idx_tenant_cache_quotas_overage
        ON tenant_cache_quotas(tenant_id)
        WHERE current_usage_mb > quota_mb
    """)

    # Foreign key to tenants table
    op.create_foreign_key(
        'fk_tenant_cache_quotas_tenant_id',
        'tenant_cache_quotas',
        'tenants',
        ['tenant_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # ========================================================================
    # Table: tenant_cache_metrics
    # ========================================================================
    op.create_table(
        'tenant_cache_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_date', sa.DATE(), nullable=False),
        sa.Column('total_requests', sa.BigInteger(), server_default='0'),
        sa.Column('cache_hits', sa.BigInteger(), server_default='0'),
        sa.Column('cache_misses', sa.BigInteger(), server_default='0'),
        sa.Column('avg_response_time_ms', sa.Numeric(10, 2)),
        sa.Column('total_bytes_served', sa.BigInteger(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'metric_date', name='uq_tenant_cache_metrics_tenant_date'),
        sa.CheckConstraint('total_requests >= 0', name='ck_tenant_cache_metrics_requests_non_negative'),
        sa.CheckConstraint('cache_hits >= 0', name='ck_tenant_cache_metrics_hits_non_negative'),
        sa.CheckConstraint('cache_misses >= 0', name='ck_tenant_cache_metrics_misses_non_negative'),
        sa.CheckConstraint('cache_hits <= total_requests', name='ck_tenant_cache_metrics_hits_lte_total'),
        sa.CheckConstraint('cache_misses <= total_requests', name='ck_tenant_cache_metrics_misses_lte_total'),
        sa.CheckConstraint('avg_response_time_ms >= 0', name='ck_tenant_cache_metrics_response_time_non_negative'),
        sa.CheckConstraint('total_bytes_served >= 0', name='ck_tenant_cache_metrics_bytes_non_negative')
    )

    # Computed column for hit_rate (GENERATED ALWAYS AS ... STORED)
    op.execute("""
        ALTER TABLE tenant_cache_metrics
        ADD COLUMN hit_rate NUMERIC(5,2)
        GENERATED ALWAYS AS (
            CASE
                WHEN total_requests > 0 THEN
                    (cache_hits::DECIMAL / total_requests * 100)
                ELSE 0
            END
        ) STORED
    """)

    # Indexes for tenant_cache_metrics
    op.create_index(
        'idx_tenant_cache_metrics_tenant_date',
        'tenant_cache_metrics',
        ['tenant_id', sa.text('metric_date DESC')]
    )

    # Foreign key to tenants table
    op.create_foreign_key(
        'fk_tenant_cache_metrics_tenant_id',
        'tenant_cache_metrics',
        'tenants',
        ['tenant_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # ========================================================================
    # Row-Level Security (RLS)
    # ========================================================================

    # Enable RLS on tenant_cache_quotas
    op.execute('ALTER TABLE tenant_cache_quotas ENABLE ROW LEVEL SECURITY')

    # RLS policy for tenant_cache_quotas
    op.execute("""
        CREATE POLICY tenant_cache_quotas_isolation_policy
        ON tenant_cache_quotas
        USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    """)

    # Enable RLS on tenant_cache_metrics
    op.execute('ALTER TABLE tenant_cache_metrics ENABLE ROW LEVEL SECURITY')

    # RLS policy for tenant_cache_metrics
    op.execute("""
        CREATE POLICY tenant_cache_metrics_isolation_policy
        ON tenant_cache_metrics
        USING (tenant_id = current_setting('app.current_tenant_id')::UUID)
    """)

    # ========================================================================
    # Trigger for updated_at timestamp (tenant_cache_quotas only)
    # ========================================================================
    op.execute("""
        CREATE OR REPLACE FUNCTION update_tenant_cache_quotas_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trigger_update_tenant_cache_quotas_updated_at
        BEFORE UPDATE ON tenant_cache_quotas
        FOR EACH ROW
        EXECUTE FUNCTION update_tenant_cache_quotas_updated_at();
    """)


def downgrade() -> None:
    """
    Drop cache quota and metrics tables.

    Removes:
    - tenant_cache_metrics table (with RLS policy, indexes, FK)
    - tenant_cache_quotas table (with RLS policy, indexes, FK, trigger)
    - Trigger function for updated_at
    """

    # Drop tenant_cache_metrics
    op.execute('DROP POLICY IF EXISTS tenant_cache_metrics_isolation_policy ON tenant_cache_metrics')
    op.drop_constraint('fk_tenant_cache_metrics_tenant_id', 'tenant_cache_metrics', type_='foreignkey')
    op.drop_index('idx_tenant_cache_metrics_tenant_date', table_name='tenant_cache_metrics')
    op.drop_table('tenant_cache_metrics')

    # Drop tenant_cache_quotas
    op.execute('DROP TRIGGER IF EXISTS trigger_update_tenant_cache_quotas_updated_at ON tenant_cache_quotas')
    op.execute('DROP FUNCTION IF EXISTS update_tenant_cache_quotas_updated_at()')
    op.execute('DROP POLICY IF EXISTS tenant_cache_quotas_isolation_policy ON tenant_cache_quotas')
    op.drop_constraint('fk_tenant_cache_quotas_tenant_id', 'tenant_cache_quotas', type_='foreignkey')
    op.drop_index('idx_tenant_cache_quotas_overage', table_name='tenant_cache_quotas')
    op.drop_table('tenant_cache_quotas')
