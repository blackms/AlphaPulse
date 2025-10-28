"""Add tenants table for multi-tenancy

Revision ID: 005_add_tenants
Revises: 6762f578afb6
Create Date: 2025-11-11

EPIC-001: Database Multi-Tenancy
User Story: US-001 (3 SP)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '005_add_tenants'
down_revision = '6762f578afb6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create tenants table with all required fields and indexes.
    """
    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('subscription_tier', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='active'),
        sa.Column('max_users', sa.Integer, server_default='5'),
        sa.Column('max_api_calls_per_day', sa.Integer, server_default='10000'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),
        sa.CheckConstraint("subscription_tier IN ('starter', 'pro', 'enterprise')", name='ck_tenants_subscription_tier'),
        sa.CheckConstraint("status IN ('active', 'suspended', 'cancelled')", name='ck_tenants_status'),
    )

    # Create indexes
    op.create_index('idx_tenants_slug', 'tenants', ['slug'])
    op.create_index('idx_tenants_status', 'tenants', ['status'])
    op.create_index('idx_tenants_tier', 'tenants', ['subscription_tier'])

    # Insert default tenant (for existing data)
    op.execute("""
        INSERT INTO tenants (id, name, slug, subscription_tier, status)
        VALUES (
            '00000000-0000-0000-0000-000000000001',
            'Default Tenant',
            'default',
            'pro',
            'active'
        )
        ON CONFLICT (id) DO NOTHING
    """)

    print("✅ Tenants table created successfully")
    print("✅ Default tenant inserted: 00000000-0000-0000-0000-000000000001")


def downgrade() -> None:
    """
    Drop tenants table and all indexes.
    """
    # Drop indexes
    op.drop_index('idx_tenants_tier', table_name='tenants')
    op.drop_index('idx_tenants_status', table_name='tenants')
    op.drop_index('idx_tenants_slug', table_name='tenants')

    # Drop table
    op.drop_table('tenants')

    print("✅ Tenants table dropped successfully")
