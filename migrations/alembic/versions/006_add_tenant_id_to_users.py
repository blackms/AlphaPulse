"""Add tenant_id to users table

Revision ID: 006_add_tenant_id_users
Revises: 005_add_tenants
Create Date: 2025-11-11

EPIC-001: Database Multi-Tenancy
User Story: US-002 (3 SP)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '006_add_tenant_id_users'
down_revision = '005_add_tenants'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add tenant_id column to users table, backfill data, and create constraints.
    """
    # Step 1: Add tenant_id column (nullable initially for backfill)
    op.add_column('users', sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
    print("✅ Added tenant_id column to users table (nullable)")

    # Step 2: Backfill existing users with default tenant
    op.execute("""
        UPDATE users
        SET tenant_id = '00000000-0000-0000-0000-000000000001'
        WHERE tenant_id IS NULL
    """)

    # Verify backfill
    result = op.get_bind().execute(sa.text("SELECT COUNT(*) FROM users WHERE tenant_id IS NULL"))
    null_count = result.scalar()
    if null_count > 0:
        raise Exception(f"❌ Backfill failed: {null_count} users still have NULL tenant_id")

    print(f"✅ Backfilled all users with default tenant")

    # Step 3: Make column NOT NULL
    op.alter_column('users', 'tenant_id', nullable=False)
    print("✅ Made tenant_id NOT NULL")

    # Step 4: Add foreign key constraint
    op.create_foreign_key(
        'fk_users_tenant',
        'users',
        'tenants',
        ['tenant_id'],
        ['id'],
        ondelete='CASCADE'
    )
    print("✅ Added foreign key constraint")

    # Step 5: Create indexes for performance
    op.create_index('idx_users_tenant_id', 'users', ['tenant_id'])
    op.create_index('idx_users_tenant_email', 'users', ['tenant_id', 'email'])  # Unique per tenant
    print("✅ Created indexes")

    # Step 6: Update unique constraint (email unique per tenant, not globally)
    # Drop existing unique constraint on email (if exists)
    try:
        op.drop_constraint('uq_users_email', 'users', type_='unique')
    except:
        pass  # Constraint may not exist

    # Create new unique constraint (email + tenant_id)
    op.create_unique_constraint('uq_users_tenant_email', 'users', ['tenant_id', 'email'])
    print("✅ Updated unique constraint (email per tenant)")


def downgrade() -> None:
    """
    Remove tenant_id from users table.
    """
    # Drop unique constraint
    op.drop_constraint('uq_users_tenant_email', 'users', type_='unique')

    # Recreate original email unique constraint
    op.create_unique_constraint('uq_users_email', 'users', ['email'])

    # Drop indexes
    op.drop_index('idx_users_tenant_email', table_name='users')
    op.drop_index('idx_users_tenant_id', table_name='users')

    # Drop foreign key
    op.drop_constraint('fk_users_tenant', 'users', type_='foreignkey')

    # Drop column
    op.drop_column('users', 'tenant_id')

    print("✅ Removed tenant_id from users table")
