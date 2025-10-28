"""Add tenant_id to domain tables

Revision ID: 003_add_tenant_id_domain
Revises: 002_add_tenant_id_users
Create Date: 2025-11-11

EPIC-001: Database Multi-Tenancy
User Story: US-003 (5 SP)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003_add_tenant_id_domain'
down_revision = '002_add_tenant_id_users'
branch_labels = None
depends_on = None

# Tables to update
DOMAIN_TABLES = [
    'trades',
    'positions',
    'orders',
    'portfolio_snapshots',
    'risk_metrics',
    'agent_signals',
]


def upgrade() -> None:
    """
    Add tenant_id to all domain tables with backfill from users.tenant_id.
    """
    for table_name in DOMAIN_TABLES:
        print(f"\n📊 Processing table: {table_name}")

        # Step 1: Add tenant_id column (nullable for backfill)
        op.add_column(table_name, sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
        print(f"  ✅ Added tenant_id column")

        # Step 2: Backfill tenant_id from users table
        # Assumption: All tables have user_id column
        op.execute(f"""
            UPDATE {table_name} t
            SET tenant_id = u.tenant_id
            FROM users u
            WHERE t.user_id = u.id
            AND t.tenant_id IS NULL
        """)

        # Verify backfill
        result = op.get_bind().execute(sa.text(f"SELECT COUNT(*) FROM {table_name} WHERE tenant_id IS NULL"))
        null_count = result.scalar()
        if null_count > 0:
            print(f"  ⚠️  Warning: {null_count} rows in {table_name} have NULL tenant_id (orphaned data?)")
            # Delete orphaned rows (no associated user)
            op.execute(f"DELETE FROM {table_name} WHERE tenant_id IS NULL")
            print(f"  ✅ Deleted {null_count} orphaned rows")
        else:
            print(f"  ✅ Backfilled all rows")

        # Step 3: Make column NOT NULL
        op.alter_column(table_name, 'tenant_id', nullable=False)
        print(f"  ✅ Made tenant_id NOT NULL")

        # Step 4: Add foreign key constraint
        op.create_foreign_key(
            f'fk_{table_name}_tenant',
            table_name,
            'tenants',
            ['tenant_id'],
            ['id'],
            ondelete='CASCADE'
        )
        print(f"  ✅ Added foreign key constraint")

        # Step 5: Create indexes for performance
        op.create_index(f'idx_{table_name}_tenant_id', table_name, ['tenant_id'])

        # Create compound index with created_at for time-series queries
        if table_name in ['trades', 'orders', 'agent_signals']:
            op.create_index(
                f'idx_{table_name}_tenant_created',
                table_name,
                ['tenant_id', 'created_at'],
                postgresql_ops={'created_at': 'DESC'}
            )
            print(f"  ✅ Created indexes (tenant_id, tenant_id + created_at DESC)")
        else:
            print(f"  ✅ Created index (tenant_id)")

    print("\n✅ All domain tables updated successfully")

    # Performance verification query
    print("\n📈 Performance check: Run this query to verify index usage:")
    print("EXPLAIN ANALYZE SELECT * FROM trades WHERE tenant_id = '00000000-0000-0000-0000-000000000001' ORDER BY created_at DESC LIMIT 100;")


def downgrade() -> None:
    """
    Remove tenant_id from all domain tables.
    """
    for table_name in DOMAIN_TABLES:
        print(f"\n📊 Reverting table: {table_name}")

        # Drop indexes
        if table_name in ['trades', 'orders', 'agent_signals']:
            op.drop_index(f'idx_{table_name}_tenant_created', table_name=table_name)
        op.drop_index(f'idx_{table_name}_tenant_id', table_name=table_name)

        # Drop foreign key
        op.drop_constraint(f'fk_{table_name}_tenant', table_name, type_='foreignkey')

        # Drop column
        op.drop_column(table_name, 'tenant_id')

        print(f"  ✅ Removed tenant_id")

    print("\n✅ All domain tables reverted successfully")
