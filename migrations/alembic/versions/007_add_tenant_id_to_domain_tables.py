"""Add tenant_id to domain tables

Revision ID: 007_add_tenant_id_domain
Revises: 006_add_tenant_id_users
Create Date: 2025-11-11

EPIC-001: Database Multi-Tenancy
User Story: US-003 (5 SP)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '007_add_tenant_id_domain'
down_revision = '006_add_tenant_id_users'
branch_labels = None
depends_on = None

# Tables to update - grouped by their relationship to tenants
ACCOUNT_BASED_TABLES = [
    'trading_accounts',
    'trades',
    'positions',
    'portfolio_snapshots',
    'risk_metrics',
]

STANDALONE_TABLES = [
    'agent_signals',
    'audit_logs',
]


def upgrade() -> None:
    """
    Add tenant_id to all domain tables with appropriate backfill strategy.
    """
    # Handle account-based tables (backfill all with default tenant for now)
    for table_name in ACCOUNT_BASED_TABLES:
        print(f"\n📊 Processing table: {table_name}")

        # Step 1: Add tenant_id column (nullable for backfill)
        op.add_column(table_name, sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
        print(f"  ✅ Added tenant_id column")

        # Step 2: Backfill all rows with default tenant
        # In a real scenario, these would be linked via a user/account ownership table
        op.execute(f"""
            UPDATE {table_name}
            SET tenant_id = '00000000-0000-0000-0000-000000000001'
            WHERE tenant_id IS NULL
        """)

        print(f"  ✅ Backfilled all rows with default tenant")

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

        # Create compound index with created_at/executed_at for time-series queries
        if table_name == 'trades':
            op.create_index(
                f'idx_{table_name}_tenant_created',
                table_name,
                ['tenant_id', 'executed_at'],
                postgresql_ops={'executed_at': 'DESC'}
            )
            print(f"  ✅ Created indexes (tenant_id, tenant_id + executed_at DESC)")
        else:
            print(f"  ✅ Created index (tenant_id)")

    # Handle standalone tables (backfill from users table where possible)
    for table_name in STANDALONE_TABLES:
        print(f"\n📊 Processing table: {table_name}")

        # Step 1: Add tenant_id column
        op.add_column(table_name, sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True))
        print(f"  ✅ Added tenant_id column")

        # Step 2: Backfill based on table structure
        if table_name == 'audit_logs':
            # Backfill from users table where user_id exists
            op.execute("""
                UPDATE audit_logs a
                SET tenant_id = u.tenant_id
                FROM users u
                WHERE a.user_id = u.id
                AND a.tenant_id IS NULL
            """)
            # Set default tenant for rows without user_id
            op.execute("""
                UPDATE audit_logs
                SET tenant_id = '00000000-0000-0000-0000-000000000001'
                WHERE tenant_id IS NULL
            """)
        else:
            # For agent_signals and others, use default tenant
            op.execute(f"""
                UPDATE {table_name}
                SET tenant_id = '00000000-0000-0000-0000-000000000001'
                WHERE tenant_id IS NULL
            """)

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

        # Step 5: Create indexes
        op.create_index(f'idx_{table_name}_tenant_id', table_name, ['tenant_id'])

        # Compound index for agent_signals
        if table_name == 'agent_signals':
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
    all_tables = ACCOUNT_BASED_TABLES + STANDALONE_TABLES

    for table_name in all_tables:
        print(f"\n📊 Reverting table: {table_name}")

        # Drop indexes
        if table_name in ['trades', 'agent_signals']:
            op.drop_index(f'idx_{table_name}_tenant_created', table_name=table_name)
        op.drop_index(f'idx_{table_name}_tenant_id', table_name=table_name)

        # Drop foreign key
        op.drop_constraint(f'fk_{table_name}_tenant', table_name, type_='foreignkey')

        # Drop column
        op.drop_column(table_name, 'tenant_id')

        print(f"  ✅ Removed tenant_id")

    print("\n✅ All domain tables reverted successfully")
