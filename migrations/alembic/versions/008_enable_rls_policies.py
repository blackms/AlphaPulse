"""Enable Row-Level Security (RLS) policies

Revision ID: 008_enable_rls
Revises: 007_add_tenant_id_domain
Create Date: 2025-11-11

EPIC-001: Database Multi-Tenancy
User Story: US-004 (5 SP)

IMPORTANT: This migration enables RLS policies for tenant isolation.
Ensure feature flag RLS_ENABLED=false before running this migration.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '008_enable_rls'
down_revision = '007_add_tenant_id_domain'
branch_labels = None
depends_on = None

# Tables to enable RLS (only tables that exist and have tenant_id)
RLS_TABLES = [
    'users',
    'trading_accounts',
    'trades',
    'positions',
    'portfolio_snapshots',
    'risk_metrics',
    'agent_signals',
    'audit_logs',
]


def upgrade() -> None:
    """
    Enable RLS and create policies for all multi-tenant tables.
    """
    print("\n🔒 Enabling Row-Level Security (RLS)")
    print("=" * 60)

    # Step 1: Enable RLS on all tables
    for table_name in RLS_TABLES:
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        print(f"✅ RLS enabled on {table_name}")

    print()

    # Step 2: Create policies for each table
    for table_name in RLS_TABLES:
        print(f"\n📋 Creating policies for {table_name}:")

        # Policy for SELECT
        op.execute(f"""
            CREATE POLICY {table_name}_tenant_isolation_select ON {table_name}
            FOR SELECT
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID)
        """)
        print(f"  ✅ SELECT policy created")

        # Policy for INSERT
        op.execute(f"""
            CREATE POLICY {table_name}_tenant_isolation_insert ON {table_name}
            FOR INSERT
            WITH CHECK (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID)
        """)
        print(f"  ✅ INSERT policy created")

        # Policy for UPDATE
        op.execute(f"""
            CREATE POLICY {table_name}_tenant_isolation_update ON {table_name}
            FOR UPDATE
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID)
            WITH CHECK (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID)
        """)
        print(f"  ✅ UPDATE policy created")

        # Policy for DELETE
        op.execute(f"""
            CREATE POLICY {table_name}_tenant_isolation_delete ON {table_name}
            FOR DELETE
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID)
        """)
        print(f"  ✅ DELETE policy created")

    print("\n" + "=" * 60)

    # Step 3: Create bypass role for admin operations (migrations, backups)
    print("\n🔓 Creating RLS bypass role")
    op.execute("CREATE ROLE rls_bypass_role")
    print("✅ rls_bypass_role created")

    # Grant bypass role to application user (use sparingly!)
    # Uncomment this line only when admin operations are needed
    # op.execute("GRANT rls_bypass_role TO alphapulse")

    # Step 4: Force RLS even for table owner (security hardening)
    for table_name in RLS_TABLES:
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
    print("✅ Forced RLS for table owners")

    print("\n" + "=" * 60)
    print("🎉 RLS policies enabled successfully!")
    print("\n⚠️  IMPORTANT NEXT STEPS:")
    print("1. Enable feature flag: RLS_ENABLED=true in application config")
    print("2. Test with multiple tenants:")
    print("   SET app.current_tenant_id = '00000000-0000-0000-0000-000000000001';")
    print("   SELECT COUNT(*) FROM trades;")
    print("3. Verify no cross-tenant data leakage")
    print("4. Monitor query performance (<25% degradation)")
    print("=" * 60)


def downgrade() -> None:
    """
    Disable RLS and drop all policies.
    """
    print("\n🔓 Disabling Row-Level Security (RLS)")
    print("=" * 60)

    # Step 1: Drop all policies
    for table_name in RLS_TABLES:
        print(f"\n📋 Dropping policies for {table_name}:")

        # Drop policies
        op.execute(f"DROP POLICY IF EXISTS {table_name}_tenant_isolation_select ON {table_name}")
        op.execute(f"DROP POLICY IF EXISTS {table_name}_tenant_isolation_insert ON {table_name}")
        op.execute(f"DROP POLICY IF EXISTS {table_name}_tenant_isolation_update ON {table_name}")
        op.execute(f"DROP POLICY IF EXISTS {table_name}_tenant_isolation_delete ON {table_name}")
        print(f"  ✅ All policies dropped")

    print()

    # Step 2: Disable RLS on all tables
    for table_name in RLS_TABLES:
        op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")
        print(f"✅ RLS disabled on {table_name}")

    # Step 3: Drop bypass role
    op.execute("DROP ROLE IF EXISTS rls_bypass_role")
    print("✅ rls_bypass_role dropped")

    print("\n" + "=" * 60)
    print("✅ RLS policies disabled successfully!")
    print("=" * 60)
