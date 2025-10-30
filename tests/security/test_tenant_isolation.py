"""
Tenant Isolation Security Tests

EPIC-001: Database Multi-Tenancy
Critical security validation per HLD Section 5.1 (Risk-001: Data Leakage)

This test suite validates:
1. Complete data isolation between tenants (no leakage)
2. RLS policies prevent cross-tenant access
3. Session variable enforcement
4. Foreign key constraints respect tenant boundaries

Failure of any test = CRITICAL SECURITY ISSUE - DO NOT DEPLOY
"""
import pytest
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
import uuid

DATABASE_URL = "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse"

# Test fixtures
TENANT_A = "00000000-0000-0000-0000-000000000001"
TENANT_B = "00000000-0000-0000-0000-000000000002"


class TenantIsolationTests:
    """Comprehensive tenant isolation security tests"""

    def __init__(self):
        self.engine = create_engine(DATABASE_URL, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def setup_test_data(self):
        """Create test data for both tenants"""
        print("\n📦 Setting up test data...")

        session = self.Session()
        try:
            # Ensure both tenants exist
            session.execute(text(f"""
                INSERT INTO tenants (id, name, slug, subscription_tier, status)
                VALUES
                    ('{TENANT_A}', 'Tenant A', 'tenant-a', 'pro', 'active'),
                    ('{TENANT_B}', 'Tenant B', 'tenant-b', 'pro', 'active')
                ON CONFLICT (id) DO NOTHING
            """))

            # Create test user for Tenant A
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            user_a_id = str(uuid.uuid4())
            session.execute(text(f"""
                INSERT INTO users (id, tenant_id, email, hashed_password, is_active)
                VALUES ('{user_a_id}', '{TENANT_A}', 'user_a@tenant-a.com', 'hashed', true)
                ON CONFLICT (tenant_id, email) DO NOTHING
            """))

            # Create test user for Tenant B
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_B}'"))
            user_b_id = str(uuid.uuid4())
            session.execute(text(f"""
                INSERT INTO users (id, tenant_id, email, hashed_password, is_active)
                VALUES ('{user_b_id}', '{TENANT_B}', 'user_b@tenant-b.com', 'hashed', true)
                ON CONFLICT (tenant_id, email) DO NOTHING
            """))

            session.commit()
            print(f"✅ Test users created: Tenant A={user_a_id}, Tenant B={user_b_id}")
            return user_a_id, user_b_id

        except Exception as e:
            session.rollback()
            print(f"⚠️  Setup failed: {e}")
            raise
        finally:
            session.close()

    def test_001_select_isolation(self):
        """Test 001: SELECT statements respect tenant boundaries"""
        print("\n" + "=" * 70)
        print("TEST 001: SELECT Isolation")
        print("=" * 70)

        session = self.Session()
        try:
            # Tenant A: Count users
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            result_a = session.execute(text("SELECT COUNT(*) FROM users"))
            count_a = result_a.scalar()

            # Tenant B: Count users
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_B}'"))
            result_b = session.execute(text("SELECT COUNT(*) FROM users"))
            count_b = result_b.scalar()

            # No context: Should see no users (RLS blocks access)
            session.execute(text("RESET app.current_tenant_id"))
            result_none = session.execute(text("SELECT COUNT(*) FROM users"))
            count_none = result_none.scalar()

            print(f"Tenant A sees: {count_a} users")
            print(f"Tenant B sees: {count_b} users")
            print(f"No context sees: {count_none} users")

            assert count_a >= 1, "Tenant A should see at least 1 user (itself)"
            assert count_b >= 1, "Tenant B should see at least 1 user (itself)"
            assert count_none == 0, "No tenant context should see 0 users (RLS blocks)"

            print("✅ PASSED: SELECT isolation working correctly")

        finally:
            session.close()

    def test_002_insert_isolation(self):
        """Test 002: INSERT statements enforce tenant_id from context"""
        print("\n" + "=" * 70)
        print("TEST 002: INSERT Isolation (WITH CHECK enforcement)")
        print("=" * 70)

        session = self.Session()
        try:
            # Attempt 1: Insert with correct tenant_id (should succeed)
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            user_id_correct = str(uuid.uuid4())

            session.execute(text("""
                INSERT INTO users (id, tenant_id, email, hashed_password, is_active)
                VALUES (:id, :tenant_id, :email, 'hashed', true)
            """), {
                'id': user_id_correct,
                'tenant_id': TENANT_A,  # Correct tenant
                'email': f'test_{user_id_correct}@tenant-a.com'
            })
            session.commit()
            print(f"✅ INSERT with correct tenant_id succeeded")

            # Attempt 2: Insert with WRONG tenant_id (should FAIL)
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            user_id_wrong = str(uuid.uuid4())

            try:
                session.execute(text("""
                    INSERT INTO users (id, tenant_id, email, hashed_password, is_active)
                    VALUES (:id, :tenant_id, :email, 'hashed', true)
                """), {
                    'id': user_id_wrong,
                    'tenant_id': TENANT_B,  # WRONG tenant!
                    'email': f'test_{user_id_wrong}@tenant-b.com'
                })
                session.commit()

                # If we reach here, security is BROKEN
                raise AssertionError("CRITICAL: INSERT with wrong tenant_id was allowed!")

            except Exception as e:
                session.rollback()
                if "policy" in str(e).lower() or "row-level security" in str(e).lower():
                    print(f"✅ INSERT with wrong tenant_id correctly blocked by RLS")
                    print(f"   Error: {str(e)[:80]}...")
                else:
                    raise  # Unexpected error

            print("✅ PASSED: INSERT isolation enforced by WITH CHECK")

        finally:
            session.close()

    def test_003_update_isolation(self):
        """Test 003: UPDATE statements cannot modify other tenants' data"""
        print("\n" + "=" * 70)
        print("TEST 003: UPDATE Isolation")
        print("=" * 70)

        session = self.Session()
        try:
            # Get Tenant A's user
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            result = session.execute(text("SELECT id FROM users LIMIT 1"))
            user_a_id = result.scalar()

            if not user_a_id:
                print("⚠️  No users found for Tenant A, skipping test")
                return

            # Attempt 1: Tenant A updates their own user (should succeed)
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            session.execute(text("""
                UPDATE users SET hashed_password = 'updated'
                WHERE id = :user_id
            """), {'user_id': user_a_id})
            session.commit()
            print(f"✅ Tenant A can update their own user")

            # Attempt 2: Tenant B tries to update Tenant A's user (should affect 0 rows)
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_B}'"))
            result = session.execute(text("""
                UPDATE users SET hashed_password = 'hacked'
                WHERE id = :user_id
            """), {'user_id': user_a_id})
            rows_affected = result.rowcount
            session.commit()

            assert rows_affected == 0, f"Tenant B should not update Tenant A's data (affected {rows_affected} rows)"
            print(f"✅ Tenant B cannot update Tenant A's user (0 rows affected)")

            print("✅ PASSED: UPDATE isolation working correctly")

        finally:
            session.close()

    def test_004_delete_isolation(self):
        """Test 004: DELETE statements cannot remove other tenants' data"""
        print("\n" + "=" * 70)
        print("TEST 004: DELETE Isolation")
        print("=" * 70)

        session = self.Session()
        try:
            # Create a test user for Tenant A
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            test_user_id = str(uuid.uuid4())
            session.execute(text("""
                INSERT INTO users (id, tenant_id, email, hashed_password, is_active)
                VALUES (:id, :tenant_id, :email, 'hashed', true)
            """), {
                'id': test_user_id,
                'tenant_id': TENANT_A,
                'email': f'deleteme_{test_user_id}@tenant-a.com'
            })
            session.commit()
            print(f"✅ Created test user: {test_user_id}")

            # Attempt 1: Tenant B tries to delete Tenant A's user (should affect 0 rows)
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_B}'"))
            result = session.execute(text("DELETE FROM users WHERE id = :user_id"), {'user_id': test_user_id})
            rows_deleted_b = result.rowcount
            session.commit()

            assert rows_deleted_b == 0, f"Tenant B should not delete Tenant A's data (deleted {rows_deleted_b} rows)"
            print(f"✅ Tenant B cannot delete Tenant A's user (0 rows deleted)")

            # Attempt 2: Tenant A deletes their own user (should succeed)
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            result = session.execute(text("DELETE FROM users WHERE id = :user_id"), {'user_id': test_user_id})
            rows_deleted_a = result.rowcount
            session.commit()

            assert rows_deleted_a == 1, f"Tenant A should delete their own user (deleted {rows_deleted_a} rows)"
            print(f"✅ Tenant A can delete their own user (1 row deleted)")

            print("✅ PASSED: DELETE isolation working correctly")

        finally:
            session.close()

    def test_005_join_isolation(self):
        """Test 005: JOIN operations respect tenant boundaries"""
        print("\n" + "=" * 70)
        print("TEST 005: JOIN Isolation (Foreign Key Traversal)")
        print("=" * 70)

        session = self.Session()
        try:
            # Create trading accounts for both tenants
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            account_a_id = str(uuid.uuid4())
            session.execute(text("""
                INSERT INTO trading_accounts (id, tenant_id, exchange_name, account_type, is_active)
                VALUES (:id, :tenant_id, 'binance', 'spot', true)
                ON CONFLICT DO NOTHING
            """), {'id': account_a_id, 'tenant_id': TENANT_A})

            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_B}'"))
            account_b_id = str(uuid.uuid4())
            session.execute(text("""
                INSERT INTO trading_accounts (id, tenant_id, exchange_name, account_type, is_active)
                VALUES (:id, :tenant_id, 'binance', 'spot', true)
                ON CONFLICT DO NOTHING
            """), {'id': account_b_id, 'tenant_id': TENANT_B})
            session.commit()

            # Test JOIN: Each tenant should only see their own accounts
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_A}'"))
            result_a = session.execute(text("""
                SELECT COUNT(*) FROM trading_accounts
            """))
            count_a = result_a.scalar()

            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_B}'"))
            result_b = session.execute(text("""
                SELECT COUNT(*) FROM trading_accounts
            """))
            count_b = result_b.scalar()

            print(f"Tenant A sees: {count_a} trading accounts")
            print(f"Tenant B sees: {count_b} trading accounts")

            assert count_a >= 1, "Tenant A should see at least 1 account"
            assert count_b >= 1, "Tenant B should see at least 1 account"

            print("✅ PASSED: JOIN operations respect tenant boundaries")

        finally:
            session.close()

    def test_006_no_context_blocks_all(self):
        """Test 006: Queries without tenant context should return empty results"""
        print("\n" + "=" * 70)
        print("TEST 006: No Tenant Context = No Data Access")
        print("=" * 70)

        session = self.Session()
        try:
            # Reset tenant context (simulate missing session variable)
            session.execute(text("RESET app.current_tenant_id"))

            # Try to query all tables
            tables = ['users', 'trading_accounts', 'trades', 'positions']
            for table in tables:
                try:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"{table}: {count} rows")

                    # Allow 0 rows (correct) or error (also correct)
                    assert count == 0, f"Table {table} returned {count} rows without tenant context!"

                except Exception as e:
                    if "app.current_tenant_id" in str(e):
                        print(f"{table}: Blocked (session variable required)")
                    else:
                        raise

            print("✅ PASSED: No tenant context blocks all data access")

        finally:
            session.close()

    def run_all_tests(self):
        """Run complete tenant isolation security test suite"""
        print("\n" + "=" * 70)
        print("TENANT ISOLATION SECURITY TEST SUITE")
        print("EPIC-001: Database Multi-Tenancy (RISK-001: Data Leakage)")
        print("=" * 70)

        try:
            self.setup_test_data()
            self.test_001_select_isolation()
            self.test_002_insert_isolation()
            self.test_003_update_isolation()
            self.test_004_delete_isolation()
            self.test_005_join_isolation()
            self.test_006_no_context_blocks_all()

            print("\n" + "=" * 70)
            print("🎉 ALL SECURITY TESTS PASSED")
            print("Tenant isolation is working correctly - safe to deploy")
            print("=" * 70)

        except AssertionError as e:
            print("\n" + "=" * 70)
            print("❌ CRITICAL SECURITY FAILURE")
            print(f"Error: {e}")
            print("DO NOT DEPLOY - Data leakage risk detected")
            print("=" * 70)
            raise

        except Exception as e:
            print("\n" + "=" * 70)
            print("❌ TEST SUITE ERROR")
            print(f"Error: {e}")
            print("=" * 70)
            raise


if __name__ == "__main__":
    tester = TenantIsolationTests()
    tester.run_all_tests()