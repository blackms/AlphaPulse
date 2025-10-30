"""
Performance Benchmarking for RLS Overhead

EPIC-001: Database Multi-Tenancy
Test validates ADR-001 requirement: RLS overhead must be < 10%

This script tests:
1. Baseline query performance (no RLS context)
2. RLS-enabled query performance (with tenant context)
3. Index usage verification
4. Cross-tenant isolation verification

Success Criteria:
- RLS overhead < 10% for all query types
- Composite indexes are being used
- No cross-tenant data leakage possible
"""
import pytest
import time
from typing import List, Dict
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
import uuid

# Database connection (adjust as needed)
DATABASE_URL = "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse"

# Test tenant IDs
TENANT_1 = "00000000-0000-0000-0000-000000000001"
TENANT_2 = "00000000-0000-0000-0000-000000000002"


class RLSPerformanceTester:
    """Test RLS performance overhead and security"""

    def __init__(self):
        self.engine = create_engine(DATABASE_URL, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def time_query(self, query: str, params: Dict = None, iterations: int = 100) -> float:
        """Execute query multiple times and return average time in milliseconds"""
        session = self.Session()
        times = []

        try:
            for _ in range(iterations):
                start = time.perf_counter()
                if params:
                    session.execute(text(query), params)
                else:
                    session.execute(text(query))
                session.commit()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            return sum(times) / len(times)
        finally:
            session.close()

    def test_primary_key_lookup(self, trade_id: uuid.UUID):
        """Test 1: Primary key lookup performance"""
        print("\n" + "=" * 70)
        print("TEST 1: Primary Key Lookup Performance")
        print("=" * 70)

        # Baseline: Query without RLS context (admin role)
        baseline_query = f"SELECT * FROM trades WHERE id = '{trade_id}'"
        baseline_time = self.time_query(baseline_query)
        print(f"Baseline (no RLS):     {baseline_time:.2f}ms")

        # RLS: Query with tenant context
        rls_query = f"""
            SET LOCAL app.current_tenant_id = '{TENANT_1}';
            SELECT * FROM trades WHERE id = '{trade_id}';
        """
        rls_time = self.time_query(rls_query)
        print(f"RLS-enabled:           {rls_time:.2f}ms")

        overhead = ((rls_time - baseline_time) / baseline_time) * 100
        print(f"Overhead:              {overhead:.2f}%")

        # Verify index usage
        explain_query = f"""
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT * FROM trades
            WHERE tenant_id = '{TENANT_1}' AND id = '{trade_id}'
        """
        session = self.Session()
        result = session.execute(text(explain_query))
        explain_output = "\n".join([str(row[0]) for row in result])
        session.close()

        print("\nQuery Plan:")
        print(explain_output)

        # Assert performance requirement
        assert overhead < 10, f"RLS overhead {overhead:.2f}% exceeds 10% threshold"
        assert "idx_trades_tenant_id_compound" in explain_output, "Composite index not being used"

        print(f"\n✅ PASSED: Overhead {overhead:.2f}% < 10%")
        return overhead

    def test_time_series_query(self):
        """Test 2: Time-series query performance (recent trades)"""
        print("\n" + "=" * 70)
        print("TEST 2: Time-Series Query Performance (Last 100 Trades)")
        print("=" * 70)

        # Baseline
        baseline_query = "SELECT * FROM trades ORDER BY executed_at DESC LIMIT 100"
        baseline_time = self.time_query(baseline_query, iterations=50)
        print(f"Baseline (no RLS):     {baseline_time:.2f}ms")

        # RLS
        rls_query = f"""
            SET LOCAL app.current_tenant_id = '{TENANT_1}';
            SELECT * FROM trades ORDER BY executed_at DESC LIMIT 100;
        """
        rls_time = self.time_query(rls_query, iterations=50)
        print(f"RLS-enabled:           {rls_time:.2f}ms")

        overhead = ((rls_time - baseline_time) / baseline_time) * 100
        print(f"Overhead:              {overhead:.2f}%")

        assert overhead < 10, f"RLS overhead {overhead:.2f}% exceeds 10% threshold"

        print(f"\n✅ PASSED: Overhead {overhead:.2f}% < 10%")
        return overhead

    def test_aggregation_query(self):
        """Test 3: Aggregation query performance"""
        print("\n" + "=" * 70)
        print("TEST 3: Aggregation Query Performance (Count trades by symbol)")
        print("=" * 70)

        # Baseline
        baseline_query = "SELECT symbol, COUNT(*) FROM trades GROUP BY symbol"
        baseline_time = self.time_query(baseline_query, iterations=50)
        print(f"Baseline (no RLS):     {baseline_time:.2f}ms")

        # RLS
        rls_query = f"""
            SET LOCAL app.current_tenant_id = '{TENANT_1}';
            SELECT symbol, COUNT(*) FROM trades GROUP BY symbol;
        """
        rls_time = self.time_query(rls_query, iterations=50)
        print(f"RLS-enabled:           {rls_time:.2f}ms")

        overhead = ((rls_time - baseline_time) / baseline_time) * 100
        print(f"Overhead:              {overhead:.2f}%")

        assert overhead < 10, f"RLS overhead {overhead:.2f}% exceeds 10% threshold"

        print(f"\n✅ PASSED: Overhead {overhead:.2f}% < 10%")
        return overhead

    def test_tenant_isolation(self, trade_id: uuid.UUID):
        """Test 4: Tenant isolation security"""
        print("\n" + "=" * 70)
        print("TEST 4: Tenant Isolation Security")
        print("=" * 70)

        session = self.Session()

        try:
            # Tenant 1 should see the trade
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_1}'"))
            result = session.execute(
                text(f"SELECT id FROM trades WHERE id = '{trade_id}'")
            )
            trades_tenant1 = result.fetchall()
            print(f"Tenant 1 can see trade: {len(trades_tenant1) > 0}")

            # Tenant 2 should NOT see the trade
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_2}'"))
            result = session.execute(
                text(f"SELECT id FROM trades WHERE id = '{trade_id}'")
            )
            trades_tenant2 = result.fetchall()
            print(f"Tenant 2 can see trade: {len(trades_tenant2) > 0}")

            # Verify isolation
            assert len(trades_tenant1) > 0, "Tenant 1 should see their own trade"
            assert len(trades_tenant2) == 0, "Tenant 2 should NOT see Tenant 1's trade"

            print("\n✅ PASSED: Tenant isolation working correctly")

        finally:
            session.close()

    def test_insert_with_wrong_tenant(self):
        """Test 5: Prevent insert with mismatched tenant_id"""
        print("\n" + "=" * 70)
        print("TEST 5: INSERT Protection - Prevent Wrong Tenant ID")
        print("=" * 70)

        session = self.Session()

        try:
            # Set tenant context to Tenant 1
            session.execute(text(f"SET LOCAL app.current_tenant_id = '{TENANT_1}'"))

            # Try to insert trade with Tenant 2's ID (should fail)
            insert_query = text("""
                INSERT INTO trades (id, tenant_id, symbol, side, quantity, price, executed_at)
                VALUES (:id, :tenant_id, :symbol, :side, :quantity, :price, NOW())
            """)

            with pytest.raises(Exception) as exc_info:
                session.execute(insert_query, {
                    'id': str(uuid.uuid4()),
                    'tenant_id': TENANT_2,  # Wrong tenant!
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'quantity': 1.0,
                    'price': 50000.0
                })
                session.commit()

            print(f"INSERT with wrong tenant_id correctly rejected")
            print(f"Error: {str(exc_info.value)[:100]}")
            print("\n✅ PASSED: RLS prevents cross-tenant inserts")

        except AssertionError:
            raise
        except Exception as e:
            # Expected to fail - this is correct behavior
            session.rollback()
            print(f"✅ PASSED: INSERT with wrong tenant_id correctly rejected")
        finally:
            session.close()

    def run_all_tests(self):
        """Run complete performance benchmark suite"""
        print("\n" + "=" * 70)
        print("EPIC-001: Database Multi-Tenancy Performance Validation")
        print("Validating ADR-001 Requirement: RLS Overhead < 10%")
        print("=" * 70)

        # Setup: Get a test trade ID
        session = self.Session()
        result = session.execute(text(f"SELECT id FROM trades LIMIT 1"))
        trade_id = result.fetchone()
        session.close()

        if not trade_id:
            print("\n⚠️  WARNING: No trades found in database. Skipping performance tests.")
            print("Please run the system to generate some trades first.")
            return

        trade_id = trade_id[0]

        # Run all tests
        overheads = []
        overheads.append(self.test_primary_key_lookup(trade_id))
        overheads.append(self.test_time_series_query())
        overheads.append(self.test_aggregation_query())
        self.test_tenant_isolation(trade_id)
        self.test_insert_with_wrong_tenant()

        # Summary
        print("\n" + "=" * 70)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 70)
        avg_overhead = sum(overheads) / len(overheads)
        print(f"Average RLS Overhead:  {avg_overhead:.2f}%")
        print(f"Max RLS Overhead:      {max(overheads):.2f}%")
        print(f"Min RLS Overhead:      {min(overheads):.2f}%")

        if avg_overhead < 5:
            print(f"\n🎉 EXCELLENT: Average overhead {avg_overhead:.2f}% is well below 10% target")
        elif avg_overhead < 10:
            print(f"\n✅ PASSED: Average overhead {avg_overhead:.2f}% meets 10% requirement")
        else:
            print(f"\n❌ FAILED: Average overhead {avg_overhead:.2f}% exceeds 10% requirement")
            raise AssertionError(f"RLS overhead {avg_overhead:.2f}% exceeds 10%")

        print("\n" + "=" * 70)
        print("All tests completed successfully!")
        print("=" * 70)


if __name__ == "__main__":
    tester = RLSPerformanceTester()
    tester.run_all_tests()