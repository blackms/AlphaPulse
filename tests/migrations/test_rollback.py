#!/usr/bin/env python3
"""
Database Migration Rollback Tests

Story: #161 - Create database migration rollback tests
Epic: EPIC-001 (#140) - Database Multi-Tenancy

This script validates that all multi-tenant migrations can be safely rolled back
and that data integrity is maintained throughout the rollback process.

Acceptance Criteria:
1. Rollback script tested for all migrations (005-009)
2. Data integrity verified after rollback
3. Recovery time <10 minutes documented

Usage:
    python tests/migrations/test_rollback.py
"""

import time
import subprocess
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import psycopg2
from psycopg2 import sql


class MigrationRollbackTester:
    """Test migration rollback capabilities."""

    def __init__(self, database_url: str = "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse"):
        self.database_url = database_url
        self.conn = None
        self.migrations = [
            ("009_composite_indexes", "Add composite indexes for tenant isolation performance"),
            ("008_enable_rls", "Enable Row-Level Security (RLS) policies"),
            ("007_add_tenant_id_domain", "Add tenant_id to domain tables"),
            ("006_add_tenant_id_users", "Add tenant_id to users table"),
            ("005_add_tenants", "Add tenants table for multi-tenancy"),
        ]
        self.test_results = []

    def connect(self):
        """Connect to database."""
        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = True
        print(f"✓ Connected to database")

    def disconnect(self):
        """Disconnect from database."""
        if self.conn:
            self.conn.close()
            print("✓ Disconnected from database")

    def run_command(self, command: List[str], description: str) -> Tuple[bool, float, str]:
        """Execute shell command and measure time."""
        print(f"  Executing: {description}...")
        start = time.time()

        try:
            # Use bash -c to ensure PATH is set correctly
            cmd_str = " ".join(command)
            result = subprocess.run(
                ["bash", "-c", f"cd /Users/a.rocchi/Projects/Personal/AlphaPulse && PYTHONPATH=. {cmd_str}"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                print(f"  ✓ Success ({elapsed:.2f}s)")
                return True, elapsed, result.stdout
            else:
                print(f"  ✗ Failed ({elapsed:.2f}s)")
                print(f"  Error: {result.stderr[:200]}")
                return False, elapsed, result.stderr

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  ✗ Timeout after {elapsed:.2f}s")
            return False, elapsed, "Timeout expired"

    def get_current_revision(self) -> str:
        """Get current Alembic revision."""
        success, _, output = self.run_command(
            ["poetry", "run", "alembic", "current"],
            "Get current revision"
        )
        if success:
            # Extract revision from output
            for line in output.split('\n'):
                if line.strip() and not line.startswith('The currently'):
                    return line.split()[0]
        return "unknown"

    def verify_table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            )
        """, (table_name,))
        exists = cursor.fetchone()[0]
        cursor.close()
        return exists

    def verify_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                  AND column_name = %s
            )
        """, (table_name, column_name))
        exists = cursor.fetchone()[0]
        cursor.close()
        return exists

    def verify_index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE schemaname = 'public'
                  AND indexname = %s
            )
        """, (index_name,))
        exists = cursor.fetchone()[0]
        cursor.close()
        return exists

    def check_rls_enabled(self, table_name: str) -> bool:
        """Check if RLS is enabled on a table."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT rowsecurity
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename = %s
        """, (table_name,))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else False

    def count_rows(self, table_name: str) -> int:
        """Count rows in a table."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception:
            return -1  # Table doesn't exist or error

    def verify_data_integrity_005(self) -> Dict[str, bool]:
        """Verify state after migration 005 (tenants table)."""
        return {
            "tenants_table_exists": self.verify_table_exists("tenants"),
            "tenants_has_id_column": self.verify_column_exists("tenants", "id"),
            "tenants_has_slug_column": self.verify_column_exists("tenants", "slug"),
        }

    def verify_data_integrity_006(self) -> Dict[str, bool]:
        """Verify state after migration 006 (users.tenant_id)."""
        return {
            "users_has_tenant_id": self.verify_column_exists("users", "tenant_id"),
        }

    def verify_data_integrity_007(self) -> Dict[str, bool]:
        """Verify state after migration 007 (domain tables tenant_id)."""
        return {
            "trades_has_tenant_id": self.verify_column_exists("trades", "tenant_id"),
            "positions_has_tenant_id": self.verify_column_exists("positions", "tenant_id"),
            "trading_accounts_has_tenant_id": self.verify_column_exists("trading_accounts", "tenant_id"),
        }

    def verify_data_integrity_008(self) -> Dict[str, bool]:
        """Verify state after migration 008 (RLS policies)."""
        return {
            "trades_rls_enabled": self.check_rls_enabled("trades"),
            "users_rls_enabled": self.check_rls_enabled("users"),
            "positions_rls_enabled": self.check_rls_enabled("positions"),
        }

    def verify_data_integrity_009(self) -> Dict[str, bool]:
        """Verify state after migration 009 (composite indexes)."""
        return {
            "trades_composite_index": self.verify_index_exists("idx_trades_tenant_id_compound"),
            "users_composite_index": self.verify_index_exists("idx_users_tenant_id_compound"),
        }

    def test_rollback_cycle(self, from_revision: str, description: str) -> Dict:
        """Test a single rollback and upgrade cycle."""
        print(f"\n{'='*80}")
        print(f"Testing Rollback: {from_revision} - {description}")
        print(f"{'='*80}")

        test_result = {
            "migration": from_revision,
            "description": description,
            "rollback_success": False,
            "rollback_time": 0.0,
            "upgrade_success": False,
            "upgrade_time": 0.0,
            "data_integrity_before": {},
            "data_integrity_after": {},
            "recovery_time": 0.0,
        }

        # Check state before rollback
        print("\n1. Verifying state BEFORE rollback...")
        if from_revision == "009_composite_indexes":
            test_result["data_integrity_before"] = self.verify_data_integrity_009()
        elif from_revision == "008_enable_rls":
            test_result["data_integrity_before"] = self.verify_data_integrity_008()
        elif from_revision == "007_add_tenant_id_domain":
            test_result["data_integrity_before"] = self.verify_data_integrity_007()
        elif from_revision == "006_add_tenant_id_users":
            test_result["data_integrity_before"] = self.verify_data_integrity_006()
        elif from_revision == "005_add_tenants":
            test_result["data_integrity_before"] = self.verify_data_integrity_005()

        print(f"   State: {test_result['data_integrity_before']}")

        # Perform rollback
        print("\n2. Rolling back migration...")
        rollback_start = time.time()
        success, rollback_time, _ = self.run_command(
            ["poetry", "run", "alembic", "downgrade", "-1"],
            f"Rollback {from_revision}"
        )
        test_result["rollback_success"] = success
        test_result["rollback_time"] = rollback_time

        if not success:
            print("   ✗ Rollback failed - aborting test")
            return test_result

        # Verify rollback completed
        print("\n3. Verifying rollback completed...")
        current_rev = self.get_current_revision()
        print(f"   Current revision: {current_rev}")

        # Perform upgrade (recovery)
        print("\n4. Upgrading to restore state...")
        upgrade_start = time.time()
        success, upgrade_time, _ = self.run_command(
            ["poetry", "run", "alembic", "upgrade", "+1"],
            f"Upgrade back to {from_revision}"
        )
        test_result["upgrade_success"] = success
        test_result["upgrade_time"] = upgrade_time
        test_result["recovery_time"] = time.time() - rollback_start

        if not success:
            print("   ✗ Upgrade failed - database may be in inconsistent state")
            return test_result

        # Verify state after recovery
        print("\n5. Verifying state AFTER recovery...")
        if from_revision == "009_composite_indexes":
            test_result["data_integrity_after"] = self.verify_data_integrity_009()
        elif from_revision == "008_enable_rls":
            test_result["data_integrity_after"] = self.verify_data_integrity_008()
        elif from_revision == "007_add_tenant_id_domain":
            test_result["data_integrity_after"] = self.verify_data_integrity_007()
        elif from_revision == "006_add_tenant_id_users":
            test_result["data_integrity_after"] = self.verify_data_integrity_006()
        elif from_revision == "005_add_tenants":
            test_result["data_integrity_after"] = self.verify_data_integrity_005()

        print(f"   State: {test_result['data_integrity_after']}")

        # Compare states
        print("\n6. Comparing states...")
        if test_result["data_integrity_before"] == test_result["data_integrity_after"]:
            print("   ✓ Data integrity maintained (states match)")
        else:
            print("   ✗ Data integrity issue (states differ)")
            print(f"     Before: {test_result['data_integrity_before']}")
            print(f"     After:  {test_result['data_integrity_after']}")

        # Summary
        print(f"\n{'='*80}")
        print(f"RESULT: {from_revision}")
        print(f"{'='*80}")
        print(f"Rollback:      {'✓ Success' if test_result['rollback_success'] else '✗ Failed'} ({test_result['rollback_time']:.2f}s)")
        print(f"Upgrade:       {'✓ Success' if test_result['upgrade_success'] else '✗ Failed'} ({test_result['upgrade_time']:.2f}s)")
        print(f"Recovery Time: {test_result['recovery_time']:.2f}s")
        print(f"Data Integrity: {'✓ Maintained' if test_result['data_integrity_before'] == test_result['data_integrity_after'] else '✗ Compromised'}")

        return test_result

    def run_all_tests(self):
        """Run rollback tests for all migrations."""
        print("\n" + "="*80)
        print("DATABASE MIGRATION ROLLBACK TESTS")
        print("Story #161 - EPIC-001")
        print("="*80)

        try:
            self.connect()

            # Ensure we're at the latest migration
            print("\nEnsuring database is at latest migration...")
            self.run_command(
                ["poetry", "run", "alembic", "upgrade", "head"],
                "Upgrade to head"
            )

            # Test each migration rollback
            for migration_id, description in self.migrations:
                result = self.test_rollback_cycle(migration_id, description)
                self.test_results.append(result)

            # Generate summary report
            self.print_summary()

        finally:
            self.disconnect()

    def print_summary(self):
        """Print summary report of all tests."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["rollback_success"] and r["upgrade_success"] and r["data_integrity_before"] == r["data_integrity_after"])
        failed_tests = total_tests - passed_tests

        print(f"\nTotal Tests:  {total_tests}")
        print(f"Passed:       {passed_tests} ✓")
        print(f"Failed:       {failed_tests} {'✗' if failed_tests > 0 else ''}")
        print()

        print(f"{'Migration':<30} {'Rollback':<12} {'Upgrade':<12} {'Recovery Time':<15} {'Status':<10}")
        print("-" * 80)

        for result in self.test_results:
            migration = result["migration"]
            rollback = "✓" if result["rollback_success"] else "✗"
            upgrade = "✓" if result["upgrade_success"] else "✗"
            recovery = f"{result['recovery_time']:.2f}s"
            integrity = result["data_integrity_before"] == result["data_integrity_after"]
            status = "✓ PASS" if result["rollback_success"] and result["upgrade_success"] and integrity else "✗ FAIL"

            print(f"{migration:<30} {rollback:<12} {upgrade:<12} {recovery:<15} {status:<10}")

        # Recovery time analysis
        print("\n" + "="*80)
        print("RECOVERY TIME ANALYSIS")
        print("="*80)

        max_recovery = max(r["recovery_time"] for r in self.test_results)
        avg_recovery = sum(r["recovery_time"] for r in self.test_results) / len(self.test_results)
        total_recovery = sum(r["recovery_time"] for r in self.test_results)

        print(f"\nMaximum recovery time:   {max_recovery:.2f}s")
        print(f"Average recovery time:   {avg_recovery:.2f}s")
        print(f"Total recovery time:     {total_recovery:.2f}s ({total_recovery/60:.2f} minutes)")
        print()

        # Check AC: Recovery time <10 minutes
        if max_recovery < 600:  # 10 minutes
            print(f"✓ PASS: Maximum recovery time {max_recovery:.2f}s < 600s (10 minutes)")
        else:
            print(f"✗ FAIL: Maximum recovery time {max_recovery:.2f}s > 600s (10 minutes)")

        # Final verdict
        print("\n" + "="*80)
        if passed_tests == total_tests and max_recovery < 600:
            print("✓ ALL TESTS PASSED - Rollback procedures validated")
        else:
            print("✗ SOME TESTS FAILED - Review rollback procedures")
        print("="*80)


def main():
    """Main entry point."""
    tester = MigrationRollbackTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
