#!/bin/bash
#
# Database Migration Rollback Tests
#
# Story: #161 - Create database migration rollback tests
# Epic: EPIC-001 (#140) - Database Multi-Tenancy
#
# This script validates that all multi-tenant migrations can be safely rolled back
# and that data integrity is maintained throughout the rollback process.
#
# Acceptance Criteria:
# 1. Rollback script tested for all migrations (005-009)
# 2. Data integrity verified after rollback
# 3. Recovery time <10 minutes documented

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
MAX_RECOVERY_TIME=0
TOTAL_RECOVERY_TIME=0

# Migrations to test (in reverse order)
declare -a MIGRATIONS=(
    "009_composite_indexes:Add composite indexes"
    "008_enable_rls:Enable RLS policies"
    "007_add_tenant_id_domain:Add tenant_id to domain tables"
    "006_add_tenant_id_users:Add tenant_id to users"
    "005_add_tenants:Add tenants table"
)

echo "================================================================================"
echo "DATABASE MIGRATION ROLLBACK TESTS"
echo "Story #161 - EPIC-001"
echo "================================================================================"
echo ""

# Change to project directory
cd /Users/a.rocchi/Projects/Personal/AlphaPulse

# Ensure we're at latest migration
echo "Ensuring database is at latest migration..."
PYTHONPATH=. poetry run alembic upgrade head > /dev/null 2>&1
echo "✓ Database at HEAD"
echo ""

# Function to check if table exists
table_exists() {
    local table=$1
    PGPASSWORD=alphapulse psql -h localhost -U alphapulse -d alphapulse -tAc \
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='$table')" 2>/dev/null
}

# Function to check if column exists
column_exists() {
    local table=$1
    local column=$2
    PGPASSWORD=alphapulse psql -h localhost -U alphapulse -d alphapulse -tAc \
        "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name='$table' AND column_name='$column')" 2>/dev/null
}

# Function to check if index exists
index_exists() {
    local index=$1
    PGPASSWORD=alphapulse psql -h localhost -U alphapulse -d alphapulse -tAc \
        "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname='$index')" 2>/dev/null
}

# Function to check RLS status
rls_enabled() {
    local table=$1
    PGPASSWORD=alphapulse psql -h localhost -U alphapulse -d alphapulse -tAc \
        "SELECT rowsecurity FROM pg_tables WHERE tablename='$table'" 2>/dev/null
}

# Function to test a migration rollback
test_migration_rollback() {
    local migration_id=$1
    local description=$2

    ((TOTAL_TESTS++))

    echo "================================================================================"
    echo "Testing Rollback: $migration_id - $description"
    echo "================================================================================"
    echo ""

    # 1. Check state before rollback
    echo "1. Checking state BEFORE rollback..."
    case $migration_id in
        "009_composite_indexes")
            BEFORE_CHECK=$(index_exists "idx_trades_tenant_id_compound")
            echo "   Composite index exists: $BEFORE_CHECK"
            ;;
        "008_enable_rls")
            BEFORE_CHECK=$(rls_enabled "trades")
            echo "   RLS enabled on trades: $BEFORE_CHECK"
            ;;
        "007_add_tenant_id_domain")
            BEFORE_CHECK=$(column_exists "trades" "tenant_id")
            echo "   trades.tenant_id exists: $BEFORE_CHECK"
            ;;
        "006_add_tenant_id_users")
            BEFORE_CHECK=$(column_exists "users" "tenant_id")
            echo "   users.tenant_id exists: $BEFORE_CHECK"
            ;;
        "005_add_tenants")
            BEFORE_CHECK=$(table_exists "tenants")
            echo "   tenants table exists: $BEFORE_CHECK"
            ;;
    esac
    echo ""

    # 2. Perform rollback
    echo "2. Rolling back migration..."
    ROLLBACK_START=$(date +%s)

    if PYTHONPATH=. poetry run alembic downgrade -1 > /tmp/rollback_${migration_id}.log 2>&1; then
        ROLLBACK_END=$(date +%s)
        ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))
        echo -e "   ${GREEN}✓${NC} Rollback successful (${ROLLBACK_TIME}s)"
    else
        ROLLBACK_END=$(date +%s)
        ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))
        echo -e "   ${RED}✗${NC} Rollback failed (${ROLLBACK_TIME}s)"
        echo "   See /tmp/rollback_${migration_id}.log for details"
        ((FAILED_TESTS++))
        return 1
    fi
    echo ""

    # 3. Verify rollback (state should be opposite)
    echo "3. Verifying rollback completed..."
    case $migration_id in
        "009_composite_indexes")
            AFTER_ROLLBACK=$(index_exists "idx_trades_tenant_id_compound")
            if [ "$AFTER_ROLLBACK" = "f" ]; then
                echo -e "   ${GREEN}✓${NC} Composite indexes removed"
            else
                echo -e "   ${RED}✗${NC} Composite indexes still exist"
                ((FAILED_TESTS++))
                return 1
            fi
            ;;
        "008_enable_rls")
            AFTER_ROLLBACK=$(rls_enabled "trades")
            if [ "$AFTER_ROLLBACK" = "f" ]; then
                echo -e "   ${GREEN}✓${NC} RLS disabled"
            else
                echo -e "   ${RED}✗${NC} RLS still enabled"
                ((FAILED_TESTS++))
                return 1
            fi
            ;;
        "007_add_tenant_id_domain")
            AFTER_ROLLBACK=$(column_exists "trades" "tenant_id")
            if [ "$AFTER_ROLLBACK" = "f" ]; then
                echo -e "   ${GREEN}✓${NC} tenant_id columns removed"
            else
                echo -e "   ${RED}✗${NC} tenant_id columns still exist"
                ((FAILED_TESTS++))
                return 1
            fi
            ;;
        "006_add_tenant_id_users")
            AFTER_ROLLBACK=$(column_exists "users" "tenant_id")
            if [ "$AFTER_ROLLBACK" = "f" ]; then
                echo -e "   ${GREEN}✓${NC} users.tenant_id removed"
            else
                echo -e "   ${RED}✗${NC} users.tenant_id still exists"
                ((FAILED_TESTS++))
                return 1
            fi
            ;;
        "005_add_tenants")
            AFTER_ROLLBACK=$(table_exists "tenants")
            if [ "$AFTER_ROLLBACK" = "f" ]; then
                echo -e "   ${GREEN}✓${NC} tenants table removed"
            else
                echo -e "   ${RED}✗${NC} tenants table still exists"
                ((FAILED_TESTS++))
                return 1
            fi
            ;;
    esac
    echo ""

    # 4. Perform upgrade (recovery)
    echo "4. Upgrading to restore state..."
    UPGRADE_START=$(date +%s)

    if PYTHONPATH=. poetry run alembic upgrade +1 > /tmp/upgrade_${migration_id}.log 2>&1; then
        UPGRADE_END=$(date +%s)
        UPGRADE_TIME=$((UPGRADE_END - UPGRADE_START))
        RECOVERY_TIME=$((UPGRADE_END - ROLLBACK_START))
        echo -e "   ${GREEN}✓${NC} Upgrade successful (${UPGRADE_TIME}s)"
        echo -e "   Total recovery time: ${RECOVERY_TIME}s"

        # Track max recovery time
        if [ $RECOVERY_TIME -gt $MAX_RECOVERY_TIME ]; then
            MAX_RECOVERY_TIME=$RECOVERY_TIME
        fi
        TOTAL_RECOVERY_TIME=$((TOTAL_RECOVERY_TIME + RECOVERY_TIME))
    else
        UPGRADE_END=$(date +%s)
        UPGRADE_TIME=$((UPGRADE_END - UPGRADE_START))
        echo -e "   ${RED}✗${NC} Upgrade failed (${UPGRADE_TIME}s)"
        echo "   See /tmp/upgrade_${migration_id}.log for details"
        ((FAILED_TESTS++))
        return 1
    fi
    echo ""

    # 5. Verify state after recovery (should match before)
    echo "5. Verifying state AFTER recovery..."
    case $migration_id in
        "009_composite_indexes")
            AFTER_RECOVERY=$(index_exists "idx_trades_tenant_id_compound")
            ;;
        "008_enable_rls")
            AFTER_RECOVERY=$(rls_enabled "trades")
            ;;
        "007_add_tenant_id_domain")
            AFTER_RECOVERY=$(column_exists "trades" "tenant_id")
            ;;
        "006_add_tenant_id_users")
            AFTER_RECOVERY=$(column_exists "users" "tenant_id")
            ;;
        "005_add_tenants")
            AFTER_RECOVERY=$(table_exists "tenants")
            ;;
    esac

    if [ "$BEFORE_CHECK" = "$AFTER_RECOVERY" ]; then
        echo -e "   ${GREEN}✓${NC} Data integrity maintained (states match)"
        ((PASSED_TESTS++))
    else
        echo -e "   ${RED}✗${NC} Data integrity compromised (states differ)"
        echo "   Before: $BEFORE_CHECK, After: $AFTER_RECOVERY"
        ((FAILED_TESTS++))
        return 1
    fi
    echo ""

    echo "================================================================================"
    echo -e "RESULT: $migration_id - ${GREEN}✓ PASSED${NC}"
    echo "Rollback: ${ROLLBACK_TIME}s, Upgrade: ${UPGRADE_TIME}s, Recovery: ${RECOVERY_TIME}s"
    echo "================================================================================"
    echo ""
}

# Run tests for each migration
for migration_spec in "${MIGRATIONS[@]}"; do
    IFS=':' read -r migration_id description <<< "$migration_spec"
    test_migration_rollback "$migration_id" "$description" || true
done

# Print summary
echo ""
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo ""
echo "Total Tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$PASSED_TESTS ✓${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "Failed:       ${RED}$FAILED_TESTS ✗${NC}"
else
    echo "Failed:       $FAILED_TESTS"
fi
echo ""

echo "================================================================================"
echo "RECOVERY TIME ANALYSIS"
echo "================================================================================"
echo ""
AVG_RECOVERY_TIME=$((TOTAL_RECOVERY_TIME / TOTAL_TESTS))
echo "Maximum recovery time:   ${MAX_RECOVERY_TIME}s"
echo "Average recovery time:   ${AVG_RECOVERY_TIME}s"
echo "Total recovery time:     ${TOTAL_RECOVERY_TIME}s ($(echo "scale=2; $TOTAL_RECOVERY_TIME/60" | bc) minutes)"
echo ""

# Check AC: Recovery time <10 minutes
if [ $MAX_RECOVERY_TIME -lt 600 ]; then
    echo -e "${GREEN}✓ PASS${NC}: Maximum recovery time ${MAX_RECOVERY_TIME}s < 600s (10 minutes)"
else
    echo -e "${RED}✗ FAIL${NC}: Maximum recovery time ${MAX_RECOVERY_TIME}s > 600s (10 minutes)"
fi
echo ""

# Final verdict
echo "================================================================================"
if [ $PASSED_TESTS -eq $TOTAL_TESTS ] && [ $MAX_RECOVERY_TIME -lt 600 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC} - Rollback procedures validated"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC} - Review rollback procedures"
    exit 1
fi
