#!/bin/bash
# Vault DR Drill: Backup and Restore Test
#
# This script automates the monthly DR drill for Vault backup/restore validation
# Expected Duration: ~20 minutes
# Environment: Staging

set -euo pipefail

# Configuration
VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-root}"
TEST_DATA_FILE="/tmp/vault-test-data.json"
SNAPSHOT_FILE="/tmp/vault-dr-test-$(date +%Y%m%d-%H%M%S).snap"
LOG_FILE="/tmp/vault-dr-test.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG_FILE}"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "${LOG_FILE}"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "${LOG_FILE}"
}

# Metrics
START_TIME=$(date +%s)
MTTR_TARGET=1800  # 30 minutes in seconds

# Cleanup on exit
cleanup() {
    log "Cleaning up test artifacts..."
    rm -f "${TEST_DATA_FILE}" "${SNAPSHOT_FILE}"
}
trap cleanup EXIT

# ============================================================================
# PHASE 1: Pre-Test Setup
# ============================================================================

log "===== Vault DR Drill Started ====="
log "Target MTTR: ${MTTR_TARGET} seconds (30 minutes)"

# Check Vault availability
log "Checking Vault connectivity..."
if ! vault status > /dev/null 2>&1; then
    error "Vault is not reachable at ${VAULT_ADDR}"
    exit 1
fi
success "Vault is reachable"

# Verify Vault is unsealed
if vault status | grep -q "Sealed.*true"; then
    error "Vault is sealed. Unseal before running DR drill."
    exit 1
fi
success "Vault is unsealed"

# Check storage backend (Raft required for snapshots)
STORAGE_TYPE=$(vault status -format=json | jq -r '.storage_type')
if [ "${STORAGE_TYPE}" != "raft" ]; then
    error "Vault storage backend is '${STORAGE_TYPE}', not 'raft'"
    error "Raft snapshots require Vault with Raft integrated storage"
    error "This test is intended for production/staging environments"
    error ""
    error "To test locally, use docker-compose with Raft storage:"
    error "  docker-compose -f docker-compose.raft.yml up vault"
    exit 1
fi
success "Vault storage backend: ${STORAGE_TYPE}"

# ============================================================================
# PHASE 2: Generate Test Data
# ============================================================================

log "Generating test data (100 tenant credentials)..."
for i in $(seq 1 100); do
    TENANT_ID=$(printf "%08d-0000-0000-0000-000000000001" $i)

    # Create API key credential
    vault kv put "secret/tenants/${TENANT_ID}/binance/api_key" \
        api_key="test_key_${i}_$(openssl rand -hex 16)" \
        api_secret="test_secret_${i}_$(openssl rand -hex 32)" \
        exchange="binance" \
        created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > /dev/null 2>&1 || {
            error "Failed to create credential for tenant ${TENANT_ID}"
            exit 1
        }
done

success "Generated 100 test credentials"

# Store credential list for verification
vault kv list -format=json secret/tenants > "${TEST_DATA_FILE}"
CREDENTIAL_COUNT=$(jq '. | length' "${TEST_DATA_FILE}")
log "Credential count before backup: ${CREDENTIAL_COUNT}"

# ============================================================================
# PHASE 3: Take Snapshot
# ============================================================================

log "Taking Vault snapshot..."
SNAPSHOT_START=$(date +%s)

vault operator raft snapshot save "${SNAPSHOT_FILE}" || {
    error "Snapshot creation failed"
    exit 1
}

SNAPSHOT_END=$(date +%s)
SNAPSHOT_DURATION=$((SNAPSHOT_END - SNAPSHOT_START))
log "Snapshot created in ${SNAPSHOT_DURATION} seconds"

# Verify snapshot integrity
log "Verifying snapshot integrity..."
vault operator raft snapshot inspect "${SNAPSHOT_FILE}" > /dev/null || {
    error "Snapshot integrity check failed"
    exit 1
}

SNAPSHOT_SIZE=$(du -h "${SNAPSHOT_FILE}" | cut -f1)
success "Snapshot verified (size: ${SNAPSHOT_SIZE})"

# ============================================================================
# PHASE 4: Simulate Disaster (Delete Data)
# ============================================================================

log "Simulating disaster: Deleting all test credentials..."
for i in $(seq 1 100); do
    TENANT_ID=$(printf "%08d-0000-0000-0000-000000000001" $i)
    vault kv delete "secret/tenants/${TENANT_ID}/binance/api_key" > /dev/null 2>&1
done

# Verify deletion
REMAINING_COUNT=$(vault kv list -format=json secret/tenants 2>/dev/null | jq '. | length' || echo 0)
if [ "${REMAINING_COUNT}" -eq 0 ]; then
    success "All test credentials deleted (disaster simulated)"
else
    warning "${REMAINING_COUNT} credentials still exist"
fi

# ============================================================================
# PHASE 5: Restore from Snapshot
# ============================================================================

log "Restoring from snapshot..."
RESTORE_START=$(date +%s)

vault operator raft snapshot restore -force "${SNAPSHOT_FILE}" || {
    error "Snapshot restore failed"
    exit 1
}

RESTORE_END=$(date +%s)
RESTORE_DURATION=$((RESTORE_END - RESTORE_START))
log "Restore completed in ${RESTORE_DURATION} seconds"

# Wait for data to be available (eventual consistency)
sleep 5

# ============================================================================
# PHASE 6: Verify Data Integrity
# ============================================================================

log "Verifying data integrity post-restore..."

# Check credential count
RESTORED_COUNT=$(vault kv list -format=json secret/tenants 2>/dev/null | jq '. | length' || echo 0)
log "Credentials after restore: ${RESTORED_COUNT}"

if [ "${RESTORED_COUNT}" -ne "${CREDENTIAL_COUNT}" ]; then
    error "Data integrity check FAILED"
    error "Expected: ${CREDENTIAL_COUNT}, Found: ${RESTORED_COUNT}"
    exit 1
fi

# Spot check 10 random credentials
log "Spot checking 10 random credentials..."
SPOT_CHECK_FAILED=0
for i in $(seq 1 10); do
    RANDOM_ID=$(shuf -i 1-100 -n 1)
    TENANT_ID=$(printf "%08d-0000-0000-0000-000000000001" $RANDOM_ID)

    if ! vault kv get "secret/tenants/${TENANT_ID}/binance/api_key" > /dev/null 2>&1; then
        error "Credential missing for tenant ${TENANT_ID}"
        ((SPOT_CHECK_FAILED++))
    fi
done

if [ "${SPOT_CHECK_FAILED}" -eq 0 ]; then
    success "All 10 spot checks passed"
else
    error "${SPOT_CHECK_FAILED}/10 spot checks failed"
    exit 1
fi

# ============================================================================
# PHASE 7: Calculate Metrics
# ============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
MTTR=$((SNAPSHOT_DURATION + RESTORE_DURATION))

log "===== DR Drill Results ====="
log "Snapshot Duration: ${SNAPSHOT_DURATION}s"
log "Restore Duration: ${RESTORE_DURATION}s"
log "MTTR (Snapshot + Restore): ${MTTR}s"
log "Total Test Duration: ${TOTAL_DURATION}s"
log "MTTR Target: ${MTTR_TARGET}s (30 minutes)"

if [ "${MTTR}" -le "${MTTR_TARGET}" ]; then
    success "MTTR target MET (${MTTR}s <= ${MTTR_TARGET}s)"
    RESULT="PASS"
else
    error "MTTR target MISSED (${MTTR}s > ${MTTR_TARGET}s)"
    RESULT="FAIL"
fi

# ============================================================================
# PHASE 8: Generate Report
# ============================================================================

cat > /tmp/vault-dr-report-$(date +%Y%m%d).md <<EOF
# Vault DR Drill Report

**Date**: $(date '+%Y-%m-%d %H:%M:%S %Z')
**Environment**: Staging
**Operator**: ${USER}

## Test Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Snapshot Duration | ${SNAPSHOT_DURATION}s | - | ✅ |
| Snapshot Size | ${SNAPSHOT_SIZE} | - | ✅ |
| Restore Duration | ${RESTORE_DURATION}s | - | ✅ |
| MTTR (Total) | ${MTTR}s | ${MTTR_TARGET}s | $([ "${MTTR}" -le "${MTTR_TARGET}" ] && echo "✅ PASS" || echo "❌ FAIL") |
| Data Integrity | ${RESTORED_COUNT}/${CREDENTIAL_COUNT} | 100% | $([ "${RESTORED_COUNT}" -eq "${CREDENTIAL_COUNT}" ] && echo "✅ PASS" || echo "❌ FAIL") |
| Spot Checks | $((10 - SPOT_CHECK_FAILED))/10 | 10/10 | $([ "${SPOT_CHECK_FAILED}" -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL") |

## Overall Result: ${RESULT}

## Timeline

1. **Test Start**: $(date -d @${START_TIME} '+%H:%M:%S')
2. **Snapshot Taken**: $(date -d @${SNAPSHOT_END} '+%H:%M:%S') (${SNAPSHOT_DURATION}s)
3. **Data Deleted**: $(date -d @${RESTORE_START} '+%H:%M:%S') (disaster simulated)
4. **Restore Complete**: $(date -d @${RESTORE_END} '+%H:%M:%S') (${RESTORE_DURATION}s)
5. **Verification Complete**: $(date -d @${END_TIME} '+%H:%M:%S')

## Observations

- Snapshot integrity: ✅ Verified
- Restore process: ✅ No errors
- Data integrity: ✅ 100% credentials restored
- Performance: $([ "${MTTR}" -le "${MTTR_TARGET}" ] && echo "✅ Within target" || echo "⚠️ Exceeds target")

## Recommendations

$(if [ "${MTTR}" -gt "${MTTR_TARGET}" ]; then
    echo "- [ ] Investigate restore performance (exceeded MTTR target)"
    echo "- [ ] Consider incremental snapshots or delta backups"
fi)
$(if [ "${SPOT_CHECK_FAILED}" -gt 0 ]; then
    echo "- [ ] Investigate ${SPOT_CHECK_FAILED} missing credentials"
    echo "- [ ] Review snapshot process for data loss"
fi)
- [x] Update runbook with any process improvements
- [x] Schedule next drill for $(date -d '+1 month' '+%Y-%m-%d')

## Action Items

- [ ] Review with operations team
- [ ] Update Prometheus MTTR dashboard
- [ ] Archive test logs and report

---

**Report Generated**: $(date '+%Y-%m-%d %H:%M:%S %Z')
**Log File**: ${LOG_FILE}
EOF

log "DR drill report generated: /tmp/vault-dr-report-$(date +%Y%m%d).md"

# ============================================================================
# PHASE 9: Cleanup Test Data
# ============================================================================

log "Cleaning up test credentials..."
for i in $(seq 1 100); do
    TENANT_ID=$(printf "%08d-0000-0000-0000-000000000001" $i)
    vault kv delete "secret/tenants/${TENANT_ID}/binance/api_key" > /dev/null 2>&1
done

success "Test credentials cleaned up"

# ============================================================================
# Exit
# ============================================================================

log "===== Vault DR Drill Complete ====="
if [ "${RESULT}" = "PASS" ]; then
    success "DR drill PASSED all criteria"
    exit 0
else
    error "DR drill FAILED - review report for details"
    exit 1
fi
