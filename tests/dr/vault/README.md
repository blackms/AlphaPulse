# Vault Disaster Recovery Testing

This directory contains automated DR testing scripts for HashiCorp Vault.

## Overview

These scripts validate the backup/restore procedures documented in the [Vault Disaster Recovery Runbook](../../../docs/runbooks/VAULT_DISASTER_RECOVERY.md).

## Test Scripts

### `test-backup-restore.sh`

Automated monthly DR drill that validates:
- Snapshot creation and integrity
- Full cluster restore from snapshot
- Data integrity verification
- MTTR measurement

**Usage**:
```bash
# Run against local dev Vault
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root"
./test-backup-restore.sh

# Run against staging Vault
export VAULT_ADDR="https://vault-staging.alphapulse.io:8200"
export VAULT_TOKEN="<staging-token>"
./test-backup-restore.sh
```

**Expected Output**:
```
[2025-11-14 10:00:00] ===== Vault DR Drill Started =====
[2025-11-14 10:00:01] [SUCCESS] Vault is reachable
[2025-11-14 10:00:01] [SUCCESS] Vault is unsealed
[2025-11-14 10:00:15] [SUCCESS] Generated 100 test credentials
[2025-11-14 10:00:20] Snapshot created in 5 seconds
[2025-11-14 10:00:21] [SUCCESS] Snapshot verified (size: 2.3M)
[2025-11-14 10:00:35] [SUCCESS] All test credentials deleted
[2025-11-14 10:00:45] Restore completed in 10 seconds
[2025-11-14 10:00:50] [SUCCESS] All 10 spot checks passed
[2025-11-14 10:00:51] ===== DR Drill Results =====
[2025-11-14 10:00:51] MTTR (Snapshot + Restore): 15s
[2025-11-14 10:00:51] [SUCCESS] MTTR target MET (15s <= 1800s)
[2025-11-14 10:00:51] ===== Vault DR Drill Complete =====
[2025-11-14 10:00:51] [SUCCESS] DR drill PASSED all criteria
```

**Success Criteria**:
- ✅ MTTR <30 minutes (1800 seconds)
- ✅ 100% data integrity (all credentials restored)
- ✅ Snapshot integrity verified
- ✅ No manual interventions required

**Exit Codes**:
- `0`: All tests passed
- `1`: One or more tests failed

## Reports

The script generates a detailed markdown report in `/tmp/vault-dr-report-YYYYMMDD.md` with:
- Test metrics and timeline
- MTTR comparison to target
- Data integrity results
- Recommendations and action items

## Testing Schedule

| Test | Frequency | Environment | Duration |
|------|-----------|-------------|----------|
| Backup/Restore | Monthly (1st Monday) | Staging | ~20 min |
| DR Failover | Quarterly | Production (maintenance window) | ~2 hours |

## Prerequisites

### Local Development

```bash
# Start Vault in dev mode
vault server -dev -dev-root-token-id=root &

# Run test
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root"
./test-backup-restore.sh
```

### Staging Environment

```bash
# SSH to bastion host
ssh bastion-staging.alphapulse.io

# Set Vault address and token
export VAULT_ADDR="https://vault-staging.alphapulse.io:8200"
export VAULT_TOKEN="$(cat ~/.vault-token)"

# Run test
./test-backup-restore.sh
```

## Troubleshooting

### Vault Sealed

```bash
# Error: Vault is sealed. Unseal before running DR drill.

# Solution: Unseal Vault
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>
```

### MTTR Target Exceeded

```bash
# Warning: MTTR target MISSED (2100s > 1800s)

# Investigation:
# 1. Check Vault node disk I/O: iostat -x 1 10
# 2. Check network latency to S3: aws s3 cp /dev/null s3://bucket/test --debug
# 3. Review snapshot size: du -h /tmp/vault-*.snap
# 4. Check Raft log size: vault operator raft list-peers

# Mitigation:
# - Increase Vault node disk IOPS (use gp3 with higher throughput)
# - Use S3 Transfer Acceleration for cross-region backups
# - Implement incremental snapshots (Vault Enterprise)
```

### Data Integrity Failure

```bash
# Error: Data integrity check FAILED
# Expected: 100, Found: 95

# Investigation:
# 1. Check snapshot integrity: vault operator raft snapshot inspect <file>
# 2. Review Vault audit logs for errors during snapshot
# 3. Check for Raft consensus issues: vault operator raft list-peers

# Mitigation:
# - Verify Raft cluster health before taking snapshot
# - Ensure snapshot taken from leader node
# - Check disk space on Vault nodes
```

## Metrics

The test script exports Prometheus metrics via Pushgateway (if configured):

```prometheus
# HELP vault_dr_test_mttr_seconds Mean Time To Recover during DR test
# TYPE vault_dr_test_mttr_seconds gauge
vault_dr_test_mttr_seconds 15.0

# HELP vault_dr_test_data_integrity_percent Data integrity after restore
# TYPE vault_dr_test_data_integrity_percent gauge
vault_dr_test_data_integrity_percent 100.0

# HELP vault_dr_test_snapshot_duration_seconds Time to create snapshot
# TYPE vault_dr_test_snapshot_duration_seconds gauge
vault_dr_test_snapshot_duration_seconds 5.0

# HELP vault_dr_test_restore_duration_seconds Time to restore snapshot
# TYPE vault_dr_test_restore_duration_seconds gauge
vault_dr_test_restore_duration_seconds 10.0
```

## Integration with CI/CD

### GitHub Actions Workflow

```yaml
name: Vault DR Test

on:
  schedule:
    - cron: '0 3 * * 1'  # Every Monday at 3 AM UTC
  workflow_dispatch:     # Manual trigger

jobs:
  dr-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Vault CLI
        run: |
          wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
          unzip vault_1.15.0_linux_amd64.zip
          sudo mv vault /usr/local/bin/

      - name: Run DR Test
        env:
          VAULT_ADDR: ${{ secrets.VAULT_STAGING_ADDR }}
          VAULT_TOKEN: ${{ secrets.VAULT_STAGING_TOKEN }}
        run: |
          ./tests/dr/vault/test-backup-restore.sh

      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: vault-dr-report
          path: /tmp/vault-dr-report-*.md
```

## See Also

- [Vault Disaster Recovery Runbook](../../../docs/runbooks/VAULT_DISASTER_RECOVERY.md)
- [Vault HA Architecture](../../../docs/architecture/VAULT_HA.md)
- [Operations Playbook](../../../docs/runbooks/OPERATIONS.md)
