# HashiCorp Vault Disaster Recovery Runbook

**Version**: 1.0
**Last Updated**: 2025-11-14
**Owner**: DevOps/Security Team
**MTTR Target**: <30 minutes

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Backup Procedures](#backup-procedures)
4. [Restore Procedures](#restore-procedures)
5. [Unsealing Procedures](#unsealing-procedures)
6. [Disaster Recovery Scenarios](#disaster-recovery-scenarios)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Testing Schedule](#testing-schedule)
9. [Appendix](#appendix)

---

## 1. Overview

This runbook provides step-by-step procedures for backing up, restoring, and recovering HashiCorp Vault in disaster scenarios. Vault stores all tenant exchange credentials for AlphaPulse and is a **critical dependency** for the trading system.

### Business Impact

- **High Severity**: Vault downtime blocks all trading operations
- **SLA**: 99.9% uptime (43 minutes downtime/month maximum)
- **RTO**: <30 minutes (Recovery Time Objective)
- **RPO**: <1 hour (Recovery Point Objective)

### Key Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| Primary On-Call | DevOps rotation | PagerDuty |
| Vault SME | Security Team Lead | Slack #vault-incidents |
| Database Admin | DBA rotation | PagerDuty |
| Incident Commander | Engineering Manager | Phone |

---

## 2. Architecture

### Production Setup (HA Configuration)

```
┌─────────────────────────────────────────────────┐
│              Load Balancer (HAProxy)            │
│              vault.alphapulse.io:8200           │
└─────────────┬───────────────────────────────────┘
              │
    ┌─────────┴──────────┬─────────────────┐
    │                    │                 │
┌───▼────┐         ┌────▼────┐      ┌────▼────┐
│ Vault  │◄────────┤ Vault   │──────┤ Vault   │
│ Node 1 │ Raft    │ Node 2  │ Raft │ Node 3  │
│(Leader)│ Cluster │(Follower│      │(Follower│
└───┬────┘         └────┬────┘      └────┬────┘
    │                   │                │
    └───────────────────┴────────────────┘
                        │
                ┌───────▼────────┐
                │   Auto-Unseal  │
                │   AWS KMS      │
                └────────────────┘
```

**Key Components**:
- **3 Vault nodes** in Raft HA mode (1 leader, 2 followers)
- **Integrated Storage** (Raft): Data stored locally on each node
- **Auto-Unseal**: AWS KMS key for automatic unsealing
- **Load Balancer**: HAProxy for failover (health checks on `/v1/sys/health`)

### Storage Details

- **Backend**: Raft integrated storage
- **Snapshot Location**: S3 bucket `alphapulse-vault-backups`
- **Encryption**: AES-256-GCM (at rest), TLS 1.3 (in transit)
- **Retention**: 90 days (daily snapshots), 1 year (monthly snapshots)

---

## 3. Backup Procedures

### 3.1 Automated Daily Backups

**Schedule**: Daily at 02:00 UTC
**Tool**: Cron job on Vault leader node
**Script**: `/opt/vault/scripts/backup.sh`

#### Backup Script

```bash
#!/bin/bash
# /opt/vault/scripts/backup.sh
# Automated Vault snapshot backup to S3

set -euo pipefail

VAULT_ADDR="https://vault.alphapulse.io:8200"
VAULT_TOKEN="$(cat /etc/vault.d/backup-token)"
BACKUP_DIR="/var/vault/backups"
S3_BUCKET="s3://alphapulse-vault-backups"
DATE=$(date +%Y%m%d-%H%M%S)
SNAPSHOT_FILE="vault-snapshot-${DATE}.snap"

# Health check
vault status || { echo "Vault is sealed or unreachable"; exit 1; }

# Take snapshot
vault operator raft snapshot save "${BACKUP_DIR}/${SNAPSHOT_FILE}" || {
    echo "Snapshot failed"
    exit 1
}

# Verify snapshot integrity
vault operator raft snapshot inspect "${BACKUP_DIR}/${SNAPSHOT_FILE}" || {
    echo "Snapshot verification failed"
    exit 1
}

# Upload to S3 with encryption
aws s3 cp "${BACKUP_DIR}/${SNAPSHOT_FILE}" \
    "${S3_BUCKET}/${SNAPSHOT_FILE}" \
    --server-side-encryption aws:kms \
    --sse-kms-key-id "arn:aws:kms:us-east-1:123456789012:key/vault-backup-key"

# Verify upload
aws s3 ls "${S3_BUCKET}/${SNAPSHOT_FILE}" || {
    echo "S3 upload verification failed"
    exit 1
}

# Clean up local backups older than 7 days
find "${BACKUP_DIR}" -name "vault-snapshot-*.snap" -mtime +7 -delete

# Log success
echo "Backup completed: ${SNAPSHOT_FILE}" | logger -t vault-backup

# Send metrics to Prometheus
echo "vault_backup_success_timestamp $(date +%s)" | curl -s -X POST \
    --data-binary @- http://pushgateway:9091/metrics/job/vault-backup
```

#### Crontab Entry

```cron
# Vault daily backup at 02:00 UTC
0 2 * * * /opt/vault/scripts/backup.sh >> /var/log/vault/backup.log 2>&1
```

### 3.2 Manual Backup (On-Demand)

Use this procedure before:
- Vault version upgrades
- Configuration changes
- Security policy updates

```bash
# 1. SSH to Vault leader node
ssh vault-leader-prod

# 2. Set environment
export VAULT_ADDR="https://vault.alphapulse.io:8200"
export VAULT_TOKEN="<your-token>"

# 3. Take snapshot
vault operator raft snapshot save /tmp/vault-manual-$(date +%Y%m%d-%H%M%S).snap

# 4. Verify snapshot
vault operator raft snapshot inspect /tmp/vault-manual-*.snap

# 5. Upload to S3 (manual folder)
aws s3 cp /tmp/vault-manual-*.snap s3://alphapulse-vault-backups/manual/

# 6. Verify upload
aws s3 ls s3://alphapulse-vault-backups/manual/ | tail -1
```

### 3.3 Backup Verification

**Weekly Schedule**: Every Monday 03:00 UTC
**Script**: `/opt/vault/scripts/verify-backup.sh`

```bash
#!/bin/bash
# Verify latest backup can be inspected

LATEST_BACKUP=$(aws s3 ls s3://alphapulse-vault-backups/ | grep vault-snapshot | tail -1 | awk '{print $4}')
aws s3 cp "s3://alphapulse-vault-backups/${LATEST_BACKUP}" /tmp/
vault operator raft snapshot inspect "/tmp/${LATEST_BACKUP}" || {
    echo "CRITICAL: Backup verification failed"
    # Trigger PagerDuty alert
    exit 1
}
echo "Backup verification successful: ${LATEST_BACKUP}"
```

---

## 4. Restore Procedures

### 4.1 Full Cluster Restore (Complete Data Loss)

**Scenario**: All 3 Vault nodes lost, need to restore from S3 backup
**Estimated Time**: 20-30 minutes
**Prerequisites**: S3 backup available, Vault binaries installed

#### Step-by-Step Restore

```bash
# === STEP 1: Provision new Vault cluster (if needed) ===
# Use Terraform or manual EC2 instances (3 nodes)
# Install Vault binary (same version as backup)

# === STEP 2: Download latest backup ===
LATEST_BACKUP=$(aws s3 ls s3://alphapulse-vault-backups/ | grep vault-snapshot | tail -1 | awk '{print $4}')
aws s3 cp "s3://alphapulse-vault-backups/${LATEST_BACKUP}" /tmp/vault-restore.snap

# Verify backup integrity
vault operator raft snapshot inspect /tmp/vault-restore.snap

# === STEP 3: Initialize first node ===
# On vault-node-1
vault operator init -key-shares=5 -key-threshold=3 > /tmp/vault-init-keys.txt

# Save unseal keys and root token to AWS Secrets Manager (CRITICAL!)
aws secretsmanager create-secret \
    --name vault-recovery-keys-$(date +%s) \
    --secret-string "$(cat /tmp/vault-init-keys.txt)"

# Unseal node-1 (use 3 of 5 keys from init output)
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# === STEP 4: Restore snapshot to node-1 ===
export VAULT_TOKEN="<root-token-from-init>"
vault operator raft snapshot restore -force /tmp/vault-restore.snap

# === STEP 5: Join nodes 2 and 3 to cluster ===
# On vault-node-2
vault operator raft join https://vault-node-1:8200

# Unseal node-2
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# Repeat for node-3
# On vault-node-3
vault operator raft join https://vault-node-1:8200
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# === STEP 6: Verify cluster health ===
vault operator raft list-peers

# Expected output:
# Node       Address              State       Voter
# ----       -------              -----       -----
# node1      10.0.1.10:8201       leader      true
# node2      10.0.1.11:8201       follower    true
# node3      10.0.1.12:8201       follower    true

# === STEP 7: Verify data integrity ===
# Test credential retrieval
vault kv get secret/tenants/00000000-0000-0000-0000-000000000001/binance/api_key

# === STEP 8: Re-enable auto-unseal (if configured) ===
# Update Vault config to use AWS KMS auto-unseal
# Restart Vault service

# === STEP 9: Update DNS/Load Balancer ===
# Point vault.alphapulse.io to new cluster

# === STEP 10: Notify operations team ===
# Post to #incidents Slack channel
# Update incident timeline
```

### 4.2 Single Node Restore (Node Failure)

**Scenario**: 1 of 3 nodes failed, cluster still operational
**Estimated Time**: 10-15 minutes

```bash
# === STEP 1: Remove failed node from cluster ===
# On healthy leader node
vault operator raft remove-peer <failed-node-id>

# === STEP 2: Provision replacement node ===
# Launch new EC2 instance, install Vault

# === STEP 3: Join replacement node to cluster ===
# On replacement node
vault operator raft join https://vault-leader:8200

# === STEP 4: Unseal replacement node ===
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# === STEP 5: Verify cluster ===
vault operator raft list-peers
# Ensure replacement node shows as follower

# === STEP 6: Monitor replication ===
# Wait for data to sync (check raft logs)
journalctl -u vault -f | grep "replication"
```

---

## 5. Unsealing Procedures

### 5.1 Auto-Unseal (Production - AWS KMS)

**Normal Operations**: Vault auto-unseals using AWS KMS key

#### Configuration

```hcl
# /etc/vault.d/vault.hcl
seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/vault-unseal-key"
}
```

#### Troubleshooting Auto-Unseal Failures

```bash
# Check Vault status
vault status

# If sealed, check KMS key access
aws kms describe-key --key-id vault-unseal-key

# Check IAM role permissions
aws sts get-caller-identity

# Check CloudWatch logs for KMS errors
aws logs filter-log-events \
    --log-group-name /aws/kms/vault-unseal \
    --start-time $(date -u -d '1 hour ago' +%s000)

# If KMS unavailable, use manual unseal (see 5.2)
```

### 5.2 Manual Unseal (Emergency / KMS Unavailable)

**Use Case**: KMS key deleted, IAM role issue, AWS region outage

#### Prerequisites

- Access to 3 of 5 unseal keys (stored in AWS Secrets Manager + offline backup)
- SSH access to Vault nodes

#### Procedure

```bash
# === STEP 1: Retrieve unseal keys ===
# From AWS Secrets Manager
aws secretsmanager get-secret-value \
    --secret-id vault-unseal-keys \
    --query SecretString \
    --output text > /tmp/unseal-keys.txt

# OR from offline secure storage (encrypted USB key)
# Decrypt: gpg --decrypt /mnt/usb/vault-keys.gpg > /tmp/unseal-keys.txt

# === STEP 2: Unseal each node (repeat for all 3 nodes) ===
export VAULT_ADDR="https://vault-node-1:8200"

# Provide 3 of 5 keys
vault operator unseal
# Enter Key 1: <paste-key>

vault operator unseal
# Enter Key 2: <paste-key>

vault operator unseal
# Enter Key 3: <paste-key>

# Verify unsealed
vault status | grep "Sealed"
# Expected: Sealed          false

# === STEP 3: Repeat for nodes 2 and 3 ===
export VAULT_ADDR="https://vault-node-2:8200"
vault operator unseal  # Key 1
vault operator unseal  # Key 2
vault operator unseal  # Key 3

export VAULT_ADDR="https://vault-node-3:8200"
vault operator unseal  # Key 1
vault operator unseal  # Key 2
vault operator unseal  # Key 3

# === STEP 4: Verify cluster health ===
vault operator raft list-peers
vault status
```

### 5.3 Seal Key Rotation

**Schedule**: Quarterly (every 90 days)
**Trigger**: Security audit, key compromise suspected

```bash
# Generate new unseal keys
vault operator rekey -init -key-shares=5 -key-threshold=3

# Follow prompts to provide existing unseal keys
# Save new keys to AWS Secrets Manager (new version)

# Verify rekey
vault status | grep "Unseal Progress"
```

---

## 6. Disaster Recovery Scenarios

### 6.1 Scenario: Complete AWS Region Failure

**Impact**: All 3 Vault nodes unreachable
**Recovery Path**: Restore in secondary region from S3 cross-region backup

#### Prerequisites

- S3 cross-region replication enabled (`alphapulse-vault-backups-dr` in `us-west-2`)
- VPC and networking in DR region pre-configured
- Vault AMI replicated to DR region

#### DR Failover Procedure

```bash
# === STEP 1: Activate DR runbook ===
# Incident Commander declares DR event
# Notify stakeholders (15-min trading halt)

# === STEP 2: Launch Vault cluster in DR region (us-west-2) ===
cd terraform/vault-dr
terraform apply -var="region=us-west-2"

# === STEP 3: Download backup from DR S3 bucket ===
LATEST_DR_BACKUP=$(aws s3 ls s3://alphapulse-vault-backups-dr/ --region us-west-2 | grep vault-snapshot | tail -1 | awk '{print $4}')
aws s3 cp "s3://alphapulse-vault-backups-dr/${LATEST_DR_BACKUP}" /tmp/vault-dr-restore.snap --region us-west-2

# === STEP 4: Restore cluster (follow Section 4.1) ===
# Initialize, unseal, restore snapshot, join nodes

# === STEP 5: Update DNS failover ===
# Route53 health check auto-failover OR manual:
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890ABC \
    --change-batch file://dns-failover.json

# dns-failover.json:
# {
#   "Changes": [{
#     "Action": "UPSERT",
#     "ResourceRecordSet": {
#       "Name": "vault.alphapulse.io",
#       "Type": "A",
#       "TTL": 60,
#       "ResourceRecords": [{"Value": "<dr-load-balancer-ip>"}]
#     }
#   }]
# }

# === STEP 6: Resume trading operations ===
# Verify credential retrieval from API
# Restart halted trading services
# Monitor for errors (15-min observation window)

# === STEP 7: Post-incident tasks ===
# Document incident timeline
# Schedule post-mortem (within 48 hours)
# Update DR runbook with lessons learned
```

**RTO Achieved**: ~25 minutes (backup download: 5 min, restore: 15 min, DNS propagation: 5 min)

### 6.2 Scenario: Vault Data Corruption

**Symptoms**: Vault unsealed but data unreadable, CRUD operations failing

```bash
# === STEP 1: Identify corruption ===
# Check Vault audit logs
tail -100 /var/log/vault/audit.log | grep "ERROR"

# Check Raft integrity
vault operator raft list-peers
# Look for "Index mismatch" or "Checksum failure"

# === STEP 2: Isolate corrupted node ===
# If single node, remove from cluster
vault operator raft remove-peer <node-id>

# === STEP 3: Restore from backup (if cluster-wide) ===
# Follow Section 4.1 (Full Cluster Restore)

# === STEP 4: Root cause analysis ===
# Check disk errors: dmesg | grep -i error
# Check file system: fsck /dev/xvdf
# Review recent deployments/changes
```

### 6.3 Scenario: Accidental Secret Deletion

**Impact**: Tenant credentials deleted from Vault
**Recovery Path**: Point-in-time restore

```bash
# === STEP 1: Identify deletion timestamp ===
# Check Vault audit log
grep "secret/tenants/<tenant-id>" /var/log/vault/audit.log | grep "delete"

# === STEP 2: Find backup before deletion ===
# List S3 backups by timestamp
aws s3 ls s3://alphapulse-vault-backups/ | grep "$(date -d '24 hours ago' +%Y%m%d)"

# === STEP 3: Restore to temporary Vault instance ===
# Launch isolated Vault node for data extraction
vault operator raft snapshot restore /tmp/vault-backup-before-deletion.snap

# === STEP 4: Extract deleted secret ===
vault kv get -format=json secret/tenants/<tenant-id>/binance/api_key > /tmp/recovered-secret.json

# === STEP 5: Re-import to production Vault ===
vault kv put secret/tenants/<tenant-id>/binance/api_key \
    api_key="$(jq -r .data.api_key /tmp/recovered-secret.json)" \
    api_secret="$(jq -r .data.api_secret /tmp/recovered-secret.json)"

# === STEP 6: Verify recovery ===
# Test credential via CCXT
python3 <<EOF
import ccxt
exchange = ccxt.binance({
    'apiKey': '<recovered-key>',
    'secret': '<recovered-secret>'
})
print(exchange.fetch_balance())
EOF
```

---

## 7. Monitoring and Alerting

### 7.1 Critical Metrics (Prometheus)

| Metric | Threshold | Action |
|--------|-----------|--------|
| `vault_core_unsealed` | 0 (sealed) | Page on-call immediately |
| `vault_raft_leader_last_contact` | >1000ms | Investigate replication lag |
| `vault_backup_success_timestamp` | >25 hours | Check backup script |
| `vault_runtime_sys_gc_pause_ns` | >500ms | Memory pressure, investigate |
| `vault_core_handle_request` | p99 >100ms | Performance degradation |

### 7.2 Health Check Endpoints

```bash
# Vault API health check
curl -s https://vault.alphapulse.io:8200/v1/sys/health | jq

# Expected response:
# {
#   "initialized": true,
#   "sealed": false,
#   "standby": false,
#   "performance_standby": false,
#   "replication_performance_mode": "disabled",
#   "replication_dr_mode": "disabled",
#   "server_time_utc": 1700000000,
#   "version": "1.15.0"
# }
```

### 7.3 Alerting Rules (Prometheus AlertManager)

```yaml
# vault-alerts.yml
groups:
  - name: vault_critical
    interval: 30s
    rules:
      - alert: VaultSealed
        expr: vault_core_unsealed == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Vault is sealed on {{ $labels.instance }}"
          description: "Unseal Vault immediately (MTTR <30 min)"

      - alert: VaultBackupFailed
        expr: (time() - vault_backup_success_timestamp) > 86400
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Vault backup not completed in 24 hours"
          description: "Check /var/log/vault/backup.log"

      - alert: VaultHighLatency
        expr: histogram_quantile(0.99, vault_core_handle_request_bucket) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Vault p99 latency >100ms"
          description: "Investigate credential cache hit rate"
```

---

## 8. Testing Schedule

### 8.1 Monthly DR Drill (First Monday)

**Objective**: Verify backup restore procedure
**Duration**: 1 hour
**Environment**: Staging

```bash
# Test Plan:
# 1. Deploy Vault cluster in staging
# 2. Populate with test data (100 tenant credentials)
# 3. Take snapshot
# 4. Destroy cluster
# 5. Restore from snapshot
# 6. Verify all 100 credentials readable
# 7. Measure MTTR (target: <30 min)
# 8. Document any issues

# Success Criteria:
# - Restore completes in <30 minutes
# - 100% data integrity (all secrets readable)
# - Zero manual interventions required
```

### 8.2 Quarterly Failover Test

**Objective**: Validate DR region failover
**Duration**: 2 hours
**Environment**: Production (during maintenance window)

```bash
# Test Plan:
# 1. Simulate primary region failure (stop all Vault nodes)
# 2. Activate DR runbook
# 3. Restore in us-west-2 from cross-region backup
# 4. Update Route53 DNS to DR cluster
# 5. Run smoke tests (10 API calls)
# 6. Measure RTO (target: <25 min)
# 7. Failback to primary region
# 8. Post-test review

# Success Criteria:
# - RTO <25 minutes
# - Zero credential retrieval errors during failover
# - DNS propagation <5 minutes
```

---

## 9. Appendix

### 9.1 Unseal Key Storage Locations

| Location | Purpose | Access Control |
|----------|---------|----------------|
| AWS Secrets Manager `vault-unseal-keys` | Primary (encrypted at rest) | IAM role: `vault-admin` |
| 1Password Vault `Engineering/Vault` | Secondary (manual access) | 2FA required |
| Encrypted USB drive (safe deposit box) | Offline backup | Physical access (2 people) |

### 9.2 Vault Configuration Files

```bash
# Production Vault config
/etc/vault.d/vault.hcl

# Backup script
/opt/vault/scripts/backup.sh

# Systemd service
/etc/systemd/system/vault.service

# TLS certificates
/etc/vault.d/tls/vault.crt
/etc/vault.d/tls/vault.key
```

### 9.3 AWS Resources

| Resource | ARN | Purpose |
|----------|-----|---------|
| S3 Backup Bucket | `arn:aws:s3:::alphapulse-vault-backups` | Daily snapshots |
| S3 DR Bucket | `arn:aws:s3:::alphapulse-vault-backups-dr` | Cross-region replication |
| KMS Unseal Key | `arn:aws:kms:us-east-1:...:key/vault-unseal-key` | Auto-unseal |
| KMS Backup Key | `arn:aws:kms:us-east-1:...:key/vault-backup-key` | S3 encryption |
| Secrets Manager | `arn:aws:secretsmanager:us-east-1:...:secret:vault-unseal-keys` | Key storage |

### 9.4 Runbook Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-14 | DevOps Team | Initial runbook creation |

---

## Emergency Contacts

- **Primary On-Call**: PagerDuty rotation (automated)
- **Vault SME**: security-team@alphapulse.io
- **AWS Support**: Enterprise Support case (Critical severity)
- **Incident Commander**: eng-manager@alphapulse.io

**Incident Slack Channel**: `#vault-incidents`
**Escalation Policy**: https://wiki.alphapulse.io/runbooks/escalation

---

**Document Status**: APPROVED
**Next Review**: 2025-02-14 (90 days)
