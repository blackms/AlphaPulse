# Key Management Guide

## Overview

AlphaPulse uses a hierarchical key management system to protect encryption keys and enable key rotation without re-encrypting all data.

## Key Hierarchy

```
Master Key (in KMS/Vault)
    ├── Data Encryption Keys (DEKs)
    │   ├── Trading Data Context
    │   ├── User PII Context  
    │   ├── API Credentials Context
    │   └── Custom Contexts
    └── Key Metadata
        ├── Version Information
        ├── Rotation Timestamps
        └── Access Logs
```

## Key Generation

### Master Key

Generate a new master key:

```python
from cryptography.fernet import Fernet

# Generate 256-bit key
master_key = Fernet.generate_key()
print(f"Master Key (base64): {master_key.decode()}")
```

Or using OpenSSL:

```bash
# Generate 32 bytes (256 bits) and encode as base64
openssl rand -base64 32
```

### Data Encryption Keys

DEKs are automatically derived from the master key:

```python
from alpha_pulse.utils.encryption import EncryptionKeyManager

manager = EncryptionKeyManager()
key, version = manager.derive_data_key("trading_data")
```

## Storage Backends

### AWS Secrets Manager (Production)

```python
# Store in AWS Secrets Manager
import boto3

client = boto3.client('secretsmanager', region_name='us-east-1')
client.create_secret(
    Name='alphapulse/encryption/master_key',
    SecretString=master_key,
    Tags=[
        {'Key': 'Environment', 'Value': 'production'},
        {'Key': 'Purpose', 'Value': 'encryption'}
    ]
)
```

AWS IAM Policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:PutSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:alphapulse/encryption/*"
    },
    {
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:*:*:key/*"
    }
  ]
}
```

### HashiCorp Vault (Staging)

```bash
# Enable KV v2 secret engine
vault secrets enable -path=alphapulse kv-v2

# Store master key
vault kv put alphapulse/encryption/master_key value="$MASTER_KEY"

# Create policy
vault policy write alphapulse-encryption - <<EOF
path "alphapulse/data/encryption/*" {
  capabilities = ["read", "create", "update"]
}
EOF
```

### Environment Variables (Development)

```bash
# .env file
ALPHAPULSE_ENCRYPTION_MASTER_KEY=your-base64-encoded-key
ALPHAPULSE_ENCRYPTION_KEY_VERSION=1
```

## Key Rotation

### Automatic Rotation

Set up automatic rotation with AWS Lambda:

```python
import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """AWS Lambda function for key rotation."""
    
    # Initialize clients
    secrets_client = boto3.client('secretsmanager')
    
    # Generate new key version
    from alpha_pulse.utils.encryption import EncryptionKeyManager
    manager = EncryptionKeyManager()
    new_version = manager.rotate_keys()
    
    # Log rotation
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'event': 'key_rotation',
        'old_version': new_version - 1,
        'new_version': new_version
    }
    
    # Store rotation log
    secrets_client.put_secret_value(
        SecretId='alphapulse/encryption/rotation_log',
        SecretString=json.dumps(log_entry)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Rotated to version {new_version}')
    }
```

### Manual Rotation

```python
from alpha_pulse.utils.encryption import EncryptionKeyManager

# Rotate keys
manager = EncryptionKeyManager()
new_version = manager.rotate_keys()

print(f"Keys rotated to version: {new_version}")

# Verify old data still decrypts
from alpha_pulse.config.database import get_db_session
session = get_db_session()

# Test decryption with old and new data
old_record = session.query(Model).first()
print(f"Old record decrypts: {old_record.encrypted_field}")
```

### Rotation Schedule

Recommended rotation schedule:
- **Production**: Quarterly (every 90 days)
- **Staging**: Monthly
- **Development**: As needed

## Key Recovery

### Backup Procedures

1. **Export Keys** (secure environment only):
```python
# Export current keys for backup
manager = EncryptionKeyManager()
backup_data = {
    'master_key': manager.get_or_create_master_key(),
    'current_version': manager.get_current_key_version(),
    'timestamp': datetime.utcnow().isoformat()
}

# Encrypt backup with separate key
backup_key = Fernet.generate_key()
f = Fernet(backup_key)
encrypted_backup = f.encrypt(json.dumps(backup_data).encode())

# Store encrypted backup and backup key separately
```

2. **Secure Storage**:
- Store encrypted backup in separate location
- Use different access controls for backup
- Test recovery procedures regularly

### Recovery Process

```python
# Recover from backup
def recover_keys(encrypted_backup, backup_key):
    f = Fernet(backup_key)
    backup_data = json.loads(f.decrypt(encrypted_backup))
    
    # Restore master key
    secrets_manager = get_secrets_manager()
    secrets_manager.primary_provider.set_secret(
        "encryption_master_key",
        backup_data['master_key']
    )
    
    # Restore version
    secrets_manager.primary_provider.set_secret(
        "encryption_key_version",
        backup_data['current_version']
    )
    
    print(f"Keys recovered from backup dated: {backup_data['timestamp']}")
```

## Access Control

### Role-Based Access

Define roles for key access:

1. **Key Administrator**: Full access to all key operations
2. **Application Service**: Read-only access to current keys
3. **Backup Service**: Write-only access to backup location
4. **Audit Service**: Read-only access to logs

### Audit Logging

All key operations are logged:

```python
from alpha_pulse.utils.encryption import EncryptionKeyManager

# Access logs are automatically created
manager = EncryptionKeyManager()
key, version = manager.derive_data_key("context")

# View audit logs
from alpha_pulse.config.secure_settings import get_secrets_manager
secrets_manager = get_secrets_manager()
audit_logs = secrets_manager.get_audit_log()

for log in audit_logs:
    print(f"{log['timestamp']}: {log['action']} - {log['secret_name']}")
```

## Monitoring

### Key Usage Metrics

Monitor key usage with CloudWatch:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Send key usage metric
cloudwatch.put_metric_data(
    Namespace='AlphaPulse/Encryption',
    MetricData=[
        {
            'MetricName': 'KeyDerivations',
            'Value': 1,
            'Unit': 'Count',
            'Dimensions': [
                {
                    'Name': 'KeyVersion',
                    'Value': str(key_version)
                },
                {
                    'Name': 'Context',
                    'Value': context
                }
            ]
        }
    ]
)
```

### Alerts

Set up alerts for:
- Failed key retrievals
- Unusual access patterns
- Approaching rotation deadlines
- Key version mismatches

## Best Practices

### Key Generation

1. **Use cryptographically secure random sources**
2. **Generate keys of appropriate length** (256 bits minimum)
3. **Never use predictable or derived keys**
4. **Document key generation procedures**

### Key Storage

1. **Never store keys in code or configuration files**
2. **Use hardware security modules (HSM) when possible**
3. **Implement key escrow for critical keys**
4. **Separate key storage from data storage**

### Key Usage

1. **Use different keys for different purposes**
2. **Implement key versioning from the start**
3. **Cache keys appropriately** (but securely)
4. **Monitor key usage patterns**

### Key Destruction

1. **Securely overwrite key material**
2. **Clear keys from memory after use**
3. **Document key lifecycle**
4. **Maintain destruction logs**

## Compliance Requirements

### PCI DSS

- Requirement 3.5: Protect encryption keys
- Requirement 3.6: Document key management procedures
- Requirement 3.6.4: Key rotation at least annually

### FIPS 140-2

- Level 2: Role-based authentication
- Level 3: Hardware security module
- Approved algorithms only

### SOX

- Documented key management procedures
- Separation of duties
- Audit trail for all key operations

## Emergency Procedures

### Key Compromise

1. **Immediate Actions**:
   ```bash
   # Rotate keys immediately
   python -m alpha_pulse.utils.encryption rotate_emergency
   
   # Disable compromised key version
   python -m alpha_pulse.utils.encryption disable_version --version X
   ```

2. **Investigation**:
   - Review audit logs
   - Identify affected data
   - Determine compromise scope

3. **Remediation**:
   - Re-encrypt affected data
   - Update access controls
   - Document incident

### Key Loss

1. **Check Backups**:
   ```python
   # Attempt recovery from backup
   python -m alpha_pulse.utils.encryption recover --backup-file /secure/backup
   ```

2. **If No Backup**:
   - Document affected data
   - Plan data recovery/recreation
   - Implement new keys for future data

## Tools and Scripts

### Key Management CLI

```bash
# Generate new master key
python -m alpha_pulse.utils.key_manager generate

# Rotate keys
python -m alpha_pulse.utils.key_manager rotate

# Check key status
python -m alpha_pulse.utils.key_manager status

# Export keys (secure environment only)
python -m alpha_pulse.utils.key_manager export --output /secure/location
```

### Health Checks

```python
# Check key accessibility
from alpha_pulse.utils.encryption import EncryptionKeyManager

manager = EncryptionKeyManager()
try:
    master_key = manager.get_or_create_master_key()
    print("✓ Master key accessible")
    
    key, version = manager.derive_data_key("test")
    print(f"✓ Can derive keys (version {version})")
    
except Exception as e:
    print(f"✗ Key management error: {e}")
```