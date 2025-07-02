# Database Encryption Guide

## Overview

AlphaPulse implements field-level encryption for sensitive data using AES-256-GCM encryption. This provides authenticated encryption with built-in integrity protection for all sensitive trading and user data.

## Architecture

### Encryption Algorithm

- **Algorithm**: AES-256-GCM (Galois/Counter Mode)
- **Key Size**: 256 bits
- **IV Size**: 96 bits (12 bytes)
- **Authentication Tag**: 128 bits (16 bytes)
- **Key Derivation**: PBKDF2 with SHA-256, 100,000 iterations

### Key Management

The system uses a hierarchical key structure:

1. **Master Key**: Stored in secure key management system (AWS KMS, Vault, etc.)
2. **Data Encryption Keys (DEKs)**: Derived from master key using context
3. **Key Versioning**: Support for key rotation without data re-encryption

### Encrypted Data Types

#### Trading Data
- Account numbers
- Position sizes  
- Trade execution details
- P&L calculations
- Risk metrics

#### User Data  
- Email addresses
- Phone numbers
- Personal information (names, DOB)
- API credentials
- Notification settings

## Implementation

### Encrypted Field Types

AlphaPulse provides custom SQLAlchemy field types for transparent encryption:

```python
from alpha_pulse.models.encrypted_fields import (
    EncryptedString,
    EncryptedInteger,
    EncryptedFloat,
    EncryptedJSON,
    EncryptedSearchableString
)

class TradingAccount(Base):
    # Encrypted account number with search capability
    account_number = Column(EncryptedSearchableString())
    
    # Encrypted balance
    balance = Column(EncryptedFloat())
    
    # Encrypted JSON metadata
    metadata = Column(EncryptedJSON())
```

### Searchable Encryption

For fields that need exact-match queries (like email addresses), use searchable encryption:

```python
class User(Base):
    # Searchable encrypted email
    email = Column(EncryptedSearchableString())
    email_search = Column(SearchTokenIndex("email"))
```

This creates a deterministic search token that allows queries without decrypting all records.

### Performance Optimization

For bulk operations, use the optimized cipher:

```python
from alpha_pulse.utils.encryption import get_optimized_cipher

cipher = get_optimized_cipher()

# Batch encrypt
encrypted_items = cipher.batch_encrypt(items)

# Batch decrypt
decrypted_items = cipher.batch_decrypt(encrypted_items)
```

## Migration Guide

### Step 1: Add Encrypted Columns

```bash
python -m alpha_pulse.migrations.add_encryption --add-columns
```

This adds encrypted columns alongside existing unencrypted columns.

### Step 2: Migrate Data

```bash
python -m alpha_pulse.migrations.add_encryption --migrate-data --batch-size 1000
```

Options:
- `--batch-size`: Number of records to process at once (default: 1000)
- `--table`: Migrate only specific table

### Step 3: Verify Migration

```bash
python -m alpha_pulse.migrations.add_encryption --verify
```

This shows migration status and any failures.

### Step 4: Create Views (Optional)

```bash
python -m alpha_pulse.migrations.add_encryption --create-views
```

Creates database views that transparently use encrypted columns.

### Step 5: Drop Unencrypted Columns

⚠️ **WARNING**: This is irreversible!

```bash
python -m alpha_pulse.migrations.add_encryption --drop-unencrypted --confirm
```

## Configuration

### Environment Variables

```bash
# Master encryption key (generate with Fernet)
ALPHAPULSE_ENCRYPTION_MASTER_KEY=your-base64-encoded-key

# Key version (for rotation)
ALPHAPULSE_ENCRYPTION_KEY_VERSION=1
```

### Database Connection

The encryption system automatically configures the database connection:

```python
from alpha_pulse.config.database import get_db_session

# Session automatically handles encryption/decryption
session = get_db_session()
```

## Key Rotation

### Rotating Keys

```python
from alpha_pulse.utils.encryption import EncryptionKeyManager

manager = EncryptionKeyManager()
new_version = manager.rotate_keys()
```

### Migration After Rotation

Data encrypted with old keys remains readable. New data uses the new key version:

```python
# Old data still decrypts
old_record = session.query(Model).filter_by(id=1).first()

# New data uses new key
new_record = Model(sensitive_field="new data")
session.add(new_record)
```

## Performance Considerations

### Query Performance

- **Encrypted fields**: ~20% slower than unencrypted
- **Searchable fields**: Use index on search token for fast lookups
- **Batch operations**: Use `batch_encrypt`/`batch_decrypt` for better performance

### Memory Usage

- **Encryption overhead**: ~15% increase in memory usage
- **Key caching**: Reduces key derivation overhead
- **Connection pooling**: Configured for optimal performance

### Optimization Tips

1. **Use appropriate field types**: Don't encrypt non-sensitive data
2. **Batch operations**: Process multiple records together
3. **Index search tokens**: For searchable fields
4. **Connection pooling**: Use configured pool settings

## Security Best Practices

### Key Management

1. **Never hardcode keys**: Use environment variables or key management systems
2. **Rotate regularly**: Implement quarterly key rotation
3. **Separate keys**: Use different keys for different environments
4. **Audit access**: Log all key access operations

### Data Protection

1. **Minimize scope**: Only encrypt sensitive fields
2. **Use contexts**: Different encryption contexts for different data types
3. **Validate integrity**: GCM mode provides built-in authentication
4. **Secure backups**: Ensure backups are also encrypted

### Application Security

1. **Input validation**: Validate before encryption
2. **Output masking**: Mask sensitive data in logs/UI
3. **Access control**: Implement role-based access
4. **Audit logging**: Log all access to encrypted data

## Troubleshooting

### Common Issues

#### Decryption Errors

```
DecryptionError: Failed to decrypt data
```

**Causes**:
- Wrong key version
- Corrupted data
- Tampered ciphertext

**Solution**:
- Check key version in metadata
- Verify data integrity
- Check audit logs

#### Performance Issues

```
Slow query performance on encrypted fields
```

**Solution**:
- Add indexes on search tokens
- Use batch operations
- Increase connection pool size

#### Migration Failures

```
Migration failed for table.column
```

**Solution**:
- Check error in `encryption_migrations` table
- Retry with smaller batch size
- Verify source data validity

### Debug Mode

Enable encryption debug logging:

```python
import logging
logging.getLogger("alpha_pulse.utils.encryption").setLevel(logging.DEBUG)
```

## Compliance

The encryption implementation meets requirements for:

- **PCI DSS**: Requirement 3.4 (strong cryptography)
- **GDPR**: Article 32 (appropriate technical measures)
- **SOX**: Section 404 (internal controls)
- **HIPAA**: If handling health data

## API Reference

### Encryption Functions

```python
# Encrypt a field
encrypted = encrypt_field(value, searchable=False, context="default")

# Decrypt a field  
decrypted = decrypt_field(encrypted_data)

# Get cipher instances
cipher = get_cipher()
searchable = get_searchable_cipher()
optimized = get_optimized_cipher()
```

### Field Types

- `EncryptedString`: Encrypted string field
- `EncryptedInteger`: Encrypted integer field
- `EncryptedFloat`: Encrypted float field
- `EncryptedJSON`: Encrypted JSON field
- `EncryptedSearchableString`: Searchable encrypted string
- `SearchTokenIndex`: Index field for search tokens

### Migration Tools

```bash
# Full migration workflow
python -m alpha_pulse.migrations.add_encryption --add-columns
python -m alpha_pulse.migrations.add_encryption --migrate-data
python -m alpha_pulse.migrations.add_encryption --verify
python -m alpha_pulse.migrations.add_encryption --create-views
```