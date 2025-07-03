# Encryption Migration Guide

## Overview

This guide walks through migrating existing AlphaPulse data to use field-level encryption. The migration process is designed to be safe, reversible, and performant.

## Pre-Migration Checklist

- [ ] **Backup your database** - Complete backup before starting
- [ ] **Test in staging** - Run full migration in staging environment first  
- [ ] **Plan downtime** - Migration requires maintenance window
- [ ] **Configure encryption keys** - Set up key management system
- [ ] **Review affected tables** - Understand which data will be encrypted
- [ ] **Allocate resources** - Ensure sufficient disk space (2x current size)

## Migration Steps

### Step 1: Prepare Environment

1. **Set up encryption keys**:
```bash
# Generate master key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set environment variable
export ALPHAPULSE_ENCRYPTION_MASTER_KEY="your-generated-key"
```

2. **Configure key storage**:

For AWS:
```bash
aws secretsmanager create-secret \
  --name alphapulse/encryption/master_key \
  --secret-string "$ALPHAPULSE_ENCRYPTION_MASTER_KEY"
```

For Vault:
```bash
vault kv put alphapulse/encryption/master_key \
  value="$ALPHAPULSE_ENCRYPTION_MASTER_KEY"
```

### Step 2: Add Encrypted Columns

Add encrypted columns alongside existing ones:

```bash
python -m alpha_pulse.migrations.add_encryption --add-columns
```

This creates:
- `column_name_encrypted` for each sensitive column
- Search token columns for searchable fields
- Indexes for performance

Expected output:
```
Adding encrypted columns to tables...
✓ Added column trading_accounts.account_number_encrypted
✓ Added column trading_accounts.balance_encrypted
✓ Added column users.email_encrypted
✓ Created index idx_user_email_search
...
```

### Step 3: Migrate Data

Start the data migration:

```bash
# Full migration
python -m alpha_pulse.migrations.add_encryption --migrate-data

# Or migrate specific table
python -m alpha_pulse.migrations.add_encryption --migrate-data --table users

# Or with custom batch size
python -m alpha_pulse.migrations.add_encryption --migrate-data --batch-size 500
```

Monitor progress:
```bash
# In another terminal
watch -n 5 'psql -c "SELECT * FROM encryption_migrations ORDER BY table_name"'
```

### Step 4: Verify Migration

Verify all data migrated successfully:

```bash
python -m alpha_pulse.migrations.add_encryption --verify
```

Expected output:
```
Migration verification results:
  Total tables: 5
  Total columns: 23
  Successful: 23
  Failed: 0
```

### Step 5: Test Application

1. **Update application code** to use encrypted fields:
```python
# Old code
user = session.query(User).filter_by(email="user@example.com").first()

# New code  
from alpha_pulse.utils.encryption import get_searchable_cipher
cipher = get_searchable_cipher()
search_token = cipher.generate_search_token("user@example.com", "user_pii")
user = session.query(User).filter_by(email_search=search_token).first()
```

2. **Run integration tests**:
```bash
pytest src/alpha_pulse/tests/test_encryption.py -v
```

3. **Verify functionality**:
- User login/registration
- Trading operations
- API endpoints
- Report generation

### Step 6: Create Database Views (Optional)

For backward compatibility, create views using encrypted columns:

```bash
python -m alpha_pulse.migrations.add_encryption --create-views
```

This allows gradual code migration.

### Step 7: Switch to Encrypted Columns

1. **Deploy updated application** that uses encrypted fields

2. **Monitor for issues**:
```bash
# Check application logs
tail -f logs/alphapulse.log | grep -E "(encrypt|decrypt|ERROR)"

# Monitor performance
psql -c "SELECT query, mean_exec_time FROM pg_stat_statements 
         WHERE query LIKE '%encrypted%' ORDER BY mean_exec_time DESC"
```

### Step 8: Drop Unencrypted Columns

⚠️ **CAUTION**: This step is irreversible!

After confirming everything works:

```bash
# Final backup
pg_dump alphapulse > backup_before_drop.sql

# Drop unencrypted columns
python -m alpha_pulse.migrations.add_encryption --drop-unencrypted --confirm
```

## Rollback Procedures

If issues arise during migration:

### Before Dropping Columns

1. **Stop application**
2. **Revert code** to use unencrypted columns
3. **Keep encrypted columns** for future retry

### After Dropping Columns

1. **Restore from backup**:
```bash
psql alphapulse < backup_before_drop.sql
```

2. **Revert application code**
3. **Investigate issues** before retry

## Performance Tuning

### During Migration

```sql
-- Increase work memory for migration
SET work_mem = '256MB';

-- Disable autovacuum temporarily
ALTER TABLE trading_accounts SET (autovacuum_enabled = false);

-- Re-enable after migration
ALTER TABLE trading_accounts SET (autovacuum_enabled = true);
```

### Post-Migration

1. **Update statistics**:
```sql
ANALYZE trading_accounts;
ANALYZE users;
```

2. **Rebuild indexes**:
```sql
REINDEX TABLE trading_accounts;
REINDEX TABLE users;
```

3. **Monitor query performance**:
```sql
-- Find slow encrypted queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
WHERE query LIKE '%_encrypted%'
ORDER BY mean_exec_time DESC
LIMIT 20;
```

## Common Issues

### Issue: Migration Too Slow

**Solution**:
- Reduce batch size
- Run during low-traffic period
- Migrate tables in parallel

### Issue: Disk Space

**Solution**:
- Encrypted data is ~30% larger
- Ensure 2x current database size available
- Consider migrating largest tables separately

### Issue: Memory Errors

**Solution**:
```bash
# Increase memory for migration
export PYTHONPATH=/path/to/alphapulse
python -m alpha_pulse.migrations.add_encryption --migrate-data --batch-size 100
```

### Issue: Key Access Errors

**Solution**:
- Verify key permissions
- Check environment variables
- Test key access separately

## Validation Scripts

### Verify Encryption Working

```python
# test_encryption_live.py
from alpha_pulse.config.database import get_db_session
from alpha_pulse.models.user_data import User

session = get_db_session()

# Test creating encrypted record
user = User(
    username="test_encrypt",
    email="test@example.com",
    password_hash="dummy"
)
session.add(user)
session.commit()

# Test retrieving
retrieved = session.query(User).filter_by(username="test_encrypt").first()
print(f"Email decrypted correctly: {retrieved.email == 'test@example.com'}")

# Check raw encrypted data
raw = session.execute(
    "SELECT email_encrypted FROM users WHERE username='test_encrypt'"
).first()
print(f"Data is encrypted: {raw[0].startswith('{')}")

# Cleanup
session.delete(retrieved)
session.commit()
```

### Performance Comparison

```python
# benchmark_encryption.py
import time
from alpha_pulse.config.database import get_db_session

session = get_db_session()

# Benchmark encrypted query
start = time.time()
users = session.execute(
    "SELECT * FROM users WHERE email_encrypted IS NOT NULL LIMIT 1000"
).fetchall()
encrypted_time = time.time() - start

print(f"Encrypted query time: {encrypted_time:.3f}s for {len(users)} records")
print(f"Average per record: {(encrypted_time/len(users)*1000):.2f}ms")
```

## Post-Migration Tasks

1. **Update documentation**
2. **Train team on encrypted fields**
3. **Set up key rotation schedule**
4. **Configure monitoring alerts**
5. **Document any custom procedures**

## Support

For migration assistance:
- Check logs in `logs/migration/`
- Review [Database Encryption Guide](./database-encryption.md)
- Contact: dba@alphapulse.io