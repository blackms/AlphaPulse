# AlphaPulse Security Documentation

## Overview

AlphaPulse implements a comprehensive security architecture designed to protect sensitive trading data, API credentials, and user information. This document outlines the security features, best practices, and configuration guidelines.

## Secret Management

### Architecture

AlphaPulse uses a multi-layered secret management system that supports different providers based on the deployment environment:

1. **Development**: Environment variables with encrypted local file fallback
2. **Staging**: HashiCorp Vault with environment variable fallback
3. **Production**: AWS Secrets Manager with environment variable fallback

### Secret Types

The system manages several types of secrets:

- **Database Credentials**: PostgreSQL connection details
- **Exchange API Keys**: Binance, Bybit, Coinbase, Kraken credentials
- **Data Provider API Keys**: IEX Cloud, Polygon.io, Alpha Vantage, Finnhub
- **Security Keys**: JWT signing keys, AES encryption keys
- **User Credentials**: Hashed passwords, API tokens

### Configuration

#### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your actual values
```

Required environment variables:
- `ALPHAPULSE_ENV`: Environment (development/staging/production)
- `ALPHAPULSE_JWT_SECRET`: JWT signing secret (generate with `secrets.token_urlsafe(32)`)
- `ALPHAPULSE_ENCRYPTION_KEY`: AES encryption key (generate with Fernet)
- Database credentials
- Exchange API credentials
- Data provider API keys

#### AWS Secrets Manager (Production)

For production deployments:

1. Configure AWS credentials:
   ```bash
   export AWS_REGION=us-east-1
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

2. Store secrets in AWS Secrets Manager:
   ```python
   from alpha_pulse.utils.secrets_manager import AWSSecretsManagerProvider
   
   provider = AWSSecretsManagerProvider()
   provider.set_secret("database_credentials", {
       "host": "prod-db.example.com",
       "user": "alphapulse",
       "password": "secure_password"
   })
   ```

#### HashiCorp Vault (Staging)

For staging environments:

1. Configure Vault access:
   ```bash
   export VAULT_URL=https://vault.your-domain.com
   export VAULT_TOKEN=your_vault_token
   ```

2. Store secrets in Vault:
   ```bash
   vault kv put secret/alphapulse/database_credentials \
       host=staging-db.example.com \
       user=alphapulse \
       password=secure_password
   ```

### Migration from Hardcoded Credentials

Use the migration script to transition from hardcoded credentials:

```bash
# Create template
python scripts/migrate_secrets.py --create-template

# Migrate existing credentials
python scripts/migrate_secrets.py --method env --output .env.secure

# Or migrate to encrypted files
python scripts/migrate_secrets.py --method encrypted --secrets-dir .secrets
```

## Authentication & Authorization

### JWT Authentication

AlphaPulse uses JWT tokens for API authentication:

- **Algorithm**: HS256 (configurable)
- **Expiration**: 30 minutes (configurable)
- **Secret Rotation**: Supported through secret management system

### Password Security

- **Hashing**: bcrypt with configurable cost factor
- **Minimum Requirements**: 
  - Length: 12 characters
  - Complexity: Mixed case, numbers, special characters
- **Password History**: Prevents reuse of last 5 passwords

### User Roles & Permissions

Three default roles with granular permissions:

1. **Admin**: Full system access
   - All viewer and trader permissions
   - User management
   - System configuration
   
2. **Trader**: Trading operations
   - View metrics and alerts
   - Execute trades
   - Manage portfolio
   
3. **Viewer**: Read-only access
   - View metrics
   - View portfolio
   - View trades

### API Security

- **Rate Limiting**: Configurable per endpoint
- **IP Whitelisting**: Optional for production
- **Request Signing**: HMAC-SHA256 for sensitive operations
- **CORS**: Configurable allowed origins

## Data Protection

### Encryption at Rest

- **Database**: Transparent Data Encryption (TDE) for PostgreSQL
- **File Storage**: AES-256 encryption for sensitive files
- **Backups**: Encrypted with separate keys

### Encryption in Transit

- **TLS 1.3**: All API communications
- **Certificate Pinning**: Optional for mobile clients
- **Perfect Forward Secrecy**: Enabled by default

## Audit & Compliance

### Audit Logging

All security-relevant events are logged:

- Authentication attempts
- Secret access
- Trading operations
- Configuration changes
- API access

### Log Format

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "secret_access",
  "user": "admin",
  "resource": "binance_api_key",
  "action": "read",
  "result": "success",
  "ip_address": "192.168.1.100",
  "user_agent": "AlphaPulse/1.0"
}
```

### Compliance

AlphaPulse security architecture supports:

- **SOC 2 Type II**: Security controls
- **PCI DSS**: If processing payments
- **GDPR**: Data protection and privacy
- **ISO 27001**: Information security management

## Security Best Practices

### Development

1. Never commit secrets to version control
2. Use `.env.example` as template
3. Rotate development credentials regularly
4. Use separate credentials for each developer

### Deployment

1. Use different credentials for each environment
2. Implement secret rotation policies
3. Monitor secret access patterns
4. Regular security audits

### Incident Response

1. **Detection**: Monitor audit logs for anomalies
2. **Containment**: Immediate credential rotation
3. **Investigation**: Analyze audit trail
4. **Recovery**: Restore from secure backups
5. **Lessons Learned**: Update security policies

## Security Checklist

### Pre-deployment

- [ ] All hardcoded credentials removed
- [ ] Environment variables configured
- [ ] Secret management initialized
- [ ] SSL/TLS certificates valid
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Audit logging active

### Post-deployment

- [ ] Credentials rotated from defaults
- [ ] Monitoring alerts configured
- [ ] Backup encryption verified
- [ ] Access logs reviewed
- [ ] Penetration testing scheduled
- [ ] Security training completed

## Troubleshooting

### Common Issues

1. **Secret Not Found**
   ```
   Error: Secret 'database_credentials' not found
   ```
   - Check environment variables
   - Verify secret exists in provider
   - Check fallback providers

2. **Authentication Failed**
   ```
   Error: JWT validation failed
   ```
   - Verify JWT secret is configured
   - Check token expiration
   - Validate token format

3. **Permission Denied**
   ```
   Error: Insufficient permissions for operation
   ```
   - Check user role
   - Verify permission mapping
   - Review audit logs

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("alpha_pulse.utils.secrets_manager").setLevel(logging.DEBUG)
logging.getLogger("alpha_pulse.api.auth").setLevel(logging.DEBUG)
```

## Contact

For security issues or questions:
- Email: security@alphapulse.io
- Security Bug Bounty: https://alphapulse.io/security
- Emergency: Use PGP-encrypted email