# HashiCorp Vault Training - AlphaPulse Team

**Duration**: 2 hours
**Date**: Sprint 4, Day 3 (2025-10-30)
**Instructor**: DevOps Engineer / Security Lead
**Audience**: Backend Engineers, DevOps Engineers

---

## Training Objectives

By the end of this training, you will be able to:
1. Understand Vault architecture and core concepts
2. Store and retrieve secrets from Vault
3. Create tenant-scoped policies (multi-tenant isolation)
4. Authenticate applications with JWT/AppRole
5. Audit and monitor Vault usage

---

## Part 1: Vault Fundamentals (30 minutes)

### 1.1 What is Vault?

HashiCorp Vault is a secrets management tool that provides:
- **Secrets Storage**: Store API keys, passwords, certificates
- **Dynamic Secrets**: Generate short-lived credentials (AWS, Database)
- **Encryption as a Service**: Encrypt/decrypt data without storing it
- **Lease Management**: Automatic secret rotation and revocation
- **Audit Logging**: Complete audit trail of all secret access

**Why Vault for AlphaPulse?**
- **Multi-tenancy**: Tenant-scoped policies prevent cross-tenant access
- **Security**: Secrets never in source code or environment variables
- **Compliance**: Audit logs for SOC2, GDPR compliance
- **Dynamic secrets**: Database credentials auto-rotate

---

### 1.2 Vault Architecture

```
┌────────────────────────────────────────────────────┐
│                 Vault Architecture                  │
├────────────────────────────────────────────────────┤
│  Clients (API, Workers)                            │
│  ├─ Authentication (JWT, AppRole)                  │
│  └─ HTTP API (port 8200)                          │
├────────────────────────────────────────────────────┤
│  Vault Core                                        │
│  ├─ Secrets Engines (KV v2, Database, AWS)        │
│  ├─ Auth Methods (JWT, AppRole, Kubernetes)       │
│  ├─ Audit Devices (file, syslog)                  │
│  └─ Policy Engine (tenant-scoped access)          │
├────────────────────────────────────────────────────┤
│  Storage Backend (Raft, Consul, S3)               │
│  └─ Encrypted data-at-rest                        │
└────────────────────────────────────────────────────┘
```

**Key Concepts**:
- **Seal/Unseal**: Vault starts sealed (encrypted), must be unsealed with keys
- **Root Token**: Initial admin token (never use in production)
- **Paths**: Secrets organized by path (e.g., `secret/data/tenant1/api-keys`)
- **Policies**: Define what paths a token can access
- **Leases**: Secrets have TTL (time-to-live), auto-expire

---

### 1.3 Install Vault CLI

**macOS**:
```bash
brew install vault
vault version
```

**Linux**:
```bash
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install vault
```

**Windows**:
```powershell
choco install vault
```

---

## Part 2: Vault Basics (30 minutes)

### 2.1 Start Vault Dev Server

**Dev mode** (in-memory, auto-unsealed, NOT for production):
```bash
# Start dev server
vault server -dev

# In another terminal, set environment variables
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='hvs.XXXXXXXXXXXXXXXXXXXX'  # Copy from server output

# Verify connection
vault status
```

**Expected output**:
```
Key             Value
---             -----
Seal Type       shamir
Initialized     true
Sealed          false
Total Shares    1
Threshold       1
Version         1.15.0
```

---

### 2.2 Write and Read Secrets (KV v2)

**Enable KV v2 secrets engine**:
```bash
# Check enabled secrets engines
vault secrets list

# Enable KV v2 at path 'secret/' (usually enabled by default)
vault secrets enable -path=secret kv-v2
```

**Write secrets**:
```bash
# Write secret (key-value pairs)
vault kv put secret/alphapulse/openai-api-key \
  api_key="sk-XXXXXXXXXXXXXXXXXXXX"

# Write multiple key-value pairs
vault kv put secret/alphapulse/database \
  username="alphapulse" \
  password="super_secret_password" \
  host="db.alphapulse.ai" \
  port="5432"

# Verification
vault kv get secret/alphapulse/database
```

**Expected output**:
```
====== Data ======
Key         Value
---         -----
host        db.alphapulse.ai
password    super_secret_password
port        5432
username    alphapulse
```

**Read secrets** (JSON output):
```bash
# Read secret as JSON
vault kv get -format=json secret/alphapulse/database | jq .data.data

# Read specific field
vault kv get -field=password secret/alphapulse/database
```

---

### 2.3 Secret Versioning

KV v2 automatically versions secrets (unlike KV v1):

```bash
# Update secret (creates version 2)
vault kv put secret/alphapulse/database \
  username="alphapulse" \
  password="new_password_v2" \
  host="db.alphapulse.ai" \
  port="5432"

# Read latest version (version 2)
vault kv get secret/alphapulse/database

# Read specific version (version 1)
vault kv get -version=1 secret/alphapulse/database

# View metadata (all versions)
vault kv metadata get secret/alphapulse/database

# Rollback to version 1
vault kv rollback -version=1 secret/alphapulse/database
```

**Delete secret**:
```bash
# Soft delete (can be undeleted)
vault kv delete secret/alphapulse/database

# Undelete
vault kv undelete -versions=2 secret/alphapulse/database

# Permanent delete (cannot be recovered)
vault kv destroy -versions=2 secret/alphapulse/database

# Delete all versions
vault kv metadata delete secret/alphapulse/database
```

---

## Part 3: Multi-Tenant Policies (45 minutes)

### 3.1 Policy Basics

**Policy HCL** (HashiCorp Configuration Language):
```hcl
# Syntax
path "secret/data/<path>" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
```

**Capabilities**:
- `create` - Create new secrets
- `read` - Read existing secrets
- `update` - Update existing secrets
- `delete` - Delete secrets
- `list` - List secret paths
- `sudo` - Administrative operations (e.g., delete policy)

---

### 3.2 Tenant-Scoped Policies

**Create policy for Tenant 1**:
```bash
# Create policy file: tenant1-policy.hcl
cat > tenant1-policy.hcl <<EOF
# Allow full access to tenant1 secrets
path "secret/data/tenant1/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/tenant1/*" {
  capabilities = ["list", "read"]
}

# Deny access to other tenants
path "secret/data/tenant2/*" {
  capabilities = ["deny"]
}

path "secret/data/tenant3/*" {
  capabilities = ["deny"]
}
EOF

# Write policy to Vault
vault policy write tenant1-policy tenant1-policy.hcl

# View policy
vault policy read tenant1-policy
```

**Create token with policy**:
```bash
# Create token with tenant1-policy
vault token create -policy=tenant1-policy -period=24h

# Copy token: s.XXXXXXXXXXXXXXXX
```

**Test policy** (in new terminal):
```bash
# Use tenant1 token
export VAULT_TOKEN='s.XXXXXXXXXXXXXXXX'

# Should succeed (tenant1 path)
vault kv put secret/tenant1/api-keys \
  exchange_api_key="tenant1_key"

vault kv get secret/tenant1/api-keys

# Should fail (tenant2 path - access denied)
vault kv get secret/tenant2/api-keys
# Error: permission denied
```

---

### 3.3 Templated Policies (Advanced)

Use templated policies for dynamic tenant ID injection:

```hcl
# templated-tenant-policy.hcl
path "secret/data/{{identity.entity.metadata.tenant_id}}/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/{{identity.entity.metadata.tenant_id}}/*" {
  capabilities = ["list", "read"]
}
```

This allows a single policy to work for all tenants by injecting `tenant_id` from JWT claims.

---

## Part 4: Authentication (30 minutes)

### 4.1 JWT Authentication (for API)

**Enable JWT auth**:
```bash
# Enable JWT auth method
vault auth enable jwt

# Configure JWT auth with JWKS URL (or public key)
vault write auth/jwt/config \
  jwks_url="https://api.alphapulse.ai/.well-known/jwks.json" \
  default_role="api-role"

# Create role
vault write auth/jwt/role/api-role \
  role_type="jwt" \
  bound_audiences="alphapulse-api" \
  user_claim="sub" \
  policies="tenant1-policy" \
  ttl="1h"
```

**Login with JWT**:
```bash
# Login (replace JWT with actual token from API)
vault write auth/jwt/login \
  role="api-role" \
  jwt="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."

# Response includes Vault token
```

**Python example**:
```python
import hvac

# Initialize Vault client
client = hvac.Client(url='http://127.0.0.1:8200')

# Login with JWT
jwt_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
response = client.auth.jwt.jwt_login(
    role='api-role',
    jwt=jwt_token
)

# Set client token
client.token = response['auth']['client_token']

# Read secret
secret = client.secrets.kv.v2.read_secret_version(
    path='tenant1/api-keys'
)
print(secret['data']['data']['exchange_api_key'])
```

---

### 4.2 AppRole Authentication (for Workers)

**Enable AppRole auth**:
```bash
# Enable AppRole auth method
vault auth enable approle

# Create role
vault write auth/approle/role/worker-role \
  secret_id_ttl=24h \
  token_ttl=1h \
  token_max_ttl=24h \
  policies="tenant1-policy,tenant2-policy"

# Get RoleID
vault read auth/approle/role/worker-role/role-id
# role_id: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Generate SecretID
vault write -f auth/approle/role/worker-role/secret-id
# secret_id: yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy
```

**Login with AppRole**:
```bash
# Login (use RoleID and SecretID)
vault write auth/approle/login \
  role_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" \
  secret_id="yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"

# Response includes Vault token
```

**Python example**:
```python
import hvac

# Initialize Vault client
client = hvac.Client(url='http://127.0.0.1:8200')

# Login with AppRole
response = client.auth.approle.login(
    role_id='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',
    secret_id='yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy'
)

# Set client token
client.token = response['auth']['client_token']

# Read secret
secret = client.secrets.kv.v2.read_secret_version(
    path='tenant1/api-keys'
)
print(secret['data']['data'])
```

---

## Part 5: Audit and Monitoring (15 minutes)

### 5.1 Enable Audit Logging

```bash
# Enable file audit device
vault audit enable file file_path=/tmp/vault-audit.log

# View audit log
tail -f /tmp/vault-audit.log | jq .
```

**Audit log entry** (JSON):
```json
{
  "time": "2025-10-30T10:15:30.123Z",
  "type": "response",
  "auth": {
    "client_token": "hmac-sha256:...",
    "accessor": "hmac-sha256:...",
    "display_name": "jwt-user@tenant1.com",
    "policies": ["default", "tenant1-policy"],
    "metadata": {
      "tenant_id": "tenant1",
      "user_id": "user123"
    }
  },
  "request": {
    "operation": "read",
    "path": "secret/data/tenant1/api-keys"
  },
  "response": {
    "data": {
      "exchange_api_key": "hmac-sha256:..."  # Sensitive data is HMAC'd
    }
  }
}
```

**Key points**:
- All sensitive data is HMAC'd (not plaintext)
- Audit log records every read/write
- Cannot be disabled once enabled (security feature)

---

### 5.2 Monitoring Vault

**Health check**:
```bash
curl http://127.0.0.1:8200/v1/sys/health
```

**Metrics** (Prometheus format):
```bash
curl http://127.0.0.1:8200/v1/sys/metrics?format=prometheus
```

**Key metrics**:
- `vault_core_unsealed` - Vault seal status (1 = unsealed, 0 = sealed)
- `vault_token_count` - Active tokens
- `vault_secret_kv_count` - Secrets stored
- `vault_audit_log_request_duration` - Audit log latency

---

## Part 6: Hands-On Exercise (15 minutes)

### Exercise: Multi-Tenant Secret Management

**Scenario**: You are setting up Vault for 3 tenants. Each tenant should only access their own secrets.

**Tasks**:
1. Create secrets for tenant1, tenant2, tenant3
2. Create tenant-scoped policies for each tenant
3. Create tokens with each policy
4. Verify tenant isolation (tenant1 cannot read tenant2 secrets)

**Solution**:

```bash
# 1. Create secrets for each tenant
vault kv put secret/tenant1/api-keys exchange_api_key="tenant1_key"
vault kv put secret/tenant2/api-keys exchange_api_key="tenant2_key"
vault kv put secret/tenant3/api-keys exchange_api_key="tenant3_key"

# 2. Create policies
cat > tenant1-policy.hcl <<EOF
path "secret/data/tenant1/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
EOF

vault policy write tenant1-policy tenant1-policy.hcl
# Repeat for tenant2, tenant3

# 3. Create tokens
vault token create -policy=tenant1-policy -period=24h
# Save token: s.TENANT1_TOKEN

# 4. Test isolation
export VAULT_TOKEN='s.TENANT1_TOKEN'
vault kv get secret/tenant1/api-keys  # ✅ Should succeed
vault kv get secret/tenant2/api-keys  # ❌ Should fail (permission denied)
```

---

## Wrap-Up and Q&A (10 minutes)

### Key Takeaways

1. **Vault stores secrets securely**: Never hardcode secrets in code
2. **Tenant-scoped policies**: Essential for multi-tenant security
3. **JWT/AppRole auth**: Applications authenticate to get Vault token
4. **Audit logging**: Complete audit trail for compliance
5. **Secret versioning**: KV v2 tracks all changes, rollback supported

### Next Steps

- **Practice**: Store AlphaPulse secrets in Vault locally
- **Read docs**: https://www.vaultproject.io/docs/
- **Integrate**: Update AlphaPulse API to read secrets from Vault
- **Production setup**: Vault HA (3 replicas) with Raft backend

### Useful Resources

- [Vault Documentation](https://www.vaultproject.io/docs/)
- [Vault Policies](https://www.vaultproject.io/docs/concepts/policies)
- [Vault Auth Methods](https://www.vaultproject.io/docs/auth/)
- [Python hvac library](https://hvac.readthedocs.io/)

---

## Clean-Up

```bash
# Stop dev server (Ctrl+C)

# Delete audit log
rm /tmp/vault-audit.log
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
