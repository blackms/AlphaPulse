#!/bin/bash
set -e

# Script to initialize Vault HA cluster
# Usage: ./scripts/init-vault-cluster.sh

VAULT_ADDR="http://localhost:8200"
VAULT_TOKEN_FILE=".vault-token"
VAULT_KEYS_FILE=".vault-keys.json"

echo "=== Initializing Vault Cluster ==="
echo ""

# Check if Vault is running
if ! curl -s -f "$VAULT_ADDR/v1/sys/health" > /dev/null 2>&1; then
  echo "❌ Error: Vault is not running at $VAULT_ADDR"
  echo "   Please start Vault with: docker-compose -f docker-compose.vault.yml up -d"
  exit 1
fi

# Check if Vault is already initialized
if vault status -address="$VAULT_ADDR" 2>&1 | grep -q "Initialized.*true"; then
  echo "✅ Vault already initialized"
  echo ""

  if [ -f "$VAULT_TOKEN_FILE" ]; then
    echo "Root token found at: $VAULT_TOKEN_FILE"
    echo "Export token with: export VAULT_TOKEN=\$(cat $VAULT_TOKEN_FILE)"
  else
    echo "⚠️  Root token file not found. You may need to use a recovery key or existing token."
  fi

  exit 0
fi

# Initialize Vault (node 1 only)
echo "Initializing Vault node 1..."
vault operator init \
  -address="$VAULT_ADDR" \
  -key-shares=5 \
  -key-threshold=3 \
  -format=json > "$VAULT_KEYS_FILE"

if [ $? -ne 0 ]; then
  echo "❌ Error: Vault initialization failed"
  exit 1
fi

# Extract root token
ROOT_TOKEN=$(cat "$VAULT_KEYS_FILE" | jq -r '.root_token')
echo "$ROOT_TOKEN" > "$VAULT_TOKEN_FILE"
chmod 600 "$VAULT_TOKEN_FILE"
chmod 600 "$VAULT_KEYS_FILE"

echo "✅ Vault initialized successfully"
echo "Root token saved to: $VAULT_TOKEN_FILE"
echo "Unseal keys saved to: $VAULT_KEYS_FILE"
echo ""
echo "⚠️  IMPORTANT: Store unseal keys in a secure location!"
echo "   In production, use AWS Secrets Manager, 1Password, or similar"
echo ""

# Unseal vault-1 (if not using auto-unseal)
if vault status -address="$VAULT_ADDR" 2>&1 | grep -q "Sealed.*true"; then
  echo "Unsealing Vault node 1..."
  for i in 0 1 2; do
    KEY=$(cat "$VAULT_KEYS_FILE" | jq -r ".unseal_keys_b64[$i]")
    vault operator unseal -address="$VAULT_ADDR" "$KEY" > /dev/null
  done
  echo "✅ Vault node 1 unsealed"
fi

# Export token for subsequent commands
export VAULT_TOKEN="$ROOT_TOKEN"

# Wait for node 1 to be ready
echo "Waiting for Vault node 1 to be ready..."
sleep 2

# Join nodes 2 and 3 to Raft cluster
echo ""
echo "Joining Vault node 2 to cluster..."
vault operator raft join \
  -address="http://localhost:8202" \
  -leader-api-addr="$VAULT_ADDR" \
  -leader-ca-cert="" \
  -leader-client-cert="" \
  -leader-client-key="" || true

# Unseal node 2
echo "Unsealing Vault node 2..."
for i in 0 1 2; do
  KEY=$(cat "$VAULT_KEYS_FILE" | jq -r ".unseal_keys_b64[$i]")
  vault operator unseal -address="http://localhost:8202" "$KEY" > /dev/null
done
echo "✅ Vault node 2 joined and unsealed"

echo ""
echo "Joining Vault node 3 to cluster..."
vault operator raft join \
  -address="http://localhost:8204" \
  -leader-api-addr="$VAULT_ADDR" \
  -leader-ca-cert="" \
  -leader-client-cert="" \
  -leader-client-key="" || true

# Unseal node 3
echo "Unsealing Vault node 3..."
for i in 0 1 2; do
  KEY=$(cat "$VAULT_KEYS_FILE" | jq -r ".unseal_keys_b64[$i]")
  vault operator unseal -address="http://localhost:8204" "$KEY" > /dev/null
done
echo "✅ Vault node 3 joined and unsealed"

# Wait for cluster to stabilize
echo ""
echo "Waiting for cluster to stabilize..."
sleep 3

# Enable audit logging
echo "Enabling audit logging..."
vault audit enable -address="$VAULT_ADDR" file file_path=/vault/logs/audit.log || echo "⚠️  Audit logging may already be enabled"
echo "✅ Audit logging enabled"

# Enable KV v2 secrets engine
echo "Enabling KV v2 secrets engine..."
vault secrets enable -address="$VAULT_ADDR" -path=secret kv-v2 || echo "⚠️  KV v2 may already be enabled"
echo "✅ KV v2 secrets engine enabled at path: secret/"

# Create initial tenant isolation policy
echo "Creating tenant isolation policy template..."
cat > /tmp/tenant-policy.hcl <<'EOF'
# Tenant-specific access policy
# This policy allows access to tenant-specific secrets only
# Path pattern: secret/data/tenants/{{identity.entity.metadata.tenant_id}}/*

path "secret/data/tenants/{{identity.entity.metadata.tenant_id}}/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/tenants/{{identity.entity.metadata.tenant_id}}/*" {
  capabilities = ["list", "read"]
}

# Deny access to other tenants' secrets
path "secret/data/tenants/*" {
  capabilities = ["deny"]
}
EOF

vault policy write -address="$VAULT_ADDR" tenant-access /tmp/tenant-policy.hcl
rm /tmp/tenant-policy.hcl
echo "✅ Tenant isolation policy created"

# Display cluster status
echo ""
echo "=== Vault Cluster Status ==="
vault operator raft list-peers -address="$VAULT_ADDR"

echo ""
echo "=== Node Health Status ==="
echo "Node 1: $(vault status -address='http://localhost:8200' -format=json 2>/dev/null | jq -r 'if .sealed then "SEALED" else "UNSEALED" end')"
echo "Node 2: $(vault status -address='http://localhost:8202' -format=json 2>/dev/null | jq -r 'if .sealed then "SEALED" else "UNSEALED" end')"
echo "Node 3: $(vault status -address='http://localhost:8204' -format=json 2>/dev/null | jq -r 'if .sealed then "SEALED" else "UNSEALED" end')"

echo ""
echo "✅ Vault cluster initialization complete!"
echo ""
echo "📝 Next steps:"
echo "1. Export token: export VAULT_TOKEN=\$(cat $VAULT_TOKEN_FILE)"
echo "2. Test access: vault kv put secret/test/demo foo=bar"
echo "3. Verify health: ./scripts/vault-health-check.sh"
echo "4. Store unseal keys securely (delete $VAULT_KEYS_FILE after backup)"
echo ""
