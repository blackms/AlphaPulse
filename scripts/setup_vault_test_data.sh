#!/bin/bash
#
# Setup Vault Test Data
#
# Populates Vault with test secrets for load testing.
#
# Usage:
#   export VAULT_ADDR=http://localhost:8200
#   export VAULT_TOKEN=root
#   ./scripts/setup_vault_test_data.sh
#

set -e

VAULT_ADDR=${VAULT_ADDR:-http://localhost:8200}
VAULT_TOKEN=${VAULT_TOKEN:-root}
NUM_TENANTS=${NUM_TENANTS:-1000}

echo "=== Vault Test Data Setup ==="
echo "Vault URL: $VAULT_ADDR"
echo "Number of tenants: $NUM_TENANTS"
echo ""

# Check Vault is accessible
echo "Checking Vault health..."
vault status > /dev/null 2>&1 || {
    echo "ERROR: Vault is not accessible at $VAULT_ADDR"
    echo "Please ensure Vault is running and unsealed."
    exit 1
}
echo "✓ Vault is healthy"

# Enable KV v2 secrets engine if not already enabled
echo "Enabling KV v2 secrets engine..."
vault secrets enable -path=secret kv-v2 2>/dev/null || echo "  (already enabled)"
echo "✓ KV v2 secrets engine ready"

# Create test secrets for each tenant
echo "Creating test secrets..."
echo "  (This may take a few minutes for $NUM_TENANTS tenants)"

START_TIME=$(date +%s)

for i in $(seq 1 $NUM_TENANTS); do
    TENANT_ID="tenant-$i"

    # Create secrets for 3 exchanges per tenant
    for EXCHANGE in binance coinbase kraken; do
        vault kv put "secret/tenants/$TENANT_ID/exchanges/$EXCHANGE" \
            api_key="test_key_${TENANT_ID}_${EXCHANGE}" \
            secret="test_secret_${TENANT_ID}_${EXCHANGE}" \
            permissions="trading" \
            created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            > /dev/null
    done

    # Progress indicator
    if [ $((i % 100)) -eq 0 ]; then
        echo "  Created secrets for $i/$NUM_TENANTS tenants..."
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "✓ Created $((NUM_TENANTS * 3)) secrets in ${ELAPSED}s"
echo ""

# Verify a few secrets
echo "Verifying secrets..."
for TENANT_ID in tenant-1 tenant-500 tenant-1000; do
    vault kv get "secret/tenants/$TENANT_ID/exchanges/binance" > /dev/null && \
        echo "  ✓ $TENANT_ID/binance exists" || \
        echo "  ✗ $TENANT_ID/binance missing"
done

echo ""
echo "=== Setup Complete ==="
echo "Ready for load testing:"
echo "  k6 run --vus 100 --duration 60s scripts/load_test_vault.js"
echo ""
