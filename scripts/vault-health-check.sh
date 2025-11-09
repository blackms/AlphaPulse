#!/bin/bash
set -e

# Health check script for Vault cluster
# Usage: ./scripts/vault-health-check.sh

VAULT_NODES=(
  "http://localhost:8200"
  "http://localhost:8202"
  "http://localhost:8204"
)

NODE_NAMES=("vault-1" "vault-2" "vault-3")

echo "=== Vault Cluster Health Check ==="
echo ""

ALL_HEALTHY=true
LEADER_COUNT=0

for i in "${!VAULT_NODES[@]}"; do
  ADDR="${VAULT_NODES[$i]}"
  NAME="${NODE_NAMES[$i]}"

  echo "Checking $NAME ($ADDR)..."

  # Check if node is reachable
  if ! curl -s -f "$ADDR/v1/sys/health" > /dev/null 2>&1; then
    echo "  ❌ Node unreachable"
    ALL_HEALTHY=false
    echo ""
    continue
  fi

  # Get detailed status
  STATUS=$(vault status -address="$ADDR" -format=json 2>/dev/null || echo '{}')

  if [ -z "$STATUS" ] || [ "$STATUS" = "{}" ]; then
    echo "  ❌ Failed to get status"
    ALL_HEALTHY=false
    echo ""
    continue
  fi

  INITIALIZED=$(echo "$STATUS" | jq -r '.initialized // false')
  SEALED=$(echo "$STATUS" | jq -r '.sealed // true')
  HA_ENABLED=$(echo "$STATUS" | jq -r '.ha_enabled // false')
  IS_LEADER=$(echo "$STATUS" | jq -r '.is_self // false')

  if [ "$INITIALIZED" = "true" ] && [ "$SEALED" = "false" ]; then
    if [ "$IS_LEADER" = "true" ]; then
      ROLE="LEADER"
      LEADER_COUNT=$((LEADER_COUNT + 1))
    else
      ROLE="FOLLOWER"
    fi
    echo "  ✅ Healthy ($ROLE)"
  else
    echo "  ❌ Unhealthy (Initialized: $INITIALIZED, Sealed: $SEALED)"
    ALL_HEALTHY=false
  fi

  echo ""
done

# Check Raft peers if token is available
if [ -n "$VAULT_TOKEN" ] || [ -f ".vault-token" ]; then
  if [ -z "$VAULT_TOKEN" ] && [ -f ".vault-token" ]; then
    export VAULT_TOKEN=$(cat .vault-token)
  fi

  echo "=== Raft Cluster Peers ==="
  if vault operator raft list-peers -address="${VAULT_NODES[0]}" 2>/dev/null; then
    echo ""
  else
    echo "⚠️  Unable to list Raft peers (token may be invalid or Vault not initialized)"
    echo ""
  fi
else
  echo "⚠️  VAULT_TOKEN not set, skipping Raft peer check"
  echo "   Set token with: export VAULT_TOKEN=\$(cat .vault-token)"
  echo ""
fi

# Validate leader count
if [ "$LEADER_COUNT" -eq 0 ]; then
  echo "❌ No leader elected (cluster may be initializing or split-brain)"
  ALL_HEALTHY=false
elif [ "$LEADER_COUNT" -gt 1 ]; then
  echo "❌ Multiple leaders detected ($LEADER_COUNT) - split-brain scenario!"
  ALL_HEALTHY=false
fi

# Final verdict
if [ "$ALL_HEALTHY" = "true" ]; then
  echo "✅ All nodes healthy"
  exit 0
else
  echo "❌ Some nodes unhealthy or cluster issues detected"
  exit 1
fi
