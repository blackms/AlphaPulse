# Vault Node 3 Configuration
# AlphaPulse Multi-Tenant Credential Management

# Storage: Raft integrated storage
storage "raft" {
  path    = "/vault/data"
  node_id = "vault-3"

  # Retry join configuration (auto-discover peers)
  retry_join {
    leader_api_addr = "http://vault-1:8200"
  }
  retry_join {
    leader_api_addr = "http://vault-2:8200"
  }
  retry_join {
    leader_api_addr = "http://vault-3:8200"
  }
}

# API listener (HTTP for dev, HTTPS for prod)
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_disable   = 1  # Development only! Enable TLS in production
}

# Cluster communication listener (Raft replication)
listener "tcp" {
  address       = "0.0.0.0:8201"
  tls_disable   = 1  # Development only!
}

# Cluster addresses (unique per node)
cluster_addr  = "http://vault-3:8201"
api_addr      = "http://vault-3:8200"

# Auto-unseal with AWS KMS (production)
# Uncomment for production deployment:
# seal "awskms" {
#   region     = "us-east-1"
#   kms_key_id = "alias/alphapulse-vault-unseal"
#   endpoint   = "https://kms.us-east-1.amazonaws.com"
# }

# Telemetry: Prometheus metrics
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname          = true
  unauthenticated_metrics_access = true  # Development only!
}

# UI (optional for development)
ui = true

# Logging
log_level = "info"
log_format = "json"

# Disable memory locking (development only)
disable_mlock = true
