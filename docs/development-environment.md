# Development Environment Setup Guide

**Date**: 2025-10-22
**Sprint**: 3 (Design & Alignment)
**Author**: Tech Lead
**Related**: [Architecture Review](architecture-review.md), [HLD](HLD-MULTI-TENANT-SAAS.md)
**Status**: Draft

---

## Purpose

This guide provides step-by-step instructions for setting up a local development environment for AlphaPulse multi-tenant SaaS platform. By the end of this guide, you will have:

- PostgreSQL 14+ with Row-Level Security (RLS) configured
- Redis 7+ Cluster (development mode)
- HashiCorp Vault (development mode)
- Python 3.11+ with Poetry
- All AlphaPulse services running locally
- Sample tenant data for testing

**Target Audience**: Backend engineers, DevOps engineers, QA engineers

**Estimated Setup Time**: 60-90 minutes

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [PostgreSQL Setup](#postgresql-setup)
3. [Redis Setup](#redis-setup)
4. [HashiCorp Vault Setup](#hashicorp-vault-setup)
5. [Python Environment Setup](#python-environment-setup)
6. [AlphaPulse Application Setup](#alphapulse-application-setup)
7. [Database Migrations](#database-migrations)
8. [Sample Data Setup](#sample-data-setup)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

### 1.1 Required Software

Before starting, ensure you have the following installed:

| Software | Minimum Version | Installation Check |
|----------|----------------|-------------------|
| macOS / Linux | macOS 12+ / Ubuntu 20.04+ | `uname -a` |
| Homebrew (macOS) | 4.0+ | `brew --version` |
| Python | 3.11+ | `python3 --version` |
| Poetry | 1.6+ | `poetry --version` |
| Git | 2.30+ | `git --version` |
| Docker (optional) | 24.0+ | `docker --version` |

### 1.2 Install Homebrew (macOS only)

```bash
# If not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.3 Install Poetry

```bash
# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

### 1.4 Clone Repository

```bash
git clone https://github.com/blackms/AlphaPulse.git
cd AlphaPulse
```

---

## 2. PostgreSQL Setup

### 2.1 Install PostgreSQL 14+

**macOS (Homebrew)**:
```bash
brew install postgresql@14

# Start PostgreSQL service
brew services start postgresql@14

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"
```

**Ubuntu/Debian**:
```bash
# Add PostgreSQL APT repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update

# Install PostgreSQL 14
sudo apt-get install -y postgresql-14 postgresql-contrib-14

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Docker (alternative)**:
```bash
docker run --name alphapulse-postgres \
  -e POSTGRES_USER=alphapulse \
  -e POSTGRES_PASSWORD=alphapulse_dev \
  -e POSTGRES_DB=alphapulse_dev \
  -p 5432:5432 \
  -d postgres:14
```

### 2.2 Create Database and User

```bash
# Connect to PostgreSQL (macOS)
psql postgres

# Or Ubuntu (switch to postgres user first)
sudo -u postgres psql
```

```sql
-- Create user
CREATE USER alphapulse WITH PASSWORD 'alphapulse_dev';

-- Create database
CREATE DATABASE alphapulse_dev OWNER alphapulse;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE alphapulse_dev TO alphapulse;

-- Enable UUID extension
\c alphapulse_dev
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Exit psql
\q
```

### 2.3 Verify Connection

```bash
# Test connection
psql -U alphapulse -d alphapulse_dev -h localhost

# Expected output: psql prompt
# alphapulse_dev=>

# Exit
\q
```

### 2.4 Configure PostgreSQL for Development

**Edit PostgreSQL configuration** (optional, for better local performance):

```bash
# macOS
vim /opt/homebrew/var/postgresql@14/postgresql.conf

# Ubuntu
sudo vim /etc/postgresql/14/main/postgresql.conf
```

**Recommended settings for development**:
```conf
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 16MB

# Connection settings
max_connections = 100

# Logging (useful for debugging)
log_statement = 'all'
log_duration = on
```

**Restart PostgreSQL**:
```bash
# macOS
brew services restart postgresql@14

# Ubuntu
sudo systemctl restart postgresql
```

---

## 3. Redis Setup

### 3.1 Install Redis 7+

**macOS (Homebrew)**:
```bash
brew install redis

# Start Redis service
brew services start redis
```

**Ubuntu/Debian**:
```bash
# Add Redis repository
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update

# Install Redis
sudo apt-get install -y redis

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Docker (alternative)**:
```bash
docker run --name alphapulse-redis \
  -p 6379:6379 \
  -d redis:7-alpine
```

### 3.2 Verify Redis

```bash
# Test connection
redis-cli ping
# Expected output: PONG

# Check version
redis-cli --version
# Expected: redis-cli 7.x.x
```

### 3.3 Configure Redis for Development

**Edit Redis configuration** (optional):

```bash
# macOS
vim /opt/homebrew/etc/redis.conf

# Ubuntu
sudo vim /etc/redis/redis.conf
```

**Recommended settings for development**:
```conf
# Bind to localhost only (security)
bind 127.0.0.1

# Disable protected mode for local dev
protected-mode no

# Memory limit (1GB for development)
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence (disable for faster dev, enable for data retention)
# save ""  # Disable RDB snapshots
# appendonly no  # Disable AOF
```

**Restart Redis**:
```bash
# macOS
brew services restart redis

# Ubuntu
sudo systemctl restart redis-server
```

---

## 4. HashiCorp Vault Setup

### 4.1 Install Vault

**macOS (Homebrew)**:
```bash
brew tap hashicorp/tap
brew install hashicorp/tap/vault
```

**Ubuntu/Debian**:
```bash
# Add HashiCorp GPG key
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -

# Add HashiCorp repository
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"

# Install Vault
sudo apt-get update
sudo apt-get install -y vault
```

**Docker (alternative)**:
```bash
docker run --name alphapulse-vault \
  --cap-add=IPC_LOCK \
  -e 'VAULT_DEV_ROOT_TOKEN_ID=dev-root-token' \
  -p 8200:8200 \
  -d vault:1.15
```

### 4.2 Start Vault in Development Mode

**Development mode** (in-memory storage, auto-unsealed):

```bash
# Start Vault dev server (run in separate terminal or tmux)
vault server -dev -dev-root-token-id="dev-root-token"

# Expected output:
# ==> Vault server configuration:
#
#              Api Address: http://127.0.0.1:8200
#                      Cgo: disabled
#          Cluster Address: https://127.0.0.1:8201
#               Go Version: go1.21.3
#                Listener 1: tcp (addr: "127.0.0.1:8200", cluster address: "127.0.0.1:8201", max_request_duration: "1m30s", max_request_size: "33554432", tls: "disabled")
#                 Log Level: info
#                     Mlock: supported: false, enabled: false
#             Recovery Mode: false
#                   Storage: inmem
#                   Version: Vault v1.15.0
#
# WARNING! dev mode is enabled! In this mode, Vault runs entirely in-memory
# and starts unsealed with a single unseal key. The root token is already
# authenticated to the CLI, so you can immediately begin using Vault.
#
# You may need to set the following environment variable:
#
#     $ export VAULT_ADDR='http://127.0.0.1:8200'
#
# The unseal key and root token are displayed below in case you want to
# seal/unseal the Vault or re-authenticate.
#
# Unseal Key: <key>
# Root Token: dev-root-token
```

### 4.3 Configure Vault Environment Variables

**Add to `~/.zshrc` or `~/.bashrc`**:
```bash
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='dev-root-token'
```

**Source the file**:
```bash
source ~/.zshrc  # or ~/.bashrc
```

### 4.4 Verify Vault

```bash
# Check Vault status
vault status

# Expected output:
# Key             Value
# ---             -----
# Seal Type       shamir
# Initialized     true
# Sealed          false
# Total Shares    1
# Threshold       1
# Version         1.15.0
# Storage Type    inmem
# Cluster Name    vault-cluster-...
# Cluster ID      ...

# Test secret write/read
vault kv put secret/test foo=bar
vault kv get secret/test
```

### 4.5 Initialize Vault for AlphaPulse

**Enable KV secrets engine**:
```bash
# Enable v2 KV secrets engine
vault secrets enable -path=secret kv-v2

# Create test tenant credentials
vault kv put secret/tenants/00000000-0000-0000-0000-000000000001/binance \
  api_key="test_api_key" \
  api_secret="test_api_secret"

# Verify
vault kv get secret/tenants/00000000-0000-0000-0000-000000000001/binance
```

**Create Vault policy for application**:
```bash
# Create policy file
cat > /tmp/alphapulse-policy.hcl <<EOF
# Allow reading tenant credentials
path "secret/data/tenants/*" {
  capabilities = ["read"]
}

# Allow listing tenants
path "secret/metadata/tenants/*" {
  capabilities = ["list"]
}
EOF

# Apply policy
vault policy write alphapulse /tmp/alphapulse-policy.hcl

# Create token with policy
vault token create -policy=alphapulse -ttl=720h
# Save the token for application use
```

---

## 5. Python Environment Setup

### 5.1 Install Python 3.11+

**macOS (Homebrew)**:
```bash
brew install python@3.11

# Add to PATH
export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
```

**Ubuntu/Debian**:
```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

### 5.2 Install Project Dependencies

```bash
cd AlphaPulse

# Install dependencies with Poetry
poetry install --no-interaction

# Activate virtual environment
poetry shell

# Verify installation
python --version
# Expected: Python 3.11.x

# Check installed packages
poetry show
```

---

## 6. AlphaPulse Application Setup

### 6.1 Configure Environment Variables

**Create `.env` file** in project root:

```bash
cat > .env <<EOF
# Database
DATABASE_URL=postgresql://alphapulse:alphapulse_dev@localhost:5432/alphapulse_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# Vault
VAULT_ADDR=http://127.0.0.1:8200
VAULT_TOKEN=dev-root-token

# API
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=dev-secret-key-change-in-production

# Feature Flags
RLS_ENABLED=false  # Disable RLS for initial setup
MULTI_TENANT_MODE=false  # Single-tenant mode initially

# Logging
LOG_LEVEL=DEBUG

# Exchange API Keys (for testing, use paper trading)
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret

# Environment
ENVIRONMENT=development
EOF
```

**Load environment variables**:
```bash
# Add to ~/.zshrc or ~/.bashrc
export $(cat .env | xargs)

# Or use direnv (recommended)
brew install direnv  # macOS
sudo apt-get install direnv  # Ubuntu

# Add to shell config
eval "$(direnv hook zsh)"  # or bash

# Allow .env loading
direnv allow .
```

### 6.2 Verify Application Configuration

```bash
# Test configuration loading
poetry run python -c "
import os
from dotenv import load_dotenv

load_dotenv()

print('Database URL:', os.getenv('DATABASE_URL'))
print('Redis URL:', os.getenv('REDIS_URL'))
print('Vault Addr:', os.getenv('VAULT_ADDR'))
print('RLS Enabled:', os.getenv('RLS_ENABLED'))
"
```

---

## 7. Database Migrations

### 7.1 Run Alembic Migrations

```bash
# Generate initial migration (if not exists)
poetry run alembic revision --autogenerate -m "initial_schema"

# Run migrations
poetry run alembic upgrade head

# Verify migration
poetry run alembic current

# Expected output:
# <revision_id> (head)
```

### 7.2 Verify Database Schema

```bash
# Connect to database
psql -U alphapulse -d alphapulse_dev -h localhost

# List tables
\dt

# Expected tables:
# public | alembic_version
# public | tenants
# public | users
# public | portfolios
# public | trades
# public | signals
# public | positions
# ... (more tables)

# Check table structure
\d+ users

# Exit
\q
```

---

## 8. Sample Data Setup

### 8.1 Create Sample Tenant and Users

```bash
# Run sample data script
poetry run python scripts/create_sample_data.py
```

**Create the script** `scripts/create_sample_data.py`:

```python
"""Create sample tenant and users for development."""

import asyncio
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

async def create_sample_data():
    """Create sample tenant and users."""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create default tenant
        tenant_id = "00000000-0000-0000-0000-000000000001"
        session.execute(text("""
            INSERT INTO tenants (id, name, slug, subscription_tier, status)
            VALUES (:id, 'Default Tenant', 'default', 'pro', 'active')
            ON CONFLICT (id) DO NOTHING
        """), {"id": tenant_id})

        # Create admin user
        user_id = str(uuid.uuid4())
        session.execute(text("""
            INSERT INTO users (id, tenant_id, email, password_hash, role)
            VALUES (:id, :tenant_id, 'admin@alphapulse.dev', 'hashed_password', 'admin')
            ON CONFLICT (email, tenant_id) DO NOTHING
        """), {"id": user_id, "tenant_id": tenant_id})

        # Create sample portfolio
        portfolio_id = str(uuid.uuid4())
        session.execute(text("""
            INSERT INTO portfolios (id, tenant_id, user_id, name, total_value)
            VALUES (:id, :tenant_id, :user_id, 'Default Portfolio', 10000.00)
            ON CONFLICT DO NOTHING
        """), {"id": portfolio_id, "tenant_id": tenant_id, "user_id": user_id})

        session.commit()
        print("✅ Sample data created successfully!")
        print(f"Tenant ID: {tenant_id}")
        print(f"User ID: {user_id}")
        print(f"Portfolio ID: {portfolio_id}")

    except Exception as e:
        session.rollback()
        print(f"❌ Error creating sample data: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    asyncio.run(create_sample_data())
```

### 8.2 Verify Sample Data

```bash
# Connect to database
psql -U alphapulse -d alphapulse_dev -h localhost

# Query sample data
SELECT * FROM tenants;
SELECT * FROM users;
SELECT * FROM portfolios;

# Exit
\q
```

---

## 9. Verification

### 9.1 Run API Server

```bash
# Start API server
poetry run python src/scripts/run_api.py

# Expected output:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 9.2 Test API Endpoints

**In another terminal**:

```bash
# Health check
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","rls_enabled":false}

# API documentation (open in browser)
open http://localhost:8000/docs
```

### 9.3 Run Tests

```bash
# Run unit tests
poetry run pytest src/alpha_pulse/tests/ -v

# Run with coverage
poetry run pytest --cov=src/alpha_pulse --cov-report=term

# Expected: All tests pass, coverage > 90%
```

### 9.4 Verify Services

**PostgreSQL**:
```bash
psql -U alphapulse -d alphapulse_dev -h localhost -c "SELECT 1"
# Expected: 1 row returned
```

**Redis**:
```bash
redis-cli ping
# Expected: PONG
```

**Vault**:
```bash
vault status
# Expected: Sealed = false
```

---

## 10. Troubleshooting

### 10.1 PostgreSQL Issues

**Cannot connect to database**:
```bash
# Check if PostgreSQL is running
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql  # Ubuntu

# Restart PostgreSQL
brew services restart postgresql@14  # macOS
sudo systemctl restart postgresql  # Ubuntu

# Check PostgreSQL logs
tail -f /opt/homebrew/var/log/postgresql@14.log  # macOS
sudo tail -f /var/log/postgresql/postgresql-14-main.log  # Ubuntu
```

**Authentication failed**:
```bash
# Edit pg_hba.conf to allow local connections
# macOS: /opt/homebrew/var/postgresql@14/pg_hba.conf
# Ubuntu: /etc/postgresql/14/main/pg_hba.conf

# Add this line (if not exists):
# local   all   alphapulse   md5

# Restart PostgreSQL
```

### 10.2 Redis Issues

**Cannot connect to Redis**:
```bash
# Check if Redis is running
brew services list | grep redis  # macOS
sudo systemctl status redis-server  # Ubuntu

# Restart Redis
brew services restart redis  # macOS
sudo systemctl restart redis-server  # Ubuntu

# Check Redis logs
tail -f /opt/homebrew/var/log/redis.log  # macOS
sudo tail -f /var/log/redis/redis-server.log  # Ubuntu
```

### 10.3 Vault Issues

**Vault is sealed**:
```bash
# In dev mode, Vault should auto-unseal
# If sealed, restart Vault dev server
vault server -dev -dev-root-token-id="dev-root-token"
```

**Cannot read secrets**:
```bash
# Check Vault token
echo $VAULT_TOKEN

# Verify token is valid
vault token lookup

# Recreate token if needed
export VAULT_TOKEN="dev-root-token"
```

### 10.4 Python/Poetry Issues

**Poetry not found**:
```bash
# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

**Dependencies install fails**:
```bash
# Clear Poetry cache
poetry cache clear pypi --all

# Reinstall dependencies
poetry install --no-cache
```

### 10.5 Application Issues

**ImportError or ModuleNotFoundError**:
```bash
# Ensure virtual environment is activated
poetry shell

# Reinstall dependencies
poetry install

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**Database migration fails**:
```bash
# Check Alembic configuration
cat alembic.ini

# Verify database URL
echo $DATABASE_URL

# Reset migrations (CAUTION: drops all tables)
poetry run alembic downgrade base
poetry run alembic upgrade head
```

---

## Appendix A: Quick Start Script

**Create `scripts/setup_dev.sh`**:

```bash
#!/bin/bash
set -e

echo "🚀 Setting up AlphaPulse development environment..."

# Check prerequisites
echo "✅ Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 not found. Install it first."; exit 1; }
command -v poetry >/dev/null 2>&1 || { echo "❌ Poetry not found. Install it first."; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "❌ PostgreSQL not found. Install it first."; exit 1; }
command -v redis-cli >/dev/null 2>&1 || { echo "❌ Redis not found. Install it first."; exit 1; }
command -v vault >/dev/null 2>&1 || { echo "❌ Vault not found. Install it first."; exit 1; }

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install --no-interaction

# Create .env file (if not exists)
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env <<EOF
DATABASE_URL=postgresql://alphapulse:alphapulse_dev@localhost:5432/alphapulse_dev
REDIS_URL=redis://localhost:6379/0
VAULT_ADDR=http://127.0.0.1:8200
VAULT_TOKEN=dev-root-token
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=dev-secret-key-change-in-production
RLS_ENABLED=false
MULTI_TENANT_MODE=false
LOG_LEVEL=DEBUG
ENVIRONMENT=development
EOF
fi

# Run database migrations
echo "🗄️ Running database migrations..."
poetry run alembic upgrade head

# Create sample data
echo "🌱 Creating sample data..."
poetry run python scripts/create_sample_data.py

# Run tests
echo "🧪 Running tests..."
poetry run pytest --cov=src/alpha_pulse --cov-report=term

echo "✅ Development environment setup complete!"
echo ""
echo "📚 Next steps:"
echo "  1. Start API server: poetry run python src/scripts/run_api.py"
echo "  2. Open API docs: http://localhost:8000/docs"
echo "  3. Run tests: poetry run pytest -v"
```

**Make it executable**:
```bash
chmod +x scripts/setup_dev.sh
```

**Run it**:
```bash
./scripts/setup_dev.sh
```

---

## Appendix B: Service Management with tmux

**Create `scripts/start_services.sh`**:

```bash
#!/bin/bash

# Start all services in tmux session
tmux new-session -d -s alphapulse

# PostgreSQL (already running as service)
tmux rename-window -t alphapulse:0 'postgres'
tmux send-keys -t alphapulse:0 'psql -U alphapulse -d alphapulse_dev' C-m

# Redis (already running as service)
tmux new-window -t alphapulse:1 -n 'redis'
tmux send-keys -t alphapulse:1 'redis-cli' C-m

# Vault dev server
tmux new-window -t alphapulse:2 -n 'vault'
tmux send-keys -t alphapulse:2 'vault server -dev -dev-root-token-id="dev-root-token"' C-m

# API server
tmux new-window -t alphapulse:3 -n 'api'
tmux send-keys -t alphapulse:3 'cd /Users/a.rocchi/Projects/Personal/AlphaPulse' C-m
tmux send-keys -t alphapulse:3 'poetry shell' C-m
tmux send-keys -t alphapulse:3 'python src/scripts/run_api.py' C-m

# Attach to session
tmux attach-session -t alphapulse
```

**Usage**:
```bash
# Start all services
./scripts/start_services.sh

# Navigate between windows: Ctrl+b then 0, 1, 2, 3
# Detach: Ctrl+b then d
# Reattach: tmux attach -t alphapulse
# Kill session: tmux kill-session -t alphapulse
```

---

## Appendix C: IDE Configuration

### VS Code

**Recommended Extensions**:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- PostgreSQL (ckolkman.vscode-postgres)
- Redis (weicheng2138.redis)
- YAML (redhat.vscode-yaml)

**Settings** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "src/alpha_pulse/tests"
  ],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

### PyCharm

**Configure Interpreter**:
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Poetry Environment
3. Select existing environment: `.venv`

**Configure Database**:
1. Database Tool Window → + → Data Source → PostgreSQL
2. Host: localhost, Port: 5432
3. Database: alphapulse_dev
4. User: alphapulse, Password: alphapulse_dev

---

**Document Status**: Draft
**Review Date**: Sprint 3, Week 2
**Owner**: Tech Lead

---

**END OF DOCUMENT**
