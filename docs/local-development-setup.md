# Local Development Setup - Multi-Tenant AlphaPulse

**Last Updated**: 2025-10-28
**Sprint**: Sprint 5 (Phase 3: Build & Validate)

---

## Overview

This guide helps you set up a local development environment for AlphaPulse's multi-tenant SaaS transformation. After following this guide, you'll have:

- ✅ PostgreSQL 14+ with Row-Level Security (RLS)
- ✅ Redis Cluster (or standalone for development)
- ✅ HashiCorp Vault (dev mode)
- ✅ All database migrations applied
- ✅ Multi-tenant middleware operational

**Estimated Time**: 30 minutes

---

## Prerequisites

- Python 3.11+ (required by Poetry)
- Poetry installed (`curl -sSL https://install.python-poetry.org | python3 -`)
- Docker Desktop (recommended) OR Homebrew
- 8GB RAM minimum
- macOS, Linux, or WSL2 on Windows

---

## Option 1: Docker Compose (Recommended)

### 1. Create Docker Compose Configuration

Create `docker-compose.dev.yml` in the project root:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14-alpine
    container_name: alphapulse-postgres
    environment:
      POSTGRES_USER: devuser
      POSTGRES_PASSWORD: devpassword
      POSTGRES_DB: backtesting
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U devuser -d backtesting"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: alphapulse-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  vault:
    image: hashicorp/vault:latest
    container_name: alphapulse-vault
    ports:
      - "8200:8200"
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: dev-root-token
      VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
    cap_add:
      - IPC_LOCK
    healthcheck:
      test: ["CMD", "vault", "status"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  postgres_data:
  redis_data:
```

### 2. Start Services

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Check status
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

### 3. Verify Services

```bash
# PostgreSQL
psql -h localhost -U devuser -d backtesting -c "SELECT version();"
# Password: devpassword

# Redis
redis-cli ping
# Expected: PONG

# Vault
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN='dev-root-token'
vault status
```

---

## Option 2: Native Installation (macOS)

### 1. Install PostgreSQL

```bash
# Install PostgreSQL 14
brew install postgresql@14

# Start PostgreSQL
brew services start postgresql@14

# Create database
createdb backtesting

# Create user (if needed)
psql postgres -c "CREATE USER devuser WITH PASSWORD 'devpassword';"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE backtesting TO devuser;"
```

### 2. Install Redis

```bash
# Install Redis
brew install redis

# Start Redis
brew services start redis

# Verify
redis-cli ping
# Expected: PONG
```

### 3. Install Vault (Dev Mode)

```bash
# Install Vault
brew tap hashicorp/tap
brew install hashicorp/tap/vault

# Start Vault in dev mode (background)
vault server -dev -dev-root-token-id="dev-root-token" > vault.log 2>&1 &

# Set environment variables
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='dev-root-token'

# Verify
vault status
```

---

## Apply Database Migrations

### 1. Update Alembic Configuration

Edit `alembic.ini` to match your setup:

```ini
# For Docker Compose
sqlalchemy.url = postgresql+asyncpg://devuser:devpassword@localhost:5432/backtesting

# For native PostgreSQL (adjust username/password)
sqlalchemy.url = postgresql+asyncpg://your_user:your_password@localhost:5432/backtesting
```

### 2. Run Migrations

```bash
# Navigate to migrations directory
cd migrations

# Check current migration status
poetry run alembic current

# Apply all migrations (001 through 008)
poetry run alembic upgrade head

# Verify migrations applied
poetry run alembic current
# Expected: 008_enable_rls (head)
```

### 3. Verify Multi-Tenant Schema

```bash
# Connect to database
psql -h localhost -U devuser -d backtesting

# Check tenants table
\d tenants

# Verify default tenant
SELECT * FROM tenants;
# Expected: 1 row (Default Tenant, UUID: 00000000-0000-0000-0000-000000000001)

# Check RLS policies
SELECT schemaname, tablename, policyname, cmd, qual
FROM pg_policies
WHERE tablename IN ('users', 'trades', 'positions');

# Exit psql
\q
```

---

## Configure Environment Variables

### 1. Create `.env` File

Create `.env` in project root:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://devuser:devpassword@localhost:5432/backtesting

# Redis
REDIS_URL=redis://localhost:6379

# Vault
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=dev-root-token

# Multi-Tenant Feature Flags
RLS_ENABLED=true
MULTI_TENANT_MODE=development

# JWT Configuration
JWT_SECRET=your-local-dev-secret-change-me
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### 2. Load Environment Variables

```bash
# Option 1: Source .env file
set -a
source .env
set +a

# Option 2: Use direnv (recommended)
brew install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc  # or ~/.zshrc
echo 'dotenv' > .envrc
direnv allow
```

---

## Install Python Dependencies

```bash
# Install dependencies
poetry install --no-interaction

# Activate virtual environment
poetry shell

# Verify installation
python --version
# Expected: Python 3.11.x or 3.12.x

poetry run python -c "import fastapi, sqlalchemy, redis, hvac; print('All imports successful')"
```

---

## Seed Test Data (Optional)

### 1. Create Test Tenants

```bash
# Run seed script
poetry run python scripts/seed_load_test_users.py --environment local

# Expected output:
# ✅ Created 5 tenants
# ✅ Created 10 users (2 per tenant)
# ✅ Created sample trades and positions
```

### 2. Verify Test Data

```sql
-- Connect to database
psql -h localhost -U devuser -d backtesting

-- Check tenants
SELECT id, name, slug, subscription_tier, status FROM tenants;

-- Check users with tenant association
SELECT username, email, tenant_id FROM users;

-- Exit
\q
```

---

## Start AlphaPulse API Server

### 1. Start Server

```bash
# Navigate to project root
cd /Users/a.rocchi/Projects/Personal/AlphaPulse

# Start API server
poetry run python src/scripts/run_api.py

# Expected output:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     AlphaPulse API started successfully
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Test API Endpoints

```bash
# Test health endpoint (no auth required)
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Test login (get JWT token)
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123!@#"

# Expected response:
# {
#   "access_token": "eyJ...",
#   "token_type": "bearer",
#   "user": {
#     "username": "admin",
#     "tenant_id": "00000000-0000-0000-0000-000000000001"
#   }
# }

# Save token
export TOKEN="<paste_token_here>"

# Test authenticated endpoint with tenant context
curl http://localhost:8000/api/v1/portfolio \
  -H "Authorization: Bearer $TOKEN"

# Check server logs - should see:
# DEBUG: Tenant context set: tenant_id=00000000-0000-0000-0000-000000000001
```

---

## Run Benchmarks

### 1. RLS Performance Benchmark

```bash
# Run benchmark
poetry run python scripts/benchmark_rls.py

# Expected output:
# Running RLS Performance Benchmark...
# ✅ Simple SELECT: P99 latency = 12.3ms (overhead: 5.2%)
# ✅ Aggregation query: P99 latency = 45.6ms (overhead: 8.1%)
# ✅ JOIN query: P99 latency = 78.9ms (overhead: 9.5%)
# ✅ Time-range query: P99 latency = 34.5ms (overhead: 6.7%)
#
# RESULT: ✅ GO - RLS overhead <10% (average: 7.4%)
```

### 2. Redis Namespace Benchmark

```bash
# Ensure Redis is running
redis-cli ping

# Run benchmark
poetry run python scripts/benchmark_redis.py

# Expected output:
# Running Redis Namespace Isolation Benchmark...
# ✅ Rolling counter: P99 latency = 0.8ms
# ✅ LRU eviction: P99 latency = 65ms
# ✅ Cache hit rate: 87.3%
# ✅ Namespace isolation: 100% (0 cross-tenant access)
#
# RESULT: ✅ GO - All targets met
```

### 3. Vault Load Test

```bash
# Ensure Vault is running
vault status

# Run load test (requires k6)
brew install k6
k6 run scripts/load_test_vault.js

# Expected output:
# ✅ checks.....................: 100.00% ✓ 45000 ✗ 0
# ✅ http_req_duration...........: P99=8.5ms
# ✅ http_reqs...................: 15000/sec
#
# RESULT: ✅ GO - Throughput >10k req/sec, P99 <10ms
```

---

## Troubleshooting

### PostgreSQL Connection Refused

```bash
# Check if PostgreSQL is running
docker ps | grep postgres
# OR
brew services list | grep postgresql

# Check port is not blocked
lsof -i :5432

# Test connection
psql -h localhost -U devuser -d backtesting -c "SELECT 1;"
```

### Migration Errors

```bash
# Check current migration
cd migrations
poetry run alembic current

# If stuck, downgrade and reapply
poetry run alembic downgrade base
poetry run alembic upgrade head

# View migration history
poetry run alembic history --verbose
```

### Redis Connection Issues

```bash
# Check Redis is running
redis-cli ping

# Check Redis logs
docker logs alphapulse-redis
# OR
tail -f /usr/local/var/log/redis.log
```

### Vault Sealed

```bash
# Vault dev mode auto-unseals, but if needed:
vault operator unseal
# Enter unseal key (not needed in dev mode)

# Check status
vault status
# Look for "Sealed: false"
```

### Middleware Not Working

```bash
# Check API server logs
tail -f logs/alphapulse_api.log

# Test JWT token structure
echo "$TOKEN" | cut -d'.' -f2 | base64 -d 2>/dev/null | jq

# Should show:
# {
#   "sub": "admin",
#   "tenant_id": "00000000-0000-0000-0000-000000000001",
#   "exp": 1234567890
# }
```

---

## Next Steps

After completing this setup, you're ready for:

1. **Sprint 5, Week 1**: Run all benchmarks, document results
2. **Sprint 5, Week 2**: Implement TenantService and API endpoints
3. **Sprint 6**: Make services tenant-aware, integration testing

---

## Quick Reference

### Start All Services (Docker)
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Stop All Services
```bash
docker-compose -f docker-compose.dev.yml down
```

### Reset Database (Clean Slate)
```bash
# Stop services
docker-compose -f docker-compose.dev.yml down -v

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Reapply migrations
cd migrations && poetry run alembic upgrade head
```

### View Logs
```bash
# All services
docker-compose -f docker-compose.dev.yml logs -f

# Specific service
docker-compose -f docker-compose.dev.yml logs -f postgres
```

---

## Support

- **Issues**: https://github.com/blackms/AlphaPulse/issues
- **Sprint 5 Tracking**: Issue #181
- **Master Issue**: #149

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
