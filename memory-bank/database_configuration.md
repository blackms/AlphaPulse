# Database Configuration for AI Hedge Fund

This document provides detailed configuration for all database components of the AI Hedge Fund system.

## PostgreSQL with TimescaleDB Configuration

### `config/database_config.yaml`

```yaml
# PostgreSQL Configuration
postgres:
  host: ${DB_HOST:-localhost}
  port: ${DB_PORT:-5432}
  database: ${DB_NAME:-alphapulse}
  username: ${DB_USER:-alphapulse}
  password: ${DB_PASSWORD:-password}
  schema: ${DB_SCHEMA:-public}
  sslmode: ${DB_SSLMODE:-disable}
  pool:
    min_connections: ${DB_POOL_MIN:-5}
    max_connections: ${DB_POOL_MAX:-20}
    max_idle_time_seconds: ${DB_POOL_IDLE:-300}
  timescale:
    enabled: ${TIMESCALE_ENABLED:-true}
    # TimescaleDB-specific configurations
    chunk_time_interval: 1d  # Time interval for chunks
    compression:
      enabled: true
      compression_interval: 7d  # Compress data older than 7 days

# Redis Configuration
redis:
  host: ${REDIS_HOST:-localhost}
  port: ${REDIS_PORT:-6379}
  password: ${REDIS_PASSWORD:-}
  database: ${REDIS_DB:-0}
  ssl: ${REDIS_SSL:-false}
  pool:
    max_connections: ${REDIS_POOL_MAX:-20}
  timeout: ${REDIS_TIMEOUT:-10}
  cache:
    ttl_seconds: ${REDIS_CACHE_TTL:-300}
  pubsub:
    channel_prefix: ${REDIS_PUBSUB_PREFIX:-alphapulse}
```

## Docker Compose Configuration

### `docker-compose.db.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: alphapulse
      POSTGRES_USER: alphapulse
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U alphapulse"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass password
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "password", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

## Database Initialization

### `scripts/init/01-schema.sql`

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schema
CREATE SCHEMA IF NOT EXISTS alphapulse;

-- Users and Authentication Tables
CREATE TABLE alphapulse.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login TIMESTAMPTZ
);

CREATE TABLE alphapulse.api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES alphapulse.users(id) ON DELETE CASCADE,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used TIMESTAMPTZ
);

-- Portfolio Tables
CREATE TABLE alphapulse.portfolios (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE alphapulse.positions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES alphapulse.portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(18, 8) NOT NULL,
    current_price DECIMAL(18, 8),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Trading Tables
CREATE TABLE alphapulse.trades (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES alphapulse.portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    fees DECIMAL(18, 8),
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    external_id VARCHAR(100)
);

-- Time Series Tables
CREATE TABLE alphapulse.metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    labels JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Make metrics a hypertable
SELECT create_hypertable('alphapulse.metrics', 'time');

-- Create index on metric_name for faster queries
CREATE INDEX idx_metrics_name ON alphapulse.metrics (metric_name, time DESC);

-- Alerts Table
CREATE TABLE alphapulse.alerts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    source VARCHAR(50) NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged BOOLEAN NOT NULL DEFAULT false,
    acknowledged_by VARCHAR(50),
    acknowledged_at TIMESTAMPTZ
);
```

## Connection Management Implementation

### Connection Pool Management

Create a connection pool manager that handles:
1. Creating and maintaining database connections
2. Distributing connections for different operations
3. Health checks and reconnection logic
4. Transaction management

### ORM Models 

Use SQLAlchemy for ORM with:
1. Base models for all entities
2. Relationship definitions
3. Validation rules
4. Default values and constraints

### Redis Client Configuration

For Redis, configure:
1. Connection pooling
2. Serialization/deserialization helpers
3. Pattern for pub/sub communication
4. Caching decorators for API methods

## Security Considerations

1. Use environment variables for all sensitive information
2. Configure TLS/SSL for database connections
3. Implement connection encryption
4. Use prepared statements to prevent SQL injection
5. Implement principle of least privilege for database users