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

-- Time Series Tables (regular table instead of hypertable)
CREATE TABLE alphapulse.metrics (
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    labels JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create index on metric_name and time for faster queries
CREATE INDEX idx_metrics_name_time ON alphapulse.metrics (metric_name, time DESC);
CREATE INDEX idx_metrics_time ON alphapulse.metrics (time DESC);

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

-- Create admin user
INSERT INTO alphapulse.users (username, password_hash, email, role)
VALUES (
    'admin',
    -- This is a bcrypt hash for 'admin123' - in production, generate this properly
    '$2a$12$1InE4/AkbV4/Ye7nKrLPOOYILsLPjpqiLjHXgH6iBp2o7QEzW.ZpG',
    'admin@alphapulse.com',
    'admin'
) ON CONFLICT DO NOTHING;

-- Create sample portfolio
INSERT INTO alphapulse.portfolios (name, description)
VALUES (
    'Main Portfolio',
    'Primary trading portfolio for the AI Hedge Fund'
) ON CONFLICT DO NOTHING;