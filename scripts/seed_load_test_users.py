#!/usr/bin/env python3
"""
Seed Load Test Users - AlphaPulse

This script creates test tenants and users for load testing.
It should be run ONCE before load testing begins.

Usage:
    poetry run python scripts/seed_load_test_users.py --staging
    poetry run python scripts/seed_load_test_users.py --local
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg
from loguru import logger

# Test data configuration
TENANTS = [
    {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Tenant 1 (Load Test)",
        "slug": "tenant1",
        "subscription_tier": "pro",
        "status": "active",
    },
    {
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "Tenant 2 (Load Test)",
        "slug": "tenant2",
        "subscription_tier": "pro",
        "status": "active",
    },
    {
        "id": "00000000-0000-0000-0000-000000000003",
        "name": "Tenant 3 (Load Test)",
        "slug": "tenant3",
        "subscription_tier": "starter",
        "status": "active",
    },
    {
        "id": "00000000-0000-0000-0000-000000000004",
        "name": "Tenant 4 (Load Test)",
        "slug": "tenant4",
        "subscription_tier": "pro",
        "status": "active",
    },
    {
        "id": "00000000-0000-0000-0000-000000000005",
        "name": "Tenant 5 (Load Test)",
        "slug": "tenant5",
        "subscription_tier": "enterprise",
        "status": "active",
    },
]

USERS_PER_TENANT = 2

# Sample portfolio data
SAMPLE_TRADES = [
    {"symbol": "BTC/USD", "side": "buy", "quantity": 1.0, "price": 45000.0, "order_type": "limit"},
    {"symbol": "ETH/USD", "side": "buy", "quantity": 10.0, "price": 3000.0, "order_type": "limit"},
    {"symbol": "SOL/USD", "side": "buy", "quantity": 100.0, "price": 150.0, "order_type": "limit"},
    {"symbol": "AVAX/USD", "side": "buy", "quantity": 50.0, "price": 80.0, "order_type": "limit"},
    {"symbol": "MATIC/USD", "side": "sell", "quantity": 200.0, "price": 1.5, "order_type": "limit"},
]


async def create_tables_if_not_exist(conn: asyncpg.Connection):
    """Create tables if they don't exist (for local development)."""

    logger.info("Checking if tables exist...")

    # Check if tenants table exists
    result = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'tenants'
        )
        """
    )

    if not result:
        logger.warning("Tables do not exist. Creating tables...")

        # Create tenants table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                slug VARCHAR(100) UNIQUE NOT NULL,
                subscription_tier VARCHAR(50) NOT NULL,
                status VARCHAR(50) NOT NULL DEFAULT 'active',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                email VARCHAR(255) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(email, tenant_id)
            )
        """)

        # Create trades table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                order_type VARCHAR(20) NOT NULL,
                status VARCHAR(50) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create positions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                symbol VARCHAR(20) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                avg_entry_price DECIMAL(20, 8) NOT NULL,
                current_price DECIMAL(20, 8),
                unrealized_pnl DECIMAL(20, 8),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(tenant_id, symbol)
            )
        """)

        logger.info("✅ Tables created successfully")
    else:
        logger.info("✅ Tables already exist")


async def seed_tenants(conn: asyncpg.Connection):
    """Create test tenants."""

    logger.info(f"Creating {len(TENANTS)} test tenants...")

    for tenant in TENANTS:
        try:
            await conn.execute(
                """
                INSERT INTO tenants (id, name, slug, subscription_tier, status)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE
                SET name = EXCLUDED.name,
                    subscription_tier = EXCLUDED.subscription_tier,
                    status = EXCLUDED.status
                """,
                tenant["id"],
                tenant["name"],
                tenant["slug"],
                tenant["subscription_tier"],
                tenant["status"],
            )
            logger.info(f"✅ Created/Updated tenant: {tenant['name']} ({tenant['id']})")
        except Exception as e:
            logger.error(f"❌ Failed to create tenant {tenant['name']}: {e}")
            raise


async def seed_users(conn: asyncpg.Connection):
    """Create test users for each tenant."""

    logger.info(f"Creating {USERS_PER_TENANT} users per tenant...")

    # Simple password hash (for testing only - use proper bcrypt in production)
    password_hash = "test123_hashed"  # In production: bcrypt.hashpw(b"test123", bcrypt.gensalt())

    for tenant in TENANTS:
        for i in range(1, USERS_PER_TENANT + 1):
            email = f"user{i}@{tenant['slug']}.com"

            try:
                await conn.execute(
                    """
                    INSERT INTO users (tenant_id, email, password_hash, role)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (email, tenant_id) DO UPDATE
                    SET password_hash = EXCLUDED.password_hash,
                        role = EXCLUDED.role
                    """,
                    tenant["id"],
                    email,
                    password_hash,
                    "admin" if i == 1 else "user",
                )
                logger.info(f"✅ Created/Updated user: {email} (tenant: {tenant['slug']})")
            except Exception as e:
                logger.error(f"❌ Failed to create user {email}: {e}")
                raise


async def seed_sample_data(conn: asyncpg.Connection):
    """Create sample trades and positions for each tenant."""

    logger.info("Creating sample portfolio data...")

    for tenant in TENANTS:
        # Get first user for this tenant
        user_id = await conn.fetchval(
            "SELECT id FROM users WHERE tenant_id = $1 LIMIT 1",
            tenant["id"]
        )

        if not user_id:
            logger.warning(f"⚠️ No users found for tenant {tenant['name']}, skipping sample data")
            continue

        # Create trades
        for trade in SAMPLE_TRADES:
            try:
                await conn.execute(
                    """
                    INSERT INTO trades (tenant_id, user_id, symbol, side, quantity, price, order_type, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 'filled')
                    """,
                    tenant["id"],
                    user_id,
                    trade["symbol"],
                    trade["side"],
                    trade["quantity"],
                    trade["price"],
                    trade["order_type"],
                )
            except Exception as e:
                logger.debug(f"Trade may already exist: {e}")

        # Create positions
        for trade in SAMPLE_TRADES:
            if trade["side"] == "buy":
                try:
                    await conn.execute(
                        """
                        INSERT INTO positions (tenant_id, symbol, quantity, avg_entry_price, current_price)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (tenant_id, symbol) DO UPDATE
                        SET quantity = EXCLUDED.quantity,
                            avg_entry_price = EXCLUDED.avg_entry_price
                        """,
                        tenant["id"],
                        trade["symbol"],
                        trade["quantity"],
                        trade["price"],
                        trade["price"] * 1.05,  # 5% profit
                    )
                except Exception as e:
                    logger.debug(f"Position may already exist: {e}")

        logger.info(f"✅ Created sample data for tenant: {tenant['name']}")


async def verify_data(conn: asyncpg.Connection):
    """Verify that all data was created successfully."""

    logger.info("Verifying data...")

    # Count tenants
    tenant_count = await conn.fetchval("SELECT COUNT(*) FROM tenants")
    logger.info(f"Tenants: {tenant_count}")

    # Count users
    user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
    logger.info(f"Users: {user_count}")

    # Count trades
    trade_count = await conn.fetchval("SELECT COUNT(*) FROM trades")
    logger.info(f"Trades: {trade_count}")

    # Count positions
    position_count = await conn.fetchval("SELECT COUNT(*) FROM positions")
    logger.info(f"Positions: {position_count}")

    # Verify expected counts
    assert tenant_count == len(TENANTS), f"Expected {len(TENANTS)} tenants, got {tenant_count}"
    assert user_count == len(TENANTS) * USERS_PER_TENANT, f"Expected {len(TENANTS) * USERS_PER_TENANT} users, got {user_count}"

    logger.info("✅ Data verification complete!")


async def main(environment: str):
    """Main entry point."""

    logger.info(f"Starting load test user seeding for environment: {environment}")

    # Get database URL
    if environment == "staging":
        db_url = os.getenv("STAGING_DATABASE_URL")
        if not db_url:
            logger.error("❌ STAGING_DATABASE_URL not set")
            sys.exit(1)
    elif environment == "local":
        db_url = os.getenv("DATABASE_URL", "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse")
    else:
        logger.error(f"❌ Unknown environment: {environment}")
        sys.exit(1)

    logger.info(f"Connecting to database: {db_url.split('@')[1] if '@' in db_url else 'localhost'}")

    # Connect to database
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("✅ Connected to database")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Create tables if needed (local development)
        if environment == "local":
            await create_tables_if_not_exist(conn)

        # Seed data
        await seed_tenants(conn)
        await seed_users(conn)
        await seed_sample_data(conn)

        # Verify
        await verify_data(conn)

        logger.info("=" * 60)
        logger.info("✅ Load test data seeding complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Test Credentials:")
        logger.info("─" * 60)
        for tenant in TENANTS:
            logger.info(f"Tenant: {tenant['name']} ({tenant['subscription_tier']})")
            for i in range(1, USERS_PER_TENANT + 1):
                email = f"user{i}@{tenant['slug']}.com"
                logger.info(f"  - {email} / test123")
        logger.info("=" * 60)

    finally:
        await conn.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed load test users and data")
    parser.add_argument(
        "--staging",
        action="store_const",
        const="staging",
        dest="environment",
        help="Seed staging environment",
    )
    parser.add_argument(
        "--local",
        action="store_const",
        const="local",
        dest="environment",
        help="Seed local environment",
    )

    args = parser.parse_args()

    if not args.environment:
        parser.print_help()
        sys.exit(1)

    asyncio.run(main(args.environment))
