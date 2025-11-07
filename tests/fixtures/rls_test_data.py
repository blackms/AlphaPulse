#!/usr/bin/env python3
"""
Production-Like Test Data Generation for RLS Benchmarking

Generates realistic test data for validating RLS performance with production-scale datasets.
Follows DEV-PROTO.yaml deterministic build principles (seeded RNG, factory functions).

EPIC-001: Database Multi-Tenancy
Story #160: Test RLS Performance Benchmarks
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID, uuid4

import asyncpg


class RLSTestDataGenerator:
    """Generate production-like test data for RLS benchmarking."""

    def __init__(self, database_url: str, seed: int = 42):
        self.database_url = database_url
        self.pool = None
        random.seed(seed)  # Deterministic data generation

        # Test tenant IDs
        self.tenant_ids = [
            UUID("00000000-0000-0000-0000-000000000001"),
            UUID("00000000-0000-0000-0000-000000000002"),
        ]

        # Realistic trading data
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT", "AVAX/USDT"]
        self.sides = ["buy", "sell"]

    async def setup(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=5, max_size=20)
        print(f"✓ Connected to database")

    async def teardown(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
        print("✓ Database connection closed")

    async def check_existing_data(self) -> Dict[str, int]:
        """Check existing row counts to avoid duplicating data."""
        async with self.pool.acquire() as conn:
            counts = {}
            for table in ["trades", "positions", "users", "trading_accounts"]:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                counts[table] = count
        return counts

    async def generate_users(self, tenant_id: UUID, count: int = 10):
        """Generate test users for a tenant."""
        print(f"  Generating {count} users for tenant {tenant_id}...")

        async with self.pool.acquire() as conn:
            for i in range(count):
                try:
                    await conn.execute("""
                        INSERT INTO users (tenant_id, email, username, created_at)
                        VALUES ($1, $2, $3, $4)
                    """, tenant_id, f"user{i}_tenant{str(tenant_id)[:8]}@test.com",
                        f"user{i}_tenant{str(tenant_id)[:8]}", datetime.now() - timedelta(days=random.randint(1, 365)))
                except Exception as e:
                    print(f"    Warning: User generation error: {e}")
                    continue

    async def generate_trading_accounts(self, tenant_id: UUID, user_id: int, count: int = 2):
        """Generate test trading accounts for a user."""
        exchanges = ["binance", "coinbase"]

        async with self.pool.acquire() as conn:
            for i in range(count):
                try:
                    await conn.execute("""
                        INSERT INTO trading_accounts (tenant_id, user_id, exchange, account_name, created_at)
                        VALUES ($1, $2, $3, $4, $5)
                    """, tenant_id, user_id, exchanges[i],
                        f"{exchanges[i]}_account", datetime.now() - timedelta(days=random.randint(1, 180)))
                except Exception as e:
                    print(f"    Warning: Trading account generation error: {e}")
                    continue

    async def generate_trades(self, tenant_id: UUID, count: int = 10000):
        """Generate test trades for a tenant with realistic time distribution."""
        print(f"  Generating {count} trades for tenant {tenant_id}...")

        # Get account_id for this tenant
        async with self.pool.acquire() as conn:
            account_id = await conn.fetchval(
                "SELECT id FROM trading_accounts WHERE tenant_id = $1 LIMIT 1", tenant_id
            )
            if not account_id:
                print(f"    ⚠️  No trading account found for tenant {tenant_id}, creating one...")
                account_id = await conn.fetchval("""
                    INSERT INTO trading_accounts (tenant_id, user_id, exchange, account_name, api_key_encrypted, api_secret_encrypted, created_at)
                    VALUES ($1, 1, 'test_exchange', 'test_account', 'encrypted', 'encrypted', NOW())
                    RETURNING id
                """, tenant_id)

        trades = []
        base_time = datetime.now() - timedelta(days=90)  # 3 months of data

        for i in range(count):
            # Realistic price movements
            symbol = random.choice(self.symbols)
            price = self._get_realistic_price(symbol)
            quantity = random.uniform(0.01, 10.0)
            side = random.choice(self.sides)
            gross_value = quantity * price
            net_value = gross_value * 0.999  # 0.1% fee

            trade = (
                f"TRADE_{i}_{str(tenant_id)[:8]}",  # trade_id
                account_id,  # account_id
                symbol,  # symbol
                side,  # side
                "limit",  # trade_type
                quantity,  # quantity
                price,  # price
                gross_value * 0.001,  # fee (0.1%)
                gross_value,  # gross_value
                net_value,  # net_value
                base_time + timedelta(minutes=random.randint(0, 90*24*60)),  # executed_at
                tenant_id  # tenant_id
            )
            trades.append(trade)

        # Batch insert for performance
        async with self.pool.acquire() as conn:
            try:
                await conn.executemany("""
                    INSERT INTO trades (trade_id, account_id, symbol, side, trade_type, quantity, price, fee, gross_value, net_value, executed_at, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, trades)
                print(f"    ✓ Inserted {count} trades")
            except Exception as e:
                print(f"    Error inserting trades: {e}")
                raise

    async def generate_positions(self, tenant_id: UUID, count: int = 50):
        """Generate test positions for a tenant."""
        print(f"  Generating {count} positions for tenant {tenant_id}...")

        # Get account_id for this tenant
        async with self.pool.acquire() as conn:
            account_id = await conn.fetchval(
                "SELECT id FROM trading_accounts WHERE tenant_id = $1 LIMIT 1", tenant_id
            )
            if not account_id:
                print(f"    ⚠️  No trading account found, skipping positions")
                return

        positions = []
        for i in range(count):
            symbol = random.choice(self.symbols)
            size = random.uniform(0.1, 100.0)
            entry_price = self._get_realistic_price(symbol)
            current_price = entry_price * random.uniform(0.9, 1.1)
            unrealized_pnl = (current_price - entry_price) * size
            side = random.choice(self.sides)

            position = (
                f"POS_{i}_{str(tenant_id)[:8]}",  # position_id
                account_id,  # account_id
                symbol,  # symbol
                side,  # side
                size,  # size
                entry_price,  # entry_price
                current_price,  # current_price
                unrealized_pnl,  # unrealized_pnl
                datetime.now() - timedelta(days=random.randint(0, 30)),  # opened_at
                tenant_id  # tenant_id
            )
            positions.append(position)

        # Batch insert
        async with self.pool.acquire() as conn:
            try:
                await conn.executemany("""
                    INSERT INTO positions (position_id, account_id, symbol, side, size, entry_price, current_price, unrealized_pnl, opened_at, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, positions)
                print(f"    ✓ Inserted {count} positions")
            except Exception as e:
                print(f"    Error inserting positions: {e}")
                raise

    def _get_realistic_price(self, symbol: str) -> float:
        """Get realistic price for a symbol."""
        price_ranges = {
            "BTC/USDT": (40000, 70000),
            "ETH/USDT": (2000, 4000),
            "SOL/USDT": (50, 200),
            "MATIC/USDT": (0.5, 2.0),
            "AVAX/USDT": (20, 50),
        }
        min_price, max_price = price_ranges.get(symbol, (10, 100))
        return random.uniform(min_price, max_price)

    async def generate_all_data(
        self,
        trades_per_tenant: int = 10000,
        positions_per_tenant: int = 50,
        users_per_tenant: int = 10
    ):
        """Generate complete test dataset for all tenants."""
        print("\n" + "="*80)
        print("GENERATING PRODUCTION-LIKE TEST DATA")
        print("="*80)

        # Check existing data
        existing = await self.check_existing_data()
        print("\nExisting data:")
        for table, count in existing.items():
            print(f"  {table}: {count:,} rows")

        if existing["trades"] >= trades_per_tenant * len(self.tenant_ids):
            print("\n⚠️  Sufficient test data already exists. Skipping generation.")
            print("   Use --force flag to regenerate data.")
            return

        print(f"\nGenerating data for {len(self.tenant_ids)} tenants...")
        print(f"  - {users_per_tenant} users per tenant")
        print(f"  - {trades_per_tenant} trades per tenant")
        print(f"  - {positions_per_tenant} positions per tenant")
        print()

        for tenant_id in self.tenant_ids:
            print(f"\nTenant: {tenant_id}")
            print("-" * 80)

            # Generate users
            await self.generate_users(tenant_id, users_per_tenant)

            # Generate trades (most critical for benchmarking)
            await self.generate_trades(tenant_id, trades_per_tenant)

            # Generate positions
            await self.generate_positions(tenant_id, positions_per_tenant)

        # Verify generated data
        print("\n" + "="*80)
        print("DATA GENERATION COMPLETE")
        print("="*80)

        final_counts = await self.check_existing_data()
        print("\nFinal row counts:")
        for table, count in final_counts.items():
            delta = count - existing[table]
            print(f"  {table}: {count:,} rows (+{delta:,})")

        print("\n✓ Test data ready for benchmarking")

    async def clean_test_data(self):
        """Clean up all test data (use with caution!)."""
        print("\n⚠️  WARNING: Cleaning all test data...")
        print("This will DELETE all rows from test tables!")

        response = input("Type 'DELETE' to confirm: ")
        if response != "DELETE":
            print("Aborted.")
            return

        async with self.pool.acquire() as conn:
            for table in ["trades", "positions", "users", "trading_accounts"]:
                count = await conn.fetchval(f"DELETE FROM {table} RETURNING COUNT(*)")
                print(f"  Deleted {count:,} rows from {table}")

        print("\n✓ Test data cleaned")

    async def run(self, action: str = "generate", **kwargs):
        """Execute data generation or cleanup."""
        try:
            await self.setup()

            if action == "generate":
                await self.generate_all_data(**kwargs)
            elif action == "clean":
                await self.clean_test_data()
            else:
                raise ValueError(f"Unknown action: {action}")

        finally:
            await self.teardown()


async def main():
    """CLI entry point for test data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate production-like test data for RLS benchmarking")
    parser.add_argument("--action", choices=["generate", "clean"], default="generate",
                       help="Action to perform (default: generate)")
    parser.add_argument("--trades", type=int, default=10000,
                       help="Number of trades per tenant (default: 10000)")
    parser.add_argument("--positions", type=int, default=50,
                       help="Number of positions per tenant (default: 50)")
    parser.add_argument("--users", type=int, default=10,
                       help="Number of users per tenant (default: 10)")
    parser.add_argument("--database-url", type=str,
                       default="postgresql://alphapulse:alphapulse@localhost:5432/alphapulse",
                       help="Database connection URL")

    args = parser.parse_args()

    generator = RLSTestDataGenerator(args.database_url)
    await generator.run(
        action=args.action,
        trades_per_tenant=args.trades,
        positions_per_tenant=args.positions,
        users_per_tenant=args.users
    )


if __name__ == "__main__":
    asyncio.run(main())
