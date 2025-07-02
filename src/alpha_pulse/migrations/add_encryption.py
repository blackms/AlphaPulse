"""
Database migration to add encryption to existing tables.

This migration adds encrypted columns alongside existing columns
and provides utilities for data migration.
"""
import argparse
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import text, inspect

from ..config.database import get_db_config, EncryptedDatabaseMigration
from ..models.trading_data import Base as TradingBase
from ..models.user_data import Base as UserBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncryptionMigration:
    """Manages the encryption migration process."""
    
    def __init__(self):
        self.db_config = get_db_config()
        self.engine = self.db_config.create_engine()
        self.migration_helper = EncryptedDatabaseMigration(self.db_config)
        self.inspector = inspect(self.engine)
        
    def get_migration_plan(self) -> List[Dict[str, Any]]:
        """Get the list of columns to encrypt."""
        migration_plan = []
        
        # Trading data migrations
        trading_migrations = [
            {
                "table": "trading_accounts",
                "migrations": [
                    ("account_number", "account_number_encrypted", "trading_account"),
                    ("balance", "balance_encrypted", "account_balance"),
                    ("available_balance", "available_balance_encrypted", "account_balance"),
                    ("api_key_reference", "api_key_reference_encrypted", "api_credentials")
                ]
            },
            {
                "table": "positions",
                "migrations": [
                    ("size", "size_encrypted", "position_size"),
                    ("entry_price", "entry_price_encrypted", "position_pricing"),
                    ("current_price", "current_price_encrypted", "position_pricing"),
                    ("unrealized_pnl", "unrealized_pnl_encrypted", "position_pnl"),
                    ("realized_pnl", "realized_pnl_encrypted", "position_pnl"),
                    ("stop_loss", "stop_loss_encrypted", "position_risk"),
                    ("take_profit", "take_profit_encrypted", "position_risk"),
                    ("metadata", "metadata_encrypted", "position_metadata")
                ]
            },
            {
                "table": "trades",
                "migrations": [
                    ("quantity", "quantity_encrypted", "trade_execution"),
                    ("price", "price_encrypted", "trade_execution"),
                    ("fee", "fee_encrypted", "trade_execution"),
                    ("gross_value", "gross_value_encrypted", "trade_financial"),
                    ("net_value", "net_value_encrypted", "trade_financial"),
                    ("order_id", "order_id_encrypted", "trade_order"),
                    ("execution_details", "execution_details_encrypted", "trade_details")
                ]
            }
        ]
        
        # User data migrations
        user_migrations = [
            {
                "table": "users",
                "migrations": [
                    ("email", "email_encrypted", "user_pii"),
                    ("phone_number", "phone_number_encrypted", "user_pii"),
                    ("first_name", "first_name_encrypted", "user_personal"),
                    ("last_name", "last_name_encrypted", "user_personal"),
                    ("address_data", "address_data_encrypted", "user_address"),
                    ("two_factor_secret", "two_factor_secret_encrypted", "user_security"),
                    ("preferences", "preferences_encrypted", "user_preferences")
                ]
            },
            {
                "table": "api_keys",
                "migrations": [
                    ("key_hash", "key_hash_encrypted", "api_key"),
                    ("permissions", "permissions_encrypted", "api_permissions"),
                    ("usage_stats", "usage_stats_encrypted", "api_usage")
                ]
            }
        ]
        
        # Combine all migrations
        all_migrations = trading_migrations + user_migrations
        
        for table_info in all_migrations:
            table_name = table_info["table"]
            
            # Check if table exists
            if not self.inspector.has_table(table_name):
                logger.warning(f"Table {table_name} does not exist, skipping")
                continue
            
            for old_col, new_col, context in table_info["migrations"]:
                migration_plan.append({
                    "table": table_name,
                    "old_column": old_col,
                    "new_column": new_col,
                    "context": context
                })
        
        return migration_plan
    
    def add_encrypted_columns(self):
        """Add encrypted columns to existing tables."""
        logger.info("Adding encrypted columns to tables...")
        
        migration_plan = self.get_migration_plan()
        
        for migration in migration_plan:
            table = migration["table"]
            new_column = migration["new_column"]
            
            # Check if column already exists
            columns = [col["name"] for col in self.inspector.get_columns(table)]
            if new_column in columns:
                logger.info(f"Column {table}.{new_column} already exists, skipping")
                continue
            
            # Add encrypted column
            try:
                with self.engine.connect() as conn:
                    conn.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {new_column} TEXT")
                    )
                    conn.commit()
                logger.info(f"Added column {table}.{new_column}")
            except Exception as e:
                logger.error(f"Failed to add column {table}.{new_column}: {e}")
    
    def add_search_indexes(self):
        """Add search token indexes for searchable fields."""
        logger.info("Adding search indexes...")
        
        search_indexes = [
            ("trading_accounts", "account_number_search", "idx_trading_account_search"),
            ("users", "email_search", "idx_user_email_search")
        ]
        
        for table, column, index_name in search_indexes:
            try:
                # Check if index exists
                indexes = self.inspector.get_indexes(table)
                if any(idx["name"] == index_name for idx in indexes):
                    logger.info(f"Index {index_name} already exists, skipping")
                    continue
                
                # Add search column if not exists
                columns = [col["name"] for col in self.inspector.get_columns(table)]
                if column not in columns:
                    with self.engine.connect() as conn:
                        conn.execute(
                            text(f"ALTER TABLE {table} ADD COLUMN {column} VARCHAR(64)")
                        )
                        conn.commit()
                
                # Create index
                with self.engine.connect() as conn:
                    conn.execute(
                        text(f"CREATE INDEX {index_name} ON {table}({column})")
                    )
                    conn.commit()
                logger.info(f"Created index {index_name}")
                
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
    
    def migrate_data(self, batch_size: int = 1000, table_filter: str = None):
        """
        Migrate data from unencrypted to encrypted columns.
        
        Args:
            batch_size: Number of records to process at once
            table_filter: Optional table name to migrate only specific table
        """
        logger.info("Starting data migration...")
        
        # Create migration tracking table
        self.migration_helper.create_migration_tables()
        
        migration_plan = self.get_migration_plan()
        
        for migration in migration_plan:
            table = migration["table"]
            
            # Apply table filter if specified
            if table_filter and table != table_filter:
                continue
            
            old_column = migration["old_column"]
            new_column = migration["new_column"]
            
            logger.info(f"Migrating {table}.{old_column} to {new_column}")
            
            try:
                self.migration_helper.migrate_table_column(
                    table_name=table,
                    old_column=old_column,
                    new_column=new_column,
                    batch_size=batch_size
                )
            except Exception as e:
                logger.error(f"Migration failed for {table}.{old_column}: {e}")
                # Continue with other migrations
    
    def verify_migration(self) -> Dict[str, Any]:
        """Verify the migration was successful."""
        logger.info("Verifying migration...")
        
        results = {
            "total_tables": 0,
            "total_columns": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "verification_details": []
        }
        
        with self.engine.connect() as conn:
            # Check migration status
            migration_status = conn.execute(
                text("SELECT * FROM encryption_migrations ORDER BY table_name, column_name")
            ).fetchall()
            
            for status in migration_status:
                results["total_columns"] += 1
                
                detail = {
                    "table": status.table_name,
                    "column": status.column_name,
                    "status": status.status,
                    "records_processed": status.records_processed,
                    "error": status.error_message
                }
                
                if status.status == "completed":
                    results["successful_migrations"] += 1
                else:
                    results["failed_migrations"] += 1
                
                results["verification_details"].append(detail)
            
            # Count unique tables
            results["total_tables"] = len(set(s.table_name for s in migration_status))
        
        return results
    
    def create_migration_views(self):
        """Create views that use encrypted columns transparently."""
        logger.info("Creating migration views...")
        
        view_definitions = [
            {
                "view_name": "v_trading_accounts",
                "definition": """
                CREATE OR REPLACE VIEW v_trading_accounts AS
                SELECT 
                    id,
                    account_id,
                    account_number_encrypted as account_number,
                    exchange_name,
                    account_type,
                    balance_encrypted as balance,
                    available_balance_encrypted as available_balance,
                    created_at,
                    updated_at,
                    is_active
                FROM trading_accounts
                """
            },
            {
                "view_name": "v_users_safe",
                "definition": """
                CREATE OR REPLACE VIEW v_users_safe AS
                SELECT 
                    id,
                    username,
                    is_active,
                    is_verified,
                    created_at,
                    last_login_at
                FROM users
                """
            }
        ]
        
        for view in view_definitions:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(view["definition"]))
                    conn.commit()
                logger.info(f"Created view {view['view_name']}")
            except Exception as e:
                logger.error(f"Failed to create view {view['view_name']}: {e}")
    
    def drop_unencrypted_columns(self, confirm: bool = False):
        """
        Drop unencrypted columns after successful migration.
        
        Args:
            confirm: Must be True to actually drop columns
        """
        if not confirm:
            logger.warning("Drop operation not confirmed. Use --confirm to proceed.")
            return
        
        logger.warning("DROPPING UNENCRYPTED COLUMNS - THIS CANNOT BE UNDONE!")
        
        migration_plan = self.get_migration_plan()
        
        for migration in migration_plan:
            table = migration["table"]
            old_column = migration["old_column"]
            
            # Check migration was successful
            with self.engine.connect() as conn:
                status = conn.execute(
                    text(
                        "SELECT status FROM encryption_migrations "
                        "WHERE table_name = :table AND column_name = :column"
                    ),
                    {"table": table, "column": old_column}
                ).fetchone()
                
                if not status or status.status != "completed":
                    logger.warning(f"Skipping {table}.{old_column} - migration not completed")
                    continue
                
                # Drop column
                try:
                    conn.execute(
                        text(f"ALTER TABLE {table} DROP COLUMN {old_column}")
                    )
                    conn.commit()
                    logger.info(f"Dropped column {table}.{old_column}")
                except Exception as e:
                    logger.error(f"Failed to drop column {table}.{old_column}: {e}")


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate database to use encryption"
    )
    parser.add_argument(
        "--add-columns",
        action="store_true",
        help="Add encrypted columns to tables"
    )
    parser.add_argument(
        "--migrate-data",
        action="store_true",
        help="Migrate data to encrypted columns"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration status"
    )
    parser.add_argument(
        "--create-views",
        action="store_true",
        help="Create database views for transparent access"
    )
    parser.add_argument(
        "--drop-unencrypted",
        action="store_true",
        help="Drop unencrypted columns (requires --confirm)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive operations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for data migration"
    )
    parser.add_argument(
        "--table",
        type=str,
        help="Migrate only specific table"
    )
    
    args = parser.parse_args()
    
    if not any([args.add_columns, args.migrate_data, args.verify, 
                args.create_views, args.drop_unencrypted]):
        parser.print_help()
        sys.exit(1)
    
    migration = EncryptionMigration()
    
    try:
        if args.add_columns:
            migration.add_encrypted_columns()
            migration.add_search_indexes()
        
        if args.migrate_data:
            migration.migrate_data(
                batch_size=args.batch_size,
                table_filter=args.table
            )
        
        if args.verify:
            results = migration.verify_migration()
            logger.info(f"Migration verification results:")
            logger.info(f"  Total tables: {results['total_tables']}")
            logger.info(f"  Total columns: {results['total_columns']}")
            logger.info(f"  Successful: {results['successful_migrations']}")
            logger.info(f"  Failed: {results['failed_migrations']}")
            
            if results['failed_migrations'] > 0:
                logger.error("Some migrations failed:")
                for detail in results['verification_details']:
                    if detail['status'] != 'completed':
                        logger.error(f"  {detail['table']}.{detail['column']}: {detail['error']}")
        
        if args.create_views:
            migration.create_migration_views()
        
        if args.drop_unencrypted:
            migration.drop_unencrypted_columns(confirm=args.confirm)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()