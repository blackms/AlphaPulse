"""
Database configuration with transparent encryption support.

This module configures SQLAlchemy with encryption support and
provides utilities for secure database operations.
"""
import os
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
import logging

from .secure_settings import get_settings
from ..utils.encryption import EncryptionKeyManager

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration with encryption support."""
    
    def __init__(self):
        self.settings = get_settings()
        self.key_manager = EncryptionKeyManager()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        
    def get_connection_url(self) -> str:
        """Get database connection URL from settings."""
        db_config = self.settings.database
        
        # Support for different database types
        db_type = os.environ.get("DB_TYPE", "postgresql")
        
        if db_type == "postgresql":
            # Use async driver for better performance
            driver = "postgresql+asyncpg"
            url = f"{driver}://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        else:
            # Default to PostgreSQL
            url = db_config.connection_string
            
        return url
    
    def create_engine(self, **kwargs) -> Engine:
        """
        Create SQLAlchemy engine with encryption support.
        
        Args:
            **kwargs: Additional engine configuration
            
        Returns:
            Configured SQLAlchemy engine
        """
        if self._engine is not None:
            return self._engine
        
        # Default engine configuration
        default_config = {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "poolclass": QueuePool,
            "echo": self.settings.debug,
            "connect_args": {
                "server_settings": {
                    "application_name": "AlphaPulse",
                    "jit": "off"
                },
                "command_timeout": 60,
                "prepared_statement_cache_size": 0,
            }
        }
        
        # Merge with provided kwargs
        config = {**default_config, **kwargs}
        
        # Create engine
        connection_url = self.get_connection_url()
        self._engine = create_engine(connection_url, **config)
        
        # Configure encryption
        self._configure_encryption(self._engine)
        
        # Configure performance optimizations
        self._configure_performance(self._engine)
        
        logger.info("Database engine created with encryption support")
        return self._engine
    
    def _configure_encryption(self, engine: Engine):
        """Configure encryption for the database engine."""
        
        @event.listens_for(engine, "connect")
        def set_encryption_key(dbapi_conn, connection_record):
            """Set encryption key for the connection."""
            # Store key manager reference in connection
            connection_record.info["key_manager"] = self.key_manager
            
            # Enable SSL if configured
            if self.settings.database.get("ssl_mode"):
                cursor = dbapi_conn.cursor()
                cursor.execute("SET ssl_mode = %s", (self.settings.database.ssl_mode,))
                cursor.close()
        
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log sensitive queries in debug mode only."""
            if self.settings.debug and logger.isEnabledFor(logging.DEBUG):
                # Mask sensitive parameters
                safe_params = self._mask_sensitive_params(parameters)
                logger.debug(f"Executing query: {statement[:100]}... with params: {safe_params}")
    
    def _configure_performance(self, engine: Engine):
        """Configure performance optimizations."""
        
        @event.listens_for(engine, "connect")
        def configure_performance(dbapi_conn, connection_record):
            """Set performance parameters."""
            cursor = dbapi_conn.cursor()
            
            # PostgreSQL specific optimizations
            performance_settings = [
                "SET work_mem = '32MB'",
                "SET maintenance_work_mem = '128MB'", 
                "SET effective_cache_size = '1GB'",
                "SET random_page_cost = 1.1",
                "SET effective_io_concurrency = 200",
                "SET max_parallel_workers_per_gather = 4"
            ]
            
            for setting in performance_settings:
                try:
                    cursor.execute(setting)
                except Exception as e:
                    logger.warning(f"Could not apply setting '{setting}': {e}")
            
            cursor.close()
    
    def _mask_sensitive_params(self, parameters: Any) -> Any:
        """Mask sensitive parameters for logging."""
        if not parameters:
            return parameters
        
        # List of parameter names to mask
        sensitive_keywords = [
            "password", "secret", "key", "token", "credential",
            "ssn", "email", "phone", "account_number"
        ]
        
        if isinstance(parameters, dict):
            masked = {}
            for key, value in parameters.items():
                if any(keyword in key.lower() for keyword in sensitive_keywords):
                    masked[key] = "***MASKED***"
                else:
                    masked[key] = value
            return masked
        
        return parameters
    
    def create_session_factory(self) -> sessionmaker:
        """Create session factory with encryption support."""
        if self._session_factory is not None:
            return self._session_factory
        
        if self._engine is None:
            self.create_engine()
        
        # Configure session
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        return self._session_factory
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if self._session_factory is None:
            self.create_session_factory()
        
        session = self._session_factory()
        
        # Attach key manager to session
        session.info["key_manager"] = self.key_manager
        
        return session
    
    def dispose_engine(self):
        """Dispose of the engine and clean up connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database engine disposed")


class EncryptedDatabaseMigration:
    """Utilities for migrating to encrypted database."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.engine = db_config.create_engine()
        
    def create_migration_tables(self):
        """Create tables for tracking migration progress."""
        migration_sql = """
        CREATE TABLE IF NOT EXISTS encryption_migrations (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(100) NOT NULL,
            column_name VARCHAR(100) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            records_processed INTEGER DEFAULT 0,
            error_message TEXT,
            UNIQUE(table_name, column_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_migration_status 
        ON encryption_migrations(status);
        """
        
        with self.engine.connect() as conn:
            conn.execute(migration_sql)
            conn.commit()
    
    def migrate_table_column(
        self,
        table_name: str,
        old_column: str,
        new_column: str,
        batch_size: int = 1000
    ):
        """
        Migrate a column to encrypted format.
        
        Args:
            table_name: Name of the table
            old_column: Name of unencrypted column
            new_column: Name of encrypted column
            batch_size: Number of records to process at once
        """
        from ..utils.encryption import encrypt_field
        
        migration_sql = f"""
        WITH batch AS (
            SELECT id, {old_column}
            FROM {table_name}
            WHERE {new_column} IS NULL
            AND {old_column} IS NOT NULL
            LIMIT %s
        )
        UPDATE {table_name} t
        SET {new_column} = %s
        FROM batch b
        WHERE t.id = b.id
        RETURNING t.id;
        """
        
        # Track migration progress
        self._update_migration_status(table_name, old_column, "in_progress")
        
        total_processed = 0
        
        try:
            with self.engine.connect() as conn:
                while True:
                    # Get batch of records
                    result = conn.execute(
                        f"SELECT id, {old_column} FROM {table_name} "
                        f"WHERE {new_column} IS NULL AND {old_column} IS NOT NULL "
                        f"LIMIT {batch_size}"
                    )
                    
                    records = result.fetchall()
                    if not records:
                        break
                    
                    # Encrypt values
                    for record in records:
                        encrypted = encrypt_field(
                            record[1],
                            context=f"{table_name}_{old_column}"
                        )
                        
                        conn.execute(
                            f"UPDATE {table_name} SET {new_column} = %s WHERE id = %s",
                            (json.dumps(encrypted), record[0])
                        )
                    
                    conn.commit()
                    total_processed += len(records)
                    
                    logger.info(f"Migrated {total_processed} records in {table_name}.{old_column}")
            
            self._update_migration_status(
                table_name, 
                old_column, 
                "completed",
                records_processed=total_processed
            )
            
        except Exception as e:
            logger.error(f"Migration failed for {table_name}.{old_column}: {str(e)}")
            self._update_migration_status(
                table_name,
                old_column,
                "failed",
                error_message=str(e)
            )
            raise
    
    def _update_migration_status(
        self,
        table_name: str,
        column_name: str,
        status: str,
        records_processed: int = 0,
        error_message: str = None
    ):
        """Update migration tracking status."""
        with self.engine.connect() as conn:
            if status == "in_progress":
                conn.execute(
                    """
                    INSERT INTO encryption_migrations 
                    (table_name, column_name, status, started_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (table_name, column_name)
                    DO UPDATE SET status = %s, started_at = NOW()
                    """,
                    (table_name, column_name, status, status)
                )
            elif status == "completed":
                conn.execute(
                    """
                    UPDATE encryption_migrations
                    SET status = %s, completed_at = NOW(), records_processed = %s
                    WHERE table_name = %s AND column_name = %s
                    """,
                    (status, records_processed, table_name, column_name)
                )
            elif status == "failed":
                conn.execute(
                    """
                    UPDATE encryption_migrations
                    SET status = %s, error_message = %s
                    WHERE table_name = %s AND column_name = %s
                    """,
                    (status, error_message, table_name, column_name)
                )
            
            conn.commit()


# Global database configuration instance
_db_config: Optional[DatabaseConfig] = None


def get_db_config() -> DatabaseConfig:
    """Get or create global database configuration."""
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig()
    return _db_config


def get_db_session() -> Session:
    """Get a new database session."""
    return get_db_config().get_session()


# Export main classes and functions
__all__ = [
    "DatabaseConfig",
    "EncryptedDatabaseMigration",
    "get_db_config",
    "get_db_session"
]