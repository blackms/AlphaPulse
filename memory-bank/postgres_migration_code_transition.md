# PostgreSQL Migration Code Transition Guide

This document provides detailed implementation specifications for transitioning AlphaPulse to PostgreSQL-only support.

## 1. Database Connection Module Changes

**File:** `src/alpha_pulse/data_pipeline/database/connection.py`

### Code to Remove

```python
# Remove these imports
import sqlite3

# Remove these constants
DB_TYPE = os.environ.get("DB_TYPE", "postgres").lower()
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "alphapulse.db")
```

### Code to Modify

Replace the `init_db()` function:

```python
# From:
async def init_db():
    """
    Initialize the database connection and tables.
    
    This function is used during application startup to ensure
    the database is properly initialized.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    thread_id = threading.get_ident()
    logger.info(f"[THREAD {thread_id}] Initializing database")
    
    try:
        if DB_TYPE == "postgres":
            # Initialize the thread-local pool
            await _get_thread_pg_pool()
            return True
        elif DB_TYPE == "sqlite":
            # For SQLite, implement initialization here
            logger.warning(f"Database initialization for {DB_TYPE} not fully implemented")
            # TODO: Implement SQLite initialization
            return True
        else:
            logger.warning(f"Unknown database type: {DB_TYPE}")
            logger.warning(f"Supported types are: postgres, sqlite")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        logger.error("The application may not function correctly without database access")
        # Return False but don't re-raise the exception to allow the application to start
        # with degraded functionality
        return False

# To:
async def init_db():
    """
    Initialize the PostgreSQL database connection and tables.
    
    This function is used during application startup to ensure
    the database is properly initialized.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    thread_id = threading.get_ident()
    logger.info(f"[THREAD {thread_id}] Initializing PostgreSQL database")
    
    try:
        # Initialize the thread-local pool
        await _get_thread_pg_pool()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {str(e)}")
        logger.error("The application may not function correctly without database access")
        # Return False but don't re-raise the exception to allow the application to start
        # with degraded functionality
        return False
```

## 2. Connection Manager Module Changes

**File:** `src/alpha_pulse/data_pipeline/database/connection_manager.py`

### Code to Remove

```python
# Remove these imports
import sqlite3
import aiosqlite

# Remove these variables 
_sqlite_pools = {}  # Special storage for SQLite connections
```

### PostgreSQL Optimizations

Optimize the PostgreSQL connection pool creation:

```python
# Enhanced PostgreSQL pool creation with better defaults
pool = await asyncpg.create_pool(
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_user,
    password=db_pass,
    min_size=5,                # Increased from 3 for better performance
    max_size=30,               # Increased from 20 for higher concurrency
    command_timeout=60.0,      # 60 second timeout for commands
    max_inactive_connection_lifetime=300.0,  # 5 minutes idle time
    # Server-side statement cache and other PostgreSQL-specific optimizations
    server_settings={
        'statement_timeout': '30s',          # 30 second statement timeout
        'idle_in_transaction_session_timeout': '60s',  # Prevent hung transactions
        'application_name': 'AlphaPulse',    # Identify app in pg_stat_activity
        'client_min_messages': 'notice'      # Reduce log noise
    }
)
```

## 3. Configuration Updates

**File:** `config/database_config.yaml`

Remove SQLite configuration options and standardize on PostgreSQL:

```yaml
# From:
database:
  type: ${DB_TYPE:postgres}
  postgres:
    host: ${DB_HOST:localhost}
    port: ${DB_PORT:5432}
    name: ${DB_NAME:alphapulse}
    user: ${DB_USER:testuser}
    password: ${DB_PASS:testpassword}
  sqlite:
    path: ${SQLITE_DB_PATH:alphapulse.db}

# To:
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  name: ${DB_NAME:alphapulse}
  user: ${DB_USER:testuser}
  password: ${DB_PASS:testpassword}
  # PostgreSQL-specific settings
  min_connections: ${DB_MIN_CONNECTIONS:5}
  max_connections: ${DB_MAX_CONNECTIONS:30}
  command_timeout: ${DB_COMMAND_TIMEOUT:60}
  connection_lifetime: ${DB_CONNECTION_LIFETIME:300}
```

## 4. Test Infrastructure Changes

**File:** `tests/conftest.py` (or similar test fixtures)

Update test fixtures to use PostgreSQL exclusively:

```python
# From:
@pytest.fixture
async def test_db():
    """Create a test database connection."""
    if os.environ.get("DB_TYPE", "postgres").lower() == "postgres":
        # Setup PostgreSQL test database
        # ...
    else:
        # Setup SQLite test database
        # ...

# To:
@pytest.fixture
async def test_db():
    """Create a PostgreSQL test database connection."""
    # Setup PostgreSQL test database
    # ...
```

## 5. Run Scripts and Documentation

**File:** `run_api.sh` (and other scripts)

Update to remove SQLite options:

```bash
# From:
#!/bin/bash
# Run API with SQLite or PostgreSQL
if [ "$1" == "sqlite" ]; then
    export DB_TYPE=sqlite
    export SQLITE_DB_PATH=alphapulse.db
    echo "Using SQLite database at $SQLITE_DB_PATH"
else
    export DB_TYPE=postgres
    echo "Using PostgreSQL database"
fi
# ...

# To:
#!/bin/bash
# Run API with PostgreSQL
echo "Using PostgreSQL database"
# ...
```

## 6. Error Handling Improvements

**File:** `src/alpha_pulse/data_pipeline/database/connection_manager.py`

Enhance PostgreSQL-specific error handling:

```python
# Add/modify error handling specific to PostgreSQL
except (asyncpg.PostgresConnectionError, 
        asyncpg.CannotConnectNowError, 
        asyncpg.TooManyConnectionsError) as e:
    # Handle specific PostgreSQL connection issues
    error_code = getattr(e, 'sqlstate', None)
    if error_code:
        # Log PostgreSQL-specific error codes for better diagnostics
        logger.error(f"PostgreSQL error code: {error_code}")
        
        # Handle specific error conditions
        if error_code == '57P01':  # admin_shutdown
            logger.warning("Database server is shutting down, retrying...")
        elif error_code == '53300':  # too_many_connections
            logger.warning("Too many database connections, waiting for availability...")
    
    # Continue with retry logic
    last_error = e
    retry_count += 1
    # ...
```

## 7. Testing Requirements

All test files that currently support both database engines should be updated to only use PostgreSQL:

1. Remove conditional DB_TYPE checking in test files
2. Update test environment setup scripts
3. Modify CI/CD configurations to ensure PostgreSQL is available
4. Update test assertions if they contain SQLite-specific logic

## 8. Performance Monitoring

Add PostgreSQL-specific monitoring capabilities:

```python
async def get_db_stats():
    """
    Get PostgreSQL database statistics for monitoring.
    """
    async with get_db_connection() as conn:
        # Query for active connections
        connections = await conn.fetch("""
            SELECT datname, usename, application_name, state, query, query_start
            FROM pg_stat_activity
            WHERE datname = $1
        """, os.environ.get("DB_NAME", DEFAULT_DB_NAME))
        
        # Query for table statistics
        table_stats = await conn.fetch("""
            SELECT relname, n_live_tup, n_dead_tup, last_vacuum, last_analyze
            FROM pg_stat_user_tables
        """)
        
        # Return formatted stats
        return {
            "connections": [dict(row) for row in connections],
            "tables": [dict(row) for row in table_stats]
        }