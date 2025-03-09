# PostgreSQL Migration Implementation Guide

This document outlines the specific code changes required to remove SQLite support and standardize exclusively on PostgreSQL in the AlphaPulse system.

## Code Change Summary

### Primary Components Requiring Changes:

1. **Database Connection Module** (`src/alpha_pulse/data_pipeline/database/connection.py`)
2. **Connection Manager Module** (`src/alpha_pulse/data_pipeline/database/connection_manager.py`)
3. **Configuration and Environment Files**
4. **Testing Infrastructure**

## Detailed Implementation Plan

### 1. Database Connection Module Changes

```python
# src/alpha_pulse/data_pipeline/database/connection.py
```

- Remove SQLite imports:
  - Remove `import sqlite3`
  - Remove any SQLite-specific libraries/modules

- Remove DB_TYPE and SQLite configuration:
  - Remove `DB_TYPE = os.environ.get("DB_TYPE", "postgres").lower()`
  - Remove `SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", "alphapulse.db")`

- Update initialization function:
  - Remove conditional logic for DB_TYPE in `init_db()`
  - Simplify to only handle PostgreSQL initialization

- Remove any SQLite-specific connection functions
  
### 2. Connection Manager Module Changes

```python
# src/alpha_pulse/data_pipeline/database/connection_manager.py
```

- Remove SQLite imports:
  - Remove `import sqlite3`
  - Remove `import aiosqlite`

- Remove SQLite connection handling:
  - Remove variable `_sqlite_pools`
  - Refactor connection pool dictionary to only handle PostgreSQL pools
  
- Update connection creation logic:
  - Remove any conditional logic checking for DB_TYPE
  - Remove any SQLite-specific code paths

- Optimize PostgreSQL connection handling:
  - Update connection parameters specifically for PostgreSQL
  - Enable PostgreSQL-specific optimizations

### 3. Configuration Updates

- Remove SQLite-related environment variables:
  - `DB_TYPE` - No longer needed as PostgreSQL is the only supported option
  - `SQLITE_DB_PATH` - No longer relevant

- Update configuration files to remove SQLite options:
  - Update `config/database_config.yaml` or similar files
  - Simplify to only include PostgreSQL settings

### 4. Testing Infrastructure Changes

- Update test fixtures to use PostgreSQL exclusively
- Remove SQLite-specific test setup and teardown
- Update CI/CD configurations to ensure PostgreSQL availability

## Migration Guide for Users

For users currently using SQLite:

1. Install PostgreSQL if not already available:
   ```bash
   # Example for Ubuntu
   sudo apt update
   sudo apt install postgresql postgresql-contrib
   ```

2. Create a database for AlphaPulse:
   ```bash
   sudo -u postgres psql
   CREATE DATABASE alphapulse;
   CREATE USER testuser WITH PASSWORD 'testpassword';
   GRANT ALL PRIVILEGES ON DATABASE alphapulse TO testuser;
   ```

3. Update environment variables:
   ```bash
   # Remove
   # export DB_TYPE=sqlite
   # export SQLITE_DB_PATH=path/to/db.sqlite
   
   # Add/Update
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=alphapulse
   export DB_USER=testuser
   export DB_PASS=testpassword
   ```

4. Run database migrations to set up schema in PostgreSQL

## Testing Requirements

1. **Unit Tests**:
   - Update unit tests to connect to PostgreSQL only
   - Set up test database in CI environment

2. **Integration Tests**:
   - Test all data pipeline operations against PostgreSQL
   - Verify connection pooling under load

3. **Migration Tests**:
   - Test data migration scripts (if needed)
   - Verify schema compatibility

4. **Error Handling Tests**:
   - Test database connection failure scenarios
   - Verify retry logic works properly with PostgreSQL

## Post-Migration Optimizations

After removing SQLite support, consider these PostgreSQL-specific optimizations:

1. **Connection Pooling Tuning**:
   - Optimize connection pool size based on workload
   - Adjust timeout and lifetime parameters for production use

2. **Query Optimizations**:
   - Leverage PostgreSQL-specific SQL features
   - Implement advanced indexing strategies

3. **Transaction Management**:
   - Fine-tune transaction isolation levels
   - Optimize for specific access patterns

4. **Performance Monitoring**:
   - Add PostgreSQL-specific metrics collection
   - Set up monitoring for connection usage and query performance