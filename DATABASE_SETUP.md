# Database Setup for AlphaPulse

This document explains how to set up the database for the AlphaPulse exchange synchronization module.

## Prerequisites

- PostgreSQL installed and running
- PostgreSQL client tools (`psql`) available in your PATH

## Automatic Setup

The easiest way to set up the database is to use the provided script:

```bash
# Make the script executable
chmod +x create_alphapulse_db.sh

# Run the script
./create_alphapulse_db.sh
```

This script will:
1. Create a database user `testuser` with password `testpassword` if it doesn't exist
2. Create a database named `alphapulse` owned by `testuser` if it doesn't exist
3. Grant all privileges on the database to the user

## Manual Setup

If you prefer to set up the database manually, follow these steps:

1. Connect to PostgreSQL as a superuser:
   ```bash
   psql -U postgres
   ```

2. Create the database user:
   ```sql
   CREATE USER testuser WITH PASSWORD 'testpassword';
   ```

3. Create the database:
   ```sql
   CREATE DATABASE alphapulse OWNER testuser;
   ```

4. Grant privileges:
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE alphapulse TO testuser;
   ```

## Configuration

The exchange sync module uses the following environment variables for database configuration:

```bash
export DB_USER="testuser"
export DB_PASS="testpassword"
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="alphapulse"
```

You can modify these values in the `test_exchange_sync.sh` script if needed.

## Tables

The following tables will be automatically created when you run the exchange sync module:

1. `portfolio_items` - Stores portfolio data for each exchange and asset
2. `sync_history` - Tracks synchronization operations

## Troubleshooting

If you encounter database connection issues:

1. Verify PostgreSQL is running:
   ```bash
   pg_isready -h localhost -p 5432
   ```

2. Check if the database exists:
   ```bash
   psql -U postgres -c "SELECT datname FROM pg_database WHERE datname='alphapulse';"
   ```

3. Verify the user has proper permissions:
   ```bash
   psql -U postgres -c "SELECT rolname, rolsuper, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname='testuser';"
   ```

4. Check PostgreSQL logs for any errors:
   ```bash
   tail -f /var/log/postgresql/postgresql-*.log