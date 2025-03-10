#!/bin/bash
# Script to create the AlphaPulse database

# Database connection parameters
DB_USER="testuser"
DB_PASS="testpassword"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="alphapulse"

echo "Creating AlphaPulse database..."

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL client not found. Please install PostgreSQL."
    exit 1
fi

# Create the database user if it doesn't exist
echo "Creating database user $DB_USER if it doesn't exist..."
PGPASSWORD="postgres" psql -h $DB_HOST -p $DB_PORT -U postgres -c "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 || \
PGPASSWORD="postgres" psql -h $DB_HOST -p $DB_PORT -U postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"

# Create the database if it doesn't exist
echo "Creating database $DB_NAME if it doesn't exist..."
PGPASSWORD="postgres" psql -h $DB_HOST -p $DB_PORT -U postgres -c "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 || \
PGPASSWORD="postgres" psql -h $DB_HOST -p $DB_PORT -U postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# Grant privileges
echo "Granting privileges to $DB_USER on $DB_NAME..."
PGPASSWORD="postgres" psql -h $DB_HOST -p $DB_PORT -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

echo "Database setup complete!"