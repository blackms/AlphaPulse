#!/bin/bash
# Script to set up the test database for AlphaPulse

# Database configuration
DB_NAME="alphapulse"
DB_USER="testuser"
DB_PASS="testpassword"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL is not installed. Please install it first."
    exit 1
fi

# Check if the database exists and is accessible
echo "Checking database connection..."
if ! sudo -u postgres psql -c "SELECT 1" $DB_NAME > /dev/null 2>&1; then
    echo "Creating database $DB_NAME..."
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;"
fi

# Check if user exists
echo "Checking if user $DB_USER exists..."
if ! sudo -u postgres psql -c "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" postgres | grep -q 1; then
    echo "Creating database user $DB_USER..."
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
fi

# Grant privileges on the database to the user
echo "Granting privileges to $DB_USER..."
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Connect to the database and grant schema privileges
echo "Setting up schema privileges..."
sudo -u postgres psql -d $DB_NAME -c "GRANT ALL ON SCHEMA public TO $DB_USER;"
sudo -u postgres psql -d $DB_NAME -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;"
sudo -u postgres psql -d $DB_NAME -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;"
sudo -u postgres psql -d $DB_NAME -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO $DB_USER;"

# Drop all tables in the database to start fresh
echo "Dropping existing tables..."
sudo -u postgres psql -d $DB_NAME -c "
DO \$\$ DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END \$\$;
"

# Determine the location of the virtual environment Python
if [ -d "venv" ]; then
    PYTHON_PATH="./venv/bin/python"
elif [ -d "../venv" ]; then
    PYTHON_PATH="../venv/bin/python"
else
    PYTHON_PATH="python3"
fi

# Run the Alembic migration to create the tables
echo "Running Alembic migration to create database tables..."
cd migrations
export DB_TYPE=postgres
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=$DB_NAME
export DB_USER=$DB_USER
export DB_PASS=$DB_PASS

echo "Using Python: $PYTHON_PATH"
$PYTHON_PATH -m alembic upgrade head

if [ $? -eq 0 ]; then
    echo "Database tables created successfully!"
else
    echo "Failed to create database tables. Check the logs for details."
    exit 1
fi

echo "Database setup completed successfully!"
echo "You can now run the tests with:"
echo "DB_TYPE=postgres DB_NAME=$DB_NAME DB_USER=$DB_USER DB_PASS=$DB_PASS python3 ./test_database_connection.py"