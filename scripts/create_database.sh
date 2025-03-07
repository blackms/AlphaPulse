#!/bin/bash
# Create the alphapulse database in PostgreSQL

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Creating AlphaPulse database...${NC}"

# Database connection parameters
DB_HOST=${DB_HOST:-"localhost"}
DB_PORT=${DB_PORT:-"5432"}
DB_USER=${DB_USER:-"postgres"}
DB_PASSWORD=${DB_PASSWORD:-""}
DB_NAME=${DB_NAME:-"alphapulse"}

# Check if PostgreSQL client is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}PostgreSQL client is not installed. Please install it first.${NC}"
    echo "On Ubuntu/Debian: sudo apt-get install postgresql-client"
    echo "On CentOS/RHEL: sudo yum install postgresql"
    echo "On macOS: brew install postgresql"
    exit 1
fi

# Create .pgpass file for passwordless authentication
if [ ! -z "$DB_PASSWORD" ]; then
    echo -e "${YELLOW}Setting up temporary .pgpass file...${NC}"
    PGPASS_FILE=~/.pgpass
    if [ -f "$PGPASS_FILE" ]; then
        # Backup existing .pgpass file
        cp $PGPASS_FILE ${PGPASS_FILE}.bak
    fi
    echo "$DB_HOST:$DB_PORT:*:$DB_USER:$DB_PASSWORD" > $PGPASS_FILE
    chmod 600 $PGPASS_FILE
fi

# Check if the database already exists
if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo -e "${YELLOW}Database '$DB_NAME' already exists.${NC}"
else
    # Create the database
    echo -e "${YELLOW}Creating database '$DB_NAME'...${NC}"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME;"
    echo -e "${GREEN}Database '$DB_NAME' created successfully.${NC}"
fi

# Check if TimescaleDB extension is available
if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb';" | grep -q 1; then
    echo -e "${GREEN}TimescaleDB extension is available.${NC}"
else
    echo -e "${YELLOW}TimescaleDB extension is not available. Some features may not work properly.${NC}"
    echo -e "${YELLOW}Consider installing TimescaleDB: https://docs.timescale.com/install/latest/self-hosted/${NC}"
fi

# Restore original .pgpass file if it was backed up
if [ -f "${PGPASS_FILE}.bak" ]; then
    mv ${PGPASS_FILE}.bak $PGPASS_FILE
    chmod 600 $PGPASS_FILE
fi

echo -e "${GREEN}Database setup complete!${NC}"
echo -e "${YELLOW}You can now initialize the database schema with:${NC}"
echo -e "${YELLOW}  python src/scripts/init_db.py${NC}"