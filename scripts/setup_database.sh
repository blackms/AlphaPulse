#!/bin/bash
# Setup script for AlphaPulse database infrastructure

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up AlphaPulse database infrastructure...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install asyncpg redis pyyaml sqlalchemy alembic

# Create directories if they don't exist
mkdir -p logs

# Start database containers
echo -e "${YELLOW}Starting database containers...${NC}"
docker-compose -f docker-compose.db.yml up -d

# Wait for PostgreSQL to be ready
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
for i in {1..30}; do
    if docker exec alphapulse-timescaledb pg_isready -U alphapulse > /dev/null 2>&1; then
        echo -e "${GREEN}PostgreSQL is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}Timed out waiting for PostgreSQL to be ready.${NC}"
        exit 1
    fi
done

# Wait for Redis to be ready
echo -e "${YELLOW}Waiting for Redis to be ready...${NC}"
for i in {1..30}; do
    if docker exec alphapulse-redis redis-cli -a password ping > /dev/null 2>&1; then
        echo -e "${GREEN}Redis is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}Timed out waiting for Redis to be ready.${NC}"
        exit 1
    fi
done

# Initialize the database
echo -e "${YELLOW}Initializing database...${NC}"
python src/scripts/init_db.py

echo -e "${GREEN}Database setup complete!${NC}"
echo -e "${YELLOW}PostgreSQL is running at localhost:5432${NC}"
echo -e "${YELLOW}Redis is running at localhost:6379${NC}"
echo -e "${YELLOW}PgAdmin is running at http://localhost:5050${NC}"
echo -e "${YELLOW}  - Email: admin@alphapulse.com${NC}"
echo -e "${YELLOW}  - Password: admin${NC}"

echo -e "${GREEN}You can now use the database infrastructure for AlphaPulse.${NC}"