#!/bin/bash
# Test the database connection with sudo

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing AlphaPulse database with sudo...${NC}"

# Test PostgreSQL connection
echo -e "${YELLOW}Testing PostgreSQL connection...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT COUNT(*) FROM alphapulse.users;"

# Test Redis connection
echo -e "${YELLOW}Testing Redis connection...${NC}"
redis-cli ping
redis-cli set test_key test_value
redis-cli get test_key
redis-cli del test_key

echo -e "${GREEN}Database tests completed!${NC}"