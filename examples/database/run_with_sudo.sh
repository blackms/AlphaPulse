#!/bin/bash
# Run the database demo script with sudo for PostgreSQL access

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running AlphaPulse database demo with sudo...${NC}"

# Run the demo script with sudo
echo -e "${YELLOW}Running database demo script...${NC}"
sudo -u postgres PYTHONPATH=../../ python3 ./demo_database.py

echo -e "${GREEN}Demo completed!${NC}"