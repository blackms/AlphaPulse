#!/bin/bash
# Run the database demo script

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running AlphaPulse database demo...${NC}"

# Run the demo script
echo -e "${YELLOW}Running database demo script...${NC}"
./demo_database.py

echo -e "${GREEN}Demo completed!${NC}"