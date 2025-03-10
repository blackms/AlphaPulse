#!/bin/bash

# Run Bybit order history test script
# This script sets up the environment and runs the test_bybit_order_history.py script

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===============================================${NC}"
echo -e "${YELLOW}  BYBIT ORDER HISTORY TEST${NC}"
echo -e "${YELLOW}===============================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking required packages...${NC}"
python3 -c "import aiohttp, ccxt, asyncio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install aiohttp ccxt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install required packages${NC}"
        exit 1
    fi
fi

# Check if credentials are set
if [ -z "$BYBIT_API_KEY" ] || [ -z "$BYBIT_API_SECRET" ]; then
    echo -e "${YELLOW}No API credentials found in environment variables.${NC}"
    echo -e "${YELLOW}Will try to use credentials from credentials_manager.${NC}"
fi

# Make the test script executable
chmod +x test_bybit_order_history.py

# Run the test script
echo -e "${YELLOW}Running Bybit order history test...${NC}"
python3 test_bybit_order_history.py

# Check the result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Test completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Test failed!${NC}"
    exit 1
fi