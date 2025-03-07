#!/bin/bash

# Set up environment
export NODE_ENV=development
export REACT_APP_API_URL=http://localhost:5000/api

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting AI Hedge Fund Dashboard...${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed.${NC}"
    echo -e "Please install Node.js to run the dashboard."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed.${NC}"
    echo -e "Please install npm to run the dashboard."
    exit 1
fi

# Navigate to dashboard directory
cd dashboard

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install dependencies.${NC}"
        exit 1
    fi
fi

# Check if API is running
echo -e "${YELLOW}Checking API connection...${NC}"
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo -e "${GREEN}API is running.${NC}"
else
    echo -e "${YELLOW}Warning: API does not appear to be running at http://localhost:5000/api${NC}"
    echo -e "You may have issues with data fetching. Would you like to continue? (y/n)"
    read -r response
    if [[ "$response" =~ ^([nN][oO]|[nN])$ ]]; then
        exit 0
    fi
fi

# Start the dashboard in development mode
echo -e "${GREEN}Starting dashboard development server...${NC}"
echo -e "${YELLOW}The dashboard will be available at http://localhost:3000${NC}"
npm start

# Exit with npm's exit code
exit $?