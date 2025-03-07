#!/bin/bash
# Run the API server and demo

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running AlphaPulse API demo...${NC}"

# Check if the API server is already running
if nc -z localhost 8000 2>/dev/null; then
    echo -e "${YELLOW}API server is already running on port 8000${NC}"
else
    echo -e "${YELLOW}Starting API server...${NC}"
    # Start the API server in the background
    cd ../../src && python -m scripts.run_api --reload &
    API_PID=$!
    
    # Wait for the server to start
    echo -e "${YELLOW}Waiting for API server to start...${NC}"
    for i in {1..10}; do
        if nc -z localhost 8000 2>/dev/null; then
            echo -e "${GREEN}API server started successfully!${NC}"
            break
        fi
        if [ $i -eq 10 ]; then
            echo -e "${RED}Failed to start API server${NC}"
            kill $API_PID 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
fi

# Install required packages if not already installed
echo -e "${YELLOW}Checking required packages...${NC}"
pip install -q requests websockets

# Run the demo script
echo -e "${YELLOW}Running API demo script...${NC}"
./demo_api.py

# If we started the API server, stop it
if [ -n "$API_PID" ]; then
    echo -e "${YELLOW}Stopping API server...${NC}"
    kill $API_PID 2>/dev/null || true
fi

echo -e "${GREEN}Demo completed!${NC}"