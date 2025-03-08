#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}  RESTARTING FRONTEND WITH WEBPACK CONFIG FIX${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${YELLOW}Backend URL: ${NC}https://platform.aigenconsult.com"
echo ""

# Kill any existing frontend process on port 3000
echo -e "${BLUE}Checking for existing processes...${NC}"
FRONTEND_PID=$(lsof -ti:3000)

if [ ! -z "$FRONTEND_PID" ]; then
  echo -e "${YELLOW}Killing existing frontend process on port 3000 (PID: $FRONTEND_PID)${NC}"
  kill -9 $FRONTEND_PID
fi

# Start the frontend with debug logging
echo -e "${BLUE}Starting frontend server with fixed webpack configuration...${NC}"
cd dashboard

# Enable browser console logging of network requests
echo "BROWSER_ARGS=--auto-open-devtools-for-tabs" > .env.development.local
echo "REACT_APP_DEBUG=true" >> .env.development.local

# Create proper webpack development configuration
echo "REACT_APP_API_URL=http://localhost:8000" > .env.development
echo "REACT_APP_WS_URL=ws://localhost:8000/ws" >> .env.development
echo "WDS_SOCKET_HOST=localhost" >> .env.development
echo "WDS_SOCKET_PORT=3000" >> .env.development
echo "HOST=localhost" >> .env.development
echo "PORT=3000" >> .env.development
echo "DANGEROUSLY_DISABLE_HOST_CHECK=true" >> .env.development

# Start the React development server
npm start > ../frontend_debug.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}Frontend started with PID: $FRONTEND_PID${NC}"

echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}  Frontend Restarted with Fixed Configuration${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${YELLOW}Frontend URL: ${NC}http://localhost:3000"
echo -e "${YELLOW}Frontend log: ${NC}frontend_debug.log"
echo ""
echo -e "${BLUE}The frontend should now connect to the backend server${NC}"
echo -e "${BLUE}through the proxy configuration with fixed webpack settings${NC}"
echo ""
echo -e "${YELLOW}To stop the frontend:${NC}"
echo -e "  kill -9 $FRONTEND_PID"
echo ""