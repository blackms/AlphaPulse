#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}  AI HEDGE FUND - CONNECTION DEBUGGING${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Function to test backend connectivity
test_backend() {
  echo -e "${YELLOW}Testing backend connectivity...${NC}"
  
  # Try to reach the backend root
  if curl -s https://platform.aigenconsult.com:9000/ > /dev/null; then
    echo -e "${GREEN}✓ Backend root endpoint is accessible${NC}"
  else
    echo -e "${RED}✗ Backend root endpoint is NOT accessible${NC}"
  fi
  
  # Try to reach the docs endpoint
  if curl -s https://platform.aigenconsult.com:9000/docs > /dev/null; then
    echo -e "${GREEN}✓ API documentation is accessible${NC}"
  else
    echo -e "${RED}✗ API documentation is NOT accessible${NC}"
  fi
  
  # Try the token endpoint
  TOKEN_RESPONSE=$(curl -s -X POST https://platform.aigenconsult.com:9000/token \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=admin&password=password")
  
  if [[ "$TOKEN_RESPONSE" == *"access_token"* ]]; then
    echo -e "${GREEN}✓ Token endpoint is working${NC}"
    TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"access_token":"[^"]*' | sed 's/"access_token":"//')
    echo -e "${BLUE}Token: ${TOKEN:0:20}...${NC}"
    
    # Test a protected endpoint
    SYSTEM_RESPONSE=$(curl -s -X GET https://platform.aigenconsult.com:9000/api/v1/system \
      -H "Authorization: Bearer $TOKEN")
    
    if [[ "$SYSTEM_RESPONSE" == *"cpu"* ]]; then
      echo -e "${GREEN}✓ System endpoint is working with authentication${NC}"
    else
      echo -e "${RED}✗ System endpoint is NOT working with authentication${NC}"
      echo -e "${YELLOW}Response: ${SYSTEM_RESPONSE}${NC}"
    fi
  else
    echo -e "${RED}✗ Token endpoint is NOT working${NC}"
    echo -e "${YELLOW}Response: ${TOKEN_RESPONSE}${NC}"
  fi
}

# Kill any existing processes on port 3000
echo -e "${BLUE}Checking for existing processes...${NC}"
FRONTEND_PID=$(lsof -ti:3000)

if [ ! -z "$FRONTEND_PID" ]; then
  echo -e "${YELLOW}Killing existing frontend process on port 3000 (PID: $FRONTEND_PID)${NC}"
  kill -9 $FRONTEND_PID
fi

# We're using a remote backend now, so no need to start a local one
echo -e "${BLUE}Using remote backend at https://platform.aigenconsult.com:9000${NC}"
echo -e "${YELLOW}Testing backend connectivity...${NC}"

# Test backend connectivity
test_backend

# Start the frontend with debug logging
echo -e "${BLUE}Starting frontend server...${NC}"
cd dashboard
# Enable browser console logging of network requests
echo "BROWSER_ARGS=--auto-open-devtools-for-tabs" > .env.development.local
# Enable detailed fetch/XHR logging
echo "REACT_APP_DEBUG=true" >> .env.development.local
npm start > ../frontend_debug.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}Frontend started with PID: $FRONTEND_PID${NC}"

echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}  Debugging Information${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${YELLOW}Backend URL: ${NC}https://platform.aigenconsult.com:9000"
echo -e "${YELLOW}Frontend URL: ${NC}http://localhost:3000"
echo -e "${YELLOW}Backend log: ${NC}backend_debug.log"
echo -e "${YELLOW}Frontend log: ${NC}frontend_debug.log"
echo ""
echo -e "${BLUE}To stop the frontend:${NC}"
echo -e "  kill -9 $FRONTEND_PID"
echo ""
echo -e "${YELLOW}Monitoring logs...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop monitoring (frontend will keep running)${NC}"
echo -e "${BLUE}===============================================${NC}"

# Monitor logs for connection issues
tail -f frontend_debug.log