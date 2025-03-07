#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}  AI HEDGE FUND - FULL APPLICATION LAUNCHER${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
  echo -e "${RED}Error: .env file not found!${NC}"
  echo "Creating a default .env file..."
  cat > .env << EOL
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
API_LOG_LEVEL=INFO

# Database Configuration
DB_TYPE=sqlite
DB_NAME=alpha_pulse.db
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres

# Exchange API Keys
BYBIT_API_KEY=rYo8Mt01xzflkn33lZ
BYBIT_API_SECRET=KcpAnOXpVh14bkxPTSpsLu0HWEioWkClCeya

# JWT Configuration
JWT_SECRET=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Feature Flags
ENABLE_REAL_DATA=true
ENABLE_DEMO_MODE=false
ENABLE_WEBSOCKETS=true
EOL
  echo -e "${GREEN}Default .env file created.${NC}"
fi

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if tmux is installed
if ! command_exists tmux; then
  echo -e "${RED}Error: tmux is not installed. Please install it first.${NC}"
  echo "On Ubuntu/Debian: sudo apt-get install tmux"
  echo "On macOS with Homebrew: brew install tmux"
  exit 1
fi

# Kill any existing tmux session with the same name
tmux kill-session -t alpha_pulse 2>/dev/null

# Create a new tmux session
echo -e "${BLUE}Starting services in tmux session...${NC}"
tmux new-session -d -s alpha_pulse

# Split the window horizontally
tmux split-window -h -t alpha_pulse

# Start the API server in the left pane
tmux send-keys -t alpha_pulse:0.0 "echo -e '${GREEN}Starting API server...${NC}'" C-m
tmux send-keys -t alpha_pulse:0.0 "./run_api_sqlite.sh" C-m

# Wait for API to start
echo "Waiting for API server to start..."
sleep 5

# Start the dashboard in the right pane
tmux send-keys -t alpha_pulse:0.1 "echo -e '${GREEN}Starting dashboard...${NC}'" C-m
tmux send-keys -t alpha_pulse:0.1 "cd dashboard && npm start" C-m

# Attach to the tmux session
echo -e "${GREEN}Services started. Attaching to tmux session...${NC}"
echo -e "${BLUE}Press Ctrl+B then D to detach from the session without stopping the services.${NC}"
echo -e "${BLUE}To reattach later, run: tmux attach -t alpha_pulse${NC}"
echo -e "${BLUE}To stop all services, run: tmux kill-session -t alpha_pulse${NC}"
echo ""

tmux attach -t alpha_pulse

# This will only execute if the user detaches from the tmux session
echo -e "${GREEN}Detached from tmux session. Services are still running.${NC}"
echo -e "${BLUE}To reattach, run: tmux attach -t alpha_pulse${NC}"
echo -e "${BLUE}To stop all services, run: tmux kill-session -t alpha_pulse${NC}"