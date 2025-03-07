#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}  AI HEDGE FUND - DASHBOARD WITH REAL API${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed. Please install it first.${NC}"
    echo "On Ubuntu/Debian: sudo apt-get install tmux"
    echo "On macOS with Homebrew: brew install tmux"
    exit 1
fi

# Kill any existing tmux session with the same name
tmux kill-session -t alphapulse 2>/dev/null

# Create a new tmux session
echo -e "${BLUE}Starting services in tmux session...${NC}"
tmux new-session -d -s alphapulse

# Split the window horizontally
tmux split-window -h -t alphapulse

# Start the API server in the left pane
tmux send-keys -t alphapulse:0.0 "echo -e '${GREEN}Starting API server...${NC}'" C-m
tmux send-keys -t alphapulse:0.0 "./run_api_sqlite.sh" C-m

# Wait for API to start
echo "Waiting for API server to start..."
sleep 5

# Start the dashboard in the right pane
tmux send-keys -t alphapulse:0.1 "echo -e '${GREEN}Starting dashboard...${NC}'" C-m
tmux send-keys -t alphapulse:0.1 "cd dashboard && npm start" C-m

# Attach to the tmux session
echo -e "${GREEN}Services started. Attaching to tmux session...${NC}"
echo -e "${BLUE}Press Ctrl+B then D to detach from the session without stopping the services.${NC}"
echo -e "${BLUE}To reattach later, run: tmux attach -t alphapulse${NC}"
echo -e "${BLUE}To stop all services, run: tmux kill-session -t alphapulse${NC}"
echo ""

tmux attach -t alphapulse

# This will only execute if the user detaches from the tmux session
echo -e "${GREEN}Detached from tmux session. Services are still running.${NC}"
echo -e "${BLUE}To reattach, run: tmux attach -t alphapulse${NC}"
echo -e "${BLUE}To stop all services, run: tmux kill-session -t alphapulse${NC}"