# AI Hedge Fund Demo Script

Below is a shell script that can be used to run all components of the AI Hedge Fund system together with real data. Copy this script to `run_demo.sh` in the project root and make it executable.

```bash
#!/bin/bash
# AI Hedge Fund System Demo Script
# This script starts all components of the AI Hedge Fund system and connects them together

echo "==============================================="
echo "  AI HEDGE FUND SYSTEM - INTEGRATED DEMO"
echo "==============================================="
echo

# Function to clean up when the script exits
cleanup() {
    echo
    echo "Shutting down all services..."
    
    # Kill background processes
    if [ ! -z "$API_PID" ]; then
        echo "Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi
    
    if [ ! -z "$DATA_GEN_PID" ]; then
        echo "Stopping data generator (PID: $DATA_GEN_PID)..."
        kill $DATA_GEN_PID 2>/dev/null
    fi
    
    echo "Cleanup complete."
    exit 0
}

# Set up trap to call cleanup function on exit
trap cleanup EXIT INT TERM

# Check for required software
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm is required but not found"
    exit 1
fi

# Setup environment
echo "Setting up environment..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env 2>/dev/null
fi

# Start the backend API server
echo "Starting backend API server..."
cd src
python3 -m alpha_pulse.api.main &
API_PID=$!
cd ..

# Wait for API server to initialize
echo "Waiting for API server to initialize (5 seconds)..."
sleep 5

# Verify API is running
echo "Checking if API is running..."
if ! curl -s http://localhost:8000/api/v1/system/health > /dev/null; then
    echo "Error: API server is not responding. Check logs for details."
    exit 1
fi
echo "API server is running at http://localhost:8000"

# Configure frontend to use the real backend
echo "Configuring frontend..."
cd dashboard
if [ ! -f ".env" ]; then
    echo "Creating dashboard .env file..."
    cat > .env << EOL
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_USE_MOCK_DATA=false
EOL
fi

# Install dependencies and start frontend
echo "Installing frontend dependencies..."
npm install --silent

echo "Starting frontend development server..."
npm start &
FRONTEND_PID=$!
cd ..

# Wait for frontend to initialize
echo "Waiting for frontend to initialize (10 seconds)..."
sleep 10

# Start data generation
echo "Starting data generation to provide real-time data..."
cd examples
python3 trading/demo_ai_hedge_fund.py --live-data &
DATA_GEN_PID=$!
cd ..

echo
echo "==============================================="
echo "  AI HEDGE FUND SYSTEM IS NOW RUNNING"
echo "==============================================="
echo
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo
echo "Login with these credentials:"
echo "  Username: admin"
echo "  Password: password"
echo
echo "Press Ctrl+C to stop all components"
echo

# Keep the script running until Ctrl+C is pressed
wait $API_PID
```

## How to Use This Script

1. Copy the script content to a file named `run_demo.sh` in the project root directory
2. Make the script executable:
   ```bash
   chmod +x run_demo.sh
   ```
3. Run the script:
   ```bash
   ./run_demo.sh
   ```

## What This Script Does

1. Starts the backend API server
2. Configures the frontend to connect to the real backend
3. Starts the React development server for the dashboard
4. Launches the data generation script to provide real-time data
5. Sets up proper cleanup when you exit with Ctrl+C

## Troubleshooting

If you encounter any issues:

1. Check the console output for error messages
2. Verify all required ports are available (8000 for API, 3000 for frontend)
3. Ensure all dependencies are installed
4. Look for any service-specific error logs