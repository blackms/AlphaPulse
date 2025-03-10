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

# Create data directory if it doesn't exist
echo "Setting up data directory..."
mkdir -p data

# Set environment variable to use PostgreSQL
echo "Configuring to use PostgreSQL database..."
export ALPHA_PULSE_DB_TYPE=postgres
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=alphapulse
export DB_USER=alessio
export DB_PASS=""

# Initialize database
echo "Initializing database..."
python3 -c "from src.alpha_pulse.data_pipeline.database.connection import init_db; import asyncio; asyncio.run(init_db())"
if [ $? -ne 0 ]; then
    echo "Database initialization failed! Please check your PostgreSQL configuration."
    echo "Make sure PostgreSQL is running and accessible with the credentials in config/database_config.yaml"
    exit 1
fi

# Start the backend API server
echo "Starting backend API server..."
python3 run_api.py &
API_PID=$!

# Wait for API server to initialize
echo "Waiting for API server to initialize (10 seconds)..."
sleep 10

# Verify API is running
echo "Checking if API is running..."
max_retries=5
retry_count=0
while ! curl -s http://localhost:8000/ > /dev/null && [ $retry_count -lt $max_retries ]; do
    echo "API not ready yet, waiting (attempt $((retry_count+1))/$max_retries)..."
    sleep 3
    retry_count=$((retry_count+1))
done

if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "Error: API server is not responding after multiple attempts. Check logs for details."
    exit 1
fi
echo "API server is running at http://localhost:8000"

# Check API endpoints (optional)
if [ "$SKIP_API_CHECK" != "true" ]; then
    echo "Checking API endpoints (set SKIP_API_CHECK=true to bypass)..."
    python3 check_api_endpoints.py || true
    echo "Continuing with demo setup..."
fi

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
python3 trading/demo_ai_hedge_fund_live.py --live-data &
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