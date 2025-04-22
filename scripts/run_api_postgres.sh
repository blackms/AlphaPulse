#!/bin/bash
# Script to run only the API component of the AI Hedge Fund system with PostgreSQL

# Display header
echo "==============================================="
echo "  AI HEDGE FUND - API SERVER (PostgreSQL)"
echo "==============================================="
echo ""

# Check if API is already running
if curl -s http://localhost:8000/ > /dev/null; then
  echo "⚠️  Warning: An API appears to be already running at http://localhost:8000"
  echo "   This might cause conflicts. Make sure to stop any existing server first."
  echo ""
  read -p "Continue anyway? (y/n): " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Please stop the existing API server first."
    exit 1
  fi
fi

# Setting up environment
echo "Setting up environment..."
export PYTHONPATH=./src:$PYTHONPATH

# Ensure data directory exists
echo "Creating data directory..."
mkdir -p data

# Set up PostgreSQL connection
echo "Configuring PostgreSQL connection..."
# Use environment variables or config defaults
export ALPHA_PULSE_DB_TYPE=postgres
# Set database connection parameters
export DB_NAME=alphapulse
export DB_USER=testuser
export DB_PASS=testpassword
# Note: Additional PostgreSQL connection parameters can be set in config/database_config.yaml

# Initialize database if needed
echo "Checking database connection..."
python -c "from src.alpha_pulse.data_pipeline.database.connection import init_db; import asyncio; asyncio.run(init_db())"
if [ $? -ne 0 ]; then
  echo "Database connection failed! Please check your PostgreSQL configuration."
  echo "Make sure PostgreSQL is running and accessible with the credentials in config/database_config.yaml"
  exit 1
fi

# Start the API server
echo "Starting API server..."
echo "The API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Available endpoints:"
echo "- Authentication: POST /token"
echo "- Portfolio Data: GET /api/v1/portfolio"
echo "- Metrics: GET /api/v1/metrics/portfolio"
echo "- Trades: GET /api/v1/trades"
echo "- Alerts: GET /api/v1/alerts"
echo "- System Status: GET /api/v1/system"
echo ""
echo "Default credentials:"
echo "  Username: admin"
echo "  Password: password"
echo ""
echo "Press Ctrl+C to stop the API server"
echo "==============================================="

# Run the API server
python src/scripts/run_api.py