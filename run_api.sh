#!/bin/bash
# Script to run only the API component of the AI Hedge Fund system

# Display header
echo "==============================================="
echo "  AI HEDGE FUND - API SERVER ONLY"
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

# Check for database
if [ ! -f "data/hedge_fund.db" ]; then
  echo "Creating database directory..."
  mkdir -p data
  
  echo "Initializing database..."
  python src/scripts/init_db.py
  if [ $? -ne 0 ]; then
    echo "Database initialization failed! Check the error above."
    exit 1
  fi
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