#!/bin/bash

# Run the Dashboard API demo
# This script starts the API server and runs the demo client

# Set environment variables
export AP_JWT_SECRET="dev-secret-key"

# Start the API server in the background
echo "Starting API server..."
python -m src.scripts.run_api --reload &
API_PID=$!

# Wait for the server to start
echo "Waiting for API server to start..."
sleep 5

# Run the demo client
echo "Running demo client..."
python examples/monitoring/demo_dashboard_api.py

# Cleanup
echo "Stopping API server..."
kill $API_PID

echo "Demo completed"