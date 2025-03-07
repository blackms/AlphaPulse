#!/bin/bash
# Script to run only the dashboard component of the AI Hedge Fund system

# Display header
echo "==============================================="
echo "  AI HEDGE FUND - DASHBOARD ONLY"
echo "==============================================="
echo ""

# Check if the API is running
echo "Checking if API is already running..."
if curl -s http://localhost:8000/ > /dev/null; then
  echo "✅ API appears to be running at http://localhost:8000"
else
  echo "⚠️  Warning: API does not seem to be running at http://localhost:8000"
  echo "   You should start the API first with: './run_api.sh'"
  echo "   The dashboard will start, but it won't have any data to display."
  echo ""
  read -p "Continue anyway? (y/n): " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Please start the API first."
    exit 1
  fi
fi

# Navigate to dashboard directory
cd dashboard || {
  echo "Error: dashboard directory not found!"
  echo "Make sure you're running this script from the project root."
  exit 1
}

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "Installing dashboard dependencies..."
  npm install
  if [ $? -ne 0 ]; then
    echo "Failed to install dependencies! Please check your npm installation."
    exit 1
  fi
fi

# Start the dashboard
echo "Starting dashboard development server..."
echo "The dashboard will be available at: http://localhost:3000"
echo ""
echo "Default login credentials:"
echo "  Username: admin"
echo "  Password: password"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "==============================================="

# Run the development server
npm start