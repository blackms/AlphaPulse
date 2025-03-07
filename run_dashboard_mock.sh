#!/bin/bash
# Script to run the dashboard with mock data

# Display header
echo "==============================================="
echo "  AI HEDGE FUND - DASHBOARD WITH MOCK DATA"
echo "==============================================="
echo ""

# Check if API is running
echo "Checking if API is running..."
if curl -s http://localhost:8000/ > /dev/null; then
  echo "✅ API is running at http://localhost:8000"
else
  echo "⚠️  Warning: API does not seem to be running at http://localhost:8000"
  echo "   The dashboard will use mock data, but some features may not work correctly."
  echo ""
fi

# Navigate to dashboard directory
cd dashboard || {
  echo "Error: dashboard directory not found!"
  echo "Make sure you're running this script from the project root."
  exit 1
}

# Copy mock environment file
echo "Setting up mock data environment..."
cp .env.mock .env

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
echo "Starting dashboard with mock data..."
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