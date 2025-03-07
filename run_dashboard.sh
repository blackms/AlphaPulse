#!/bin/bash

# Navigate to the dashboard directory
cd dashboard

# Install dependencies if they don't exist
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Start the development server
echo "Starting dashboard on http://localhost:3000"
npm start