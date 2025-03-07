#!/bin/bash

# AlphaPulse Dashboard Start Script

echo "Starting AlphaPulse Dashboard Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js v16 or higher."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Dependencies not found. Running setup script..."
    ./setup.sh
fi

# Start the development server
echo "Starting development server..."
npm start