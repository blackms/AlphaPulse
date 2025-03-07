#!/bin/bash

# AlphaPulse Dashboard Test Script

echo "Running AlphaPulse Dashboard Tests..."

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

# Run tests
echo "Running tests..."
npm test -- --watchAll=false

# Check if tests were successful
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Please check the errors above."
    exit 1
fi