#!/bin/bash

# AlphaPulse Dashboard Build Script

echo "Building AlphaPulse Dashboard Frontend..."

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

# Build the application
echo "Building production bundle..."
npm run build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "The build artifacts are in the 'build' directory."
else
    echo "Build failed. Please check the errors above."
    exit 1
fi