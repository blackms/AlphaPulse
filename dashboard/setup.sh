#!/bin/bash

# AlphaPulse Dashboard Setup Script

echo "Setting up AlphaPulse Dashboard Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js v16 or higher."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d 'v' -f 2 | cut -d '.' -f 1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "Node.js version is too old. Please install Node.js v16 or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install npm."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
npm install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "REACT_APP_API_URL=http://localhost:8000/api/v1" > .env
    echo "REACT_APP_WS_URL=ws://localhost:8000" >> .env
    echo ".env file created with default values."
fi

echo "Setup complete!"
echo "To start the development server, run: npm start"