#!/bin/bash

# AlphaPulse Dashboard Lint Script

echo "Running AlphaPulse Dashboard Linting..."

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

# Run linting
echo "Running ESLint..."
npm run lint

# Check if linting was successful
if [ $? -eq 0 ]; then
    echo "Linting passed!"
else
    echo "Linting failed. Please fix the issues above."
    exit 1
fi

# Run formatting check
echo "Running Prettier check..."
npx prettier --check "src/**/*.{js,jsx,ts,tsx,json,css,scss,md}"

# Check if formatting check was successful
if [ $? -eq 0 ]; then
    echo "Formatting check passed!"
else
    echo "Formatting check failed. Run 'npm run format' to fix formatting issues."
    exit 1
fi