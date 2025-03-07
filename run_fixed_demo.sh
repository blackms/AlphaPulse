#!/bin/bash
# Script to run the AI Hedge Fund demo with the PortfolioData fix applied

# Display header
echo "==============================================="
echo "  AI HEDGE FUND SYSTEM - FIXED DEMO"
echo "==============================================="
echo ""

# Apply the fix for PortfolioData
echo "Applying PortfolioData fix..."
python fix_portfolio_data.py

if [ $? -ne 0 ]; then
    echo "Failed to apply the PortfolioData fix! Aborting."
    exit 1
fi

echo "Fix applied successfully!"
echo ""

# Run the demo with the fixed components
echo "Starting the demo with fixed components..."
echo ""

# Skip API checks to avoid unnecessary failures during demo
export SKIP_API_CHECK=true

# Execute the main demo script
./run_demo.sh

# Restore environment
unset SKIP_API_CHECK