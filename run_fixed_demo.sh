#!/bin/bash
# Script to run the AI Hedge Fund demo with all fixes applied

# Display header
echo "==============================================="
echo "  AI HEDGE FUND SYSTEM - FIXED DEMO"
echo "==============================================="
echo ""

# Apply fixes
echo "Applying PortfolioData fix..."
python patch_portfolio_data.py
if [ $? -ne 0 ]; then
  echo "❌ Failed to apply PortfolioData fix"
  exit 1
fi
echo "Fix applied successfully!"
echo ""

echo "Applying Portfolio Rebalancing fix..."
python fix_portfolio_rebalance.py
if [ $? -ne 0 ]; then
  echo "❌ Failed to apply Portfolio Rebalancing fix"
  exit 1
fi
echo "Fix applied successfully!"
echo ""

echo "Starting the demo with fixed components..."
echo ""

# Run the original demo script
./run_demo.sh