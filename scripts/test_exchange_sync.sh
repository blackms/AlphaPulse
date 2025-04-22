#!/bin/bash
# Test script for the exchange sync module

# Create the database if it doesn't exist
echo "Setting up the database..."
chmod +x create_alphapulse_db.sh
./create_alphapulse_db.sh

if [ $? -ne 0 ]; then
    echo "Failed to create database. Please check the error messages above."
    exit 1
fi

# Set database environment variables
export DB_USER="testuser"
export DB_PASS="testpassword"
export DB_HOST="localhost"
export DB_PORT="5432" 
export DB_NAME="alphapulse"

# Set exchange sync configuration
export EXCHANGE_SYNC_EXCHANGES="bybit"
export EXCHANGE_SYNC_INTERVAL_MINUTES="1"
export EXCHANGE_SYNC_ENABLED="true"
export EXCHANGE_SYNC_LOG_LEVEL="DEBUG"

# Set API credentials as fallback (the module will first try to use AlphaPulse's credential manager)
echo "Note: The exchange sync module will first attempt to use credentials from AlphaPulse's credential manager"
echo "      before falling back to these environment variables."
export BYBIT_API_KEY="test_api_key"
export BYBIT_API_SECRET="test_api_secret"
export BYBIT_TESTNET="true"

# Run the example in one-time mode
echo "Running exchange sync test..."
python examples/exchange_sync/sync_example.py