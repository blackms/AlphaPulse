#!/bin/bash

# Run the AlphaPulse Alerting System demo

# Set environment variables for the demo
export AP_ALERTING_ENABLED=true
export AP_ALERTING_CHECK_INTERVAL=5

# Create data directory if it doesn't exist
mkdir -p data

# Run the demo
echo "Starting AlphaPulse Alerting System demo..."
python examples/monitoring/demo_alerting.py

# Run the tests
echo ""
echo "Running alerting system tests..."
python -m unittest src/alpha_pulse/tests/test_alerting.py