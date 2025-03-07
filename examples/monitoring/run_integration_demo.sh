#!/bin/bash

# Run the AlphaPulse Metrics Collector and Alerting System integration demo

# Set environment variables for the demo
export AP_ALERTING_ENABLED=true
export AP_ALERTING_CHECK_INTERVAL=5
export AP_MONITORING_INTERVAL=5

# Create data directory if it doesn't exist
mkdir -p data

# Run the demo
echo "Starting AlphaPulse Metrics Collector and Alerting System integration demo..."
python examples/monitoring/demo_collector_alerting_integration.py

# Run the integration test
echo ""
echo "Running integration test..."
python -m unittest src/alpha_pulse/tests/test_monitoring_alerting_integration.py