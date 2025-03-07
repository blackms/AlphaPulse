#!/bin/bash
# Run the monitoring system demonstration

# Ensure we're in the project root directory
cd "$(dirname "$0")/../.."

# Install required packages if needed
pip install -r src/alpha_pulse/monitoring/requirements.txt

# Run the demo
python examples/monitoring/demo_monitoring.py