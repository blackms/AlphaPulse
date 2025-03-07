#!/bin/bash
# Run the alerting system demo

# Make the script executable
chmod +x demo_alerting.py

# Set up environment variables for testing
export AP_EMAIL_ENABLED=false
export AP_SLACK_ENABLED=false
export AP_SMS_ENABLED=false

# Run the demo
python demo_alerting.py