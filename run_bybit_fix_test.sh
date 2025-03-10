#!/bin/bash
# Simple script to test the fixed Bybit order history functionality

echo "Testing Bybit order history fix..."
python test_bybit_fix.py

if [ $? -eq 0 ]; then
    echo "SUCCESS: Test passed!"
    exit 0
else
    echo "ERROR: Test failed!"
    exit 1
fi