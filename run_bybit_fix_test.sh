#!/bin/bash

# Run the Bybit fix and test script

# Ensure script is executable
chmod +x test_bybit_fix.py

# Run the script
python3 test_bybit_fix.py

# Check exit code
if [ $? -eq 0 ]; then
  echo "Test completed successfully. The fix has been applied."
else
  echo "Test failed. Please check the logs for details."
fi