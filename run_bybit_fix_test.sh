#!/bin/bash

# Run the Bybit fix and test script

# Ensure script is executable
chmod +x test_bybit_fix.py 2>/dev/null

# Run the script (main logs will be written to bybit_fix_results.log)
python3 test_bybit_fix.py

LOG_FILE="bybit_fix_results.log"

# Check exit code
if [ $? -eq 0 ]; then
  echo "Test completed successfully. The fix has been applied. See $LOG_FILE for details."
else
  echo "Test failed. Please check $LOG_FILE for details."
fi