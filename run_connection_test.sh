#!/bin/bash
# Script to test database connection handling in AlphaPulse
# This will verify our fixes for concurrent operation errors

echo "===== AlphaPulse Database Connection Pool Test ====="
echo "Running tests to verify connection pooling, transaction handling, and error recovery"
echo "Looking for 'InterfaceError: cannot perform operation: another operation is in progress'"
echo "and 'ConnectionDoesNotExistError: connection was closed in the middle of operation'"

# Run the connection test script with extended timeouts
python3 test_connection_pool.py

# Check for errors in the log file
echo ""
echo "===== Test Results Summary ====="
echo "Checking logs for connection errors..."
newest_log=$(ls -t connection_pool_test_*.log | head -1)

if [ -f "$newest_log" ]; then
  # Count error occurrences
  interface_errors=$(grep -c "InterfaceError" "$newest_log")
  connection_closed_errors=$(grep -c "ConnectionDoesNotExistError" "$newest_log")
  loop_errors=$(grep -c "attached to a different loop" "$newest_log")
  
  # Show summary
  echo "Log file: $newest_log"
  echo "InterfaceError occurrences: $interface_errors"
  echo "ConnectionDoesNotExistError occurrences: $connection_closed_errors"
  echo "Event loop conflict occurrences: $loop_errors"
  
  # Success/failure message
  total_errors=$((interface_errors + connection_closed_errors + loop_errors))
  if [ $total_errors -eq 0 ]; then
    echo "SUCCESS: No database connection errors detected! The fixes appear effective."
  else
    echo "WARNING: $total_errors database connection errors detected. Further improvements may be needed."
  fi
else
  echo "No log file found. Check if the test script ran correctly."
fi

echo ""
echo "Test completed. See full logs for details."