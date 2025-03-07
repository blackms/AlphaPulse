#!/usr/bin/env python
"""
Run API tests for the AlphaPulse Dashboard Backend.

This script runs the API tests with various options for filtering and reporting.
"""
import os
import sys
import argparse
import subprocess
import pytest


def main():
    """Run API tests."""
    parser = argparse.ArgumentParser(description="Run API tests for AlphaPulse Dashboard Backend")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--filter", "-f", help="Filter tests by name")
    parser.add_argument("--mark", "-m", help="Run tests with specific mark (e.g., integration, performance)")
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = ["pytest", "src/alpha_pulse/tests/api"]
    
    if args.verbose:
        pytest_args.append("-v")
        
    if args.coverage:
        pytest_args.extend(["--cov=alpha_pulse.api", "--cov-report=term"])
        
    if args.html:
        pytest_args.append("--html=reports/api_tests.html")
        
    if args.filter:
        pytest_args.append(f"-k {args.filter}")
        
    if args.mark:
        pytest_args.append(f"-m {args.mark}")
    
    # Run tests
    print(f"Running API tests with command: {' '.join(pytest_args)}")
    result = subprocess.run(pytest_args)
    
    # Return exit code
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())