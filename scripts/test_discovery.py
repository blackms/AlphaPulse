#!/usr/bin/env python3
"""
Script to discover which tests can be collected without errors.
"""
import subprocess
import sys
from pathlib import Path

def test_single_file(test_file):
    """Try to collect tests from a single file."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "--collect-only", "-q",
        str(test_file)
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        env={**subprocess.os.environ, "PYTHONPATH": "./src"}
    )
    
    return result.returncode == 0, result.stdout, result.stderr

def main():
    test_dir = Path("src/alpha_pulse/tests")
    test_files = sorted(test_dir.glob("test_*.py"))
    
    working_tests = []
    failed_tests = []
    
    print(f"Testing {len(test_files)} test files...\n")
    
    for test_file in test_files:
        success, stdout, stderr = test_single_file(test_file)
        
        if success and "error" not in stderr.lower():
            # Count collected tests
            collected = 0
            for line in stdout.split('\n'):
                if '<Module' in line or '<Function' in line:
                    collected += 1
            
            if collected > 0:
                working_tests.append((test_file, collected))
                print(f"✓ {test_file.name}: {collected} tests")
        else:
            failed_tests.append(test_file)
            print(f"✗ {test_file.name}: Collection failed")
            if stderr.strip():
                print(f"  Error: {stderr.strip()[:100]}...")
    
    print(f"\n\nSummary:")
    print(f"Working test files: {len(working_tests)}")
    print(f"Failed test files: {len(failed_tests)}")
    print(f"Total tests that can be collected: {sum(count for _, count in working_tests)}")
    
    print(f"\n\nWorking test files for CI:")
    for test_file, count in working_tests:
        print(f"  {test_file}")

if __name__ == "__main__":
    main()