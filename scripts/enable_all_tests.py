#!/usr/bin/env python3
"""
Script to progressively enable all tests and identify what needs to be fixed.
"""
import subprocess
import sys
from pathlib import Path
import json

def run_pytest_collect(test_file):
    """Try to collect tests from a file and return detailed error info."""
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
    
    # Parse the error to identify the issue
    error_type = None
    error_detail = None
    
    if result.returncode != 0:
        stderr = result.stderr
        if "ModuleNotFoundError" in stderr:
            error_type = "missing_module"
            # Extract module name
            for line in stderr.split('\n'):
                if "No module named" in line:
                    error_detail = line.split("'")[1]
                    break
        elif "ImportError" in stderr:
            error_type = "import_error"
            error_detail = stderr.split('\n')[0]
        elif "AttributeError" in stderr:
            error_type = "attribute_error"
            error_detail = stderr.split('\n')[0]
        elif "metadata" in stderr and "reserved" in stderr:
            error_type = "metadata_error"
            error_detail = "SQLAlchemy metadata attribute conflict"
        else:
            error_type = "other"
            error_detail = stderr.split('\n')[0] if stderr else "Unknown error"
    
    return result.returncode == 0, error_type, error_detail

def main():
    test_dir = Path("src/alpha_pulse/tests")
    test_files = sorted(test_dir.glob("test_*.py"))
    
    results = {
        "working": [],
        "fixable": {
            "missing_module": [],
            "import_error": [],
            "attribute_error": [],
            "metadata_error": [],
            "other": []
        }
    }
    
    print(f"Analyzing {len(test_files)} test files...\n")
    
    for test_file in test_files:
        success, error_type, error_detail = run_pytest_collect(test_file)
        
        if success:
            results["working"].append(str(test_file))
            print(f"✓ {test_file.name}")
        else:
            results["fixable"][error_type].append({
                "file": str(test_file),
                "error": error_detail
            })
            print(f"✗ {test_file.name} - {error_type}: {error_detail}")
    
    # Generate report
    print("\n" + "="*80)
    print("TEST ENABLEMENT REPORT")
    print("="*80)
    
    print(f"\nWorking tests: {len(results['working'])}")
    for test in results["working"]:
        print(f"  - {Path(test).name}")
    
    print(f"\nTests needing fixes:")
    for error_type, tests in results["fixable"].items():
        if tests:
            print(f"\n{error_type.upper()} ({len(tests)} files):")
            for test_info in tests[:5]:  # Show first 5
                print(f"  - {Path(test_info['file']).name}: {test_info['error']}")
            if len(tests) > 5:
                print(f"  ... and {len(tests) - 5} more")
    
    # Save detailed report
    with open("test_analysis_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to test_analysis_report.json")
    
    # Generate CI config
    print("\n" + "="*80)
    print("RECOMMENDED CI CONFIGURATION")
    print("="*80)
    
    if results["working"]:
        print("\n# Add these tests to CI immediately:")
        for test in results["working"]:
            print(f"  {test} \\")

if __name__ == "__main__":
    main()