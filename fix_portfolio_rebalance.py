#!/usr/bin/env python3
"""
Patch to fix the coroutine error in PortfolioManager._retry_with_timeout method.

The issue is in the _retry_with_timeout method where it's trying to call a coroutine
object directly with (). This script patches the method to properly await the coroutine.
"""

import sys
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PORTFOLIO_MANAGER_PATH = Path("src/alpha_pulse/portfolio/portfolio_manager.py")

def read_file(path):
    """Read a file and return its contents."""
    try:
        with open(path, "r") as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        sys.exit(1)

def write_file(path, content):
    """Write content to a file."""
    try:
        with open(path, "w") as file:
            file.write(content)
    except Exception as e:
        logger.error(f"Error writing to file {path}: {e}")
        sys.exit(1)

def fix_retry_with_timeout(file_content):
    """
    Fix the _retry_with_timeout method to properly handle coroutines.
    
    The issue is in this line:
    return await asyncio.wait_for(coro_func(), timeout=timeout)
    
    Where coro_func is already a coroutine object, not a function returning a coroutine.
    """
    # Look for the problematic method
    if "_retry_with_timeout" not in file_content:
        logger.error("Could not find _retry_with_timeout method in file")
        return None
    
    # Find the method definition and the problematic line
    lines = file_content.split("\n")
    method_start = None
    problematic_line = None
    
    for i, line in enumerate(lines):
        if "async def _retry_with_timeout" in line:
            method_start = i
        if method_start is not None and "asyncio.wait_for(coro_func()" in line:
            problematic_line = i
            break
    
    if problematic_line is None:
        logger.error("Could not find the problematic line in _retry_with_timeout method")
        return None
    
    # Fix the problematic line
    lines[problematic_line] = lines[problematic_line].replace("coro_func()", "coro_func")
    
    return "\n".join(lines)

async def main():
    """Main function to apply the fix."""
    logger.info(f"Reading {PORTFOLIO_MANAGER_PATH}")
    file_content = read_file(PORTFOLIO_MANAGER_PATH)
    
    logger.info("Fixing _retry_with_timeout method")
    fixed_content = fix_retry_with_timeout(file_content)
    
    if fixed_content:
        logger.info(f"Writing fixed content to {PORTFOLIO_MANAGER_PATH}")
        write_file(PORTFOLIO_MANAGER_PATH, fixed_content)
        logger.info("Fix applied successfully!")
    else:
        logger.error("Failed to apply fix")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())