#!/usr/bin/env python3
"""
Simple database connection check script without external dependencies.
"""
import asyncio
import os
import sys
from datetime import datetime

# Import from the new connection manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.alpha_pulse.data_pipeline.database.connection_manager import get_db_connection


async def check_database():
    """Check database connection."""
    print(f"[{datetime.now().isoformat()}] Checking database connection...")
    
    try:
        async with get_db_connection() as conn:
            # Run a simple query
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                print(f"[{datetime.now().isoformat()}] Database connection successful!")
                return True
            else:
                print(f"[{datetime.now().isoformat()}] Database returned unexpected result: {result}")
                return False
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Database connection failed: {str(e)}")
        print(f"[{datetime.now().isoformat()}] Exception type: {type(e).__name__}")
        return False


async def main():
    """Main function."""
    success = await check_database()
    if success:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    asyncio.run(main())