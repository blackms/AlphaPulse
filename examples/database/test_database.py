#!/usr/bin/env python
"""
Test script for AlphaPulse database infrastructure.

This script tests the database connection and performs basic operations.
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_database')


async def test_database_connection():
    """Test the database connection."""
    import asyncpg
    
    logger.info("Testing PostgreSQL connection...")
    
    try:
        # Connect to the database
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='postgres',
            database='alphapulse'
        )
        
        # Test the connection
        version = await conn.fetchval('SELECT version()')
        logger.info(f"Connected to PostgreSQL: {version}")
        
        # Get schema information
        schemas = await conn.fetch("SELECT schema_name FROM information_schema.schemata")
        logger.info(f"Available schemas: {[s['schema_name'] for s in schemas]}")
        
        # Get table information
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'alphapulse'
        """)
        logger.info(f"Tables in alphapulse schema: {[t['table_name'] for t in tables]}")
        
        # Test a query on the users table
        users = await conn.fetch("SELECT * FROM alphapulse.users")
        logger.info(f"Found {len(users)} users")
        for user in users:
            logger.info(f"User: {user['username']}, Role: {user['role']}")
        
        # Close the connection
        await conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def test_redis_connection():
    """Test the Redis connection."""
    import redis
    
    logger.info("Testing Redis connection...")
    
    try:
        # Connect to Redis
        r = redis.Redis(
            host='localhost',
            port=6379,
            db=0
        )
        
        # Test the connection
        r.ping()
        logger.info("Connected to Redis")
        
        # Set a test value
        r.set('test_key', 'test_value')
        logger.info("Set test key in Redis")
        
        # Get the test value
        value = r.get('test_key')
        logger.info(f"Retrieved test key from Redis: {value}")
        
        # Delete the test value
        r.delete('test_key')
        logger.info("Deleted test key from Redis")
        
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False


async def main():
    """Main function to run the tests."""
    logger.info("Starting database tests...")
    
    # Test PostgreSQL connection
    pg_success = await test_database_connection()
    
    # Test Redis connection
    redis_success = await test_redis_connection()
    
    # Report results
    if pg_success and redis_success:
        logger.info("All database tests passed!")
        return 0
    else:
        logger.error("Some database tests failed")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)