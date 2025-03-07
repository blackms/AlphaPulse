#!/usr/bin/env python
"""
Database initialization script for AlphaPulse.

This script initializes the database, creates necessary tables,
and sets up initial data if needed.
"""
import os
import sys
import yaml
import argparse
import logging
import asyncio
import asyncpg
import redis
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('init_db')

# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config' / 'database_config.yaml'


def load_config(config_path=None):
    """Load database configuration from YAML file."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Process environment variables in config
        process_env_vars(config)
        
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def process_env_vars(config, prefix=''):
    """Process environment variables in configuration."""
    if isinstance(config, dict):
        for key, value in list(config.items()):
            if isinstance(value, (dict, list)):
                process_env_vars(value, f"{prefix}{key}_")
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Extract environment variable name and default value
                env_var = value[2:-1]
                if ':-' in env_var:
                    env_name, default = env_var.split(':-')
                    config[key] = os.environ.get(env_name, default)
                else:
                    config[key] = os.environ.get(env_var, '')
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, (dict, list)):
                process_env_vars(item, prefix)


async def check_postgres_connection(config):
    """Check if PostgreSQL connection can be established."""
    pg_config = config['postgres']
    
    connection_params = {
        'host': pg_config['host'],
        'port': pg_config['port'],
        'user': pg_config['username'],
        'password': pg_config['password'],
        'database': pg_config['database']
    }
    
    logger.info(f"Checking PostgreSQL connection to {pg_config['host']}:{pg_config['port']}")
    
    try:
        conn = await asyncpg.connect(**connection_params)
        await conn.execute('SELECT 1')
        await conn.close()
        logger.info("PostgreSQL connection successful")
        return True
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False


def check_redis_connection(config):
    """Check if Redis connection can be established."""
    redis_config = config['redis']
    
    logger.info(f"Checking Redis connection to {redis_config['host']}:{redis_config['port']}")
    
    try:
        r = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config['password'] or None,
            db=redis_config['database'],
            ssl=redis_config['ssl'],
            socket_timeout=redis_config['timeout']
        )
        r.ping()
        logger.info("Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


async def execute_sql_file(conn, file_path):
    """Execute SQL statements from a file."""
    logger.info(f"Executing SQL file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            sql = f.read()
        
        # Split the SQL file into individual statements
        statements = sql.split(';')
        
        for statement in statements:
            statement = statement.strip()
            if statement:
                await conn.execute(statement)
        
        logger.info(f"SQL file executed successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to execute SQL file {file_path}: {e}")
        return False


async def initialize_database(config, sql_dir=None):
    """Initialize the database with schema and initial data."""
    pg_config = config['postgres']
    
    connection_params = {
        'host': pg_config['host'],
        'port': pg_config['port'],
        'user': pg_config['username'],
        'password': pg_config['password'],
        'database': pg_config['database']
    }
    
    # Default SQL directory
    if sql_dir is None:
        sql_dir = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'init'
    
    logger.info(f"Initializing database from SQL files in {sql_dir}")
    
    try:
        # Connect to the database
        conn = await asyncpg.connect(**connection_params)
        
        # Get all SQL files in the directory, sorted by name
        sql_files = sorted([f for f in os.listdir(sql_dir) if f.endswith('.sql')])
        
        for sql_file in sql_files:
            file_path = os.path.join(sql_dir, sql_file)
            success = await execute_sql_file(conn, file_path)
            if not success:
                logger.error(f"Failed to initialize database with {sql_file}")
                return False
        
        await conn.close()
        logger.info("Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


async def main():
    """Main function to initialize the database."""
    parser = argparse.ArgumentParser(description='Initialize the AlphaPulse database')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--sql-dir', help='Directory containing SQL initialization files')
    parser.add_argument('--check-only', action='store_true', help='Only check connections, do not initialize')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check database connections
    pg_ok = await check_postgres_connection(config)
    redis_ok = check_redis_connection(config)
    
    if not pg_ok or not redis_ok:
        logger.error("Database connection checks failed")
        return 1
    
    if args.check_only:
        logger.info("Connection checks passed")
        return 0
    
    # Initialize database
    success = await initialize_database(config, args.sql_dir)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)