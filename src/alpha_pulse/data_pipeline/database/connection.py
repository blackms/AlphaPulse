"""
Database connection management for AlphaPulse.

This module provides connection management for PostgreSQL and Redis databases.
"""
import os
import yaml
import logging
import asyncio
import asyncpg
import redis
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_connection')

# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / 'config' / 'database_config.yaml'


class ConnectionManager:
    """
    Database connection manager for PostgreSQL and Redis.
    
    This class manages connection pools and provides methods to acquire and release connections.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of ConnectionManager."""
        if cls._instance is None:
            cls._instance = ConnectionManager()
        return cls._instance
    
    def __init__(self, config_path=None):
        """Initialize the connection manager."""
        self.config = self._load_config(config_path)
        self.pg_pool = None
        self.redis_client = None
        self._initialized = False
    
    def _load_config(self, config_path=None):
        """Load database configuration from YAML file."""
        config_path = config_path or DEFAULT_CONFIG_PATH
        
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Process environment variables in config
            self._process_env_vars(config)
            
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _process_env_vars(self, config, prefix=''):
        """Process environment variables in configuration."""
        if isinstance(config, dict):
            for key, value in list(config.items()):
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value, f"{prefix}{key}_")
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
                    self._process_env_vars(item, prefix)
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing database connections")
        
        # Initialize PostgreSQL connection pool
        await self._initialize_pg_pool()
        
        # Initialize Redis client
        self._initialize_redis_client()
        
        self._initialized = True
        logger.info("Database connections initialized")
    
    async def _initialize_pg_pool(self):
        """Initialize PostgreSQL connection pool."""
        pg_config = self.config['postgres']
        
        connection_params = {
            'host': pg_config['host'],
            'port': pg_config['port'],
            'user': pg_config['username'],
            'password': pg_config['password'],
            'database': pg_config['database'],
            'min_size': pg_config['pool']['min_connections'],
            'max_size': pg_config['pool']['max_connections']
        }
        
        logger.info(f"Initializing PostgreSQL connection pool to {pg_config['host']}:{pg_config['port']}")
        
        try:
            self.pg_pool = await asyncpg.create_pool(**connection_params)
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise
    
    def _initialize_redis_client(self):
        """Initialize Redis client."""
        redis_config = self.config['redis']
        
        logger.info(f"Initializing Redis client to {redis_config['host']}:{redis_config['port']}")
        
        try:
            connection_pool = redis.ConnectionPool(
                host=redis_config['host'],
                port=redis_config['port'],
                password=redis_config['password'] or None,
                db=redis_config['database'],
                ssl=redis_config['ssl'],
                socket_timeout=redis_config['timeout'],
                max_connections=redis_config['pool']['max_connections']
            )
            
            self.redis_client = redis.Redis(connection_pool=connection_pool)
            self.redis_client.ping()  # Test connection
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    @asynccontextmanager
    async def get_pg_connection(self):
        """Get a PostgreSQL connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self.pg_pool.acquire() as connection:
            yield connection
    
    def get_redis_connection(self):
        """Get the Redis client."""
        if not self._initialized:
            asyncio.create_task(self.initialize())
        
        return self.redis_client
    
    @asynccontextmanager
    async def transaction(self):
        """Execute operations in a transaction."""
        if not self._initialized:
            await self.initialize()
        
        async with self.pg_pool.acquire() as connection:
            async with connection.transaction():
                yield connection
    
    async def close(self):
        """Close all database connections."""
        logger.info("Closing database connections")
        
        if self.pg_pool:
            await self.pg_pool.close()
            self.pg_pool = None
        
        if self.redis_client:
            self.redis_client.connection_pool.disconnect()
            self.redis_client = None
        
        self._initialized = False
        logger.info("Database connections closed")


# Singleton instance
connection_manager = ConnectionManager()


async def execute_with_retry(operation, max_retries=3, retry_delay=1):
    """Execute a database operation with retry logic."""
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            return await operation()
        except (asyncpg.exceptions.ConnectionDoesNotExistError, 
                asyncpg.exceptions.InterfaceError,
                redis.exceptions.ConnectionError,
                redis.exceptions.TimeoutError) as e:
            retries += 1
            last_error = e
            if retries < max_retries:
                logger.warning(f"Database operation failed, retrying ({retries}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay * (2 ** (retries - 1)))  # Exponential backoff
    
    # Log the failure after all retries
    logger.error(f"Database operation failed after {max_retries} retries: {last_error}")
    raise last_error


# Convenience functions
async def get_pg_connection():
    """Get a PostgreSQL connection from the pool."""
    return connection_manager.get_pg_connection()


def get_redis_connection():
    """Get the Redis client."""
    return connection_manager.get_redis_connection()


async def execute_in_transaction(callback):
    """Execute operations in a transaction."""
    async with connection_manager.transaction() as conn:
        return await callback(conn)