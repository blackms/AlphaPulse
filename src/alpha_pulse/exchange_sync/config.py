"""
Configuration management for the exchange synchronization module.

This module handles loading and validating configuration from environment
variables, providing default values where appropriate, and configuring
the logging system.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration from environment variables.
    
    Returns:
        Dictionary with database connection parameters
    """
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASS', 'postgres'),
        'database': os.getenv('DB_NAME', 'alphapulse')
    }


def get_exchange_config(exchange_id: str) -> Dict[str, Any]:
    """
    Get exchange-specific configuration from environment variables.
    
    Args:
        exchange_id: The identifier of the exchange (e.g. 'bybit', 'binance')
        
    Returns:
        Dictionary with exchange-specific configuration
    """
    # Convert to uppercase for environment variables
    exchange_upper = exchange_id.upper()
    
    return {
        'api_key': os.getenv(f'{exchange_upper}_API_KEY', ''),
        'api_secret': os.getenv(f'{exchange_upper}_API_SECRET', ''),
        'testnet': os.getenv(f'{exchange_upper}_TESTNET', 'false').lower() == 'true'
    }


def get_sync_config() -> Dict[str, Any]:
    """
    Get synchronization configuration from environment variables.
    
    Returns:
        Dictionary with synchronization parameters
    """
    return {
        'interval_minutes': int(os.getenv('EXCHANGE_SYNC_INTERVAL_MINUTES', '30')),
        'enabled': os.getenv('EXCHANGE_SYNC_ENABLED', 'true').lower() == 'true',
        'exchanges': os.getenv('EXCHANGE_SYNC_EXCHANGES', 'bybit').split(','),
        'log_level': os.getenv('EXCHANGE_SYNC_LOG_LEVEL', 'INFO'),
        'log_dir': os.getenv('EXCHANGE_SYNC_LOG_DIR', 'logs')
    }


def configure_logging(log_dir: Optional[str] = None, 
                      log_level: Optional[str] = None) -> None:
    """
    Configure the logging system.
    
    Args:
        log_dir: Directory to store log files (default from env or 'logs')
        log_level: Log level (default from env or 'INFO')
    """
    # Use provided values or get from environment
    sync_config = get_sync_config()
    log_dir = log_dir or sync_config['log_dir']
    log_level = log_level or sync_config['log_level']
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Determine the log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create a file handler
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, 'exchange_sync.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log configuration completion
    logger.info(f"Logging configured with level {log_level}")