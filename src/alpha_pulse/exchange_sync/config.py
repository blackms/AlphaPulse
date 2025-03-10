"""
Configuration management for the exchange synchronization module.

This module handles loading and validating configuration from environment
variables, providing default values where appropriate, and configuring
the logging system.
"""
import os
import sys
from typing import Dict, Any, Optional, Tuple

from loguru import logger

# Import the credentials manager from AlphaPulse
try:
    from alpha_pulse.exchanges.credentials.manager import credentials_manager
    CREDENTIALS_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIALS_MANAGER_AVAILABLE = False


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration from environment variables.
    
    Returns:
        Dictionary with database connection parameters
    """
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'user': os.getenv('DB_USER', 'testuser'),
        'password': os.getenv('DB_PASS', 'testpassword'),
        'database': os.getenv('DB_NAME', 'alphapulse')
    }


def get_exchange_config(exchange_id: str) -> Dict[str, Any]:
    """
    Get exchange-specific configuration from credentials manager or environment variables.
    
    Args:
        exchange_id: The identifier of the exchange (e.g. 'bybit', 'binance')
        
    Returns:
        Dictionary with exchange-specific configuration
    """
    api_key, api_secret, testnet = get_exchange_credentials(exchange_id)
    
    return {
        'api_key': api_key,
        'api_secret': api_secret,
        'testnet': testnet
    }


def get_exchange_credentials(exchange_id: str) -> Tuple[str, str, bool]:
    """
    Get exchange credentials from credentials manager or environment variables.
    
    Args:
        exchange_id: The identifier of the exchange (e.g. 'bybit', 'binance')
        
    Returns:
        Tuple of (api_key, api_secret, testnet)
    """
    # Try to get credentials from the credentials manager first
    if CREDENTIALS_MANAGER_AVAILABLE:
        logger.debug(f"Attempting to get credentials for {exchange_id} from credentials manager")
        creds = credentials_manager.get_credentials(exchange_id)
        if creds:
            logger.info(f"Using credentials from credentials manager for {exchange_id}")
            return creds.api_key, creds.api_secret, creds.testnet
    
    # Fall back to environment variables if credentials manager is not available
    # or if no credentials were found
    exchange_upper = exchange_id.upper()
    api_key = os.getenv(f'{exchange_upper}_API_KEY', '')
    api_secret = os.getenv(f'{exchange_upper}_API_SECRET', '')
    testnet = os.getenv(f'{exchange_upper}_TESTNET', 'false').lower() == 'true'
    
    if api_key and api_secret:
        logger.info(f"Using credentials from environment variables for {exchange_id}")
    else:
        logger.warning(f"No credentials found for {exchange_id}")
        
    return api_key, api_secret, testnet


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
    Configure the logging system using loguru.
    
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
    level = log_level.upper()
    
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    logger.add(
        os.path.join(log_dir, "exchange_sync_{time}.log"),
        rotation="10 MB",
        retention="1 week",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Log configuration completion
    logger.info(f"Logging configured with level {log_level}")