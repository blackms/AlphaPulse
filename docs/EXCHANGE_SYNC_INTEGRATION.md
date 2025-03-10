# Exchange Sync Integration Guide

This document provides information on how the exchange_sync module is integrated into the AlphaPulse application.

## Overview

The exchange_sync module provides a clean, reliable way to synchronize data from cryptocurrency exchanges. It has been designed with simplicity and maintainability in mind, avoiding the complex connection pooling and caching mechanisms that caused issues in the previous implementation.

## Key Components

1. **Repository Layer**: Handles database operations with simple, reliable connection management
2. **Service Layer**: Coordinates data synchronization operations
3. **Exchange Client**: Abstracts communication with exchange APIs
4. **Scheduler**: Manages periodic synchronization tasks
5. **Runner**: Provides standalone execution capability
6. **API Integration**: Connects the module to the FastAPI application

## Logging with Loguru

The exchange_sync module uses [Loguru](https://github.com/Delgan/loguru) for logging, which provides several advantages:

- Simple, intuitive API
- Structured logging with rich formatting
- Automatic rotation of log files
- Exception tracebacks with variable values
- Customizable log levels and handlers

### Logging Configuration

Logging is configured in the `config.py` module:

```python
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
```

### Using Loguru in Your Code

To use Loguru in your code:

1. Import the logger:
   ```python
   from loguru import logger
   ```

2. Use the logger with appropriate log levels:
   ```python
   logger.debug("Detailed information for debugging")
   logger.info("General information about program execution")
   logger.warning("Warning about potential issues")
   logger.error("Error that doesn't prevent the program from running")
   logger.critical("Critical error that may prevent the program from running")
   ```

3. Log exceptions with context:
   ```python
   try:
       # Some code that might raise an exception
       result = some_function()
   except Exception as e:
       logger.exception(f"Error in some_function: {e}")
       # or
       logger.error(f"Error in some_function: {e}")
   ```

## Integration with FastAPI

The exchange_sync module is integrated with the FastAPI application through the `exchange_sync_integration.py` module, which provides:

1. **Initialization**: Sets up the database tables and configures logging
2. **Startup**: Starts the scheduler as a background task
3. **Shutdown**: Stops the scheduler gracefully
4. **Lifespan Management**: Provides a context manager for FastAPI's lifespan
5. **Event Registration**: Registers startup and shutdown events with FastAPI
6. **Manual Triggering**: Allows manual triggering of synchronization

### Usage in FastAPI Application

```python
from fastapi import FastAPI
from alpha_pulse.api.exchange_sync_integration import register_exchange_sync_events

# Create the FastAPI application
app = FastAPI()

# Register exchange sync events
register_exchange_sync_events(app)
```

## Configuration

The exchange_sync module is configured through environment variables:

- `DB_HOST`: Database host (default: "localhost")
- `DB_PORT`: Database port (default: 5432)
- `DB_USER`: Database user (default: "testuser")
- `DB_PASS`: Database password (default: "testpassword")
- `DB_NAME`: Database name (default: "alphapulse")
- `EXCHANGE_SYNC_INTERVAL_MINUTES`: Synchronization interval in minutes (default: 30)
- `EXCHANGE_SYNC_ENABLED`: Whether synchronization is enabled (default: true)
- `EXCHANGE_SYNC_EXCHANGES`: Comma-separated list of exchanges to synchronize (default: "bybit")
- `EXCHANGE_SYNC_LOG_LEVEL`: Logging level (default: "INFO")
- `EXCHANGE_SYNC_LOG_DIR`: Directory for log files (default: "logs")

## Standalone Execution

The exchange_sync module can be run as a standalone process using the `runner.py` module:

```bash
python -m alpha_pulse.exchange_sync.runner --exchanges bybit,binance --log-level DEBUG
```

This is useful for testing and manual synchronization.