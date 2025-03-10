# Exchange Synchronization Module

A simplified, reliable module for synchronizing exchange data with the AlphaPulse database.

## Overview

This module provides a clean, maintainable solution for fetching portfolio and price data from cryptocurrency exchanges and storing it in the database. It was designed with simplicity, reliability, and maintainability in mind, addressing issues with the previous implementation related to connection pooling and thread synchronization.

## Key Features

- **Simple Architecture**: Clear component boundaries and separation of concerns
- **Reliable Database Access**: Single-connection approach avoids complex connection pooling issues
- **Scheduled Execution**: Runs automatically every 30 minutes (configurable)
- **Proper Error Handling**: Comprehensive logging and error recovery
- **Symbol Format Handling**: Properly handles exchange-specific symbol formats
- **Clean API**: Easy to use programmatically or as a standalone service

## Credential Management

The module integrates with AlphaPulse's credential management system:

1. **Primary Source**: First checks for credentials in AlphaPulse's credential manager
2. **Fallback**: If not found, falls back to environment variables
3. **Transparent**: Logs the source of credentials without exposing sensitive data

This approach ensures:
- Consistent credential handling across the entire AlphaPulse system
- Secure storage of API keys and secrets
- Compatibility with existing configuration methods

## Components

- **Models**: Data structures representing portfolio items and sync results
- **Repository**: Database operations with simple connection management
- **Exchange Client**: Handles communication with exchange APIs
- **Portfolio Service**: Business logic for portfolio data synchronization
- **Scheduler**: Manages periodic execution of sync operations
- **Runner**: Command-line interface for running the module

## Usage

### As a Scheduled Service

To run the synchronization as a scheduled service:

```python
from alpha_pulse.exchange_sync.scheduler import ExchangeSyncScheduler

async def main():
    # Create and start the scheduler with default settings
    scheduler = ExchangeSyncScheduler()
    await scheduler.start()

# Run the scheduler
import asyncio
asyncio.run(main())
```

### One-time Synchronization

To run a one-time synchronization:

```python
from alpha_pulse.exchange_sync.scheduler import ExchangeSyncScheduler

async def main():
    # Run a one-time sync and get the results
    results = await ExchangeSyncScheduler.run_once()
    
    # Process results
    for exchange_id, result in results.items():
        print(f"{exchange_id}: {'Success' if result.success else 'Failed'}")
        if not result.success:
            print(f"Errors: {', '.join(result.errors)}")

# Run the one-time sync
import asyncio
asyncio.run(main())
```

### Command Line Interface

The module includes a command-line interface:

```
# Run as a scheduled service
python -m alpha_pulse.exchange_sync.runner

# Run a one-time synchronization and exit
python -m alpha_pulse.exchange_sync.runner --one-time

# Specify custom interval (in minutes)
python -m alpha_pulse.exchange_sync.runner --interval 60

# Set custom log level
python -m alpha_pulse.exchange_sync.runner --log-level DEBUG
```

## Configuration

The module uses environment variables for configuration:

- **Database Configuration**:
  - `DB_HOST`: Database hostname (default: localhost)
  - `DB_PORT`: Database port (default: 5432)
  - `DB_USER`: Database username (default: postgres)
  - `DB_PASS`: Database password (default: postgres)
  - `DB_NAME`: Database name (default: alphapulse)

- **Exchange Configuration**:
  - `EXCHANGE_SYNC_EXCHANGES`: Comma-separated list of exchanges (default: bybit)
  - `EXCHANGE_SYNC_INTERVAL_MINUTES`: Sync interval in minutes (default: 30)
  - `EXCHANGE_SYNC_ENABLED`: Enable/disable scheduled sync (default: true)
  - `EXCHANGE_SYNC_LOG_LEVEL`: Logging level (default: INFO)
  - `EXCHANGE_SYNC_LOG_DIR`: Directory for log files (default: logs)

- **Exchange API Credentials**:
  - `{EXCHANGE}_API_KEY`: API key for the exchange (e.g., BYBIT_API_KEY)
  - `{EXCHANGE}_API_SECRET`: API secret for the exchange
  - `{EXCHANGE}_TESTNET`: Use testnet (true/false)

> **Note**: The module will first attempt to use credentials from AlphaPulse's credential manager before falling back to environment variables.

## Database Schema

The module uses the following database tables:

### portfolio_items

Stores portfolio data for each exchange and asset.

| Column          | Type      | Description                       |
|-----------------|-----------|-----------------------------------|
| id              | SERIAL    | Primary key                       |
| exchange_id     | VARCHAR   | Exchange identifier               |
| asset           | VARCHAR   | Asset symbol                      |
| quantity        | NUMERIC   | Quantity held                     |
| current_price   | NUMERIC   | Current asset price               |
| avg_entry_price | NUMERIC   | Average entry price               |
| created_at      | TIMESTAMP | Record creation timestamp         |
| updated_at      | TIMESTAMP | Record update timestamp           |

### sync_history

Tracks synchronization operations.

| Column          | Type      | Description                       |
|-----------------|-----------|-----------------------------------|
| id              | SERIAL    | Primary key                       |
| exchange_id     | VARCHAR   | Exchange identifier               |
| sync_type       | VARCHAR   | Type of sync (e.g., portfolio)    |
| items_processed | INTEGER   | Number of items processed         |
| items_synced    | INTEGER   | Number of items successfully synced|
| success         | BOOLEAN   | Whether sync was successful       |
| start_time      | TIMESTAMP | When sync started                 |
| end_time        | TIMESTAMP | When sync completed               |
| error_message   | TEXT      | Error message, if any             |
| created_at      | TIMESTAMP | Record creation timestamp         |

## Example

See the `examples/exchange_sync/sync_example.py` for a complete example of using the exchange sync module.