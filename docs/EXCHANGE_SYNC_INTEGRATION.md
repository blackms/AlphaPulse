# Exchange Synchronization Integration

This document explains the integration of the new `exchange_sync` module into the AlphaPulse application.

## Overview

The `exchange_sync` module provides a simplified, reliable approach to synchronizing exchange data with the AlphaPulse database. It replaces the legacy complex logic with a cleaner, more maintainable implementation that follows SOLID principles.

## Key Improvements

- **Simplified Architecture**: Clear component boundaries and separation of concerns
- **Reliable Database Access**: Single-connection approach avoids complex connection pooling issues
- **Scheduled Execution**: Runs automatically every 30 minutes (configurable)
- **Proper Error Handling**: Comprehensive logging and error recovery
- **Clean API**: Easy to use programmatically or as a standalone service

## Integration Methods

The `exchange_sync` module can be integrated into the AlphaPulse application in two ways:

### 1. As a FastAPI Background Task

The module is integrated into the FastAPI application using the event lifecycle hooks:

```python
from alpha_pulse.api.exchange_sync_integration import register_exchange_sync_events

app = FastAPI()
register_exchange_sync_events(app)
```

This approach:
- Starts the exchange synchronization when the API starts
- Stops it gracefully when the API shuts down
- Handles errors appropriately

### 2. As a Separate Process

The module can also be run as a separate process:

```bash
python -m alpha_pulse.exchange_sync.runner
```

This approach:
- Runs independently of the API
- Can be managed by a process supervisor (systemd, supervisor, etc.)
- Provides command-line options for customization

## API Endpoints

The following API endpoints have been updated to use the new `exchange_sync` module:

### GET /api/v1/portfolio

Gets portfolio data from the database or directly from the exchange.

**Query Parameters**:
- `include_history` (boolean): Whether to include historical data
- `refresh` (boolean): Whether to force a refresh from the exchange

### POST /api/v1/portfolio/reload

Triggers an immediate synchronization of exchange data.

### POST /api/v1/system/exchange/reload

Triggers an immediate synchronization of exchange data (system-level endpoint).

## Configuration

The `exchange_sync` module uses environment variables for configuration:

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

## Implementation Details

### Components

1. **exchange_sync_integration.py**: Integrates the exchange_sync module with FastAPI
2. **PortfolioDataAccessor**: Enhanced to use the exchange_sync module
3. **System and Portfolio Routers**: Updated to use the new integration

### Database Schema

The module uses the following database tables:

#### portfolio_items

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

#### sync_history

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

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check database credentials in environment variables
   - Verify database is running and accessible

2. **Exchange API Errors**:
   - Verify API credentials are correct
   - Check exchange status and API rate limits
   - Look for specific error messages in the logs

3. **Synchronization Not Running**:
   - Check if `EXCHANGE_SYNC_ENABLED` is set to `true`
   - Verify the scheduler is running (check logs)
   - Try triggering a manual sync via the API

### Logs

Logs are stored in the directory specified by `EXCHANGE_SYNC_LOG_DIR` (default: `logs`).

The main log file is `exchange_sync.log`, which contains detailed information about synchronization operations, errors, and warnings.

## Migration from Legacy System

If you're migrating from the legacy exchange synchronization system, note the following changes:

1. The connection manager is no longer used - each database operation uses a fresh connection
2. The complex caching mechanisms have been removed in favor of a simpler approach
3. The DataType enum is no longer used for triggering specific sync types
4. Error handling is more robust and provides clearer error messages

## Future Improvements

1. Add support for more exchanges
2. Implement more sophisticated retry strategies
3. Add metrics collection for monitoring
4. Enhance the API with more detailed status information