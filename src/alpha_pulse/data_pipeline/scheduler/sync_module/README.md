# Exchange Data Synchronization Module

This module provides functionality to synchronize exchange data on a regular schedule or on demand, serving as a background worker for the portfolio API.

## Module Structure

The module is organized following SOLID principles, with each component having a single responsibility:

- **__init__.py**: Exports the main classes and types
- **types.py**: Defines the data types and enums used in the module
- **synchronizer.py**: Main orchestrator class that manages the synchronization process
- **exchange_manager.py**: Handles exchange creation and initialization
- **data_sync.py**: Handles the actual data synchronization with exchanges
- **task_manager.py**: Manages synchronization tasks and scheduling
- **event_loop_manager.py**: Handles event loop management in a multi-threaded environment

## Usage

```python
from alpha_pulse.data_pipeline.scheduler import exchange_data_synchronizer, DataType

# Start the synchronizer (non-blocking, runs in background thread)
exchange_data_synchronizer.start()

# Trigger a manual synchronization
exchange_data_synchronizer.trigger_sync(exchange_id="bybit", data_type=DataType.ALL)

# Stop the synchronizer
exchange_data_synchronizer.stop()
```

## Design Patterns

This module uses several design patterns:

1. **Singleton Pattern**: The `ExchangeDataSynchronizer` class is a singleton to ensure only one instance exists.
2. **Factory Pattern**: The `ExchangeManager` uses a factory to create exchange instances.
3. **Strategy Pattern**: Different synchronization strategies for different data types.
4. **Circuit Breaker Pattern**: Prevents repeated failures by temporarily disabling operations after multiple failures.
5. **Repository Pattern**: Data access is abstracted through repository classes.

## Error Handling

The module implements robust error handling:

1. **Retry Logic**: Failed operations are retried with exponential backoff.
2. **Circuit Breaker**: Prevents repeated failures by temporarily disabling operations.
3. **Graceful Degradation**: The system continues to function even if some components fail.
4. **Detailed Logging**: All errors are logged with detailed information for troubleshooting.

## Threading Model

The module uses a background thread for the main synchronization loop, with asyncio for asynchronous operations. Event loops are carefully managed to prevent issues with asyncio in a multi-threaded environment.

## Maintenance

When making changes to this module:

1. Follow the SOLID principles
2. Keep each file focused on a single responsibility
3. Add appropriate error handling
4. Update tests to cover new functionality
5. Document any changes in this README