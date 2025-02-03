# AlphaPulse Data Pipeline 📊

A modular and extensible data pipeline for fetching, storing, and managing market data from cryptocurrency exchanges.

## 🏗️ Architecture

```
data_pipeline/
├── core/               # Core abstractions and utilities
│   ├── interfaces.py   # Component interfaces
│   ├── models.py      # Domain models
│   ├── config.py      # Configuration management
│   ├── errors.py      # Error hierarchy
│   └── validation.py  # Data validation
├── storage/           # Data persistence
│   ├── sql.py        # SQLAlchemy implementation
├── fetcher/           # Data fetching
│   ├── exchange.py    # Exchange-based fetcher
├── providers/         # Real-time data
│   ├── exchange.py    # Exchange data provider
└── managers/          # Coordination layer
    ├── historical.py  # Historical data management
    └── real_time.py   # Real-time data management
```

## 🚀 Features

- **Modular Design**: Each component has a single, well-defined responsibility
- **Type Safety**: Comprehensive type hints and validation
- **Error Handling**: Detailed error hierarchy for better debugging
- **Async Support**: Built for high-performance async operations
- **Extensible**: Easy to add new implementations

## 💡 Usage Example

```python
from alpha_pulse.data_pipeline import (
    SQLAlchemyStorage,
    ExchangeFetcher,
    ExchangeDataProvider,
    HistoricalDataManager,
    RealTimeDataManager
)

# Initialize components
storage = SQLAlchemyStorage()
fetcher = ExchangeFetcher()
provider = ExchangeDataProvider(ExchangeType.BINANCE, testnet=True)

# Create managers
historical_mgr = HistoricalDataManager(storage, fetcher)
realtime_mgr = RealTimeDataManager(provider, storage)

# Use managers to handle data
await historical_mgr.ensure_data_available(
    ExchangeType.BINANCE,
    "BTC/USDT",
    "1h",
    start_time,
    end_time
)

await realtime_mgr.start(["BTC/USDT", "ETH/USDT"])
```

## 🔧 Configuration

Configuration is managed through dataclasses in `core/config.py`:

```python
from alpha_pulse.data_pipeline.core.config import (
    StorageConfig,
    DataFetchConfig,
    MarketDataConfig
)

# Configure storage
storage_config = StorageConfig(
    batch_size=500,
    max_connections=10,
    timeout=30.0
)

# Configure data fetching
fetch_config = DataFetchConfig(
    batch_size=1000,
    max_retries=3,
    retry_delay=1.0
)

# Configure market data
market_config = MarketDataConfig(
    update_interval=1.0,
    max_symbols=100,
    cache_duration=60
)
```

## ⚠️ Error Handling

The pipeline uses a comprehensive error hierarchy defined in `core/errors.py`:

```python
try:
    await historical_mgr.ensure_data_available(...)
except StorageError as e:
    logger.error(f"Storage error: {e}")
except DataFetchError as e:
    logger.error(f"Data fetch error: {e}")
except ValidationError as e:
    logger.error(f"Validation error: {e}")
```

## 🔍 Validation

Data validation is centralized in `core/validation.py`:

```python
from alpha_pulse.data_pipeline.core.validation import (
    validate_timeframe,
    validate_symbol,
    validate_ohlcv
)

# Validate inputs
validate_timeframe(timeframe, valid_timeframes)
validate_symbol(symbol)
validate_ohlcv(data)
```

## 📈 Performance Considerations

- Uses connection pooling for database operations
- Implements caching for real-time data
- Supports batch operations for better throughput
- Provides async interfaces for non-blocking operations

## 🤝 Contributing

1. Follow SOLID principles
2. Add comprehensive type hints
3. Include unit tests
4. Update documentation
5. Use provided error hierarchy