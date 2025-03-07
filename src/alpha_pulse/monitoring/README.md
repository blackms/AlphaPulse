# AlphaPulse Monitoring System

A comprehensive monitoring system for tracking performance, risk, trade execution, and system metrics in the AlphaPulse trading platform.

## Features

- **Time Series Storage**: Multiple storage backends (InfluxDB, TimescaleDB, in-memory)
- **Comprehensive Metrics**: Performance, risk, trade execution, agent performance, API, and system metrics
- **Real-time Monitoring**: Automatic periodic collection of metrics
- **Historical Analysis**: Query and analyze historical metrics
- **Configurable**: Flexible configuration via YAML or environment variables

## Architecture

```
monitoring/
├── collector.py         # Enhanced metrics collector
├── config.py            # Configuration management
├── metrics_calculations.py  # Metric calculation functions
├── storage/             # Storage implementations
│   ├── interfaces.py    # Storage interfaces
│   ├── influxdb.py      # InfluxDB implementation
│   ├── timescaledb.py   # TimescaleDB implementation
│   └── memory.py        # In-memory implementation
```

## Usage

### Basic Usage

```python
from alpha_pulse.monitoring.collector import EnhancedMetricsCollector
from alpha_pulse.monitoring.config import load_config

# Load configuration
config = load_config("path/to/config.yaml")

# Initialize collector
collector = EnhancedMetricsCollector(config=config)

# Start collector
await collector.start()

# Collect metrics
await collector.collect_and_store(
    portfolio_data=portfolio,
    trade_data=trade,
    agent_data=agent_data,
    system_data=True
)

# Query historical metrics
metrics = await collector.get_metrics_history(
    metric_type="performance",
    start_time=start_date,
    end_time=end_date,
    aggregation="1h"  # 1 hour aggregation
)

# Get latest metrics
latest = await collector.get_latest_metrics("system")

# Stop collector
await collector.stop()
```

### Configuration

The monitoring system can be configured via YAML file or environment variables:

```yaml
# Storage configuration
storage:
  type: "influxdb"  # "influxdb", "timescaledb", or "memory"
  retention_days: 30
  
  # InfluxDB settings
  influxdb_url: "http://localhost:8086"
  influxdb_token: "${AP_INFLUXDB_TOKEN}"
  influxdb_org: "alpha_pulse"
  influxdb_bucket: "metrics"
  
  # TimescaleDB settings
  timescaledb_host: "localhost"
  timescaledb_port: 5432
  timescaledb_user: "postgres"
  timescaledb_password: "${AP_TIMESCALEDB_PASSWORD}"
  timescaledb_database: "alpha_pulse"
  
  # Memory storage settings
  memory_max_points: 10000

# Collection settings
collection_interval: 60  # seconds
enable_realtime: true
collect_api_latency: true
collect_trade_metrics: true
collect_agent_metrics: true
```

### Metrics Types

The monitoring system collects and stores the following types of metrics:

1. **Performance Metrics**:
   - Sharpe Ratio, Sortino Ratio, Max Drawdown
   - Value at Risk (VaR), Expected Shortfall
   - Alpha, Beta, R-squared
   - Total Return, Annualized Return, Volatility

2. **Risk Metrics**:
   - Position Count, Concentration (HHI)
   - Leverage, Cash Percentage
   - Portfolio Value

3. **Trade Metrics**:
   - Slippage, Fill Rate
   - Commission, Execution Time

4. **Agent Metrics**:
   - Signal Accuracy, Confidence
   - Error Metrics, Signal Agreement

5. **API Metrics**:
   - Latency (mean, median, p95, p99)
   - Error Rate, Latency Trend

6. **System Metrics**:
   - CPU Usage, Memory Usage
   - Disk Usage, Network I/O
   - Process Metrics

## Examples

See the `examples/monitoring` directory for usage examples:

- `demo_monitoring.py`: Demonstrates the full monitoring system with sample data

## Dependencies

- `asyncio`: For asynchronous operations
- `numpy`: For numerical calculations
- `psutil`: For system metrics
- `aiohttp`: For HTTP API calls (InfluxDB)
- `asyncpg`: For PostgreSQL access (TimescaleDB)