# AlphaPulse Monitoring System

A comprehensive monitoring system for tracking performance, risk, trade execution, and system metrics in the AlphaPulse trading platform.

## Features

- **Time Series Storage**: Multiple storage backends (InfluxDB, TimescaleDB, in-memory)
- **Comprehensive Metrics**: Performance, risk, trade execution, agent performance, API, and system metrics
- **Real-time Monitoring**: Automatic periodic collection of metrics
- **Historical Analysis**: Query and analyze historical metrics
- **Alerting System**: Rule-based alerting with multiple notification channels
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
├── alerting/            # Alerting system
│   ├── manager.py       # Alert manager
│   ├── models.py        # Alert and rule models
│   ├── evaluator.py     # Rule evaluation
│   ├── history.py       # Alert history storage
│   ├── config.py        # Alerting configuration
│   └── channels/        # Notification channels
│       ├── base.py      # Base channel interface
│       ├── email.py     # Email notifications
│       ├── slack.py     # Slack notifications
│       └── web.py       # Web notifications
```

## Usage

### Basic Monitoring Usage

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

### Alerting System Usage

```python
from alpha_pulse.monitoring.alerting import (
    AlertManager, AlertRule, AlertSeverity, load_alerting_config
)

# Load alerting configuration
config = load_alerting_config("path/to/alerting_config.yaml")

# Create alert manager
alert_manager = AlertManager(config)

# Start alert manager
await alert_manager.start()

# Process metrics
metrics = {
    "sharpe_ratio": 0.3,
    "max_drawdown": 0.15
}
alerts = await alert_manager.process_metrics(metrics)

# Check for triggered alerts
if alerts:
    print(f"Generated {len(alerts)} alerts")
    for alert in alerts:
        print(f"Alert: {alert.severity.value.upper()} - {alert.message}")

# Acknowledge an alert
await alert_manager.acknowledge_alert(alert_id, "user_name")

# Get alert history
history = await alert_manager.get_alert_history(
    start_time=start_date,
    end_time=end_date,
    filters={"severity": "error", "acknowledged": False}
)

# Stop alert manager
await alert_manager.stop()
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

# Alerting configuration
alerting:
  enabled: true
  check_interval: 60  # seconds
  
  # Notification channels
  channels:
    email:
      enabled: true
      smtp_server: "smtp.example.com"
      smtp_port: 587
      smtp_user: "${AP_EMAIL_USER}"
      smtp_password: "${AP_EMAIL_PASSWORD}"
      from_address: "alerts@example.com"
      to_addresses: ["admin@example.com"]
    
    slack:
      enabled: true
      webhook_url: "${AP_SLACK_WEBHOOK}"
      channel: "#alerts"
    
    web:
      enabled: true
      max_alerts: 100
  
  # Alert rules
  rules:
    - rule_id: "sharpe_ratio_low"
      name: "Low Sharpe Ratio"
      description: "Alerts when the Sharpe ratio falls below threshold"
      metric_name: "sharpe_ratio"
      condition: "< 0.5"
      severity: "warning"
      message_template: "Sharpe ratio is {value:.2f}, below threshold of 0.5"
      channels: ["email", "slack", "web"]
      cooldown_period: 3600  # 1 hour
      enabled: true
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

## Alerting System

The alerting system monitors metrics and sends notifications when predefined conditions are met.

### Alert Rules

Alert rules define when alerts should be triggered:

- **Condition**: Comparison operators (>, <, >=, <=, ==, !=) with threshold values
- **Severity**: INFO, WARNING, ERROR, CRITICAL
- **Channels**: Which notification channels to use
- **Cooldown**: Minimum time between repeated alerts

### Notification Channels

The system supports multiple notification channels:

- **Email**: SMTP-based email notifications
- **Slack**: Webhook-based Slack notifications
- **Web**: In-memory storage for dashboard integration
- **Custom**: Extensible interface for custom channels

### Alert History

The system maintains a history of triggered alerts:

- Store alerts in memory or file
- Query alerts with filtering
- Acknowledge alerts and track who acknowledged them

## Examples

See the `examples/monitoring` directory for usage examples:

- `demo_monitoring.py`: Demonstrates the full monitoring system with sample data
- `demo_alerting.py`: Demonstrates the alerting system with simulated metrics

## Dependencies

- `asyncio`: For asynchronous operations
- `numpy`: For numerical calculations
- `psutil`: For system metrics
- `aiohttp`: For HTTP API calls (InfluxDB, Slack)
- `asyncpg`: For PostgreSQL access (TimescaleDB)
- `aiosmtplib`: For email notifications