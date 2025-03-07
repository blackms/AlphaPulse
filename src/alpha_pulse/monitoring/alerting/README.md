# AlphaPulse Alerting System

The AlphaPulse Alerting System is a comprehensive solution for monitoring metrics and sending notifications when predefined conditions are met. It is designed to be flexible, extensible, and easy to integrate with the existing monitoring infrastructure.

## Features

- **Rule-Based Alerting**: Define alert rules with conditions, severity levels, and notification channels
- **Multiple Notification Channels**: Email, Slack, and Web notifications (extensible to other channels)
- **Alert History**: Store and query alert history
- **Acknowledgment System**: Acknowledge alerts and track who acknowledged them
- **Configurable**: Configure via YAML, environment variables, or programmatically

## Architecture

The alerting system consists of the following components:

- **AlertManager**: Central component that coordinates alerting activities
- **AlertRule**: Defines conditions for triggering alerts
- **Alert**: Represents a triggered alert
- **NotificationChannel**: Interface for sending notifications
- **RuleEvaluator**: Evaluates metrics against rules
- **AlertHistoryStorage**: Stores and retrieves alert history

## Usage

### Basic Usage

```python
from alpha_pulse.monitoring.alerting import (
    Alert, AlertRule, AlertSeverity, AlertManager, AlertingConfig
)

# Create alert rules
rules = [
    AlertRule(
        rule_id="sharpe_ratio_low",
        name="Low Sharpe Ratio",
        description="Alerts when the Sharpe ratio falls below threshold",
        metric_name="sharpe_ratio",
        condition="< 0.5",
        severity=AlertSeverity.WARNING,
        message_template="Sharpe ratio is {value:.2f}, below threshold of 0.5",
        channels=["email", "slack"],
        cooldown_period=3600  # 1 hour
    ),
    # Add more rules...
]

# Create configuration
config = AlertingConfig({
    "enabled": True,
    "check_interval": 60,  # seconds
    "channels": {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "password",
            "from_address": "alerts@example.com",
            "to_addresses": ["admin@example.com"]
        },
        "slack": {
            "enabled": True,
            "webhook_url": "https://hooks.slack.com/services/...",
            "channel": "#alerts"
        }
    },
    "rules": [rule.to_dict() for rule in rules]
})

# Create and start alert manager
alert_manager = AlertManager(config)
await alert_manager.start()

# Process metrics
metrics = {
    "sharpe_ratio": 0.3,
    "max_drawdown": 0.15
}
alerts = await alert_manager.process_metrics(metrics)

# Stop alert manager when done
await alert_manager.stop()
```

### Configuration via YAML

Create a YAML configuration file:

```yaml
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

Load the configuration:

```python
from alpha_pulse.monitoring.alerting import AlertingConfig, AlertManager, load_alerting_config

# Load from file
config = load_alerting_config("config/alerting_config.yaml")

# Create alert manager
alert_manager = AlertManager(config)
```

### Integration with Metrics Collector

The alerting system can be integrated with the metrics collector:

```python
from alpha_pulse.monitoring import EnhancedMetricsCollector, AlertManager, load_alerting_config

# Load alerting configuration
alerting_config = load_alerting_config()

# Create alert manager
alert_manager = AlertManager(alerting_config)
await alert_manager.start()

# Create metrics collector
metrics_collector = EnhancedMetricsCollector()
await metrics_collector.start()

# Hook up metrics collection to alert processing
async def process_metrics_callback(metrics):
    await alert_manager.process_metrics(metrics)

# Register callback with metrics collector
metrics_collector.register_callback(process_metrics_callback)

# Later, stop both components
await metrics_collector.stop()
await alert_manager.stop()
```

## Extending the System

### Creating Custom Notification Channels

You can create custom notification channels by implementing the `NotificationChannel` interface:

```python
from alpha_pulse.monitoring.alerting.channels.base import NotificationChannel
from alpha_pulse.monitoring.alerting.models import Alert

class CustomNotificationChannel(NotificationChannel):
    async def initialize(self) -> bool:
        # Initialize the channel
        return True
        
    async def send_notification(self, alert: Alert) -> bool:
        # Send notification
        print(f"Custom notification: {alert.message}")
        return True
        
    async def close(self) -> None:
        # Clean up resources
        pass
```

Register the custom channel with the alert manager:

```python
alert_manager.register_channel("custom", CustomNotificationChannel({}))
```

### Creating Custom Alert History Storage

You can create custom alert history storage by implementing the `AlertHistoryStorage` interface:

```python
from alpha_pulse.monitoring.alerting.history import AlertHistoryStorage
from alpha_pulse.monitoring.alerting.models import Alert

class CustomAlertHistory(AlertHistoryStorage):
    async def store_alert(self, alert: Alert) -> None:
        # Store alert
        pass
        
    async def get_alerts(self, start_time=None, end_time=None, filters=None) -> List[Alert]:
        # Retrieve alerts
        return []
        
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        # Update alert
        return True
```

## Examples

See the `examples/monitoring/demo_alerting.py` file for a complete example of using the alerting system.