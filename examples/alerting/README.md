# AlphaPulse Alerting System Examples

This directory contains examples demonstrating how to use the AlphaPulse alerting system.

## Available Examples

### 1. Basic Alerting Demo

The `demo_alerting.py` script demonstrates the core functionality of the alerting system:

- Configuring and initializing the alerting system
- Creating and registering alert rules
- Processing metrics and generating alerts
- Acknowledging alerts
- Retrieving alert history

Run the example:

```bash
python demo_alerting.py
```

## Alerting System Features

The AlphaPulse alerting system provides:

- **Rule-Based Alerts**: Define conditions that trigger alerts based on metric values
- **Multiple Notification Channels**: Send alerts via email, Slack, SMS, and web interfaces
- **Alert History**: Store and query alert history with filtering options
- **Acknowledgment**: Track which alerts have been acknowledged and by whom
- **Cooldown Periods**: Prevent alert storms by setting minimum time between alerts

## Notification Channels

The alerting system supports the following notification channels:

- **Email**: Send alerts via SMTP
- **Slack**: Post alerts to Slack channels via webhooks
- **SMS**: Send text message alerts via Twilio
- **Web**: Store alerts for retrieval by the dashboard or API

## Alert Storage Options

Alerts can be stored using different storage backends:

- **Memory**: In-memory storage (for testing or short-lived alerts)
- **File**: JSON file-based storage
- **Database**: SQLite database storage for better persistence and querying

## Configuration Example

```yaml
alerting:
  enabled: true
  check_interval: 60  # seconds
  
  # Storage configuration
  history:
    type: "database"  # "memory", "file", or "database"
    db_path: "alerts.db"
  
  # Notification channels
  channels:
    email:
      enabled: true
      smtp_server: "smtp.example.com"
      smtp_port: 587
      smtp_user: "${AP_EMAIL_USER}"
      smtp_password: "${AP_EMAIL_PASSWORD}"
      from_address: "alerts@example.com"
      to_addresses: ["user@example.com"]
      use_tls: true
    
    slack:
      enabled: true
      webhook_url: "${AP_SLACK_WEBHOOK}"
      channel: "#alerts"
      username: "AlphaPulse Alerting"
    
    sms:
      enabled: true
      account_sid: "${AP_TWILIO_SID}"
      auth_token: "${AP_TWILIO_TOKEN}"
      from_number: "+15551234567"
      to_numbers: ["+15557654321"]
    
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
      message_template: "Sharpe ratio is {value}, below threshold of 0.5"
      channels: ["email", "slack", "web"]
      cooldown_period: 3600  # 1 hour
      enabled: true
```

## Integration with Monitoring System

The alerting system integrates with the AlphaPulse monitoring system to automatically process metrics as they are collected. See the monitoring examples for more information on how to set up the complete monitoring and alerting pipeline.