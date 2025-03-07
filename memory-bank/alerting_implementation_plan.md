# Alerting System Implementation Plan

This document outlines the specific implementation steps for completing the AI Hedge Fund Alerting System (Task 1.3 from our implementation plan).

## Current Status

The alerting system implementation has begun with some core files in place:
- Base directory structure created
- Alert models implementation started
- Basic evaluator framework in place
- Channel base classes defined

## Implementation Tasks

### 1. Complete Alert Models (0.5 day)
- [ ] Finalize `Alert` and `AlertRule` data models
- [ ] Add serialization/deserialization methods
- [ ] Implement validation logic
- [ ] Add rule parsing functionality

### 2. Finish Rule Evaluator (1 day)
- [ ] Complete condition parser
- [ ] Implement rule matching logic
- [ ] Add cooldown period handling
- [ ] Implement rule prioritization

### 3. Implement Alert Manager (1.5 days)
- [ ] Create main `AlertManager` class
- [ ] Implement rule configuration loading
- [ ] Add metrics evaluation pipeline
- [ ] Implement channel routing
- [ ] Add persistence integration

### 4. Complete Notification Channels (2 days)
- [ ] Implement Email channel
  - [ ] SMTP configuration
  - [ ] Template rendering
  - [ ] Failure handling
- [ ] Implement Slack channel
  - [ ] Webhook integration
  - [ ] Message formatting
  - [ ] Rate limiting
- [ ] Implement Web channel
  - [ ] In-memory alert queue
  - [ ] WebSocket integration
  - [ ] Alert persistence
- [ ] (Optional) Implement SMS channel
  - [ ] Provider integration
  - [ ] Message formatting
  - [ ] Cost optimization

### 5. Alert History Storage (1 day)
- [ ] Implement database schema
- [ ] Create repository pattern implementation
- [ ] Add query capabilities
- [ ] Implement alert updates
- [ ] Add expiry/cleanup

### 6. API Integration (1 day)
- [ ] Create REST endpoints for alerts
  - [ ] GET alerts with filtering
  - [ ] PUT acknowledge alert
  - [ ] POST create manual alert
  - [ ] GET alert configurations
- [ ] Implement WebSocket notification service
- [ ] Add authorization for alert endpoints

### 7. Testing & Documentation (2 days)
- [ ] Write unit tests for all components
- [ ] Create integration tests
- [ ] Update API documentation
- [ ] Create usage examples
- [ ] Document configuration options

## File Structure Updates

```
src/alpha_pulse/monitoring/
├── alerting/
│   ├── __init__.py                # Exports and version
│   ├── manager.py                 # AlertManager implementation
│   ├── models.py                  # Alert and AlertRule models
│   ├── evaluator.py               # Rule evaluation logic
│   ├── storage.py                 # Alert history storage
│   ├── config.py                  # Configuration handling
│   └── channels/                  # Notification channels
│       ├── __init__.py            # Channel registry
│       ├── base.py                # Base channel interface
│       ├── email.py               # Email implementation
│       ├── slack.py               # Slack implementation
│       ├── web.py                 # Web notification
│       └── sms.py                 # (Optional) SMS channel
```

## Integration Points

### 1. Metrics Collector Integration
```python
# In metrics collector
from alpha_pulse.monitoring.alerting import AlertManager

alert_manager = AlertManager(config)

# After collecting metrics
async def process_metrics(metrics):
    # Process and store metrics
    store_metrics(metrics)
    
    # Forward to alert manager
    await alert_manager.process_metrics(metrics)
```

### 2. API Integration
```python
# In FastAPI router
from alpha_pulse.monitoring.alerting import AlertManager, Alert

alert_router = APIRouter(prefix="/api/v1/alerts")

@alert_router.get("/")
async def get_alerts(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    user = Depends(get_current_user)
):
    """Get alerts with optional filtering."""
    filters = {
        "severity": severity,
        "acknowledged": acknowledged
    }
    return await alert_manager.get_alert_history(start_time, end_time, filters)

@alert_router.put("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user = Depends(get_current_user)
):
    """Acknowledge an alert."""
    return await alert_manager.acknowledge_alert(alert_id, user.username)
```

### 3. WebSocket Integration
```python
# In WebSocket manager
async def send_alert_updates(websocket, user):
    """Send alert updates to connected clients."""
    # Subscribe to alert channel
    alert_manager.subscribe("web", on_alert)
    
    try:
        while True:
            # Handle other WebSocket operations
            await asyncio.sleep(0.1)
    finally:
        # Unsubscribe on disconnect
        alert_manager.unsubscribe("web", on_alert)

async def on_alert(alert: Alert):
    """Handle new alert notification."""
    # Format alert for WebSocket
    alert_data = alert.dict()
    
    # Send to all subscribed clients
    await websocket_broadcast("alerts", alert_data)
```

## Configuration Example

```yaml
# In config/monitoring_config.yaml
alerting:
  enabled: true
  check_interval: 60  # seconds
  
  # Database settings
  storage:
    type: "postgres"  # or "sqlite", "memory"
    connection_string: "${DB_CONNECTION_STRING}"
    table_name: "alerts"
    max_history: 10000  # Max number of alerts to store
  
  # Notification channels
  channels:
    email:
      enabled: true
      smtp_server: "smtp.example.com"
      smtp_port: 587
      username: "${AP_EMAIL_USER}"
      password: "${AP_EMAIL_PASSWORD}"
      from_address: "alerts@example.com"
      recipients:
        - "user1@example.com"
        - "user2@example.com"
    
    slack:
      enabled: true
      webhook_url: "${AP_SLACK_WEBHOOK}"
      channel: "#alerts"
      
    web:
      enabled: true
      max_alerts: 100  # Maximum alerts to keep in memory
  
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

## Testing Approach

1. **Unit Tests**:
   - Test rule evaluation with different conditions
   - Test alert generation and routing
   - Test notification channel formatting
   - Mock dependencies for isolation

2. **Integration Tests**:
   - Test end-to-end alert flow
   - Test persistence and retrieval
   - Test with actual metrics data
   - Test channel delivery (with mocked endpoints)

3. **Manual Testing**:
   - Verify email formatting and delivery
   - Test Slack message appearance
   - Validate WebSocket delivery and UI updates

## Next Steps

1. Complete the Alert Models implementation
2. Implement the Rule Evaluator
3. Create the AlertManager class
4. Begin implementing the notification channels
5. Set up a meeting to review progress after steps 1-3 are complete