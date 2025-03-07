# Dashboard Integration with Alerting System

This document outlines the integration plan for connecting our newly implemented alerting system with the existing dashboard backend.

## Current Status Summary

Based on our code exploration, we've discovered that:

1. We have a well-structured Dashboard Backend implementation already in place:
   - FastAPI application structure
   - REST API endpoints for metrics, alerts, portfolio, and trades
   - WebSocket server for real-time updates
   - Authentication and authorization
   - Caching layer
   - Data access layer

2. We've successfully implemented the Alerting System with:
   - Alert Manager
   - Multiple notification channels
   - Rule-based alert generation
   - Alert history storage
   - Integration with the monitoring system

## Integration Requirements

To properly integrate the alerting system with the dashboard backend, we need to:

1. **Data Access Integration**:
   - Ensure the API alert endpoints use our new alerting system for data
   - Update alert data models to match our alerting system implementation
   - Implement caching for alert history queries

2. **Real-time Updates**:
   - Connect the WebSocket alert channel to our alerting system
   - Ensure alert notifications are broadcast to subscribed clients

3. **Alert Management**:
   - Update alert acknowledgment to work with our new storage system
   - Add alert filtering capabilities

4. **Authentication**:
   - Ensure proper authentication checks for alert operations
   - Implement role-based permissions for alert management

## Integration Tasks

### 1. Update Alert Data Accessor

```python
# Update src/alpha_pulse/api/data/alerts.py to use the new alerting system

from alpha_pulse.monitoring.alerting.storage import AlertStorage

class AlertDataAccessor:
    """Alert data accessor for the API."""
    
    def __init__(self, alert_storage: AlertStorage):
        """Initialize the alert data accessor."""
        self.alert_storage = alert_storage
    
    async def get_alerts(self, start_time=None, end_time=None, filters=None):
        """Get alerts with optional filtering."""
        return await self.alert_storage.get_alerts(
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )
    
    async def acknowledge_alert(self, alert_id, user):
        """Acknowledge an alert."""
        return await self.alert_storage.acknowledge_alert(alert_id, user)
```

### 2. Connect WebSocket to Alerting System

```python
# Update WebSocket subscription manager to listen for alerts

from alpha_pulse.monitoring.alerting import AlertManager

class AlertsSubscription:
    """Subscription for alert updates."""
    
    def __init__(self, alert_manager: AlertManager):
        """Initialize the subscription."""
        self.alert_manager = alert_manager
        self.subscribers = set()
    
    async def start(self):
        """Start listening for alert notifications."""
        self.alert_manager.add_notification_handler(self._handle_alert)
    
    async def _handle_alert(self, alert):
        """Handle an alert notification."""
        # Broadcast to all subscribers
        for subscriber in self.subscribers:
            await subscriber.send_json({
                "type": "alert",
                "data": alert.to_dict()
            })
```

### 3. Update API Dependencies

```python
# Update src/alpha_pulse/api/dependencies.py

from alpha_pulse.monitoring.alerting import get_alert_manager
from alpha_pulse.monitoring.alerting.storage import get_alert_storage
from .data.alerts import AlertDataAccessor

async def get_alert_accessor():
    """Get the alert data accessor."""
    alert_storage = get_alert_storage()
    return AlertDataAccessor(alert_storage)

async def get_alert_subscription():
    """Get the alert subscription manager."""
    alert_manager = get_alert_manager()
    # Return the subscription manager
```

## Testing Plan

1. **Alert Endpoint Testing**:
   - Verify GET /api/alerts returns the correct alert history
   - Test filtering by severity, acknowledgment status, and time range
   - Ensure alert acknowledgment works correctly

2. **WebSocket Testing**:
   - Test WebSocket connection to /ws/alerts
   - Verify authentication and authorization
   - Ensure real-time alerts appear when triggered

3. **Integration Testing**:
   - Create an alert through the monitoring system
   - Verify it appears in both the API and WebSocket
   - Acknowledge the alert through the API
   - Verify acknowledgment status is updated

## Implementation Timeline

The integration tasks should take approximately 2-3 days:

1. Day 1: Update alert data accessor and API endpoints
2. Day 2: Connect WebSocket system to alerting system
3. Day 3: Testing and bug fixes

Once these integration tasks are complete, we'll be ready to move on to the Dashboard Frontend implementation (Task 1.5).

## Next Steps

1. Implement the updates above to connect the alerting system with the dashboard backend
2. Test the integration points
3. Update documentation
4. Begin planning for Dashboard Frontend implementation