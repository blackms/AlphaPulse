# Audit Logging Guide

## Overview

AlphaPulse implements comprehensive audit logging for all security-relevant events, trading decisions, and API access. The audit system provides:

- **Structured logging** with consistent event formats
- **Asynchronous writes** for minimal performance impact
- **Compliance support** for GDPR, SOX, PCI, and other regulations
- **Anomaly detection** for security monitoring
- **Flexible querying** and reporting capabilities

## Architecture

### Components

1. **AuditLogger**: Core logging service with batching and async writes
2. **AuditEventTypes**: Standardized event taxonomy
3. **AuditContext**: Thread-local context propagation
4. **AuditQueries**: Query builder and reporting utilities
5. **API Middleware**: Automatic request/response logging

### Event Categories

- **Authentication** (`auth.*`): Login, logout, token operations
- **Trading** (`trade.*`): Decisions, executions, cancellations
- **Risk** (`risk.*`): Limit triggers, overrides, alerts
- **API** (`api.*`): Requests, responses, errors
- **System** (`system.*`): Startup, shutdown, configuration
- **Agent** (`agent.*`): AI decisions and signals
- **Data** (`data.*`): Access, modifications, exports

## Implementation

### Basic Usage

```python
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType

audit_logger = get_audit_logger()

# Log a simple event
audit_logger.log(
    event_type=AuditEventType.TRADE_DECISION,
    event_data={
        'symbol': 'BTC/USD',
        'action': 'buy',
        'quantity': 0.5,
        'reasoning': {'rsi': 30, 'macd': 'bullish'}
    },
    data_classification="confidential"
)
```

### Context Management

Use context managers to propagate user and request information:

```python
# Set audit context for a request
with audit_logger.context(
    user_id="trader123",
    ip_address="192.168.1.100",
    request_id="req-12345"
):
    # All logs within this block include the context
    audit_logger.log_trade_execution(...)
```

### Convenience Methods

Pre-defined methods for common events:

```python
# Authentication
audit_logger.log_login(
    user_id="user123",
    success=True,
    method="password"
)

# Trading
audit_logger.log_trade_decision(
    agent="TechnicalAgent",
    symbol="ETH/USD",
    action="sell",
    quantity=2.0,
    reasoning={'ma_cross': 'bearish'},
    confidence=0.75
)

# Risk events
audit_logger.log_risk_event(
    risk_type="position_size",
    threshold=0.1,
    actual_value=0.15,
    action_taken="reduce_position"
)
```

## API Integration

### Automatic Request Logging

Add middleware to FastAPI for automatic logging:

```python
from alpha_pulse.api.middleware.audit_middleware import AuditLoggingMiddleware

app.add_middleware(AuditLoggingMiddleware)
```

This logs:
- All API requests with method, path, and timing
- Response status codes and errors
- User context from JWT tokens
- Performance metrics

### Permission-Based Logging

Sensitive operations are automatically logged:

```python
from alpha_pulse.api.auth import PermissionChecker

# This dependency logs permission checks
require_trading = PermissionChecker(["execute_trades"])

@router.post("/trades")
async def execute_trade(
    trade: TradeRequest,
    user: User = Depends(require_trading)  # Logged automatically
):
    ...
```

## Agent Integration

### Auditable Agents

Make agents auditable by inheriting from `AuditableAgent`:

```python
from alpha_pulse.agents.audit_wrapper import AuditableAgent

class TechnicalAgent(AuditableAgent, BaseAgent):
    def analyze(self, symbol: str):
        # Analysis logic...
        
        # Log the decision
        self.audit_decision(
            decision_type="entry_signal",
            symbol=symbol,
            action="buy",
            reasoning={'indicators': indicators},
            confidence=confidence
        )
```

### Method Decorators

Use decorators for automatic method auditing:

```python
from alpha_pulse.agents.audit_wrapper import audit_agent_method

class FundamentalAgent(AuditableAgent):
    @audit_agent_method("analyze_financials")
    def analyze_financials(self, symbol: str):
        # Method execution is automatically logged
        return analysis_results
```

## Querying Audit Logs

### Using the Query Builder

```python
from alpha_pulse.utils.audit_queries import AuditQueryBuilder
from datetime import datetime, timedelta

builder = AuditQueryBuilder()

# Find failed logins in the last 24 hours
failed_logins = (
    builder
    .time_range(datetime.now() - timedelta(days=1))
    .event_types(AuditEventType.AUTH_FAILED)
    .order_by_time(descending=True)
    .limit(100)
    .execute()
)

# Get high-severity events for a user
user_events = (
    builder
    .user("trader123")
    .severity(AuditSeverity.WARNING)
    .execute()
)
```

### Generating Reports

```python
from alpha_pulse.utils.audit_queries import AuditReporter

reporter = AuditReporter()

# Security summary
security_report = reporter.security_summary(
    start_time=datetime.now() - timedelta(days=7)
)

# Trading activity
trading_report = reporter.trading_activity(
    start_time=datetime.now() - timedelta(days=30),
    user_id="trader123"
)

# Detect anomalies
anomalies = reporter.detect_anomalies(
    lookback_days=7,
    threshold_multiplier=3.0
)
```

## API Endpoints

### View Audit Logs

```bash
# Get recent audit logs
GET /api/audit/logs?limit=100

# Filter by event type
GET /api/audit/logs?event_type=trade.executed&start_date=2024-01-01

# Filter by user
GET /api/audit/logs?user_id=trader123
```

### Generate Reports

```bash
# Security report
GET /api/audit/reports/security?days=7

# Trading activity report
GET /api/audit/reports/trading?days=30

# Compliance report
GET /api/audit/reports/compliance?regulations=SOX,GDPR

# Anomaly detection
GET /api/audit/anomalies?lookback_days=7&threshold=3.0
```

### Export Data

```bash
# Export audit logs
POST /api/audit/export?start_date=2024-01-01&end_date=2024-01-31&format=csv
```

## Performance Considerations

### Asynchronous Logging

The audit system uses background threads for writes:

- Events are queued in memory
- Batch writes occur every 5 seconds or 100 events
- No blocking of application code

### Optimization Settings

```python
# Configure batching
audit_logger = AuditLogger(
    batch_size=100,        # Events per batch
    flush_interval=5.0,    # Seconds between flushes
    max_queue_size=10000   # Maximum queued events
)
```

### Database Indexes

Ensure proper indexes for query performance:

```sql
-- Time-based queries
CREATE INDEX idx_audit_timestamp_type ON audit_logs(timestamp, event_type);

-- User queries
CREATE INDEX idx_audit_user_timestamp ON audit_logs(user_id, timestamp);

-- Correlation tracking
CREATE INDEX idx_audit_correlation ON audit_logs(correlation_id);
```

## Compliance Features

### Data Classification

Mark sensitive data appropriately:

```python
audit_logger.log(
    event_type=AuditEventType.DATA_ACCESS,
    event_data={'table': 'user_pii'},
    data_classification="restricted",  # public, internal, confidential, restricted
    regulatory_flags={'GDPR': True}
)
```

### Regulatory Flags

Flag events for compliance reporting:

```python
# SOX compliance for financial operations
audit_logger.log_trade_execution(
    ...,
    regulatory_flags={'SOX': True}
)

# GDPR for personal data
audit_logger.log(
    ...,
    regulatory_flags={'GDPR': True, 'data_subject': user_id}
)
```

### Retention Policies

Implement data retention:

```python
# Archive old audit logs
from datetime import datetime, timedelta

cutoff_date = datetime.now() - timedelta(days=365)
old_logs = session.query(AuditLog).filter(
    AuditLog.timestamp < cutoff_date
)

# Archive to cold storage
archive_logs(old_logs)

# Delete after retention period
old_logs.delete()
```

## Security Best Practices

### Sensitive Data Handling

1. **Never log passwords or secrets directly**
   ```python
   # Bad
   audit_logger.log(event_data={'password': user_password})
   
   # Good
   audit_logger.log(event_data={'password_changed': True})
   ```

2. **Hash sensitive identifiers**
   ```python
   audit_logger.log_secret_access(
       secret_name="api_key",  # Automatically hashed
       purpose="trading"
   )
   ```

3. **Use encryption for audit logs**
   - Audit logs containing sensitive data are encrypted
   - Search tokens allow querying without decryption

### Access Control

Restrict audit log access:

```python
# Only users with view_audit_logs permission
@router.get("/audit/logs")
async def get_logs(user: User = Depends(require_audit_access)):
    ...
```

### Monitoring

Set up alerts for security events:

```python
# Alert on excessive failed logins
if failed_login_count > threshold:
    send_security_alert("Potential brute force attack")

# Alert on privilege escalation attempts
if event.type == "permission_denied" and event.data.get("sensitive"):
    send_security_alert("Unauthorized access attempt")
```

## Troubleshooting

### Common Issues

1. **Missing audit logs**
   - Check batch size and flush interval
   - Verify database connection
   - Check for queue overflow

2. **Performance impact**
   - Increase batch size
   - Reduce flush frequency
   - Consider archiving old logs

3. **Storage growth**
   - Implement retention policies
   - Archive to cheaper storage
   - Compress old logs

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("alpha_pulse.utils.audit_logger").setLevel(logging.DEBUG)
```

### Health Checks

Monitor audit system health:

```python
# Check queue size
queue_size = audit_logger._queue.qsize()
if queue_size > 8000:  # 80% of max
    logger.warning(f"Audit queue high: {queue_size}")

# Check flush thread
if not audit_logger._flush_thread.is_alive():
    logger.error("Audit flush thread died")
```

## Migration from Previous System

If migrating from file-based logging:

```python
# Import old logs
from alpha_pulse.migrations.import_audit_logs import import_from_files

import_from_files(
    log_directory="/var/log/alphapulse",
    start_date="2024-01-01",
    batch_size=1000
)
```

## Future Enhancements

Planned improvements:

1. **Real-time streaming** of audit events via WebSocket
2. **Machine learning** for advanced anomaly detection
3. **Distributed tracing** integration with OpenTelemetry
4. **Blockchain anchoring** for tamper-proof audit trails
5. **Natural language queries** for audit log search