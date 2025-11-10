# Celery Setup Guide

## Overview

AlphaPulse uses Celery for distributed background tasks, including:
- **Credential Health Checks**: Periodic validation of tenant credentials (every 6 hours)
- **Webhook Notifications**: Alerting tenants of credential failures

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Redis     │────>│    Celery    │────>│   Workers   │
│   Broker    │     │     Beat     │     │   (Tasks)   │
│  (Queue)    │     │  (Scheduler) │     └─────────────┘
└─────────────┘     └──────────────┘
      ▲                                          │
      │                                          ▼
      │                                  ┌─────────────┐
      └──────────────────────────────────│   Vault     │
                                         │ (Credentials)│
                                         └─────────────┘
```

## Prerequisites

1. **Redis Server** (for message broker and result backend)
   ```bash
   # macOS
   brew install redis
   brew services start redis

   # Linux (Ubuntu/Debian)
   sudo apt-get install redis-server
   sudo systemctl start redis

   # Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

2. **HashiCorp Vault** (for credential storage)
   ```bash
   # See docs/VAULT_SETUP.md for Vault HA cluster setup
   # Or use dev mode for testing:
   docker-compose -f docker-compose.vault.yml up -d
   ```

3. **Environment Variables**
   ```bash
   # Redis configuration
   export CELERY_BROKER_URL="redis://localhost:6379/0"
   export CELERY_RESULT_BACKEND="redis://localhost:6379/1"

   # Vault configuration
   export VAULT_ADDR="http://localhost:8200"
   export VAULT_TOKEN="your-vault-token"

   # Health check settings (optional - defaults shown)
   export CREDENTIAL_HEALTH_CHECK_INTERVAL_HOURS=6
   export CREDENTIAL_CONSECUTIVE_FAILURES_BEFORE_ALERT=3
   export WEBHOOK_TIMEOUT_SECONDS=10
   export WEBHOOK_RETRY_ATTEMPTS=3
   ```

## Running Celery Workers

### 1. Start Celery Worker

Processes background tasks from the queue:

```bash
# Development (single worker with auto-reload)
poetry run celery -A alpha_pulse.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --queues=health_checks

# Production (multiple workers)
poetry run celery -A alpha_pulse.celery_app worker \
  --loglevel=warning \
  --concurrency=8 \
  --queues=health_checks \
  --hostname=worker1@%h
```

**Worker Options:**
- `--loglevel`: Log level (debug, info, warning, error)
- `--concurrency`: Number of worker processes (default: CPU count)
- `--queues`: Comma-separated list of queues to consume
- `--hostname`: Unique worker name for multi-worker setup

### 2. Start Celery Beat

Schedules periodic tasks (runs scheduled jobs):

```bash
# Development
poetry run celery -A alpha_pulse.celery_app beat \
  --loglevel=info \
  --scheduler=redbeat.RedBeatScheduler

# Production (with persistent schedule)
poetry run celery -A alpha_pulse.celery_app beat \
  --loglevel=warning \
  --scheduler=redbeat.RedBeatScheduler \
  --pidfile=/var/run/celery/beat.pid
```

**Beat Options:**
- `--scheduler`: Use RedBeatScheduler for Redis-backed persistence
- `--pidfile`: PID file location (prevents duplicate beat instances)

### 3. Monitor with Flower (Optional)

Real-time web-based monitoring tool:

```bash
# Install Flower
poetry add flower

# Start Flower
poetry run celery -A alpha_pulse.celery_app flower \
  --port=5555 \
  --basic_auth=admin:password
```

Access dashboard at: http://localhost:5555

## Scheduled Tasks

### Credential Health Check
- **Task**: `alpha_pulse.tasks.credential_health.check_all_credentials_health`
- **Schedule**: Every 6 hours (configurable via `CREDENTIAL_HEALTH_CHECK_INTERVAL_HOURS`)
- **Queue**: `health_checks`
- **Purpose**: Validates all stored credentials and sends webhook alerts on consecutive failures

**Cron Schedule:**
```python
crontab(minute=0, hour='*/6')  # Every 6 hours at minute 0
```

## Manual Task Invocation

### Check Specific Credential

```python
from alpha_pulse.tasks.credential_health import check_credential_health_manual

# Async (via Celery)
task = check_credential_health_manual.delay(
    tenant_id="12345678-1234-1234-1234-123456789abc",
    exchange="binance",
    credential_type="trading"
)

# Wait for result
result = task.get(timeout=30)
print(result)  # {'valid': True, 'credential_type': 'trading', ...}
```

### Check All Credentials (Manual Trigger)

```python
from alpha_pulse.tasks.credential_health import check_all_credentials_health

# Trigger task immediately
task = check_all_credentials_health.delay()

# Monitor status
print(task.state)  # 'PENDING', 'STARTED', 'SUCCESS', 'FAILURE'
```

## Production Deployment

### Using Supervisor

Create `/etc/supervisor/conf.d/celery.conf`:

```ini
[program:alphapulse-celery-worker]
command=/path/to/.venv/bin/celery -A alpha_pulse.celery_app worker --loglevel=warning --concurrency=8 --queues=health_checks
directory=/path/to/AlphaPulse
user=alphapulse
numprocs=1
autostart=true
autorestart=true
startsecs=10
stopwaitsecs=600
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/celery/worker.log
stderr_logfile=/var/log/celery/worker.err

[program:alphapulse-celery-beat]
command=/path/to/.venv/bin/celery -A alpha_pulse.celery_app beat --loglevel=warning --scheduler=redbeat.RedBeatScheduler --pidfile=/var/run/celery/beat.pid
directory=/path/to/AlphaPulse
user=alphapulse
numprocs=1
autostart=true
autorestart=true
startsecs=10
stopwaitsecs=10
stdout_logfile=/var/log/celery/beat.log
stderr_logfile=/var/log/celery/beat.err
```

```bash
# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start alphapulse-celery-worker
sudo supervisorctl start alphapulse-celery-beat
```

### Using systemd

Create `/etc/systemd/system/alphapulse-celery-worker.service`:

```ini
[Unit]
Description=AlphaPulse Celery Worker
After=network.target redis.target

[Service]
Type=forking
User=alphapulse
Group=alphapulse
EnvironmentFile=/etc/alphapulse/celery.env
WorkingDirectory=/opt/alphapulse
ExecStart=/opt/alphapulse/.venv/bin/celery -A alpha_pulse.celery_app worker --loglevel=warning --concurrency=8 --queues=health_checks --pidfile=/var/run/celery/worker.pid --detach
ExecStop=/opt/alphapulse/.venv/bin/celery -A alpha_pulse.celery_app control shutdown
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable alphapulse-celery-worker
sudo systemctl start alphapulse-celery-worker
```

## Monitoring & Observability

### Prometheus Metrics

Celery tasks expose Prometheus metrics:

```
# Credential health checks
credential_health_check_total{tenant_id, exchange, result}
credential_health_check_duration_seconds{exchange}

# Webhook deliveries
webhook_delivery_total{event_type, status}
```

Scrape endpoint: `/metrics` (if Prometheus exporter enabled)

### Logs

```bash
# Worker logs
tail -f /var/log/celery/worker.log

# Beat logs
tail -f /var/log/celery/beat.log

# Or via Docker
docker logs -f alphapulse-celery-worker
```

### Health Checks

```bash
# Check worker status
poetry run celery -A alpha_pulse.celery_app inspect active

# Check scheduled tasks
poetry run celery -A alpha_pulse.celery_app inspect scheduled

# Check registered tasks
poetry run celery -A alpha_pulse.celery_app inspect registered
```

## Troubleshooting

### Workers not processing tasks

1. **Check Redis connection:**
   ```bash
   redis-cli ping  # Should return "PONG"
   ```

2. **Check worker is running:**
   ```bash
   poetry run celery -A alpha_pulse.celery_app inspect active_queues
   ```

3. **Check task routing:**
   Ensure tasks are sent to the correct queue (`health_checks`)

### Beat not scheduling tasks

1. **Check only ONE beat instance is running:**
   ```bash
   ps aux | grep "celery.*beat"
   ```

2. **Check Redis for schedule:**
   ```bash
   redis-cli keys "redbeat:*"
   ```

3. **Check beat logs for errors:**
   ```bash
   tail -f /var/log/celery/beat.log
   ```

### Webhook deliveries failing

1. **Check webhook URL is accessible:**
   ```bash
   curl -X POST https://tenant-webhook-url.com/webhook \
     -H "Content-Type: application/json" \
     -d '{"test": true}'
   ```

2. **Check webhook secret is correct** (verify HMAC signature)

3. **Check retry settings** (3 attempts by default)

## Security Considerations

1. **Celery Broker Security:**
   - Use Redis with authentication: `redis://user:password@localhost:6379/0`
   - Enable Redis TLS for production
   - Isolate Celery Redis instance from application cache

2. **Task Serialization:**
   - Only JSON serialization is enabled (not pickle)
   - Prevents code injection attacks

3. **Webhook Signatures:**
   - All webhooks include HMAC-SHA256 signature
   - Tenants must verify signature to prevent spoofing

4. **Credential Access:**
   - Tasks use Vault for secure credential retrieval
   - No credentials logged or exposed in task results

## Next Steps

- Configure database models for health check tracking
- Set up webhook URLs in tenant configuration
- Enable Prometheus metrics scraping
- Configure alerting for task failures
