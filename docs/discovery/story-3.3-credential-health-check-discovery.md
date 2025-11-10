# Story 3.3: Credential Health Check Job - Discovery

**Epic**: EPIC-003 - Credential Management (#142)
**Story Points**: 5
**Status**: Discovery
**Date**: 2025-11-10

## Overview

Implement an automated background job that periodically validates all stored tenant credentials by performing test API calls via CCXT. The job should detect credential failures (revoked API keys, permission changes, etc.) and notify tenants via webhook when their credentials are no longer valid.

## Requirements

### Acceptance Criteria
1. ✅ Celery job runs every 6 hours (configurable)
2. ✅ Tests each credential stored in Vault via CCXT
3. ✅ Webhook sent to tenant on credential failure
4. ✅ Retries 3 times before sending alert
5. ✅ Logs all health check results for audit trail

### Related ADRs
- **ADR-003**: Multi-Tenant Credential Management (Vault + validation)

## Current State Analysis

### Existing Infrastructure
✅ **TenantCredentialService** (Story 3.2)
- get_credentials() - Retrieves credentials from Vault
- Integrated with HashiCorpVaultProvider

✅ **CredentialValidator** (Story 3.2)
- validate() - Tests credentials via CCXT (fetch_balance)
- Returns ValidationResult with status and error details

❌ **Background Job Framework**
- No Celery setup
- No Redis broker
- No task scheduler

❌ **Webhook System**
- No webhook delivery mechanism
- No retry logic
- No tenant notification system

### Gaps to Fill
1. **Celery + Redis Setup**
   - Add dependencies to pyproject.toml
   - Create Celery app configuration
   - Setup Redis as message broker

2. **Health Check Task**
   - Periodic task that validates all credentials
   - Retry logic (3 attempts with exponential backoff)
   - Error tracking and reporting

3. **Webhook Notification System**
   - Webhook delivery mechanism (HTTP POST)
   - Retry logic with exponential backoff
   - Webhook signature for security (HMAC)

4. **Monitoring & Observability**
   - Prometheus metrics for health check results
   - Audit logging for all checks
   - Alerting for repeated failures

## Technical Design

### Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Celery     │─────>│  Health      │─────>│   Vault      │
│   Beat       │      │  Check Task  │      │   Provider   │
│  (Scheduler) │      └──────────────┘      └──────────────┘
└──────────────┘            │                       │
                            │                       ▼
                            │              ┌──────────────┐
                            │              │ Credentials  │
                            │              │  (by tenant) │
                            │              └──────────────┘
                            ▼
                   ┌──────────────┐
                   │  Credential  │
                   │  Validator   │
                   │   (CCXT)     │
                   └──────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  Validation Results    │
                │  (success/failure)     │
                └────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        ┌──────────────┐        ┌──────────────┐
        │   Webhook    │        │  Audit Log   │
        │  Notifier    │        │   + Metrics  │
        └──────────────┘        └──────────────┘
```

### Component Breakdown

#### 1. Celery Configuration (`src/alpha_pulse/celery_app.py`)
```python
from celery import Celery
from celery.schedules import crontab

app = Celery('alpha_pulse')
app.config_from_object('alpha_pulse.config.celery_config')

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'check-credentials-health': {
        'task': 'alpha_pulse.tasks.credentials.check_all_credentials_health',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
    },
}
```

#### 2. Health Check Task (`src/alpha_pulse/tasks/credential_health.py`)
```python
from celery import shared_task
from alpha_pulse.services.credentials import TenantCredentialService
from alpha_pulse.services.webhook_notifier import WebhookNotifier

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},  # Retry 3 times with 60s delay
)
def check_all_credentials_health(self):
    """
    Check health of all stored credentials.

    For each tenant:
    1. Retrieve all credentials from Vault
    2. Validate each credential via CCXT
    3. Send webhook on failure
    4. Log results
    """
    pass
```

#### 3. Webhook Notifier (`src/alpha_pulse/services/webhook_notifier.py`)
```python
import httpx
import hmac
import hashlib
from typing import Dict, Any
from loguru import logger

class WebhookNotifier:
    """
    Sends webhook notifications to tenants with retry logic.
    """

    async def send_credential_failure_webhook(
        self,
        tenant_id: UUID,
        webhook_url: str,
        payload: Dict[str, Any],
        webhook_secret: str,
    ) -> bool:
        """
        Send webhook with HMAC signature for security.

        Retry logic:
        - 3 attempts with exponential backoff (1s, 2s, 4s)
        - Timeout: 10 seconds per attempt
        """
        pass
```

#### 4. Tenant Webhook Configuration
**Database Table**: `tenant_webhooks`
```sql
CREATE TABLE tenant_webhooks (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    event_type VARCHAR(50) NOT NULL,  -- 'credential.failed', etc.
    webhook_url TEXT NOT NULL,
    webhook_secret TEXT NOT NULL,  -- For HMAC signature
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, event_type)
);
```

### Data Flow

**Happy Path (All credentials valid):**
```
1. Celery Beat triggers task every 6 hours
2. Task queries Vault for all tenant credentials
3. For each credential:
   - Validate via CCXT (fetch_balance)
   - Log success result
   - Update last_validated_at timestamp
4. Emit Prometheus metrics (success count)
5. Task completes
```

**Failure Path (Credential invalid):**
```
1. Celery Beat triggers task
2. Task validates credential -> ValidationResult(valid=False)
3. Check if this is first failure or repeated:
   - First failure: Log warning, don't send webhook yet
   - Second failure (6h later): Log error, don't send webhook yet
   - Third failure (12h later): Send webhook to tenant
4. Webhook delivery:
   - POST to tenant's webhook_url
   - Include HMAC signature for security
   - Retry 3 times with exponential backoff
   - Log webhook delivery status
5. Update credential status in tracking table
```

### Database Schema

**Table**: `credential_health_checks`
```sql
CREATE TABLE credential_health_checks (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    credential_type VARCHAR(20) NOT NULL,  -- 'trading' or 'readonly'
    check_timestamp TIMESTAMP NOT NULL,
    is_valid BOOLEAN NOT NULL,
    validation_error TEXT,  -- Error message if invalid
    consecutive_failures INT DEFAULT 0,
    last_success_at TIMESTAMP,
    webhook_sent BOOLEAN DEFAULT FALSE,
    webhook_sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_tenant_exchange (tenant_id, exchange)
);
```

### Configuration

**New settings in `src/alpha_pulse/config/settings.py`:**
```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Celery Configuration
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Credential Health Check
    credential_health_check_interval_hours: int = 6
    credential_health_check_retries: int = 3
    credential_health_check_retry_delay_seconds: int = 60
    credential_consecutive_failures_before_alert: int = 3

    # Webhook Configuration
    webhook_timeout_seconds: int = 10
    webhook_retry_attempts: int = 3
    webhook_retry_backoff_base: int = 1  # 1s, 2s, 4s exponential backoff
```

## Dependencies

### New Python Packages
```toml
[tool.poetry.dependencies]
celery = "^5.3.0"  # Task queue
redis = "^5.0.0"  # Already present (message broker)
httpx = "^0.27.0"  # Already present (webhook HTTP client)
celery-redbeat = "^2.1.0"  # Redis-backed Celery Beat scheduler
```

### Infrastructure
- **Redis**: Already in use for caching, will add Celery broker usage
- **Celery Worker**: New process to run background tasks
- **Celery Beat**: New process to schedule periodic tasks

## Security Considerations

### 1. Webhook Signature (HMAC-SHA256)
```python
def generate_webhook_signature(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
```

**Webhook Headers:**
```
X-AlphaPulse-Signature: sha256=<hmac_signature>
X-AlphaPulse-Timestamp: <unix_timestamp>
X-AlphaPulse-Event: credential.failed
```

### 2. Rate Limiting
- Max 1 webhook per credential per hour (prevent spam)
- Circuit breaker for tenant webhooks (disable after 10 consecutive failures)

### 3. PII Protection
- Don't include API keys/secrets in webhook payload
- Only include exchange name, credential type, and error summary

## Metrics & Observability

### Prometheus Metrics
```python
credential_health_check_total = Counter(
    'credential_health_check_total',
    'Total credential health checks performed',
    ['tenant_id', 'exchange', 'result']  # result: success/failure
)

credential_health_check_duration_seconds = Histogram(
    'credential_health_check_duration_seconds',
    'Time taken to check credential health',
    ['exchange']
)

webhook_delivery_total = Counter(
    'webhook_delivery_total',
    'Total webhook deliveries attempted',
    ['event_type', 'status']  # status: success/failure/timeout
)
```

### Audit Logging
```python
logger.info(
    "Credential health check completed",
    tenant_id=tenant_id,
    exchange=exchange,
    is_valid=result.valid,
    error=result.error,
    duration_ms=duration,
)
```

## Implementation Plan

### Phase 1: Celery Setup (1 SP)
1. Add Celery dependencies to pyproject.toml
2. Create `src/alpha_pulse/celery_app.py`
3. Create `src/alpha_pulse/config/celery_config.py`
4. Update docker-compose to include Redis broker (if needed)

### Phase 2: Health Check Task (2 SP)
1. Create `src/alpha_pulse/tasks/__init__.py`
2. Implement `credential_health.py` with health check logic
3. Create database migration for `credential_health_checks` table
4. Add Prometheus metrics

### Phase 3: Webhook Notifier (1.5 SP)
1. Implement `src/alpha_pulse/services/webhook_notifier.py`
2. Create database migration for `tenant_webhooks` table
3. Add webhook signature generation
4. Add retry logic with exponential backoff

### Phase 4: Testing & Documentation (0.5 SP)
1. Unit tests for health check task (mocked Vault + CCXT)
2. Unit tests for webhook notifier (mocked HTTP)
3. Integration test with local Celery worker
4. Update README with Celery setup instructions

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Celery worker crashes | Medium | High | Use supervisor/systemd for auto-restart |
| Redis unavailable | Low | High | Celery graceful degradation, alert on failure |
| Webhook endpoint down | High | Medium | Retry logic + circuit breaker |
| CCXT rate limits | Medium | Medium | Implement rate limiting per exchange |
| Task queue overflow | Low | Medium | Set max queue size, alert on threshold |

## Alternatives Considered

### Alternative 1: Use APScheduler instead of Celery
**Pros:**
- Simpler setup (no Redis required)
- Lightweight for single-server deployments

**Cons:**
- No distributed task queue
- Harder to scale horizontally
- No built-in retry logic

**Decision:** Use Celery for production-grade reliability and scalability

### Alternative 2: Immediate webhook on first failure
**Pros:**
- Faster notification to tenant

**Cons:**
- Higher false positive rate (transient network errors)
- More webhook spam

**Decision:** Use 3-failure threshold (12 hours) to reduce false positives

### Alternative 3: Store webhooks in Vault
**Pros:**
- Centralized secret storage

**Cons:**
- Overkill for webhook URLs (not as sensitive as API keys)
- Additional Vault reads on every webhook

**Decision:** Store webhook configuration in PostgreSQL, secrets still in Vault if needed

## Open Questions & Decisions

### Q1: Should we validate credentials on-demand (user-triggered)?
**Decision**: YES - Add manual health check endpoint: `POST /api/v1/credentials/{tenant_id}/check`
- Useful for debugging
- Immediate feedback after credential updates
- Same retry logic as scheduled job

### Q2: What if a tenant has 100+ credentials?
**Decision**: Batch process credentials with rate limiting
- Process max 10 credentials per second per exchange
- Use CCXT enableRateLimit=True
- Distribute load across 6-hour window

### Q3: Should we persist webhook payloads?
**Decision**: YES - Store webhook attempts in `webhook_delivery_log` table
- Helps with debugging
- Supports webhook replay on failure
- Retention: 30 days

### Q4: How to handle tenant without webhook configured?
**Decision**: Log failure to audit log only, no webhook sent
- Tenant can retrieve failures via API: `GET /api/v1/credentials/health`
- Dashboard can show credential health status

## Success Criteria

1. ✅ Celery worker starts successfully and processes tasks
2. ✅ Health check task runs every 6 hours without errors
3. ✅ All tenant credentials validated via CCXT
4. ✅ Webhook delivered successfully on credential failure (3 consecutive failures)
5. ✅ Prometheus metrics exported for monitoring
6. ✅ Audit logs show all health check results
7. ✅ Unit tests achieve >90% coverage
8. ✅ Integration test validates end-to-end flow

## Timeline Estimate

- **Phase 1 (Celery Setup)**: 2-3 hours
- **Phase 2 (Health Check Task)**: 4-5 hours
- **Phase 3 (Webhook Notifier)**: 3-4 hours
- **Phase 4 (Testing)**: 2-3 hours

**Total Estimate**: 11-15 hours (aligns with 5 SP)

## Next Steps

1. Review and approve discovery document
2. Create high-level design (HLD) document
3. Create delivery plan with implementation tasks
4. Begin Phase 1 implementation
