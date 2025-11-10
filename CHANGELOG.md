# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Multi-tenant support for AgentManager** (Story 2.2, EPIC-002)
  - Created `@require_tenant_id` decorator for consistent tenant validation
  - Added `tenant_id` parameter to 4 AgentManager methods:
    - `generate_signals()` - Primary signal generation method
    - `_aggregate_signals_with_ensemble()` - Ensemble aggregation method
    - `register_agent()` - Agent registration method
    - `create_and_register_agent()` - Agent factory method
  - All log messages now include tenant context `[Tenant: {tenant_id}]`
  - Signal metadata enriched with `tenant_id` field
  - Test infrastructure: 6 decorator tests (all passing)
  - Closes issue #163 (Story 2.2)

- **Multi-tenant support for RiskManager** (Story 2.3, EPIC-002)
  - Added `tenant_id` parameter to 3 RiskManager methods:
    - `calculate_risk_exposure()` - Risk exposure calculation
    - `evaluate_trade()` - Trade validation with risk limits
    - `calculate_position_size()` - Position sizing calculation
  - All log messages now include tenant context `[Tenant: {tenant_id}]`
  - Risk metrics enriched with `tenant_id` field
  - Reused `@require_tenant_id` decorator from Story 2.2
  - Closes issue #189 (Story 2.3)

- **Tenant context integration for API routes - Phase 1 (P0)** (Story 2.4, EPIC-002)
  - Integrated `get_current_tenant_id` dependency injection in 14 critical API endpoints
  - **risk.py router** (5 endpoints):
    - GET /exposure - Risk exposure metrics with tenant isolation
    - GET /metrics - Detailed risk metrics per tenant
    - GET /limits - Tenant-specific risk limits
    - GET /position-size/{asset} - Position sizing recommendations
    - GET /report - Comprehensive risk reports
  - **risk_budget.py router** (6 endpoints):
    - GET /current - Current risk budget allocations
    - GET /utilization - Budget utilization metrics
    - POST /rebalance - Trigger budget rebalancing
    - GET /recommendations - Rebalancing recommendations
    - GET /history - Historical budget data
    - GET /regime/{regime_type} - Regime-specific budgets
  - **portfolio.py router** (3 endpoints):
    - GET /portfolio - Portfolio data with tenant isolation
    - POST /portfolio/reload - Tenant-specific exchange sync
  - All endpoints include tenant context in log messages `[Tenant: {tenant_id}]`
  - Comprehensive test coverage: 603 lines of integration tests


- **Tenant context integration for API routes - Phase 2 (P1) - COMPLETE** (Story 2.4, EPIC-002)
  - Integrated `get_current_tenant_id` dependency injection in all 11 P1 endpoints
  - **alerts.py router** (2 endpoints):
    - GET /alerts - Alert retrieval with tenant isolation
    - POST /alerts/{id}/acknowledge - Alert acknowledgment with tenant context
  - **metrics.py router** (4 endpoints):
    - GET /metrics/{type} - Metrics retrieval per tenant
    - GET /metrics/{type}/latest - Latest metrics per tenant
    - GET /metrics/cache - Cache metrics with tenant logging
    - GET /metrics/database - Database metrics with tenant logging
  - **system.py router** (2 endpoints):
    - GET /system - System metrics with tenant context
    - POST /system/exchange/reload - Exchange sync with tenant isolation
  - **trades.py router** (1 endpoint):
    - GET /trades - Trade data with tenant isolation and filtering
  - **correlation.py router** (2 endpoints):
    - GET /correlation/matrix - Correlation matrix with tenant context and caching
    - GET /correlation/rolling - Rolling correlations with tenant isolation
  - All endpoints include tenant context in log messages `[Tenant: {tenant_id}]`
  - Test coverage: 1,537 lines of integration tests (21 test classes, 100% pattern consistency)
  - Design documents: Discovery (267 lines), HLD (860 lines), Delivery Plan (800 lines)
  - RICE Score: 57.6 (HIGH priority)
  - Closes issues #204 (Router Updates), #205 (Integration Tests), #206 (Documentation)

- **Tenant context integration for API routes - Phase 3 (P2) - COMPLETE** (Story 2.4, EPIC-002)
  - **CRITICAL SECURITY FIX**: Added authentication to hedging.py endpoints (was unauthenticated)
  - Integrated `get_current_tenant_id` and `get_current_user` in all 12 P2 endpoints
  - **hedging.py router** (3 endpoints - CRITICAL):
    - GET /analysis - Hedge position analysis (ADDED AUTHENTICATION + tenant context)
    - POST /execute - Execute hedge adjustments (ADDED AUTHENTICATION + tenant context)
    - POST /close - Close all hedge positions (ADDED AUTHENTICATION + tenant context)
  - **ensemble.py router** (9 endpoints):
    - POST /create - Create ensemble configuration with tenant isolation
    - POST /{ensemble_id}/register-agent - Register agent with tenant context
    - POST /{ensemble_id}/predict - Get ensemble prediction per tenant
    - GET /{ensemble_id}/performance - Performance metrics with tenant isolation
    - GET /{ensemble_id}/weights - Agent weights per tenant
    - POST /{ensemble_id}/optimize-weights - Weight optimization with tenant context
    - GET / - List ensembles (tenant-filtered)
    - GET /agent-rankings - Agent rankings (tenant-scoped)
    - DELETE /{ensemble_id} - Delete ensemble with tenant isolation
  - All endpoints include tenant context in log messages `[Tenant: {tenant_id}] [User: {user.get('sub')}]`
  - Test coverage: 1,117 lines of integration tests (14 test classes, 35 tests total)
  - Design documents: Discovery (409 lines), HLD (573 lines), Delivery Plan (1,040 lines)
  - RICE Score: 64.0 (CRITICAL - highest due to security gap)
  - Progress: 36/43 endpoints complete (84% of Story 2.4)

- **Tenant context integration for API routes - Phase 4 (P3) - COMPLETE - Story 2.4 FINISHED!** (Story 2.4, EPIC-002)
  - **CRITICAL SECURITY FIX**: Added user authentication to positions.py endpoints (was completely unauthenticated)
  - Integrated `get_current_user` and `get_current_tenant_id` in all 7 P3 endpoints
  - **positions.py router** (3 endpoints - CRITICAL):
    - GET /spot - Spot positions (ADDED USER AUTHENTICATION + tenant context)
    - GET /futures - Futures positions (ADDED USER AUTHENTICATION + tenant context)
    - GET /metrics - Position metrics (ADDED USER AUTHENTICATION + tenant context)
  - **regime.py router** (4 endpoints):
    - GET /current - Current market regime state with tenant context
    - GET /history - Historical regime data with tenant context
    - GET /analysis/{regime_type} - Regime-specific analysis with tenant context
    - GET /alerts - Regime transition alerts with tenant context
  - All endpoints include tenant context in log messages `[Tenant: {tenant_id}] [User: {user.get('sub')}]`
  - Test coverage: 669 lines of integration tests (10 test classes, 24 tests total)
  - Design documents: Discovery (409 lines), HLD (1,684 lines), Delivery Plan (1,808 lines)
  - RICE Score: 84.0 (CRITICAL - highest due to authentication gap)
  - **Story 2.4 Progress: 43/43 endpoints complete (100%)** - All API endpoints now have tenant context!

- **Credential Health Check System** (Story 3.3, EPIC-003)
  - Implemented Celery-based background job infrastructure for periodic credential validation
  - **Celery Configuration** (`src/alpha_pulse/celery_app.py`):
    - Redis-backed message broker and result backend
    - Celery Beat scheduler with RedBeat for persistent task schedules
    - Periodic health checks every 6 hours (configurable via `CREDENTIAL_HEALTH_CHECK_INTERVAL_HOURS`)
    - Dedicated `health_checks` queue for task routing
  - **Health Check Task** (`src/alpha_pulse/tasks/credential_health.py`):
    - Validates all tenant credentials via CCXT exchange integration
    - Tracks consecutive failures and sends webhook notifications on threshold breach
    - Prometheus metrics: `credential_health_check_total`, `credential_health_check_duration_seconds`
    - Manual on-demand check via `check_credential_health_manual` task
  - **Webhook Notifier Service** (`src/alpha_pulse/services/webhook_notifier.py`):
    - HMAC-SHA256 signed webhook notifications for credential failures
    - Exponential backoff retry logic (1s → 2s → 4s delays)
    - Timing-safe signature verification via `verify_webhook_signature()` helper
    - Configurable timeout (10s) and retry attempts (3)
  - **Database Models** (`src/alpha_pulse/models/credential_health.py`):
    - `CredentialHealthCheck`: Tracks validation history, consecutive failures, webhook delivery
    - `TenantWebhook`: Stores webhook configurations with encrypted secrets
    - 9 indexes for query performance, 9 check constraints for data integrity
    - Utility functions: `get_latest_health_check()`, `get_failing_credentials()`, `get_tenant_webhook()`
  - **Database Migration** (Alembic `010_add_credential_health_tables.py`):
    - Creates `credential_health_checks` and `tenant_webhooks` tables
    - Safe rollback support via `downgrade()` function
  - **Configuration Settings** (11 new settings in `secure_settings.py`):
    - Celery broker/backend URLs, health check intervals, webhook timeouts/retries
  - **Dependencies Added**:
    - `celery==5.4.0` - Distributed task queue
    - `celery-redbeat==2.2.0` - Redis-backed Beat scheduler
    - `redis==5.2.1` - Redis client for broker/backend
  - **Documentation**:
    - `docs/CELERY_SETUP.md` (417 lines) - Production deployment guide with Supervisor config
    - `docs/discovery/story-3.3-credential-health-check-discovery.md` (430 lines)
  - **Test Coverage**:
    - `test_webhook_notifier.py`: 25/25 tests passing (100% coverage)
    - `test_credential_health.py`: 10/16 tests passing (62% - mock signature issues documented)
  - **Known Limitations** (tracked in GitHub issue #229):
    - Core integration pending: `list_all_credentials()` returns empty list (4-6 hour fix)
    - Hardcoded Vault address needs settings integration
    - Webhook URL retrieval not implemented
  - **QA Verdict**: CONDITIONAL APPROVE (77/100 score) - Safe to merge, unsafe to deploy without issue #229
  - Closes issue #169 (Story 3.3)

### Changed
- Audited documentation: removed multi-tenant sprint artefacts, refreshed README
  and CLAUDE guidance to match the Poetry-based workflow.
- **BREAKING**: `AgentManager` methods now require `tenant_id` parameter
  - See MIGRATION.md for upgrade guide
- **BREAKING**: `RiskManager` methods now require `tenant_id` parameter
  - See MIGRATION.md for upgrade guide
- **BREAKING**: positions.py endpoints now require JWT authentication (Phase 4)
  - GET /spot, GET /futures, GET /metrics now require valid JWT token
  - Returns 401 Unauthorized if token missing/invalid
  - All API clients must include Authorization header with valid token
  - See Phase 4 HLD for migration guide

### Fixed
- REAL docs now reference the running API (`uvicorn src.alpha_pulse.api.main:app`)
  and updated testing/linting commands.

## [2.0.1] - 2025-11-02

### Fixed
- **Circular import in data_lake module** (Issue #191)
  - Resolved circular dependency between `lake_manager.py` and `storage_layers.py`
  - Created `types.py` module to hold `DataFormat` and `StorageBackend` enums
  - Maintained backward compatibility via re-exports
  - Excluded data_lake tests pending lineage module implementation
  - Closes issue #191
