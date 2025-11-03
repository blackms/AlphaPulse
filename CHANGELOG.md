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
  - Partial progress on issue #165 (Story 2.4)

### Changed
- Audited documentation: removed multi-tenant sprint artefacts, refreshed README
  and CLAUDE guidance to match the Poetry-based workflow.
- **BREAKING**: `AgentManager` methods now require `tenant_id` parameter
  - See MIGRATION.md for upgrade guide
- **BREAKING**: `RiskManager` methods now require `tenant_id` parameter
  - See MIGRATION.md for upgrade guide

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
