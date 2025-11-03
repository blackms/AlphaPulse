# Story 2.4 Phase 2: High-Level Design

**Date**: 2025-11-03
**Story**: EPIC-002 Story 2.4 Phase 2
**Phase**: Design the Solution
**Status**: In Review
**Version**: 1.0

---

## Executive Summary

This document details the high-level design for integrating tenant context into 16 P1 priority API endpoints across 5 routers (alerts, metrics, system, trades, correlation). The design reuses the proven pattern from Phase 1 (v2.1.0), minimizing complexity and risk while completing the multi-tenant security architecture.

**Key Decision**: Reuse Phase 1 patterns with zero architectural changes
**Complexity**: LOW
**Risk Level**: LOW
**Estimated Effort**: 2-3 days

---

## Table of Contents

1. [Discovery and Scope](#discovery-and-scope)
2. [Architecture Blueprint](#architecture-blueprint)
3. [Decision Log](#decision-log)
4. [Delivery Plan](#delivery-plan)
5. [Validation Strategy](#validation-strategy)
6. [Collaboration](#collaboration)

---

## Discovery and Scope

### Scenario Clarification

**Capability Summary**: Extend tenant context integration from 14 Phase 1 endpoints (risk, risk_budget, portfolio) to 16 additional P1 endpoints (alerts, metrics, system, trades, correlation), completing tenant isolation for all critical API operations.

**Source Artifacts to Mirror**:
- [src/alpha_pulse/api/routers/risk.py](../src/alpha_pulse/api/routers/risk.py:54) - Phase 1 pattern reference
- [src/alpha_pulse/api/routers/risk_budget.py](../src/alpha_pulse/api/routers/risk_budget.py:68) - Phase 1 pattern reference
- [src/alpha_pulse/api/routers/portfolio.py](../src/alpha_pulse/api/routers/portfolio.py:29) - Phase 1 pattern reference
- [src/alpha_pulse/api/middleware/tenant_context.py](../src/alpha_pulse/api/middleware/tenant_context.py:171) - `get_current_tenant_id` dependency
- [tests/api/test_risk_tenant_context.py](../tests/api/test_risk_tenant_context.py) - Test pattern reference

**Stakeholders and Decision Owners**:
- **Technical Lead**: Architecture approval, risk review, final sign-off
- **Security Lead**: Multi-tenant security validation
- **Engineering Team**: Implementation, testing, code review
- **QA/Release Manager**: CI/CD validation, deployment approval

### Constraints

#### Functional Constraints
- **Behavioral Guarantees**: All endpoints must enforce tenant isolation with zero cross-tenant data leakage
- **SLAs**: No degradation in API response times (maintain p99 <200ms)
- **Regulatory Requirements**: Support SOC 2, ISO 27001 compliance requirements
- **Domain Guardrails**: JWT token must contain valid `tenant_id` claim; fallback to default tenant not allowed

#### Technical Constraints
- **Tech Stack Boundaries**: FastAPI Depends() pattern, JWT middleware, Python 3.11+
- **Legacy Coupling**: Existing permission-based auth system must remain intact
- **Integration Limits**: No changes to database schema or JWT token structure
- **External Contracts**: API responses remain unchanged (tenant_id is internal only)

#### Operational Constraints
- **Rollout Window**: Deploy during business hours (zero-downtime requirement)
- **Support Model**: Self-service deployment via GitHub Actions CI/CD
- **Maintenance Expectations**: Follow existing code review, testing, and documentation standards
- **Sustainability**: Patterns must be maintainable by any team member

### Assumptions

1. **JWT Token Structure**: All authenticated requests include `tenant_id` claim in JWT token
   - **Validation Method**: Integration tests with mocked JWT tokens
   - **Retirement Trigger**: If tenant claim structure changes (requires ADR)

2. **Existing Middleware**: `tenant_context.py` middleware correctly extracts `tenant_id` from JWT
   - **Validation Method**: Unit tests for `get_current_tenant_id()`
   - **Retirement Trigger**: If middleware refactoring planned

3. **Data Accessor Compatibility**: Current data accessors support tenant-scoped queries without modification
   - **Validation Method**: Review data accessor implementations
   - **Retirement Trigger**: If data accessor refactoring required

4. **Test Infrastructure**: Existing pytest fixtures and mocks support Phase 2 testing
   - **Validation Method**: Validate fixtures work with new routers
   - **Retirement Trigger**: If test framework changes

### Dependencies

#### Upstream Dependencies
| Dependency | Owner | Status | Risk | Mitigation |
|------------|-------|--------|------|------------|
| `alpha_pulse.api.middleware.tenant_context` | Auth Team | ✅ Stable | LOW | Phase 1 proven |
| JWT middleware + auth.py | Auth Team | ✅ Stable | LOW | No changes required |
| pytest + conftest fixtures | QA Team | ✅ Stable | LOW | Phase 1 patterns reusable |

#### Downstream Dependencies
| System | Impact | Contact | Risk | Mitigation |
|--------|--------|---------|------|------------|
| Frontend Dashboard | None (internal change) | Frontend Team | NONE | API contract unchanged |
| External API Clients | None (internal change) | API Consumers | NONE | No breaking changes |
| Monitoring/Alerting | Log format change | Ops Team | LOW | Update log parsing |

---

## Architecture Blueprint

### Architecture Views

#### Context View

```
┌────────────────────────────────────────────────────────┐
│                   External Actors                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Web App  │  │ CLI Tool │  │ 3rd Party│             │
│  │  Client  │  │  Client  │  │   API    │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
└───────│─────────────│─────────────│───────────────────┘
        │             │             │
        │ HTTPS + JWT │             │
        ▼             ▼             ▼
┌────────────────────────────────────────────────────────┐
│           FastAPI Application (AlphaPulse)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │         JWT Middleware Layer                      │  │
│  │  (Extracts tenant_id from token, validates)      │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       │                                 │
│  ┌────────────────────▼─────────────────────────────┐  │
│  │  API Routers (Phase 2 Integration Points)        │  │
│  │  ┌─────────────────────────────────────────┐    │  │
│  │  │ alerts.py     (3 endpoints)              │    │  │
│  │  │ metrics.py    (4 endpoints)              │    │  │
│  │  │ system.py     (2 endpoints)              │    │  │
│  │  │ trades.py     (5 endpoints)              │    │  │
│  │  │ correlation.py (2 endpoints)             │    │  │
│  │  └────────────────┬────────────────────────┘    │  │
│  │                   │ Depends(get_current_tenant_id)│ │
│  │                   ▼                               │  │
│  │  ┌─────────────────────────────────────────┐    │  │
│  │  │ Tenant Context (tenant_id extracted)     │    │  │
│  │  └────────────────┬────────────────────────┘    │  │
│  └───────────────────┼────────────────────────────────┘
│                      │                                 │
│  ┌───────────────────▼────────────────────────────┐   │
│  │    Data Accessors (tenant-scoped queries)      │   │
│  │  AlertDataAccessor, MetricsDataAccessor, etc.  │   │
│  └───────────────────┬────────────────────────────┘   │
└────────────────────────│───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│              External Systems                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │PostgreSQL│  │  Redis   │  │  Vault   │             │
│  │   DB     │  │  Cache   │  │ Secrets  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└────────────────────────────────────────────────────────┘

Value Exchanged:
- Actors → System: Authenticated API requests with tenant_id in JWT
- System → Actors: Tenant-scoped data responses (alerts, metrics, trades)
- System → PostgreSQL: Tenant-isolated database queries (RLS enforced)
```

#### Container View

```
┌─────────────────────────────────────────────────────────┐
│  AlphaPulse API Service (FastAPI + Uvicorn)             │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Router Layer (Phase 2 Targets)                    │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ alerts.py                                    │   │  │
│  │ │  - GET /alerts                               │   │  │
│  │ │  - POST /alerts                              │   │  │
│  │ │  - GET /alerts/{alert_id}                    │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ metrics.py                                   │   │  │
│  │ │  - GET /metrics/system                       │   │  │
│  │ │  - GET /metrics/trading                      │   │  │
│  │ │  - GET /metrics/risk                         │   │  │
│  │ │  - GET /metrics/portfolio                    │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ system.py                                    │   │  │
│  │ │  - GET /health                               │   │  │
│  │ │  - GET /status                               │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ trades.py                                    │   │  │
│  │ │  - GET /trades                               │   │  │
│  │ │  - POST /trades                              │   │  │
│  │ │  - GET /trades/{trade_id}                    │   │  │
│  │ │  - GET /trades/history                       │   │  │
│  │ │  - GET /trades/pending                       │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  │ ┌─────────────────────────────────────────────┐   │  │
│  │ │ correlation.py                               │   │  │
│  │ │  - GET /correlation/matrix                   │   │  │
│  │ │  - GET /correlation/analysis                 │   │  │
│  │ └─────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Middleware Layer                                   │  │
│  │  - JWT Authentication Middleware                   │  │
│  │  - Tenant Context Middleware                       │  │
│  │  - Audit Logging Middleware                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Dependency Injection Layer                         │  │
│  │  - get_current_tenant_id() → extracts tenant_id    │  │
│  │  - get_current_user() → extracts user context      │  │
│  │  - Data accessor factories                         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Data Access Layer                                  │  │
│  │  - AlertDataAccessor (tenant-scoped)               │  │
│  │  - MetricsDataAccessor (tenant-scoped)             │  │
│  │  - TradeDataAccessor (tenant-scoped)               │  │
│  │  - SystemDataAccessor (tenant logging)             │  │
│  │  - CorrelationAnalyzer (tenant-aware)              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### Component View

**Critical Modules to Touch**:

1. **Router Files (5 files to modify)**
   - `src/alpha_pulse/api/routers/alerts.py`
   - `src/alpha_pulse/api/routers/metrics.py`
   - `src/alpha_pulse/api/routers/system.py`
   - `src/alpha_pulse/api/routers/trades.py`
   - `src/alpha_pulse/api/routers/correlation.py`

2. **Test Files (5 files to create)**
   - `tests/api/test_alerts_tenant_context.py`
   - `tests/api/test_metrics_tenant_context.py`
   - `tests/api/test_system_tenant_context.py`
   - `tests/api/test_trades_tenant_context.py`
   - `tests/api/test_correlation_tenant_context.py`

3. **Dependencies (No changes required - reuse Phase 1)**
   - `src/alpha_pulse/api/middleware/tenant_context.py` ← **READ ONLY**
   - `src/alpha_pulse/api/auth.py` ← **READ ONLY**
   - `tests/api/conftest.py` ← **READ ONLY**

**Module Interaction Diagram**:

```
┌─────────────────────────────────────────────────────────┐
│ Router Endpoint (e.g., GET /alerts)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ 1. Depends(get_current_tenant_id)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ tenant_context.py::get_current_tenant_id()              │
│  - Extracts tenant_id from request.state.tenant_id      │
│  - Validates tenant_id exists                           │
│  - Returns tenant_id: str                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ 2. tenant_id passed to endpoint
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Endpoint Function (e.g., async def get_alerts(...))     │
│  - Logs: f"[Tenant: {tenant_id}] ..."                  │
│  - Calls data accessor with tenant context             │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ 3. Tenant-scoped query
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Data Accessor (e.g., AlertDataAccessor)                │
│  - Adds tenant_id to WHERE clause                       │
│  - Executes tenant-isolated query                       │
│  - Returns filtered results                             │
└─────────────────────────────────────────────────────────┘
```

#### Runtime View

**Request Flow (Phase 2 Pattern)**:

```
┌─────┐                                                         ┌─────┐
│User │                                                         │ DB  │
└──┬──┘                                                         └──┬──┘
   │                                                               │
   │ 1. GET /alerts?severity=critical                             │
   │    Authorization: Bearer <JWT_TOKEN>                         │
   ├──────────────────────────────────────────────────────────────┤
   │                                                               │
   │ 2. JWT Middleware extracts tenant_id from token              │
   │    request.state.tenant_id = "tenant-abc-123"                │
   │                                                               │
   │ 3. get_current_tenant_id() retrieves tenant_id               │
   │    tenant_id = request.state.tenant_id                       │
   │                                                               │
   │ 4. Endpoint: async def get_alerts(                           │
   │       tenant_id: str = Depends(get_current_tenant_id)        │
   │    )                                                          │
   │    logger.info(f"[Tenant: {tenant_id}] Fetching alerts")     │
   │                                                               │
   │ 5. Data Accessor: alert_accessor.get_alerts(filters)         │
   │    Query: SELECT * FROM alerts                               │
   │           WHERE tenant_id = 'tenant-abc-123'                 │
   │           AND severity = 'critical'                          │
   │    ──────────────────────────────────────────────────────────┤
   │                                                               │
   │                              6. Return tenant-scoped results │
   │    ◀─────────────────────────────────────────────────────────┤
   │                                                               │
   │ 7. Response: [{"id": 1, "severity": "critical", ...}]        │
   │    (No tenant_id exposed in response)                        │
   ◀─────────────────────────────────────────────────────────────┤
   │                                                               │
```

**State Transitions**: None (stateless HTTP requests)

**Tick Cadence**: N/A (request/response model)

#### Deployment View

**Environments**: No changes to existing deployment topology

```
┌────────────────────────────────────────────────────────┐
│ Production Environment (AWS)                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Load Balancer (ALB)                             │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ FastAPI App (ECS/Fargate)                       │   │
│  │  - Auto-scaling: 2-10 instances                 │   │
│  │  - Health check: GET /health                    │   │
│  │  - Rolling deployment (zero downtime)           │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │ PostgreSQL RDS (Multi-AZ)                       │   │
│  │  - RLS policies active per tenant               │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘

Staging Environment: Identical topology (1-2 instances)
Development Environment: Local Docker Compose
```

**Scaling Strategy**: No changes (horizontal scaling via ECS)

**Resilience Expectations**:
- Health checks monitor tenant context extraction
- Circuit breakers on database queries
- Graceful degradation if tenant_id missing (return 401)

### Data Design

#### Lifecycle

**Entity Model**: No new entities required

**Ownership Boundaries**: Each tenant owns their own:
- Alerts (alerts table, tenant_id FK)
- Metrics (metrics table, tenant_id FK)
- Trades (trades table, tenant_id FK)
- System logs (audit_logs table, tenant_id FK)

**Retention Policy**: Per existing tenant data retention policies (90 days default)

**Archival Path**: No changes to existing archival strategy

#### Duplication Strategy

**Decision**: **SHARE** - Do not duplicate state across tenants

- Phase 2 endpoints read from shared multi-tenant tables
- Tenant isolation enforced via RLS policies + application-level filtering
- No data duplication (single source of truth per tenant)

#### Consistency Model

**Synchronous Updates**: All Phase 2 endpoints use synchronous writes

- Alerts: Immediate consistency (POST /alerts → DB write)
- Metrics: Eventual consistency (buffered writes, acceptable)
- Trades: Immediate consistency (critical path)
- System: Read-only (no writes)

**Conflict Resolution**: N/A (tenant_id prevents conflicts by design)

### Integration Points

#### APIs

**APIs to Consume** (No changes):
- JWT token validation endpoint (existing)
- Database connection pool (existing)
- Redis cache (if used by metrics/correlation)

**APIs to Expose** (Modified endpoints):

| Endpoint | Method | Changes | Versioning |
|----------|--------|---------|------------|
| `/alerts` | GET, POST | Add tenant_id parameter | No version change |
| `/alerts/{alert_id}` | GET | Add tenant_id parameter | No version change |
| `/metrics/system` | GET | Add tenant_id parameter | No version change |
| `/metrics/trading` | GET | Add tenant_id parameter | No version change |
| `/metrics/risk` | GET | Add tenant_id parameter | No version change |
| `/metrics/portfolio` | GET | Add tenant_id parameter | No version change |
| `/health` | GET | Add tenant_id to logs only | No version change |
| `/status` | GET | Add tenant_id parameter | No version change |
| `/trades` | GET, POST | Add tenant_id parameter | No version change |
| `/trades/{trade_id}` | GET | Add tenant_id parameter | No version change |
| `/trades/history` | GET | Add tenant_id parameter | No version change |
| `/trades/pending` | GET | Add tenant_id parameter | No version change |
| `/correlation/matrix` | GET | Add tenant_id parameter | No version change |
| `/correlation/analysis` | GET | Add tenant_id parameter | No version change |

**API Contract Changes**: NONE (internal parameter, not exposed in responses)

#### Automation

**Schedulers**: No new schedulers required

**Background Jobs**: No changes to existing background engines

**Tick Systems**: N/A

#### Controls

**Feature Flags**: Not required (feature is additive and backward-compatible)

**Configuration Switches**: No new configuration needed

**Access Policies**:
- Existing RBAC permissions remain unchanged
- Tenant isolation enforced transparently via middleware

**Safe Rollout Strategy**:
1. Deploy Phase 2 behind CI/CD checks
2. Monitor logs for `[Tenant: {tenant_id}]` patterns
3. Smoke test each router after deployment
4. Rollback via standard deployment rollback (< 5 minutes)

---

## Decision Log

### Option 1: Reuse Phase 1 Pattern (SELECTED ✅)

**Description**: Apply the exact same `Depends(get_current_tenant_id)` pattern used in Phase 1 to all Phase 2 endpoints.

**Pros**:
- ✅ Proven pattern (50 tests passing in Phase 1)
- ✅ Zero architectural changes (minimal risk)
- ✅ Fast implementation (pattern reuse)
- ✅ Consistent codebase (same pattern everywhere)
- ✅ Leverages existing middleware and auth infrastructure

**Cons**:
- ❌ Requires modifying 16 endpoint signatures (manual work)
- ❌ Test coverage duplication (5 new test files)

**Trade-Off Matrix**:

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Complexity** | 9/10 | Lowest complexity option |
| **Cost** | 10/10 | Zero infrastructure cost |
| **Time-to-Market** | 9/10 | 2-3 days (fastest) |
| **Risk** | 10/10 | Lowest risk (proven pattern) |
| **Opportunity Impact** | 10/10 | Completes multi-tenant architecture |

**Total Score**: 48/50 ⭐ **HIGHEST SCORE**

### Option 2: Decorator-Based Approach

**Description**: Create a `@require_tenant_context` decorator to automatically inject `tenant_id` without modifying endpoint signatures.

**Pros**:
- ✅ Less boilerplate (no signature changes)
- ✅ Could be reused in future endpoints

**Cons**:
- ❌ New abstraction layer (increased complexity)
- ❌ Requires comprehensive testing (decorator edge cases)
- ❌ Inconsistent with Phase 1 (maintenance burden)
- ❌ Harder to understand for new developers

**Trade-Off Matrix**:

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Complexity** | 6/10 | New abstraction increases complexity |
| **Cost** | 10/10 | Zero infrastructure cost |
| **Time-to-Market** | 7/10 | 3-4 days (extra testing) |
| **Risk** | 7/10 | Untested pattern in codebase |
| **Opportunity Impact** | 10/10 | Completes multi-tenant architecture |

**Total Score**: 40/50

**Decision**: ❌ **REJECTED** - Inconsistent with Phase 1, increases complexity

### Option 3: Middleware-Only Approach

**Description**: Handle tenant isolation entirely in middleware without explicit endpoint parameters.

**Pros**:
- ✅ No endpoint signature changes (cleaner code)
- ✅ Centralized tenant logic

**Cons**:
- ❌ Hidden tenant context (harder to debug)
- ❌ No explicit documentation of tenant dependency
- ❌ Harder to mock in tests
- ❌ Inconsistent with Phase 1 (major refactor)

**Trade-Off Matrix**:

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Complexity** | 5/10 | Hidden dependencies increase cognitive load |
| **Cost** | 10/10 | Zero infrastructure cost |
| **Time-to-Market** | 6/10 | 4-5 days (middleware refactor) |
| **Risk** | 6/10 | Higher risk (changes middleware behavior) |
| **Opportunity Impact** | 10/10 | Completes multi-tenant architecture |

**Total Score**: 37/50

**Decision**: ❌ **REJECTED** - Too risky, inconsistent with Phase 1

### Related ADRs

- **ADR-001**: Multi-Tenant Architecture via JWT Middleware (Phase 1) → **REUSED**
- No new ADR required for Phase 2 (reusing existing decision)

### Open Questions

| Question | Owner | Due Date | Status |
|----------|-------|----------|--------|
| ~~Do we need to update API documentation?~~ | Tech Lead | 2025-11-03 | ✅ RESOLVED: No, tenant_id is internal |
| ~~Should we add tenant_id to audit logs?~~ | Security Lead | 2025-11-03 | ✅ RESOLVED: Yes, use `[Tenant: {tenant_id}]` format |

---

## Delivery Plan

### Phases

Per HLD-PROTO.yaml delivery plan structure:

#### Phase 1: Inception (Completed in Discovery Phase)
- ✅ Refined scope (16 endpoints across 5 routers)
- ✅ Spiked unknowns (validated Phase 1 patterns apply)
- ✅ Updated backlog (GitHub Issue #204 created)

#### Phase 2: Design & Alignment (Current Phase)
- ✅ Socialized HLD (this document)
- ⏳ Secured stakeholder sign-off (pending review)
- ⏳ Updated diagrams (included in this document)

#### Phase 3: Build (Next Phase)
**Focus**:
- Implement tenant context integration for 16 endpoints
- Maintain TDD RED-GREEN-REFACTOR cycle
- Iterate on feedback from code reviews

**Entry Criteria**:
- HLD approved by Tech Lead
- All open questions resolved
- Team capacity confirmed (1 engineer, 2-3 days)

**Exit Criteria**:
- All 16 endpoints integrated with tenant context
- All unit and integration tests passing locally
- Code reviewed by at least 2 team members

#### Phase 4: Stabilization
**Focus**:
- Harden implementation (CI/CD iterations)
- Perform load testing (p99 latency validation)
- Complete operational readiness (monitoring, alerts)

**Entry Criteria**:
- Build phase complete
- All tests passing in CI
- No critical security vulnerabilities

**Exit Criteria**:
- CI/CD green for 2 consecutive runs
- Load tests show no performance degradation
- Security scan passes (bandit, safety)

#### Phase 5: Rollout
**Focus**:
- Run progressive release plan (deploy to staging → production)
- Monitor telemetry (log analysis, error rates)
- Conduct post-launch review (retrospective)

**Entry Criteria**:
- Stabilization phase complete
- Deployment approved by Release Manager
- Rollback plan documented

**Exit Criteria**:
- Production deployment successful
- Zero critical incidents within 24 hours
- Monitoring shows expected tenant isolation behavior

### Work Breakdown

#### Epic: Story 2.4 Phase 2 - Tenant Context Integration (GitHub Issue #204)

**Story 1: alerts.py Tenant Integration** (2 story points)
- **Tasks**:
  1. Write integration tests for 3 alerts endpoints (RED)
  2. Add `tenant_id: str = Depends(get_current_tenant_id)` to all endpoints (GREEN)
  3. Add tenant logging `[Tenant: {tenant_id}]` (GREEN)
  4. Refactor for code quality (REFACTOR)
- **Acceptance Criteria**:
  - All 3 endpoints enforce tenant isolation
  - Integration tests pass (mock JWT with tenant_id)
  - Logs include tenant context

**Story 2: metrics.py Tenant Integration** (2 story points)
- **Tasks**:
  1. Write integration tests for 4 metrics endpoints (RED)
  2. Add `tenant_id: str = Depends(get_current_tenant_id)` to all endpoints (GREEN)
  3. Add tenant logging (GREEN)
  4. Refactor for code quality (REFACTOR)
- **Acceptance Criteria**:
  - All 4 endpoints enforce tenant isolation
  - Integration tests pass
  - Logs include tenant context

**Story 3: system.py Tenant Integration** (1 story point)
- **Tasks**:
  1. Write integration tests for 2 system endpoints (RED)
  2. Add `tenant_id` to logging (GET /health, GET /status) (GREEN)
  3. Validate tenant context extraction (GREEN)
  4. Refactor for code quality (REFACTOR)
- **Acceptance Criteria**:
  - Both endpoints log tenant context
  - Integration tests pass
  - System health checks remain functional

**Story 4: trades.py Tenant Integration** (2 story points)
- **Tasks**:
  1. Write integration tests for 5 trades endpoints (RED)
  2. Add `tenant_id: str = Depends(get_current_tenant_id)` to all endpoints (GREEN)
  3. Add tenant logging (GREEN)
  4. Refactor for code quality (REFACTOR)
- **Acceptance Criteria**:
  - All 5 endpoints enforce tenant isolation
  - Integration tests pass
  - Trade execution scoped to tenant

**Story 5: correlation.py Tenant Integration** (1 story point)
- **Tasks**:
  1. Write integration tests for 2 correlation endpoints (RED)
  2. Add `tenant_id: str = Depends(get_current_tenant_id)` to both endpoints (GREEN)
  3. Add tenant logging (GREEN)
  4. Refactor for code quality (REFACTOR)
- **Acceptance Criteria**:
  - Both endpoints enforce tenant isolation
  - Integration tests pass
  - Correlation analysis scoped to tenant portfolio

**Total Effort**: 8 story points (same as Phase 1)

### Sequencing

**Critical Path** (sequential dependencies):
1. alerts.py → Independent (can start immediately)
2. metrics.py → Independent (can start immediately)
3. system.py → Independent (can start immediately)
4. trades.py → Independent (can start immediately)
5. correlation.py → Independent (can start immediately)

**Parallelization Opportunity**:
- All 5 routers can be implemented in parallel (no dependencies)
- However, recommend sequential implementation for quality control
- Suggested order: alerts → metrics → system → trades → correlation

**Dependency Map**:
```
┌─────────────┐
│ Discovery   │ (Completed)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Design/HLD  │ (Current)
└──────┬──────┘
       │
       ├─────────┬─────────┬─────────┬─────────┐
       ▼         ▼         ▼         ▼         ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│alerts.py ││metrics.py││system.py ││trades.py ││correlation│
│   (2 SP) ││   (2 SP) ││   (1 SP) ││   (2 SP) ││    (1 SP) │
└──────┬───┘└──────┬───┘└──────┬───┘└──────┬───┘└──────┬───┘
       │           │           │           │           │
       └───────────┴───────────┴───────────┴───────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │ CI/CD Tests │
                    │  (All pass) │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Release   │
                    │   (v2.2.0)  │
                    └─────────────┘
```

### Resources

**Team Model**:
- **Roster**: 1 engineer (full-time, 2-3 days)
- **Skill Coverage**: Python, FastAPI, pytest, JWT authentication
- **Onboarding**: None required (Phase 1 knowledge transfer complete)
- **FTE Allocations**: 100% dedicated to Phase 2

**Rotations**:
- **On-Call**: No special on-call required (standard rotation)
- **Code Review**: Minimum 2 reviewers (1 senior engineer required)
- **SME Availability**: Tech Lead available for questions

**Calendar**:
- **Events**: None blocking (no holidays, no change freezes)
- **Cadences**: Daily standup (15 min), final demo before release
- **Constraints**: Target completion by 2025-11-06 (3 days from discovery)

**Budget**: No additional costs (uses existing infrastructure)

---

## Validation Strategy

### Testing

#### Unit Tests
**Coverage**: Key domain rules and calculations

- **Tenant Context Extraction**: Unit test `get_current_tenant_id()` with valid/invalid tokens
- **Data Accessor Filtering**: Unit test tenant-scoped queries
- **Error Handling**: Unit test 401 when tenant_id missing

**Estimated Tests**: ~20 unit tests (4 per router)

#### Integration Tests
**Coverage**: Exercise API flows and tenant isolation

Pattern (per Phase 1):
```python
@pytest.mark.asyncio
async def test_get_alerts_tenant_isolation(client, mock_tenant_token):
    """Test that GET /alerts enforces tenant isolation."""
    # Arrange
    tenant_id = "tenant-abc-123"
    token = generate_jwt_token(tenant_id=tenant_id)

    # Act
    response = await client.get(
        "/alerts",
        headers={"Authorization": f"Bearer {token}"}
    )

    # Assert
    assert response.status_code == 200
    # Validate response only contains tenant-abc-123 alerts
```

**Estimated Tests**: ~50 integration tests (10 per router)

#### E2E Tests
**Coverage**: Validate user journeys through API

- **User Login → Get Alerts → Acknowledge Alert** (tenant isolation verified)
- **User Login → Get Metrics → View Portfolio** (tenant isolation verified)
- **User Login → Execute Trade → Get Trade History** (tenant isolation verified)

**Estimated Tests**: ~10 E2E tests (2 per router)

#### Non-Functional Tests

**Performance Testing**:
- Measure p99 latency before/after Phase 2 integration
- Target: No degradation (maintain <200ms)
- Tool: pytest-benchmark or locust

**Resilience Testing**:
- Invalid JWT token → 401 response
- Missing tenant_id claim → 401 response
- Expired JWT token → 401 response

**Observability Testing**:
- Validate `[Tenant: {tenant_id}]` appears in logs
- Validate tenant_id NOT exposed in API responses
- Validate audit logs capture tenant context

**Security Testing**:
- Cross-tenant data leakage test (tenant A cannot access tenant B data)
- SQL injection test with tenant_id parameter
- JWT tampering test (modified tenant_id claim)

### Readiness Checklist

#### QA Sign-Off (per QA.yaml)
- [ ] All unit tests passing (100% pass rate)
- [ ] All integration tests passing (100% pass rate)
- [ ] Code coverage ≥ 90% for new code
- [ ] No critical or high security vulnerabilities (bandit scan)
- [ ] Manual smoke test completed for all 5 routers

#### Observability
**Metrics**:
- `api_requests_total{router="alerts", tenant_id="<tenant>"}` (Prometheus counter)
- `api_request_duration_seconds{router="alerts"}` (Prometheus histogram)
- `tenant_isolation_errors_total` (Prometheus counter for 401 errors)

**Logs**:
- Standard format: `[Tenant: {tenant_id}] <log message>`
- Log level: INFO for successful operations, ERROR for failures

**Alerts**:
- Alert on `tenant_isolation_errors_total > 10/min` (potential JWT issue)
- Alert on p99 latency > 250ms (performance degradation)

**Dashboards**:
- Tenant-specific request volume (Grafana dashboard)
- API error rates per tenant (Grafana dashboard)

**On-Call Expectations**:
- Standard on-call rotation (no special requirements)
- Escalation path: Tech Lead → Engineering Manager

#### Documentation
- [ ] HLD document (this document)
- [ ] Delivery plan (this document)
- [ ] API documentation unchanged (no external changes)
- [ ] Runbook updated (tenant context troubleshooting steps)
- [ ] Change management notes (deployment steps in RELEASE-PROTO.yaml)

### Risk Controls

#### Risks

| Risk ID | Failure Mode | Likelihood | Impact | Mitigation | Owner |
|---------|--------------|------------|--------|------------|-------|
| R2P-001 | Import chain issues (like Phase 1) | LOW | HIGH | Check imports locally before CI push | Engineer |
| R2P-002 | CI collection errors | LOW | MEDIUM | Use `--continue-on-collection-errors` flag | CI/CD Lead |
| R2P-003 | Pattern inconsistency across routers | LOW | MEDIUM | Code review checklist, pair programming | Tech Lead |
| R2P-004 | Test flakiness (mock failures) | MEDIUM | LOW | Strict mock validation, fixture review | QA Lead |
| R2P-005 | Performance degradation | LOW | MEDIUM | Benchmark tests pre/post integration | Engineer |
| R2P-006 | Cross-tenant data leakage | LOW | CRITICAL | Security testing, integration tests | Security Lead |

#### Fallback Plan

**Rollback Strategy**:
1. **Trigger**: Any critical bug in production OR p99 latency > 300ms
2. **Steps**:
   - Revert to previous release (v2.1.0) via GitHub release tag
   - Redeploy using standard CI/CD pipeline (~5 minutes)
   - Verify rollback with smoke tests
3. **Data Recovery**: Not required (no data model changes)
4. **Manual Overrides**: None needed
5. **Kill Switches**: Not required (no feature flags)

**Monitoring Plan**:

**Success Metrics**:
- Zero cross-tenant data leakage incidents (verify via logs)
- p99 latency < 200ms (verify via Prometheus)
- 100% test pass rate (verify via CI/CD)
- Zero 500 errors related to tenant context (verify via logs)

**Leading Indicators** (first 24 hours):
- `[Tenant: {tenant_id}]` log pattern appears for all Phase 2 endpoints
- No 401 errors from valid JWT tokens
- API request volume unchanged (no drop-off)

**Alert Thresholds**:
- `tenant_isolation_errors_total > 10/min` → Page on-call engineer
- `api_request_duration_seconds{p99} > 0.3` → Slack alert to team
- `http_requests_total{status="500"} > 5/min` → Page on-call engineer

**Dashboards**:
- Grafana: "Phase 2 Tenant Context Monitoring" (real-time)
- Includes: Request volume, error rates, latency, tenant distribution

**Steady State Monitoring** (post-launch):
- Weekly review of tenant isolation logs
- Monthly review of performance metrics
- Quarterly review of security audit logs

---

## Collaboration

### Communication Plan

**Cadence**:
- **Daily Sync**: 15-minute standup (async Slack update acceptable)
- **Async Updates**: Post progress in #story-2-4 Slack channel
- **Decision Reviews**: Tech Lead reviews HLD (this document)
- **Stakeholder Demos**: Demo tenant isolation in staging before production deploy

**Async Notes Format**:
```
**Phase 2 Update - Day X**
✅ Completed: <list>
🚧 In Progress: <list>
⚠️ Blockers: <list>
📅 Next: <list>
```

**Decision Reviews**:
- HLD approval required before Build phase
- Code review required before merge (2 approvals)
- Release approval required before production deploy

**Stakeholder Demos**:
- Staging demo: Show tenant isolation in action (cURL examples)
- Production smoke test: Verify tenant context in production logs

### Approvals

**Gatekeepers**:

| Gate | Gatekeeper | Required Evidence | Approval Deadline |
|------|-----------|------------------|-------------------|
| **Design Sign-Off** | Tech Lead | HLD document (this doc) | 2025-11-03 EOD |
| **Security Review** | Security Lead | Security test results | Before prod deploy |
| **QA Sign-Off** | QA Lead | All tests passing, coverage ≥90% | Before prod deploy |
| **Release Readiness** | Release Manager | CI/CD green, runbook updated | Before prod deploy |

**Go/No-Go Decision**:
- **Owner**: Tech Lead + Release Manager
- **Criteria**:
  - All quality gates passed (lint, security, tests, coverage)
  - HLD approved
  - Security review complete
  - Runbook updated
- **Decision Point**: End of Stabilization phase (before Rollout)

### Artifact Checklist

- [x] HLD document stored in `docs/design/story-2.4-phase2-hld.md`
- [ ] Diagram sources versioned (N/A - using ASCII diagrams in Markdown)
- [x] Backlog items linked to scope (GitHub Issue #204)
- [ ] Delivery plan reviewed for completeness (pending stakeholder review)

---

## Completion Criteria

Per HLD-PROTO.yaml completion criteria:

- [ ] **Stakeholders approve the HLD and phased plan**
  - Tech Lead: ⏳ Pending review
  - Security Lead: ⏳ Pending review
  - Engineering Team: ⏳ Pending review

- [x] **Risks have owners and mitigation tracked to closure**
  - All 6 risks documented with owners and mitigations (see Risk Controls)

- [x] **Validation strategy and monitoring hooks are in place before build starts**
  - Testing strategy defined (unit, integration, E2E, non-functional)
  - Monitoring metrics defined (Prometheus counters/histograms)
  - Alert thresholds defined (tenant_isolation_errors, latency, 500 errors)

- [ ] **Go-live requires sign-off from engineering, QA, and operations**
  - Engineering: ⏳ Pending build completion
  - QA: ⏳ Pending test results
  - Operations: ⏳ Pending deployment readiness

---

## Appendix

### Phase 1 Reference Pattern

**Example from risk_budget.py** (line 68):
```python
@router.get("/risk/budget/current", response_model=RiskBudgetResponse)
async def get_current_budget(
    budget_type: Optional[str] = Query(None, description="Specific budget type"),
    tenant_id: str = Depends(get_current_tenant_id),  # ← ADD THIS
    current_user: Dict = Depends(get_current_user),
    risk_budgeting_service = Depends(get_risk_budgeting_service)
) -> RiskBudgetResponse:
    """Get the current risk budget allocation."""
    logger.info(f"[Tenant: {tenant_id}] Retrieved current risk budget")  # ← ADD THIS
    # ... rest of endpoint logic
```

### Test Pattern Reference

**Example from test_risk_tenant_context.py**:
```python
@pytest.mark.asyncio
async def test_get_current_budget_tenant_isolation(client, mock_tenant_token):
    """Test that get_current_budget enforces tenant isolation."""
    # Arrange
    tenant_id = "test-tenant-123"

    # Act
    response = await client.get(
        "/risk/budget/current",
        headers={"Authorization": f"Bearer {mock_tenant_token(tenant_id)}"}
    )

    # Assert
    assert response.status_code == 200
    # Validate tenant context was used (check logs or response behavior)
```

### Glossary

- **Tenant**: A logical isolation boundary in multi-tenant SaaS (e.g., a customer organization)
- **Tenant Context**: The `tenant_id` extracted from JWT token and used to scope data queries
- **RLS**: Row-Level Security (database-level tenant isolation)
- **JWT**: JSON Web Token (authentication token containing user + tenant claims)
- **Depends()**: FastAPI dependency injection pattern
- **TDD**: Test-Driven Development (RED-GREEN-REFACTOR cycle)
- **P0/P1/P2**: Priority levels (P0=critical, P1=high, P2=medium)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-03
**Next Review**: After stakeholder approval
**Approved By**: ⏳ Pending Tech Lead review

**References**:
- [Story 2.4 Phase 2 Discovery](../discovery/story-2.4-phase2-discovery.md)
- [v2.1.0 Phase 1 Retrospective](../retrospectives/v2.1.0-phase1-retrospective.md)
- [GitHub Issue #204](https://github.com/blackms/AlphaPulse/issues/204)
- [HLD-PROTO.yaml](../../dev-prompts/HLD-PROTO.yaml)
- [DELIVERY-PLAN-PROTO.yaml](../../dev-prompts/DELIVERY-PLAN-PROTO.yaml)
