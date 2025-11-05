# Story 2.4 Phase 4: API Tenant Context Integration - Delivery Plan

**Date**: 2025-11-05
**Story**: EPIC-002 Story 2.4 Phase 4
**Phase**: Build and Deliver
**Status**: Ready for Execution
**Version**: 1.0

---

## Executive Summary

This delivery plan translates the approved HLD for Story 2.4 Phase 4 into an executable roadmap with prioritized backlog, TDD task breakdowns, dependency orchestration, and readiness gates. **Phase 4 is CRITICAL**: it completes Story 2.4 (100% API tenant context coverage) while fixing a **CRITICAL security gap in positions.py** (3 endpoints exposing trading data without user authentication).

**Key Milestones**:
- Build Start: 2025-11-06 (immediate)
- Phase 4 Complete: 2025-11-08 EOD (~3 days, 5-7 story points)
- Production Deploy: 2025-11-09
- Story 2.4 Complete: 43/43 endpoints (100% tenant context)

**Critical Priority**: positions.py authentication fix is a pre-deployment blocker. Any authentication gap prevents production release.

---

## Table of Contents

1. [Scope Refinement](#scope-refinement)
2. [Stories and TDD Breakdown](#stories-and-tdd-breakdown)
3. [Dependency Orchestration](#dependency-orchestration)
4. [Sequencing Plan](#sequencing-plan)
5. [Capacity and Resourcing](#capacity-and-resourcing)
6. [Quality and Readiness](#quality-and-readiness)
7. [Risk Management](#risk-management)
8. [Completion Criteria](#completion-criteria)
9. [Appendices](#appendices)

---

## Scope Refinement

### Backlog

#### Epic: Story 2.4 Phase 4 - Final Tenant Context Integration + CRITICAL Security Fix

**Epic ID**: GitHub Issue #205
**Business Outcome**: Complete multi-tenant architecture (100% API coverage) while closing CRITICAL authentication gaps in positions.py
**Definition of Done**:
- All 7 endpoints (positions.py + regime.py) enforce user authentication AND tenant isolation
- Zero authentication bypasses in positions.py
- Zero cross-tenant data leakage
- 100% test coverage (unit + integration)
- All CI/CD quality gates pass
- Production deployment successful with zero security incidents

#### Stories

---

##### Story 4.1: CRITICAL - positions.py Authentication Fix

**GitHub Issue**: #205 (subtask 1 - CRITICAL, pre-deployment blocker)
**Priority**: **P0 - CRITICAL**
**Story Points**: 2
**Owner**: Engineer (TBD)
**Timeline**: Day 1 (2.5 hours)

**User Story**: As an AlphaPulse Security Officer, I need all positions.py endpoints to enforce user authentication so that trading position data is protected from unauthorized API access.

**Acceptance Criteria**:
1. GET `/positions/spot` requires `get_current_user` dependency (401 if missing)
2. GET `/positions/futures` requires `get_current_user` dependency (401 if missing)
3. GET `/positions/metrics` requires `get_current_user` dependency (401 if missing)
4. All 3 endpoints include `tenant_id: str = Depends(get_current_tenant_id)` parameter
5. All 3 endpoints log `[Tenant: {tenant_id}] [User: {username}]` at start and completion
6. Integration tests validate authentication enforcement
7. Cross-tenant isolation validated (tenant A cannot access tenant B positions)
8. Test coverage ≥90% for modified code

**Regulatory Impact**: SOC 2 Type II, GDPR compliance requirement

**Test Notes**:
- Unauthenticated requests must return 401 Unauthorized
- Authenticated requests must return 200 OK
- Cross-tenant access attempts must be prevented
- Audit logs must contain full tenant + user context

**Traceability**: HLD Section "3.2.1 positions.py Router (CRITICAL SECURITY FIX)"

#### TDD Breakdown - Story 4.1

**RED Phase** (Write failing tests first):

1. **Test 1.1: Authentication Requirement - GET /spot**
   ```python
   # File: src/alpha_pulse/tests/api/test_positions_router_tenant.py
   # Lines: 1-30
   @pytest.mark.asyncio
   async def test_get_spot_requires_authentication(client):
       """Test that GET /positions/spot requires valid JWT token."""
       # Arrange - No Authorization header

       # Act
       response = await client.get("/positions/spot")

       # Assert
       assert response.status_code == 401
       assert "Unauthorized" in response.json()["detail"]
   ```
   Expected: FAIL (endpoint currently allows unauthenticated access)

2. **Test 1.2: Authentication Requirement - GET /futures**
   ```python
   # Lines: 31-60
   @pytest.mark.asyncio
   async def test_get_futures_requires_authentication(client):
       """Test that GET /positions/futures requires valid JWT token."""
       # Arrange - No Authorization header

       # Act
       response = await client.get("/positions/futures")

       # Assert
       assert response.status_code == 401
       assert "Unauthorized" in response.json()["detail"]
   ```
   Expected: FAIL

3. **Test 1.3: Authentication Requirement - GET /metrics**
   ```python
   # Lines: 61-90
   @pytest.mark.asyncio
   async def test_get_metrics_requires_authentication(client):
       """Test that GET /positions/metrics requires valid JWT token."""
       # Arrange - No Authorization header

       # Act
       response = await client.get("/positions/metrics")

       # Assert
       assert response.status_code == 401
       assert "Unauthorized" in response.json()["detail"]
   ```
   Expected: FAIL

4. **Test 1.4: Tenant Context Extraction**
   ```python
   # Lines: 91-130
   @pytest.mark.asyncio
   async def test_get_spot_uses_tenant_context(client, auth_tokens):
       """Test that GET /positions/spot enforces tenant isolation."""
       # Arrange
       tenant_id = "test-tenant-123"
       token = auth_tokens[tenant_id]
       headers = {"Authorization": f"Bearer {token}"}

       # Act
       response = await client.get("/positions/spot", headers=headers)

       # Assert
       assert response.status_code == 200
       positions = response.json()
       assert isinstance(positions, list)
       # All positions should be scoped to tenant
   ```
   Expected: FAIL (endpoint doesn't extract tenant_id yet)

5. **Test 1.5: Cross-Tenant Isolation**
   ```python
   # Lines: 131-180
   @pytest.mark.asyncio
   async def test_positions_cross_tenant_isolation(client, auth_tokens):
       """Test that tenant A cannot access tenant B's positions."""
       # Arrange
       token_a = auth_tokens["tenant-a"]
       token_b = auth_tokens["tenant-b"]

       # Act
       response_a = await client.get(
           "/positions/spot",
           headers={"Authorization": f"Bearer {token_a}"}
       )
       response_b = await client.get(
           "/positions/spot",
           headers={"Authorization": f"Bearer {token_b}"}
       )

       # Assert
       assert response_a.status_code == 200
       assert response_b.status_code == 200
       positions_a = response_a.json()
       positions_b = response_b.json()
       # Verify no overlap (tenants isolated)
       assert positions_a != positions_b
   ```
   Expected: FAIL (no tenant isolation yet)

6. **Test 1.6: Audit Logging**
   ```python
   # Lines: 181-230
   @pytest.mark.asyncio
   async def test_positions_logs_tenant_context(client, auth_tokens, caplog):
       """Test that endpoints log [Tenant: X] [User: Y] context."""
       # Arrange
       token = auth_tokens["test-tenant-123"]
       headers = {"Authorization": f"Bearer {token}"}

       # Act
       with caplog.at_level(logging.INFO):
           response = await client.get(
               "/positions/spot",
               headers=headers
           )

       # Assert
       assert response.status_code == 200
       log_output = caplog.text
       assert "[Tenant: test-tenant-123]" in log_output
       assert "[User:" in log_output  # Contains user ID
   ```
   Expected: FAIL (logging not implemented)

---

**GREEN Phase** (Implement to make tests pass):

1. **Update positions.py imports** (lines 1-12):
   ```python
   from typing import List, Dict
   from fastapi import APIRouter, Depends, HTTPException, status
   from pydantic import BaseModel
   from loguru import logger

   from alpha_pulse.exchanges.interfaces import BaseExchange
   from alpha_pulse.hedging.common.types import SpotPosition, FuturesPosition
   from alpha_pulse.hedging.execution.position_fetcher import ExchangePositionFetcher
   from ..dependencies import get_exchange_client, get_api_client, get_current_user
   from ..middleware.tenant_context import get_current_tenant_id

   router = APIRouter()
   ```

2. **Update GET /spot endpoint** (lines 26-44 → 26-52):
   ```python
   @router.get("/spot", response_model=List[SpotPosition])
   async def get_spot_positions(
       tenant_id: str = Depends(get_current_tenant_id),
       user: dict = Depends(get_current_user),
       exchange: BaseExchange = Depends(get_exchange_client),
       _api_client = Depends(get_api_client)
   ):
       """Get current spot positions."""
       logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Fetching spot positions")

       try:
           position_fetcher = ExchangePositionFetcher(exchange)
           positions = await position_fetcher.get_spot_positions()

           logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Retrieved {len(positions)} spot positions")
           return positions
       except Exception as e:
           logger.error(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Error fetching spot positions: {str(e)}")
           raise HTTPException(status_code=500, detail="Failed to fetch spot positions")
   ```

3. **Update GET /futures endpoint** (lines 46-64 → 54-72):
   ```python
   @router.get("/futures", response_model=List[FuturesPosition])
   async def get_futures_positions(
       tenant_id: str = Depends(get_current_tenant_id),
       user: dict = Depends(get_current_user),
       exchange: BaseExchange = Depends(get_exchange_client),
       _api_client = Depends(get_api_client)
   ):
       """Get current futures positions."""
       logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Fetching futures positions")

       try:
           position_fetcher = ExchangePositionFetcher(exchange)
           positions = await position_fetcher.get_futures_positions()

           logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Retrieved {len(positions)} futures positions")
           return positions
       except Exception as e:
           logger.error(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Error fetching futures positions: {str(e)}")
           raise HTTPException(status_code=500, detail="Failed to fetch futures positions")
   ```

4. **Update GET /metrics endpoint** (lines 66-103 → 74-111):
   ```python
   @router.get("/metrics")
   async def get_position_metrics(
       tenant_id: str = Depends(get_current_tenant_id),
       user: dict = Depends(get_current_user),
       exchange: BaseExchange = Depends(get_exchange_client),
       _api_client = Depends(get_api_client)
   ):
       """Get position metrics (value, exposure, hedge ratio)."""
       logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Fetching position metrics")

       try:
           position_fetcher = ExchangePositionFetcher(exchange)
           spot_positions = await position_fetcher.get_spot_positions()
           futures_positions = await position_fetcher.get_futures_positions()

           # Calculate metrics
           total_value = sum(p.value for p in spot_positions) + sum(p.value for p in futures_positions)
           hedge_ratio = len(futures_positions) / max(len(spot_positions), 1)

           metrics = {
               "total_value": total_value,
               "spot_count": len(spot_positions),
               "futures_count": len(futures_positions),
               "hedge_ratio": hedge_ratio
           }

           logger.info(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Retrieved position metrics")
           return metrics
       except Exception as e:
           logger.error(f"[Tenant: {tenant_id}] [User: {user.get('sub')}] Error fetching position metrics: {str(e)}")
           raise HTTPException(status_code=500, detail="Failed to fetch position metrics")
   ```

Expected Result: All 6 tests PASS

---

**REFACTOR Phase** (Improve code quality):

1. **Code Review Checklist**:
   - [ ] All 3 endpoints follow consistent pattern (tenant_id → user → other params)
   - [ ] All logging includes both `[Tenant: X]` and `[User: Y]`
   - [ ] All error handling consistent (HTTPException with 500)
   - [ ] No hardcoded values, all dependencies injected
   - [ ] Docstrings updated for authentication requirement

2. **Remove any duplication** (position_fetcher initialization):
   - Consider extracting to helper if code repeats in regime.py

3. **Validate imports** (no circular dependencies):
   - Verify `get_current_user` is properly exported from `dependencies.py`
   - Verify `get_current_tenant_id` is properly exported from `middleware.tenant_context`

---

##### Story 4.2: positions.py Tenant Context Integration

**GitHub Issue**: #205 (subtask 2 - CRITICAL, part of auth fix)
**Priority**: **P0 - CRITICAL**
**Story Points**: 1.5
**Owner**: Engineer (TBD)
**Timeline**: Day 1 (1.5 hours)

**User Story**: As an AlphaPulse SaaS operator, I need positions.py endpoints to enforce tenant isolation so tenants can only access their own trading positions.

**Acceptance Criteria**:
1. All 3 endpoints extract `tenant_id` via `Depends(get_current_tenant_id)`
2. All operations scoped to current tenant (service layer handles filtering)
3. Cross-tenant access returns no data leakage (tenant isolation validated)
4. All endpoints log tenant context with user context
5. Integration tests validate tenant isolation
6. Test coverage ≥90%

**Note**: This story's TDD is integrated with Story 4.1 (both modify same 3 endpoints). Tests 1.4, 1.5, 1.6 validate this story.

---

##### Story 4.3: regime.py Tenant Context Integration

**GitHub Issue**: #205 (subtask 3 - HIGH priority)
**Priority**: **P1 - HIGH**
**Story Points**: 1.5
**Owner**: Engineer (TBD)
**Timeline**: Day 2 (2.5 hours)

**User Story**: As an AlphaPulse SaaS operator, I want regime analysis endpoints to return tenant-specific regime data so that regime analysis is scoped to tenant's trading context.

**Acceptance Criteria**:
1. GET `/current` extracts and uses `tenant_id` via `Depends(get_current_tenant_id)`
2. GET `/history` extracts and uses `tenant_id`
3. GET `/analysis/{regime_type}` extracts and uses `tenant_id`
4. GET `/alerts` extracts and uses `tenant_id`
5. All 4 endpoints log `[Tenant: {tenant_id}] [User: {user}]` at start and completion
6. Service layer assumes tenant filtering is implemented
7. Integration tests validate tenant isolation
8. Test coverage ≥90%

**Traceability**: HLD Section "3.2.2 regime.py Router (Tenant Context Addition)"

#### TDD Breakdown - Story 4.3

**RED Phase**:

1. **Test 3.1: Tenant Context - GET /current**
   ```python
   # File: src/alpha_pulse/tests/api/test_regime_router_tenant.py
   # Lines: 1-40
   @pytest.mark.asyncio
   async def test_get_current_regime_uses_tenant_context(client, auth_tokens):
       """Test that GET /current enforces tenant isolation."""
       # Arrange
       tenant_id = "test-tenant-123"
       token = auth_tokens[tenant_id]
       headers = {"Authorization": f"Bearer {token}"}

       # Act
       response = await client.get(
           "/regime/current",
           headers=headers
       )

       # Assert
       assert response.status_code == 200
       regime_data = response.json()
       assert "current_regime" in regime_data
       # Service should filter by tenant internally
   ```
   Expected: FAIL (tenant_id not extracted yet)

2. **Test 3.2: Cross-Tenant Isolation - Regime History**
   ```python
   # Lines: 41-90
   @pytest.mark.asyncio
   async def test_regime_history_cross_tenant_isolation(client, auth_tokens):
       """Test that regime history is scoped to tenant."""
       # Arrange
       token_a = auth_tokens["tenant-a"]
       token_b = auth_tokens["tenant-b"]

       # Act
       response_a = await client.get(
           "/regime/history",
           headers={"Authorization": f"Bearer {token_a}"}
       )
       response_b = await client.get(
           "/regime/history",
           headers={"Authorization": f"Bearer {token_b}"}
       )

       # Assert
       assert response_a.status_code == 200
       assert response_b.status_code == 200
       history_a = response_a.json()
       history_b = response_b.json()
       # Verify data is different per tenant
       assert history_a != history_b
   ```
   Expected: FAIL

3. **Test 3.3: Tenant Logging**
   ```python
   # Lines: 91-140
   @pytest.mark.asyncio
   async def test_regime_endpoints_log_tenant_context(client, auth_tokens, caplog):
       """Test that regime endpoints log tenant context."""
       # Arrange
       token = auth_tokens["test-tenant-123"]
       headers = {"Authorization": f"Bearer {token}"}

       # Act
       with caplog.at_level(logging.INFO):
           for endpoint in ["/regime/current", "/regime/history", "/regime/alerts"]:
               response = await client.get(endpoint, headers=headers)
               assert response.status_code == 200

       # Assert
       log_output = caplog.text
       assert "[Tenant: test-tenant-123]" in log_output
       assert "[User:" in log_output
   ```
   Expected: FAIL

4. **Test 3.4: Regime Analysis**
   ```python
   # Lines: 141-190
   @pytest.mark.asyncio
   async def test_regime_analysis_uses_tenant_context(client, auth_tokens):
       """Test that regime analysis is tenant-scoped."""
       # Arrange
       token = auth_tokens["test-tenant-123"]
       headers = {"Authorization": f"Bearer {token}"}

       # Act
       response = await client.get(
           "/regime/analysis/trending",
           headers=headers
       )

       # Assert
       assert response.status_code == 200
       analysis = response.json()
       assert "regime_type" in analysis
       assert "strategies" in analysis
   ```
   Expected: FAIL

---

**GREEN Phase**:

1. **Update regime.py imports** (add):
   ```python
   from ..middleware.tenant_context import get_current_tenant_id
   ```

2. **Update GET /current endpoint** (lines 46-92):
   ```python
   @router.get("/current", response_model=RegimeStateResponse)
   async def get_current_regime(
       tenant_id: str = Depends(get_current_tenant_id),
       current_user: Dict = Depends(get_current_user),
       regime_service = Depends(get_regime_detection_service)
   ) -> RegimeStateResponse:
       """Get current market regime state."""
       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Getting current regime")

       regime_info = regime_service.current_regime_info

       if regime_info:
           logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Current regime: {regime_info.current_regime}")

       return regime_info
   ```

3. **Update GET /history endpoint** (lines 95-161):
   ```python
   @router.get("/history", response_model=List[RegimeHistoryResponse])
   async def get_regime_history(
       tenant_id: str = Depends(get_current_tenant_id),
       current_user: Dict = Depends(get_current_user),
       days: int = Query(30),
       regime_service = Depends(get_regime_detection_service)
   ) -> List[RegimeHistoryResponse]:
       """Get historical regime data."""
       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Fetching regime history ({days} days)")

       history = regime_service.get_regime_history(days=days)

       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Retrieved {len(history)} regime records")
       return history
   ```

4. **Update GET /analysis/{regime_type}** (lines 164-311):
   ```python
   @router.get("/analysis/{regime_type}", response_model=RegimeAnalysisResponse)
   async def get_regime_analysis(
       tenant_id: str = Depends(get_current_tenant_id),
       current_user: Dict = Depends(get_current_user),
       regime_type: str = Path(...),
       regime_service = Depends(get_regime_detection_service)
   ) -> RegimeAnalysisResponse:
       """Get regime-specific analysis and strategies."""
       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Analyzing regime: {regime_type}")

       analysis = regime_service.analyze_regime(regime_type=regime_type)

       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Analysis complete for {regime_type}")
       return analysis
   ```

5. **Update GET /alerts endpoint** (lines 314-336):
   ```python
   @router.get("/alerts", response_model=List[RegimeAlertResponse])
   async def get_regime_alerts(
       tenant_id: str = Depends(get_current_tenant_id),
       current_user: Dict = Depends(get_current_user),
       regime_service = Depends(get_regime_detection_service)
   ) -> List[RegimeAlertResponse]:
       """Get regime transition alerts."""
       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Fetching regime alerts")

       alerts = regime_service.get_alerts()

       logger.info(f"[Tenant: {tenant_id}] [User: {current_user.get('sub')}] Retrieved {len(alerts)} alerts")
       return alerts
   ```

Expected Result: All 4 tests PASS

---

**REFACTOR Phase**:

1. **Code Review Checklist**:
   - [ ] All 4 endpoints follow consistent pattern
   - [ ] Parameter order identical: tenant_id → user → other params
   - [ ] Logging format matches positions.py
   - [ ] Error handling consistent
   - [ ] Imports organized (middleware first, then services)

2. **Documentation**:
   - [ ] Docstrings updated to indicate tenant context

---

##### Story 4.4: Integration Tests - positions.py

**GitHub Issue**: #205 (subtask 4 - CRITICAL, security validation)
**Priority**: **P0 - CRITICAL**
**Story Points**: 1
**Owner**: Engineer (TBD)
**Timeline**: Day 2 (1.5 hours)

**User Story**: As a QA engineer, I need comprehensive integration tests for positions.py to validate authentication enforcement and tenant isolation.

**Acceptance Criteria**:
1. Test all 3 positions.py endpoints with valid JWT token (200 OK)
2. Test all 3 endpoints without JWT token (401 Unauthorized)
3. Test cross-tenant isolation (tenant A cannot access tenant B data)
4. Test invalid/expired JWT tokens (401)
5. Test audit logging format
6. Test error handling (500 on service failure)
7. Minimum 400-500 lines of test code
8. Test coverage ≥90%

**Test Files**:
- `src/alpha_pulse/tests/api/test_positions_router_tenant.py` (400-500 lines)

#### TDD Breakdown - Story 4.4

Tests 1.1-1.6 from Story 4.1 form the foundation. Additional tests:

7. **Test 4.1: Expired Token Handling**
   ```python
   @pytest.mark.asyncio
   async def test_positions_rejects_expired_token(client, expired_token):
       """Test that expired tokens are rejected."""
       headers = {"Authorization": f"Bearer {expired_token}"}
       response = await client.get("/positions/spot", headers=headers)
       assert response.status_code == 401
   ```

8. **Test 4.2: Invalid Signature**
   ```python
   @pytest.mark.asyncio
   async def test_positions_rejects_tampered_token(client):
       """Test that tampered tokens are rejected."""
       tampered_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
       headers = {"Authorization": f"Bearer {tampered_token}"}
       response = await client.get("/positions/spot", headers=headers)
       assert response.status_code == 401
   ```

9. **Test 4.3: All Endpoints Require Auth**
   ```python
   @pytest.mark.asyncio
   async def test_all_position_endpoints_require_auth(client):
       """Test that all position endpoints require authentication."""
       endpoints = ["/positions/spot", "/positions/futures", "/positions/metrics"]
       for endpoint in endpoints:
           response = await client.get(endpoint)
           assert response.status_code == 401, f"Endpoint {endpoint} allowed unauthenticated access!"
   ```

10. **Test 4.4: Service Error Handling**
    ```python
    @pytest.mark.asyncio
    async def test_positions_handles_service_errors(client, auth_tokens, mock_exchange_error):
        """Test that service errors return 500."""
        token = auth_tokens["test-tenant-123"]
        headers = {"Authorization": f"Bearer {token}"}

        with mock_exchange_error():
            response = await client.get("/positions/spot", headers=headers)
            assert response.status_code == 500
            assert "Failed to fetch" in response.json()["detail"]
    ```

---

##### Story 4.5: Integration Tests - regime.py

**GitHub Issue**: #205 (subtask 5 - HIGH priority validation)
**Priority**: **P1 - HIGH**
**Story Points**: 1
**Owner**: Engineer (TBD)
**Timeline**: Day 2 (1.5 hours)

**User Story**: As a QA engineer, I need comprehensive integration tests for regime.py to validate tenant isolation.

**Acceptance Criteria**:
1. Test all 4 regime endpoints with valid JWT token (200 OK)
2. Test cross-tenant isolation (tenant A cannot access tenant B regime data)
3. Test audit logging format
4. Test error handling
5. Minimum 600-700 lines of test code (more complex than positions)
6. Test coverage ≥90%

**Test Files**:
- `src/alpha_pulse/tests/api/test_regime_router_tenant.py` (600-700 lines)

#### TDD Breakdown - Story 4.5

Tests 3.1-3.4 from Story 4.3 form the foundation. Additional tests:

5. **Test 5.1: All Regime Endpoints Require Auth**
   ```python
   @pytest.mark.asyncio
   async def test_all_regime_endpoints_require_auth(client):
       """Test that all regime endpoints require authentication."""
       endpoints = [
           "/regime/current",
           "/regime/history",
           "/regime/analysis/trending",
           "/regime/alerts"
       ]
       for endpoint in endpoints:
           response = await client.get(endpoint)
           assert response.status_code == 401
   ```

6. **Test 5.2: Query Parameter Handling**
   ```python
   @pytest.mark.asyncio
   async def test_regime_history_with_query_params(client, auth_tokens):
       """Test regime history with days parameter."""
       token = auth_tokens["test-tenant-123"]
       headers = {"Authorization": f"Bearer {token}"}

       response = await client.get(
           "/regime/history?days=60",
           headers=headers
       )
       assert response.status_code == 200
       history = response.json()
       assert len(history) <= 60  # Or appropriate validation
   ```

7. **Test 5.3: Path Parameter Validation**
   ```python
   @pytest.mark.asyncio
   async def test_regime_analysis_invalid_type(client, auth_tokens):
       """Test regime analysis with invalid regime type."""
       token = auth_tokens["test-tenant-123"]
       headers = {"Authorization": f"Bearer {token}"}

       response = await client.get(
           "/regime/analysis/invalid_regime_type",
           headers=headers
       )
       # Should either return 400 or 404, or 200 with empty data
       assert response.status_code in [200, 400, 404]
   ```

8. **Test 5.4: Multi-Tenant Isolation Matrix**
   ```python
   @pytest.mark.asyncio
   async def test_regime_data_matrix_cross_tenant(client, auth_tokens):
       """Test cross-tenant isolation for all regime endpoints."""
       endpoints = [
           "/regime/current",
           "/regime/history",
           "/regime/analysis/trending",
           "/regime/alerts"
       ]

       token_a = auth_tokens["tenant-a"]
       token_b = auth_tokens["tenant-b"]

       for endpoint in endpoints:
           response_a = await client.get(endpoint, headers={"Authorization": f"Bearer {token_a}"})
           response_b = await client.get(endpoint, headers={"Authorization": f"Bearer {token_b}"})

           assert response_a.status_code == 200
           assert response_b.status_code == 200
           # Verify data is different
           assert response_a.json() != response_b.json()
   ```

---

### Estimation

**Sizing Approach**: Story Points (Fibonacci scale: 1, 2, 3, 5, 8)

**Calibration**:
- 1 SP = ~4 hours (simple changes, straightforward testing)
- 1.5 SP = ~6 hours (moderate complexity)
- 2 SP = ~8 hours (complex, high-risk changes)

**Total Effort**: 5-7 story points = 3-4 days (1 engineer, full-time)

**Effort Breakdown**:
- Story 4.1 (positions.py auth fix): 2 SP = 8 hours
- Story 4.2 (positions.py tenant context): 1.5 SP = 6 hours (integrated with 4.1)
- Story 4.3 (regime.py tenant context): 1.5 SP = 6 hours
- Story 4.4 (positions tests): 1 SP = 4 hours
- Story 4.5 (regime tests): 1 SP = 4 hours

**Total**: ~28 hours = 3.5 days (accounting for code review, CI/CD, integration)

### Value Prioritisation

**Method**: RICE Scoring (Reach × Impact × Confidence / Effort)

| Story | Reach | Impact | Confidence | Effort | RICE Score | Priority |
|-------|-------|--------|------------|--------|------------|----------|
| Story 4.1: positions.py Auth | 3 endpoints | **CRITICAL** | 95% | 2 SP | **142.5** | **CRITICAL** |
| Story 4.2: positions.py Tenant | 3 endpoints | CRITICAL | 95% | 1.5 SP | 95.0 | CRITICAL |
| Story 4.3: regime.py Tenant | 4 endpoints | HIGH | 90% | 1.5 SP | 80.0 | HIGH |
| Story 4.4: positions Tests | 3 endpoints | CRITICAL | 95% | 1 SP | 95.0 | CRITICAL |
| Story 4.5: regime Tests | 4 endpoints | HIGH | 90% | 1 SP | 72.0 | HIGH |

**RICE Baseline** (Phase 3: hedging.py auth fix): 64.0
**Phase 4 Average**: 96.9 (50% higher priority than Phase 3!)

**Guardrails**:
- **Non-Negotiable Items**:
  - Story 4.1 MUST be 100% complete (no authentication gaps permitted)
  - All 7 endpoints must be tested (no partial test coverage)
  - Security review required before production deploy
- **Blocking Dependencies**:
  - Phase 1-3 middleware must remain stable
  - JWT authentication system must be functional
  - get_current_user and get_current_tenant_id must work correctly

### Alignment

**Stakeholder Confirmation**:
- **Scope Boundaries**: 7 endpoints across 2 routers, CRITICAL security fix (no scope creep)
- **Success Metrics**: Zero authentication bypasses, zero cross-tenant leakage, 100% test pass
- **Definition of Done**: All quality gates pass, security review approved, production deployment successful

**Sign-Off**: ⏳ Pending Tech Lead + Security Lead review

---

## Dependency Orchestration

### Catalogue

#### Internal Dependencies

| Dependency | Owner | Contact | SLA | Change Lead-Time | Risk |
|------------|-------|---------|-----|------------------|------|
| `alpha_pulse.api.middleware.tenant_context` | Auth Team | @auth-team | 99.9% uptime | N/A (stable) | LOW |
| `alpha_pulse.api.dependencies` (get_current_user) | Auth Team | @auth-team | 99.9% uptime | N/A (stable) | LOW |
| `alpha_pulse.api.auth` (JWT middleware) | Auth Team | @auth-team | 99.9% uptime | N/A (stable) | LOW |
| `alpha_pulse.exchanges.interfaces.BaseExchange` | Exchange Team | @exchange-team | 99.9% uptime | N/A (stable) | LOW |
| `alpha_pulse.hedging.execution.position_fetcher` | Execution Team | @execution-team | 99.9% uptime | N/A (stable) | LOW |
| `alpha_pulse.services.regime_detection_service` | Services Team | @services-team | 99.9% uptime | N/A (stable) | LOW |
| PostgreSQL database | Data Team | @data-team | 99.99% uptime | N/A (no changes) | NONE |
| CI/CD Pipeline (.github/workflows/) | DevOps Team | @devops-team | 99% uptime | Immediate | LOW |

**Risk Assessment**: All internal dependencies are stable (Phase 1-3 proven). No blocking issues expected.

#### External Dependencies

| Dependency | Vendor/Domain | Contact | SLA | Change Lead-Time | Risk |
|------------|---------------|---------|-----|------------------|------|
| GitHub Actions CI/CD | GitHub | support@github.com | 99.9% uptime | N/A | LOW |
| pytest Testing Framework | OSS (stable) | N/A | N/A | N/A | NONE |
| FastAPI Framework | OSS (stable) | N/A | N/A | N/A | NONE |
| loguru Library | OSS (stable) | N/A | N/A | N/A | NONE |

**Risk Assessment**: All external dependencies are stable OSS or cloud services. No blocking issues expected.

### Risk Matrix

| Dependency | Delivery Risk | Likelihood | Impact | Mitigation | Contingency |
|------------|---------------|------------|--------|------------|-------------|
| JWT authentication system fails | Blocking | VERY LOW (2%) | CRITICAL | Freeze auth system during Phase 4 | Revert to Phase 3 version |
| get_current_user dependency broken | Blocking | VERY LOW (2%) | CRITICAL | Validate before build starts | Fix dependency first |
| Service layer tenant filtering fails | Blocking | LOW (5%) | CRITICAL | Integration tests validate isolation | Investigate service layer |
| CI/CD outage | Delay | LOW (10%) | MEDIUM | Use local pytest runs | Wait for recovery |

**Overall Risk**: **MEDIUM** - Authentication dependencies are stable but CRITICAL if they fail

### Handshake

**Dependency Agreements**:

1. **Auth Team** (JWT + get_current_user owner):
   - **Agreement**: Freeze changes to JWT middleware and get_current_user during Phase 4 build (2025-11-06 to 2025-11-08)
   - **Validation**: Pre-build validation that get_current_user works correctly (2025-11-06 AM)
   - **Checkpoints**: Daily Slack check-in (#story-2-4 channel)
   - **Contact**: @auth-lead (Slack)

2. **Security Team** (Security review):
   - **Agreement**: Perform security review before production deploy
   - **Validation**: Cross-tenant isolation tests must pass (100% success rate)
   - **Checkpoints**: Security review by 2025-11-08 EOD
   - **Contact**: @security-lead (Slack)

3. **DevOps Team** (CI/CD owner):
   - **Agreement**: Monitor CI/CD stability during Phase 4 build
   - **Timeline**: 2025-11-06 through 2025-11-08
   - **Checkpoints**: Alert on CI/CD issues via PagerDuty
   - **Contact**: @devops-oncall (PagerDuty)

---

## Sequencing Plan

### Roadmap

#### Day 1 (2025-11-06): CRITICAL Security Fix

**Focus**: Fix positions.py authentication gap (pre-deployment blocker)

**Tasks**:
- [ ] Story 4.1: positions.py Authentication Fix (2 SP, 8 hours)
  - [ ] RED: Write 6 failing tests
  - [ ] GREEN: Implement authentication for all 3 endpoints
  - [ ] REFACTOR: Code review and quality checks
  - **Milestone**: positions.py CRITICAL gap closed

- [ ] Story 4.2: positions.py Tenant Context (1.5 SP, 6 hours - integrated with 4.1)
  - Tests for tenant context validation included in Story 4.1 tests

**Estimated Time**: 14 hours (2 FTE-days)
**Target Exit Criteria**:
- All 3 positions.py endpoints require authentication (401 if missing)
- All 3 endpoints extract and log tenant context
- All integration tests passing locally
- Code review approved (2+ approvals)

---

#### Day 2 (2025-11-07): regime.py Tenant Context

**Focus**: Complete remaining endpoints (regime.py tenant context)

**Tasks**:
- [ ] Story 4.3: regime.py Tenant Context Integration (1.5 SP, 6 hours)
  - [ ] RED: Write 4 failing tests
  - [ ] GREEN: Add tenant context to all 4 endpoints
  - [ ] REFACTOR: Code review and quality checks
  - **Milestone**: All regime.py endpoints have tenant context

**Estimated Time**: 6 hours (0.75 FTE-days)
**Target Exit Criteria**:
- All 4 regime.py endpoints extract and log tenant context
- Integration tests passing locally
- Code review approved

---

#### Day 2-3 (2025-11-07 to 2025-11-08): Integration Tests

**Focus**: Comprehensive test coverage for both routers

**Tasks**:
- [ ] Story 4.4: Integration Tests - positions.py (1 SP, 4 hours)
  - Additional tests beyond Story 4.1 (authentication, error handling, edge cases)
  - **Milestone**: 400-500 lines of positions.py tests complete

- [ ] Story 4.5: Integration Tests - regime.py (1 SP, 4 hours)
  - Comprehensive cross-tenant isolation tests
  - **Milestone**: 600-700 lines of regime.py tests complete

**Estimated Time**: 8 hours (1 FTE-day)
**Target Exit Criteria**:
- All integration tests passing
- Code coverage ≥90%
- Security tests passing (zero cross-tenant leakage)

---

#### Day 3 (2025-11-08): Stabilization & Validation

**Focus**: Harden implementation, run full test suite, security review

**Tasks**:
- [ ] Run full test suite locally (1 hour)
- [ ] Run CI/CD pipeline (1 hour)
- [ ] Resolve any CI/CD issues (2 hours, if needed)
- [ ] Security review sign-off (1 hour)
- [ ] Update CHANGELOG.md (0.5 hours)
- [ ] Create Pull Request (0.5 hours)

**Estimated Time**: 5-6 hours (0.75 FTE-day)
**Target Exit Criteria**:
- CI/CD green for 2 consecutive runs
- Test coverage ≥90%
- Security review approved
- All quality gates passed
- PR ready for merge

---

### Critical Path

**Tasks Driving End Date** (sequential dependencies):

```
Day 1: positions.py Auth Fix (CRITICAL - 2 SP)
   ↓
Day 1: positions.py Tenant Context (CRITICAL - 1.5 SP, integrated)
   ↓
Day 2: regime.py Tenant Context (1.5 SP)
   ↓
Day 2-3: Integration Tests (2 SP)
   ↓
Day 3: Stabilization & CI/CD Green (0.75 SP)
   ↓
Production Deploy (2025-11-09)
```

**Critical Path Duration**: 3 days
**Buffer Protection**:
- **Risk Owner**: Tech Lead
- **Buffer**: 0.5 days built into estimate
- **Trigger**: If Day 1 incomplete by EOD, escalate to Tech Lead for re-planning

### Parallelisation

**Limited Parallelisation** (sequential preferred for quality):

```
Day 1:
┌─────────────────────┐
│ Story 4.1 + 4.2     │  ← Must complete Day 1 (CRITICAL, pre-deployment blocker)
│ positions.py Auth   │
│ Estimated: 14 hrs   │
└──────────┬──────────┘
           │
Day 2:     │
           └────────────────┬──────────────────────┐
                            │                      │
                   ┌────────▼────────┐    ┌───────▼────────┐
                   │ Story 4.3       │    │ Story 4.4      │
                   │ regime.py       │    │ positions      │
                   │ Tenant Context  │    │ Tests          │
                   │ (6 hrs)         │    │ (4 hrs)        │
                   └────────┬────────┘    └───────┬────────┘
                            │                      │
Day 3:                      └───────────┬──────────┘
                                        │
                            ┌───────────▼───────────┐
                            │ Story 4.5             │
                            │ regime Tests          │
                            │ (4 hrs)               │
                            └───────────┬───────────┘
                                        │
                            ┌───────────▼───────────┐
                            │ Stabilization         │
                            │ CI/CD Green           │
                            │ (5-6 hrs)             │
                            └───────────────────────┘
```

**Recommendation**: Implement sequentially for quality control. Stories 4.3 and 4.4 could run in parallel if resources allow, but recommend sequential for code review thoroughness.

---

## Capacity and Resourcing

### Team Model

**Roster**:
- **Engineer (1 FTE)**: Python backend engineer with FastAPI + JWT experience
  - **Skills Required**: Python 3.11+, FastAPI, pytest, JWT authentication, SQL
  - **Onboarding**: None required (Phase 1-3 context transfer complete)
  - **FTE Allocation**: 100% dedicated to Phase 4 (3 days)

**Code Review Pairing**:
- **Primary Reviewer**: Tech Lead (senior engineer)
- **Secondary Reviewer**: Any team member with Python experience
- **Requirement**: Minimum 2 approvals per PR (1 must be senior)

**SME Availability**:
- Auth Team: Available for JWT/get_current_user questions (Slack)
- Security Team: Available for security review (Slack)
- QA Team: Available for test fixture questions (Slack)

### Calendar

**Events**:
- **Holidays**: None during 2025-11-06 to 2025-11-08
- **Change Freezes**: None (normal deployment window)
- **Training**: None scheduled
- **Other Constraints**: None

**Cadences**:
- **Sprint Length**: N/A (Phase 4 is standalone 3-day effort)
- **Daily Standup**: 15-minute sync at 10:00 AM (async Slack update acceptable)
- **Code Review**: Continuous (as PRs submitted)
- **Security Review**: 2025-11-08 EOD (before deploy)

**Working Hours**:
- **Standard Hours**: 9:00 AM - 5:00 PM (8 hours/day)
- **Availability**: Engineer available for questions during working hours
- **Flexibility**: No overtime expected (3-day estimate includes buffer)

### Budget

**Tooling Costs**: $0 (using existing infrastructure)
**Infrastructure Costs**: $0 (no new infrastructure)
**Vendor Costs**: $0 (no external vendors)
**Approval Gates**: None required (zero budget change)

---

## Quality and Readiness

### Gates

#### QA Sign-Offs (per QA.yaml)

**Checklist** (must pass before production deploy):
- [ ] All unit tests passing (pytest)
- [ ] All integration tests passing (pytest) - 1,000+ lines of Phase 4 tests
- [ ] Code coverage ≥90% for Phase 4 code (codecov)
- [ ] No critical or high security vulnerabilities (bandit)
- [ ] No lint errors (ruff, black, mypy, flake8)
- [ ] Manual smoke test completed for all 7 endpoints
- [ ] Cross-tenant isolation validated (security test)
- [ ] Authentication enforcement validated (401 on missing token)

**Evidence Required**:
- CI/CD logs showing all tests passed
- Codecov report showing coverage ≥90%
- Bandit scan report (zero critical/high vulns)
- Smoke test checklist (signed off by QA Lead)

**Owner**: QA Lead
**Due Date**: Before production deploy (2025-11-08 EOD)

---

#### Security Sign-Offs (per SECURITY-PROTO.yaml)

**Checklist** (must pass before production deploy):
- [ ] **CRITICAL**: positions.py authentication enforcement validated (401 on missing token, all 3 endpoints)
- [ ] Cross-tenant data leakage test passed (tenant A cannot access tenant B data)
- [ ] JWT tampering test passed (modified tenant_id claim rejected)
- [ ] SQL injection test passed (tenant_id parameter sanitized)
- [ ] Audit log validation (all endpoints log `[Tenant: {tenant_id}] [User: {user}]`)
- [ ] No tenant_id exposed in API responses (privacy check)
- [ ] No authentication bypass identified in code review

**Evidence Required**:
- Security test results (automated + manual)
- Audit log samples showing tenant + user context
- API response samples (verify no tenant_id leak)
- Code review sign-off (security-focused)
- Positions.py authentication validation report

**Owner**: Security Lead
**Due Date**: Before production deploy (2025-11-08 EOD)

**CRITICAL REQUIREMENT**: Zero authentication gaps allowed. Any endpoint allowing unauthenticated access = blocker for production deploy.

---

#### Release Sign-Offs (per RELEASE-PROTO.yaml)

**Checklist** (must pass before production deploy):
- [ ] All quality gates passed (QA + Security)
- [ ] Staging deployment successful
- [ ] Staging smoke tests passed (all 7 endpoints)
- [ ] Security review approved
- [ ] Runbook updated with tenant context troubleshooting
- [ ] Monitoring dashboards configured
- [ ] Rollback plan documented and tested
- [ ] Release notes drafted (CHANGELOG.md)
- [ ] Tech Lead approval obtained
- [ ] Security Lead approval obtained
- [ ] Release Manager approval obtained

**Evidence Required**:
- Staging deployment logs
- Smoke test results (100% pass rate on all 7 endpoints)
- Security review approval
- Runbook link (docs/runbooks/)
- Grafana dashboard link
- Rollback test results (verify can revert to v2.3.0)
- Release notes draft (CHANGELOG.md)

**Owner**: Release Manager
**Due Date**: Before production deploy (2025-11-08 EOD)

---

### Validation Plan

#### Metrics

**Success Metrics** (monitor post-launch):

1. **Zero Authentication Gaps**
   - **Leading Indicator**: All 3 positions.py endpoints return 401 on missing token
   - **Lagging Indicator**: Zero unauthenticated access incidents in production

2. **Zero Cross-Tenant Data Leakage**
   - **Leading Indicator**: Integration tests pass (100% success rate)
   - **Lagging Indicator**: Zero tenant isolation incidents within 30 days

3. **p99 Latency <200ms (no degradation)**
   - **Leading Indicator**: Load tests show p99 <200ms
   - **Lagging Indicator**: Prometheus metrics show p99 <200ms in production

4. **100% Test Pass Rate**
   - **Leading Indicator**: All pytest tests pass (CI/CD green)
   - **Lagging Indicator**: No test regressions in production

**Monitoring Dashboards**:
- Grafana: "Phase 4 Tenant Context Monitoring" (real-time)
- Panels:
  - Authentication failures per endpoint
  - Cross-tenant access attempts (should be 0)
  - p99 latency per endpoint
  - Error rates per endpoint

**Alert Thresholds**:
- `api_auth_failures_total > 10/min` → Page on-call engineer
- `api_cross_tenant_access_attempts > 0` → **CRITICAL INCIDENT**
- `api_request_duration_seconds{p99} > 0.25` → Slack alert
- `http_requests_total{status="401"} > 100/min` → Investigate

---

#### Smoke Tests

**Smoke Test Checklist** (staging + production):

**positions.py** (CRITICAL):
- [ ] GET /positions/spot with token → 200 OK
- [ ] GET /positions/spot without token → 401 Unauthorized
- [ ] GET /positions/futures with token → 200 OK
- [ ] GET /positions/futures without token → 401 Unauthorized
- [ ] GET /positions/metrics with token → 200 OK
- [ ] GET /positions/metrics without token → 401 Unauthorized

**regime.py** (HIGH):
- [ ] GET /regime/current with token → 200 OK
- [ ] GET /regime/history with token → 200 OK
- [ ] GET /regime/analysis/trending with token → 200 OK
- [ ] GET /regime/alerts with token → 200 OK
- [ ] Verify audit logs contain `[Tenant: X] [User: Y]` for all endpoints

---

#### Acceptance

**Go/No-Go Decision**:
- **Owner**: Tech Lead + Security Lead + Release Manager
- **Criteria**:
  - ✅ All quality gates passed (QA + Security + Release)
  - ✅ Staging smoke tests 100% pass rate (all 7 endpoints)
  - ✅ **CRITICAL**: Zero authentication gaps verified
  - ✅ **CRITICAL**: Zero cross-tenant leakage verified
  - ✅ Security review approved
  - ✅ Monitoring dashboards operational
  - ✅ On-call engineer briefed and available
- **Decision Point**: 2025-11-08 4:00 PM (before production deploy)
- **Format**: Slack poll in #story-2-4 (unanimous approval required)

**Readiness Sign-Off Checklist**:
- [ ] QA Lead: ✅ APPROVED / ❌ BLOCKED
- [ ] Security Lead: ✅ APPROVED / ❌ BLOCKED (CRITICAL SIGN-OFF)
- [ ] Tech Lead: ✅ APPROVED / ❌ BLOCKED
- [ ] Release Manager: ✅ APPROVED / ❌ BLOCKED

**Approval Authority**:
- **GO Decision**: All 4 stakeholders approve (unanimous, esp. Security Lead)
- **NO-GO Decision**: Any 1 stakeholder blocks (with reason)
- **Escalation**: If blocked, escalate to Engineering Manager + Security Lead for resolution

---

## Risk Management

### Risk Register

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Owner |
|----|------|----------|------------|--------|-----------|-------|
| **R4-001** | Breaking change in positions.py | MEDIUM | MEDIUM | Client integration failures, production incidents | API gateway logs reviewed pre-release, client notification 1 week before, migration guide provided | Tech Lead |
| **R4-002** | Authentication gap remains in positions.py | **CRITICAL** | **LOW** | Unprotected trading data access | Security review checklist, code review for all 3 endpoints, automated tests for 401 enforcement | Security Lead |
| **R4-003** | Service layer doesn't filter by tenant | **CRITICAL** | LOW | Cross-tenant data leakage | Integration tests validate tenant isolation, database RLS as secondary defense, security review | Security Lead |
| **R4-004** | Performance regression | MEDIUM | LOW | SLA violations (>200ms p99) | Load tests before/after integration, p99 latency monitoring | Engineering Lead |
| **R4-005** | Test coverage gaps | MEDIUM | MEDIUM | Hidden security vulnerabilities | Minimum 90% coverage, security-focused test cases, cross-tenant isolation tests | QA Lead |
| **R4-006** | Dependency injection failures | LOW | LOW | Runtime 500 errors on missing dependency | Pre-deployment CI/CD tests, local validation, pytest collection test | Engineer |

### Detailed Risk Analysis

**Risk R4-002: Authentication Gap Remains in positions.py (CRITICAL)**

**Description**: One or more of the 3 positions.py endpoints fail to enforce authentication, leaving trading data accessible without JWT token.

**Current Exposure**: 3 endpoints (GET /spot, /futures, /metrics)

**Probability Assessment**:
- **LOW** (5%) - Clear security gap identified, straightforward fix
- But manual implementation can miss details

**Impact if Occurs**:
- **CRITICAL** - Regulatory violation (SOC 2 Type II, GDPR)
- Production incident, mandatory disclosure
- Immediate need for emergency hotfix

**Mitigation Strategy**:
1. **Code Review Checklist** (per endpoint):
   - [ ] `user: dict = Depends(get_current_user)` parameter present
   - [ ] Parameter is NOT optional (no default value)
   - [ ] No `dependencies=[...]` workarounds in decorator
   - [ ] All 3 endpoints have same pattern

2. **Automated Testing**:
   - Test 1.1, 1.2, 1.3: Verify 401 on missing token (all 3 endpoints)
   - Test 4.3: Matrix test all endpoints require auth

3. **Security Review**:
   - Security Lead manually verifies all 3 endpoints
   - Security team code review sign-off required before deploy

4. **Pre-Deployment Validation**:
   - Smoke test without token returns 401 (all 3 endpoints)
   - Manual curl test: `curl https://api/positions/spot` → 401

**Owner**: Security Lead (CRITICAL sign-off required)

**Status**: OPEN (mitigated by checklist + testing + security review)

---

**Risk R4-003: Service Layer Doesn't Filter by tenant_id (CRITICAL)**

**Description**: ExchangePositionFetcher or RegimeDetectionService may not properly filter data by tenant_id, leading to cross-tenant data leakage despite API-layer tenant context.

**Current Exposure**: Phase 1-3 endpoints already rely on service layer filtering. If issue exists, it affects all 43 endpoints.

**Probability Assessment**:
- **LOW** (5%) - Phase 1-3 endpoints show no cross-tenant leakage in production
- Service layer has been validated
- But new endpoint integrations may expose issues

**Impact if Occurs**:
- **CRITICAL** - Direct violation of multi-tenant security model
- Regulatory compliance breach
- Data breach affecting all tenants
- Immediate incident response required

**Mitigation Strategy**:
1. **Pre-Development Code Review**:
   - Review ExchangePositionFetcher.get_spot_positions() for tenant filtering
   - Review RegimeDetectionService for tenant filtering

2. **Integration Tests** (per endpoint):
   - Test 1.5: Cross-tenant isolation (positions.py)
   - Test 5.4: Multi-tenant isolation matrix (regime.py)
   - Each test uses different JWT tokens with different tenant_ids
   - Verify returned data is scoped to correct tenant

3. **Secondary Defense**:
   - Database RLS policies enforce tenant boundaries
   - Even if service layer fails, DB-level protection catches leakage

4. **Security Review**:
   - Security team performs security review of test results
   - Validates tenant isolation in all cross-tenant tests

**Owner**: Security Lead

**Status**: OPEN (mitigated by integration tests + RLS + security review)

---

### Rollback Plan

**Trigger Criteria** (any of the following):

- **CRITICAL Authentication Gap**: Any positions.py endpoint allows unauthenticated access
- **Cross-Tenant Data Leakage**: Tenant A can access tenant B data via API
- **p99 Latency**: Consistently >300ms (>50% degradation)
- **Error Rate**: >1% sustained for 5+ minutes
- **Data Corruption**: Any data integrity issues reported

**Rollback Steps**:

```bash
# 1. Announce incident
#    Slack: "#incidents" channel, PagerDuty page

# 2. Stop rollout (if in progress)
#    Cancel deployment to remaining regions

# 3. Trigger rollback (automatic via GitHub Actions)
git revert <v2.4.0-commit-hash>
git push origin main

# 4. Monitor rollback completion
#    Watch logs for deployment completion
#    Verify endpoint health checks pass
#    Confirm error rates return to baseline

# 5. Post-incident review
#    Schedule within 24 hours
#    Document root cause
#    Update mitigation strategy
```

**Estimated Rollback Time**: 5 minutes (via CI/CD automation)

**Notification**:
- Slack #incidents: Incident declared
- PagerDuty: On-call team notified
- Email: Executive summary sent to stakeholders
- Follow-up: Post-mortem document within 24 hours

---

## Timeline Summary

### High-Level Schedule

```
2025-11-05 (TODAY): Planning complete, Delivery Plan approved
2025-11-06 (Day 1): Build positions.py CRITICAL security fix (14 hours)
2025-11-07 (Day 2): Build regime.py + positions tests (10 hours)
2025-11-08 (Day 3): Integration tests + stabilization (5-6 hours)
2025-11-08 (EOD):   Staging validation + security review
2025-11-09:        Production deploy (Story 2.4 complete)
2025-11-10:        Post-launch monitoring (24 hours)
```

### Detailed Daily Schedule

**Day 1 (2025-11-06)**
- 09:00-09:15: Standup (review plan, address blockers)
- 09:15-12:30: Implement Story 4.1 + 4.2 (positions.py auth + tenant context)
  - Write 6 RED tests (1 hour)
  - Implement GREEN (1.5 hours)
  - REFACTOR + code review (1 hour)
- 12:30-13:30: Lunch break
- 13:30-17:00: Testing + CI/CD validation
  - Run full test suite (0.5 hours)
  - Fix any failures (1.5 hours)
  - Final code review approval (1 hour)
- 17:00: Standup update (progress report)

**Day 2 (2025-11-07)**
- 09:00-09:15: Standup
- 09:15-12:30: Implement Story 4.3 (regime.py tenant context)
  - Write 4 RED tests (1 hour)
  - Implement GREEN (1.5 hours)
  - REFACTOR (0.5 hour)
- 12:30-13:30: Lunch break
- 13:30-17:00: Implement Story 4.4 (positions.py integration tests)
  - Additional tests beyond Story 4.1 (2 hours)
  - Code review (0.5 hour)
- 17:00: Standup + daily PR review

**Day 3 (2025-11-08)**
- 09:00-09:15: Standup
- 09:15-12:30: Implement Story 4.5 (regime.py integration tests)
  - Comprehensive cross-tenant isolation tests (2.5 hours)
  - Code review (0.5 hour)
- 12:30-13:30: Lunch break
- 13:30-16:00: Stabilization
  - Run full test suite (1 hour)
  - Run CI/CD pipeline (1 hour)
  - Fix any failures (0.5 hour)
- 16:00-16:30: CHANGELOG.md update
- 16:30-17:00: Final security review coordination
- 17:00: Standup + Day 3 completion report
- EOD: Security Lead reviews + signs off (async)

---

## Appendices

### Appendix A: Test File Structure

```
src/alpha_pulse/tests/api/
├── test_positions_router_tenant.py (400-500 lines)
│   ├── Test 1.1: get_spot_requires_authentication
│   ├── Test 1.2: get_futures_requires_authentication
│   ├── Test 1.3: get_metrics_requires_authentication
│   ├── Test 1.4: get_spot_uses_tenant_context
│   ├── Test 1.5: positions_cross_tenant_isolation
│   ├── Test 1.6: positions_logs_tenant_context
│   ├── Test 4.1: positions_rejects_expired_token
│   ├── Test 4.2: positions_rejects_tampered_token
│   ├── Test 4.3: all_position_endpoints_require_auth
│   └── Test 4.4: positions_handles_service_errors
│
└── test_regime_router_tenant.py (600-700 lines)
    ├── Test 3.1: get_current_regime_uses_tenant_context
    ├── Test 3.2: regime_history_cross_tenant_isolation
    ├── Test 3.3: regime_endpoints_log_tenant_context
    ├── Test 3.4: regime_analysis_uses_tenant_context
    ├── Test 5.1: all_regime_endpoints_require_auth
    ├── Test 5.2: regime_history_with_query_params
    ├── Test 5.3: regime_analysis_invalid_type
    └── Test 5.4: regime_data_matrix_cross_tenant
```

**Total Test Lines**: 1,000-1,200 lines across 2 files

---

### Appendix B: Commit Strategy

1. **Commit 1** (Day 1 PM):
   ```
   fix(api): CRITICAL - Add authentication to positions.py endpoints

   - Add get_current_user dependency to GET /spot, /futures, /metrics
   - Add tenant context (get_current_tenant_id) to all endpoints
   - Add comprehensive audit logging [Tenant: X] [User: Y]
   - Closes pre-deployment security gap (CRITICAL)

   Fixes: #205
   ```

2. **Commit 2** (Day 2 AM):
   ```
   feat(api): Add tenant context to regime.py endpoints

   - Add get_current_tenant_id to GET /current, /history, /analysis, /alerts
   - Add tenant logging to all endpoints
   - Service layer assumes tenant filtering implemented

   Completes: Story 2.4 (43/43 endpoints = 100% coverage)
   ```

3. **Commit 3** (Day 2 PM):
   ```
   test(api): Add integration tests for positions.py

   - 400-500 lines of comprehensive tests
   - Validates authentication enforcement
   - Validates tenant isolation
   - Tests error handling and edge cases

   Related: #205
   ```

4. **Commit 4** (Day 3 AM):
   ```
   test(api): Add integration tests for regime.py

   - 600-700 lines of comprehensive tests
   - Multi-tenant isolation matrix tests
   - Query parameter and path parameter handling
   - Cross-tenant access prevention validation

   Related: #205
   ```

5. **Commit 5** (Day 3 PM):
   ```
   docs: Update CHANGELOG for v2.4.0 Phase 4

   - CRITICAL: Authentication enforcement in positions.py
   - FEATURE: Tenant context integration (43/43 endpoints = 100%)
   - BREAKING: positions.py endpoints now require JWT token
   - Migration guide: https://docs.alphapulse.com/migration-v2.4

   Completes: Story 2.4 Phase 4
   ```

---

### Appendix C: PR Template

```markdown
## Story 2.4 Phase 4: API Tenant Context Integration (CRITICAL Security Fix)

### Summary

This PR completes Story 2.4 by:
1. **CRITICAL FIX**: Adding authentication to positions.py (closes security gap)
2. Adding tenant context to all Phase 4 endpoints (regime.py + positions.py)
3. Implementing 1,000+ lines of integration tests
4. Achieving 100% API tenant context coverage (43/43 endpoints)

### Type of Change

- [x] CRITICAL Security Fix (positions.py authentication)
- [x] Feature (tenant context integration)
- [x] Tests (integration tests)
- [x] Documentation (CHANGELOG)

### Changes Made

**positions.py** (CRITICAL):
- [ ] All 3 endpoints require `get_current_user` dependency
- [ ] All 3 endpoints extract `tenant_id` via `get_current_tenant_id`
- [ ] All 3 endpoints log `[Tenant: X] [User: Y]` context
- [ ] Unauthenticated requests return 401 Unauthorized

**regime.py**:
- [ ] All 4 endpoints extract `tenant_id`
- [ ] All 4 endpoints log tenant context
- [ ] Tenant filtering implemented in service layer

**Tests**:
- [ ] 400-500 lines for positions.py tests
- [ ] 600-700 lines for regime.py tests
- [ ] All cross-tenant isolation tests passing
- [ ] Authentication enforcement tests passing

### Verification

**Security Checklist**:
- [ ] Zero authentication gaps (all 3 positions endpoints return 401 on missing token)
- [ ] Zero cross-tenant data leakage (tenant isolation validated)
- [ ] All endpoints log tenant + user context
- [ ] No sensitive data in logs

**Test Coverage**:
- [ ] Code coverage ≥90% for Phase 4 code
- [ ] All pytest tests passing
- [ ] CI/CD pipeline green

**Code Quality**:
- [ ] No lint errors (ruff, black, mypy, flake8)
- [ ] No security vulnerabilities (bandit)
- [ ] 100% pattern consistency with Phases 1-3

### Breaking Changes

**BREAKING**: positions.py endpoints now require authentication

**Migration Guide**:
- Obtain JWT token via `/auth/token` endpoint
- Include token in `Authorization: Bearer <token>` header
- See docs/migration-v2.4.md for details

### Related Issues

Fixes #205 (Story 2.4 Phase 4)
Closes #200 (positions.py security gap)

### Additional Notes

- Story 2.4 completion (100% API tenant context)
- Pre-deployment blocker resolved
- Security review required before production deploy
```

---

### Appendix D: CHANGELOG Entry

```markdown
## v2.4.0 (2025-11-09)

### CRITICAL SECURITY FIX

- **positions.py**: Added user authentication to all 3 endpoints
  - GET /positions/spot now requires valid JWT token (401 if missing)
  - GET /positions/futures now requires valid JWT token (401 if missing)
  - GET /positions/metrics now requires valid JWT token (401 if missing)
  - Closes CRITICAL security gap (unauthorized trading data access)
  - BREAKING CHANGE: Existing clients without authentication will now receive 401

### Features

- **Tenant Context Integration**: All 7 Phase 4 endpoints now enforce tenant isolation
  - regime.py: GET /current, /history, /analysis/{type}, /alerts
  - positions.py: GET /spot, /futures, /metrics
  - 100% API coverage: 43/43 endpoints with tenant context (Story 2.4 Complete)

- **Audit Logging**: All endpoints log tenant and user context
  - Format: `[Tenant: {tenant_id}] [User: {username}] <operation>`
  - Complete audit trail for compliance

### Breaking Changes

**BREAKING**: positions.py endpoints now require authentication
- **Affected Endpoints**:
  - GET /positions/spot
  - GET /positions/futures
  - GET /positions/metrics

- **Migration Path**:
  1. Obtain JWT token: `POST /auth/token`
  2. Include in requests: `Authorization: Bearer <token>`
  3. See migration guide: https://docs.alphapulse.com/migration-v2.4

### Testing

- 1,000+ lines of new integration tests
- Cross-tenant isolation validated for all 7 endpoints
- Authentication enforcement validated for all endpoints
- Code coverage: ≥90% for Phase 4 code

### Related Issues

- Closes #205 (Story 2.4 Phase 4)
- Closes #200 (positions.py security gap)

### Deployment Notes

- No database schema changes
- No data migration required
- Monitoring dashboards: Grafana "Phase 4 Tenant Context Monitoring"
- Rollback: Available within 5 minutes via git revert

### Migration Guide

For positions.py endpoint users, see detailed migration guide:
https://docs.alphapulse.com/migration-v2.4

Contact support@alphapulse.com with questions.
```

---

### Appendix E: Definition of Done

Phase 4 is considered DONE when:

1. **All Stories Implemented**
   - [ ] Story 4.1: positions.py authentication (CRITICAL)
   - [ ] Story 4.2: positions.py tenant context
   - [ ] Story 4.3: regime.py tenant context
   - [ ] Story 4.4: positions.py integration tests
   - [ ] Story 4.5: regime.py integration tests

2. **Code Quality Gates Passed**
   - [ ] All unit tests passing (100% success rate)
   - [ ] All integration tests passing (1,000+ lines tested)
   - [ ] Code coverage ≥90% for Phase 4 code
   - [ ] No lint errors (ruff, black, mypy, flake8)
   - [ ] No security vulnerabilities (bandit scan)
   - [ ] Code reviewed (2+ approvals, 1 senior)

3. **Security Validation Passed**
   - [ ] Zero authentication gaps (all 3 positions endpoints 401-protected)
   - [ ] Zero cross-tenant data leakage (tenant isolation tests pass)
   - [ ] All endpoints log tenant + user context correctly
   - [ ] No sensitive data in logs or responses
   - [ ] Security review approved

4. **CI/CD Green**
   - [ ] GitHub Actions pipeline passing for 2 consecutive runs
   - [ ] Staging deployment successful
   - [ ] Staging smoke tests 100% pass rate (all 7 endpoints)
   - [ ] Load tests show p99 latency <200ms

5. **Documentation Complete**
   - [ ] CHANGELOG.md updated (v2.4.0 entry)
   - [ ] Migration guide published (positions.py clients)
   - [ ] Runbook updated (tenant context troubleshooting)
   - [ ] Monitoring dashboards configured
   - [ ] All comments/docstrings updated

6. **Deployment Readiness**
   - [ ] Rollback plan documented and tested
   - [ ] On-call engineer briefed
   - [ ] Support team trained
   - [ ] Stakeholder sign-off obtained (QA, Security, Release Manager)

7. **Story 2.4 Completion**
   - [ ] 43/43 endpoints with tenant context integration (100% coverage)
   - [ ] All phases 1-4 complete and integrated
   - [ ] Production deployment successful
   - [ ] Post-launch monitoring 24-hour window complete

---

### Appendix F: Success Criteria

**Quantitative Metrics**:

| Metric | Target | Evidence |
|--------|--------|----------|
| Endpoints with tenant context | 43/43 (100%) | Deployment log showing all routers loaded |
| Authentication gaps in positions.py | 0/3 | Security test results (401 on all 3) |
| Cross-tenant leakage incidents | 0 | Integration test results + 24-hour production monitoring |
| Code coverage | ≥90% | Codecov report for Phase 4 code |
| Test pass rate | 100% | CI/CD pipeline green report |
| p99 latency | <200ms | Load test results + Prometheus metrics |
| Authentication gap violations | 0 | Security review sign-off |

**Qualitative Metrics**:

| Metric | Target | Evidence |
|--------|--------|----------|
| Security posture | Improved | Positions trading data now protected by authentication |
| Tenant isolation | Complete | All 43 endpoints enforce tenant boundaries |
| Audit compliance | Achieved | All endpoints log tenant + user context |
| Code consistency | 100% match Phases 1-3 | Code review checklist verification |
| Team confidence | High | Retrospective + team feedback |

---

## Document Information

**Version**: 1.0
**Last Updated**: 2025-11-05
**Next Review**: After implementation begins
**Status**: Ready for Execution

**Approved By**:
- [ ] Tech Lead (pending)
- [ ] Security Lead (pending - CRITICAL sign-off)
- [ ] Engineering Manager (pending)

**References**:
- [Phase 4 Discovery Document](../discovery/story-2.4-phase4-discovery.md)
- [Phase 4 HLD Document](../design/story-2.4-phase4-hld.md)
- [Phase 2 Delivery Plan (template)](./story-2.4-phase2-delivery-plan.md)
- [Phase 3 HLD](../design/story-2.4-phase3-hld.md)
- [GitHub Issue #205](https://github.com/blackms/AlphaPulse/issues/205)

---

**Status**: Ready for Build Phase
**Target Build Start**: 2025-11-06
**Target Build Complete**: 2025-11-08
**Target Production Deploy**: 2025-11-09
**Target Story 2.4 Complete**: 2025-11-09 (43/43 endpoints = 100%)
