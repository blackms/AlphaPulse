"""
Tests for risk_budget router tenant context integration.

Story 2.4 - Phase 1: Risk Budget Router
Tests that risk budget endpoints properly extract and use tenant_id from middleware.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt
from datetime import datetime, timedelta

from alpha_pulse.api.main import app


@pytest.fixture
def tenant_1_id():
    """Tenant 1 UUID for testing."""
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def tenant_2_id():
    """Tenant 2 UUID for testing."""
    return "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def jwt_secret():
    """JWT secret for testing."""
    return "test-secret-key-for-testing"


@pytest.fixture
def create_test_token(jwt_secret):
    """Factory to create JWT tokens with tenant_id."""
    def _create_token(username: str, tenant_id: str) -> str:
        payload = {
            "sub": username,
            "tenant_id": tenant_id,
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }
        return jwt.encode(payload, jwt_secret, algorithm="HS256")
    return _create_token


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings(jwt_secret):
    """Mock application settings for testing."""
    settings = Mock()
    settings.JWT_SECRET = jwt_secret
    settings.JWT_ALGORITHM = "HS256"
    settings.RLS_ENABLED = True
    return settings


@pytest.fixture
def mock_risk_budgeting_service():
    """Mock risk budgeting service."""
    service = AsyncMock()

    # Mock portfolio budget
    mock_budget = Mock()
    mock_budget.total_budget = 1000000.0
    mock_budget.allocations = [
        Mock(
            entity_id="strategy_1",
            entity_type="strategy",
            allocated_amount=500000.0,
            utilized_amount=300000.0,
            constraints={}
        )
    ]
    mock_budget.budget_type = "VOLATILITY"
    mock_budget.allocation_method = "EQUAL_WEIGHT"
    mock_budget.volatility_target = 0.15
    mock_budget.leverage_limit = 2.0
    mock_budget.last_updated = datetime.utcnow()

    service.get_portfolio_budget = AsyncMock(return_value=mock_budget)
    service.calculate_utilization = AsyncMock(return_value={
        "total_utilization": 0.6,
        "by_strategy": {"strategy_1": 0.6},
        "by_asset_class": {"crypto": 0.6},
        "available_budget": 400000.0,
        "warnings": []
    })
    service.rebalance_budgets = AsyncMock(return_value={
        "rebalanced": True,
        "changes": []
    })
    service.get_recommendations = AsyncMock(return_value={
        "requires_rebalancing": False,
        "reason": None,
        "target_allocations": {},
        "current_allocations": {},
        "expected_improvement": {},
        "priority": "low"
    })
    service.get_budget_history = AsyncMock(return_value={
        "snapshots": [],
        "performance_metrics": {},
        "regime_changes": []
    })
    service.get_regime_budget = AsyncMock(return_value=mock_budget)

    return service


class TestCurrentBudgetEndpoint:
    """Test /api/v1/risk-budget/current endpoint."""

    @patch('alpha_pulse.api.routers.risk_budget.get_risk_budgeting_service')
    def test_current_budget_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings, mock_risk_budgeting_service
    ):
        """Test that /current endpoint uses tenant_id."""
        mock_get_service.return_value = mock_risk_budgeting_service
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/risk-budget/current",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should succeed with tenant context
        assert response.status_code == 200
        data = response.json()
        assert "total_budget" in data


    def test_current_budget_rejects_missing_tenant(
        self, client, jwt_secret, mock_settings
    ):
        """Test that /current rejects requests without tenant_id."""
        # Create token WITHOUT tenant_id
        payload = {
            "sub": "admin",
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/risk-budget/current",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should return 401 Unauthorized
        assert response.status_code == 401
        assert "tenant_id" in response.json()["detail"].lower()


class TestUtilizationEndpoint:
    """Test /api/v1/risk-budget/utilization endpoint."""

    @patch('alpha_pulse.api.routers.risk_budget.get_risk_budgeting_service')
    def test_utilization_endpoint_isolates_by_tenant(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings, mock_risk_budgeting_service
    ):
        """Test that /utilization endpoint returns tenant-specific data."""
        mock_get_service.return_value = mock_risk_budgeting_service

        # Request for tenant 1
        token_1 = create_test_token("user1", tenant_1_id)
        with patch.object(app.state, 'settings', mock_settings):
            response_1 = client.get(
                "/api/v1/risk-budget/utilization",
                headers={"Authorization": f"Bearer {token_1}"}
            )

        # Request for tenant 2
        token_2 = create_test_token("user2", tenant_2_id)
        with patch.object(app.state, 'settings', mock_settings):
            response_2 = client.get(
                "/api/v1/risk-budget/utilization",
                headers={"Authorization": f"Bearer {token_2}"}
            )

        # Both should succeed
        assert response_1.status_code == 200
        assert response_2.status_code == 200


class TestRebalanceEndpoint:
    """Test /api/v1/risk-budget/rebalance endpoint."""

    @patch('alpha_pulse.api.routers.risk_budget.get_risk_budgeting_service')
    def test_rebalance_endpoint_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings, mock_risk_budgeting_service
    ):
        """Test that /rebalance endpoint uses tenant_id."""
        mock_get_service.return_value = mock_risk_budgeting_service
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/risk-budget/rebalance",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should succeed
        assert response.status_code == 200


class TestRecommendationsEndpoint:
    """Test /api/v1/risk-budget/recommendations endpoint."""

    @patch('alpha_pulse.api.routers.risk_budget.get_risk_budgeting_service')
    def test_recommendations_endpoint_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings, mock_risk_budgeting_service
    ):
        """Test that /recommendations endpoint uses tenant_id."""
        mock_get_service.return_value = mock_risk_budgeting_service
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/risk-budget/recommendations",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should succeed
        assert response.status_code == 200


class TestHistoryEndpoint:
    """Test /api/v1/risk-budget/history endpoint."""

    @patch('alpha_pulse.api.routers.risk_budget.get_risk_budgeting_service')
    def test_history_endpoint_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings, mock_risk_budgeting_service
    ):
        """Test that /history endpoint uses tenant_id."""
        mock_get_service.return_value = mock_risk_budgeting_service
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/risk-budget/history",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should succeed
        assert response.status_code == 200


class TestRegimeBudgetEndpoint:
    """Test /api/v1/risk-budget/regime/{regime_type} endpoint."""

    @patch('alpha_pulse.api.routers.risk_budget.get_risk_budgeting_service')
    def test_regime_budget_endpoint_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings, mock_risk_budgeting_service
    ):
        """Test that /regime endpoint uses tenant_id."""
        mock_get_service.return_value = mock_risk_budgeting_service
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/risk-budget/regime/BULL_MARKET",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should succeed
        assert response.status_code == 200


class TestTenantIsolationAcrossEndpoints:
    """Test that tenant context is consistent across all risk budget endpoints."""

    def test_all_endpoints_require_authentication(self, client):
        """Test that all risk budget endpoints reject unauthenticated requests."""
        endpoints = [
            ("/api/v1/risk-budget/current", "GET"),
            ("/api/v1/risk-budget/utilization", "GET"),
            ("/api/v1/risk-budget/rebalance", "POST"),
            ("/api/v1/risk-budget/recommendations", "GET"),
            ("/api/v1/risk-budget/history", "GET"),
            ("/api/v1/risk-budget/regime/BULL_MARKET", "GET")
        ]

        for endpoint, method in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            # Should return 401 Unauthorized
            assert response.status_code == 401, f"Endpoint {method} {endpoint} should require auth"
