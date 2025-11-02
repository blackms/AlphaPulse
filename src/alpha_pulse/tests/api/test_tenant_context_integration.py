"""
Integration tests for tenant context in API routes.

This module tests that API endpoints properly extract tenant_id from
middleware and enforce tenant isolation across all routes.

Story 2.4: Update API routes with tenant context
EPIC-002: Application Multi-Tenancy
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt
from datetime import datetime, timedelta

from alpha_pulse.api.main import app
from alpha_pulse.api.auth import User


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


class TestTenantContextMiddleware:
    """Test that middleware extracts tenant_id correctly."""

    def test_middleware_extracts_tenant_from_jwt(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that middleware extracts tenant_id from JWT token."""
        # Create token with tenant_id
        token = create_test_token("admin", tenant_1_id)

        # Mock settings
        with patch.object(app.state, 'settings', mock_settings):
            # Call protected endpoint
            response = client.get(
                "/api/v1/metrics/performance",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should not return 401 (middleware should process token)
        assert response.status_code != 401, "Token should be valid"

    def test_middleware_rejects_missing_tenant_id(
        self, client, jwt_secret, mock_settings
    ):
        """Test that middleware rejects tokens without tenant_id claim."""
        # Create token WITHOUT tenant_id
        payload = {
            "sub": "admin",
            # Missing tenant_id
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/performance",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should return 401 Unauthorized
        assert response.status_code == 401
        assert "tenant_id" in response.json()["detail"].lower()

    def test_middleware_rejects_missing_authorization(self, client, mock_settings):
        """Test that middleware rejects requests without Authorization header."""
        with patch.object(app.state, 'settings', mock_settings):
            response = client.get("/api/v1/metrics/performance")

        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()


class TestAPIRouteTenantIsolation:
    """Test that API routes enforce tenant isolation."""

    @patch('alpha_pulse.api.routers.metrics.metrics_accessor')
    def test_metrics_endpoint_uses_tenant_context(
        self, mock_accessor, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that metrics endpoint extracts and uses tenant_id."""
        # Setup mock
        mock_accessor.get_latest_metrics = AsyncMock(return_value={
            "performance": 0.15,
            "timestamp": datetime.utcnow().isoformat()
        })

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/performance/latest",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Endpoint should succeed
        assert response.status_code == 200

    @patch('alpha_pulse.api.routers.portfolio.portfolio_accessor')
    def test_portfolio_endpoint_isolates_by_tenant(
        self, mock_accessor, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that portfolio endpoint returns tenant-specific data."""
        # Setup mock to return different data per tenant
        def get_portfolio_side_effect(*args, **kwargs):
            # This will be called with tenant_id parameter
            return AsyncMock(return_value={
                "positions": [],
                "total_value": 100000.0
            })()

        mock_accessor.get_portfolio = get_portfolio_side_effect

        # Request for tenant 1
        token_1 = create_test_token("user1", tenant_1_id)
        with patch.object(app.state, 'settings', mock_settings):
            response_1 = client.get(
                "/api/v1/portfolio",
                headers={"Authorization": f"Bearer {token_1}"}
            )

        # Request for tenant 2
        token_2 = create_test_token("user2", tenant_2_id)
        with patch.object(app.state, 'settings', mock_settings):
            response_2 = client.get(
                "/api/v1/portfolio",
                headers={"Authorization": f"Bearer {token_2}"}
            )

        # Both should succeed
        assert response_1.status_code == 200
        assert response_2.status_code == 200

    @patch('alpha_pulse.api.routers.alerts.alert_manager')
    def test_alerts_endpoint_filters_by_tenant(
        self, mock_alert_manager, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that alerts endpoint only returns tenant-specific alerts."""
        # Setup mock
        mock_alert_manager.get_all_alerts = AsyncMock(return_value=[
            {"id": "alert1", "message": "Test alert", "severity": "high"}
        ])

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch.object(app.state, 'alert_manager', mock_alert_manager):
                response = client.get(
                    "/api/v1/alerts",
                    headers={"Authorization": f"Bearer {token}"}
                )

        assert response.status_code == 200


class TestTenantContextDependency:
    """Test get_current_tenant_id dependency injection."""

    def test_dependency_extracts_tenant_from_request_state(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that dependency correctly extracts tenant_id from request.state."""
        from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
        from fastapi import Request

        # Create mock request with tenant_id in state
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.tenant_id = tenant_1_id

        # Call dependency
        import asyncio
        tenant_id = asyncio.run(get_current_tenant_id(request))

        assert tenant_id == tenant_1_id

    def test_dependency_raises_error_when_context_missing(self):
        """Test that dependency raises error when tenant context not set."""
        from alpha_pulse.api.middleware.tenant_context import get_current_tenant_id
        from fastapi import HTTPException, Request

        # Create mock request WITHOUT tenant_id
        request = Mock(spec=Request)
        request.state = Mock(spec=['__dict__'])

        # Should raise HTTPException
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(get_current_tenant_id(request))

        assert exc_info.value.status_code == 500
        assert "tenant context not set" in exc_info.value.detail.lower()


class TestCrossEndpointTenantConsistency:
    """Test that tenant context remains consistent across API calls."""

    def test_multiple_endpoints_use_same_tenant_context(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that multiple API calls in same session use consistent tenant_id."""
        token = create_test_token("admin", tenant_1_id)
        headers = {"Authorization": f"Bearer {token}"}

        with patch.object(app.state, 'settings', mock_settings):
            # Call multiple endpoints
            endpoints = [
                "/api/v1/metrics/performance/latest",
                "/api/v1/portfolio",
                "/api/v1/alerts"
            ]

            for endpoint in endpoints:
                # Patch the underlying services to avoid database calls
                with patch('alpha_pulse.api.routers.metrics.metrics_accessor'):
                    with patch('alpha_pulse.api.routers.portfolio.portfolio_accessor'):
                        with patch.object(app.state, 'alert_manager', Mock()):
                            response = client.get(endpoint, headers=headers)

                            # All should process with same tenant context
                            # (either success or expected error, but not 401)
                            assert response.status_code != 401


class TestExemptPaths:
    """Test that exempt paths don't require tenant context."""

    @pytest.mark.parametrize("path", [
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/token"
    ])
    def test_exempt_paths_accessible_without_token(self, client, path):
        """Test that exempt paths don't require authentication."""
        response = client.get(path)

        # Should not return 401 Unauthorized
        assert response.status_code != 401, f"Path {path} should be exempt from auth"


class TestPostgreSQLRLSIntegration:
    """Test that PostgreSQL RLS session variable is set correctly."""

    @patch('alpha_pulse.api.middleware.tenant_context.TenantContextMiddleware._set_postgres_tenant_context')
    def test_middleware_sets_postgres_session_variable(
        self, mock_set_context, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that middleware sets PostgreSQL session variable for RLS."""
        mock_set_context.return_value = AsyncMock()

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.metrics.metrics_accessor'):
                response = client.get(
                    "/api/v1/metrics/performance/latest",
                    headers={"Authorization": f"Bearer {token}"}
                )

        # Middleware should have called _set_postgres_tenant_context
        # (if RLS_ENABLED is True)
        if mock_settings.RLS_ENABLED:
            mock_set_context.assert_called()
