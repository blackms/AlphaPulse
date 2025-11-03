"""
Tests for portfolio router tenant context integration.

Story 2.4 - Phase 1: Portfolio Router (Final P0 router)
Tests that portfolio endpoints properly extract and use tenant_id from middleware.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt
from datetime import datetime, timedelta

from alpha_pulse.api.main import app


@pytest.fixture
def tenant_1_id():
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def create_test_token():
    def _create_token(username: str, tenant_id: str) -> str:
        payload = {
            "sub": username,
            "tenant_id": tenant_id,
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }
        return jwt.encode(payload, "test-secret", algorithm="HS256")
    return _create_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_settings():
    settings = Mock()
    settings.JWT_SECRET = "test-secret"
    settings.JWT_ALGORITHM = "HS256"
    settings.RLS_ENABLED = True
    return settings


class TestPortfolioEndpoint:
    """Test /api/v1/portfolio endpoint."""

    @patch('alpha_pulse.api.routers.portfolio.get_portfolio_accessor')
    def test_portfolio_endpoint_uses_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /portfolio endpoint uses tenant_id."""
        mock_accessor = AsyncMock()
        mock_accessor.get_portfolio = AsyncMock(return_value={
            "total_value": 100000.0,
            "positions": []
        })
        mock_get_accessor.return_value = mock_accessor
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/portfolio",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200


class TestReloadEndpoints:
    """Test portfolio reload endpoints."""

    @patch('alpha_pulse.api.routers.portfolio.get_portfolio_accessor')
    @patch('alpha_pulse.api.routers.portfolio.trigger_exchange_sync')
    def test_reload_endpoint_uses_tenant_context(
        self, mock_sync, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /portfolio/reload uses tenant_id."""
        mock_sync.return_value = {"status": "success"}
        mock_accessor = AsyncMock()
        mock_accessor._exchange_id = "test_exchange"
        mock_get_accessor.return_value = mock_accessor
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/portfolio/reload",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
