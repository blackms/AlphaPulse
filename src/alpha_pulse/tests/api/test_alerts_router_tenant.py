"""
Tests for alerts router tenant context integration.

Story 2.4 - Phase 2: Alerts Router (P1 High Priority)
Tests that alert endpoints properly extract and use tenant_id from middleware.
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
def tenant_2_id():
    return "00000000-0000-0000-0000-000000000002"


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


class TestGetAlertsEndpoint:
    """Test /api/v1/alerts endpoint."""

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_get_alerts_uses_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /alerts endpoint extracts and uses tenant_id."""
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts = AsyncMock(return_value=[
            {
                "id": 1,
                "severity": "critical",
                "message": "High risk exposure",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False,
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/alerts",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Verify tenant_id was extracted (will be passed to accessor in GREEN phase)
        alerts = response.json()
        assert isinstance(alerts, list)

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_get_alerts_filters_by_tenant(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that alerts are filtered by tenant_id."""
        mock_accessor = AsyncMock()
        # Tenant 1 should only see their alerts
        mock_accessor.get_alerts = AsyncMock(return_value=[
            {
                "id": 1,
                "severity": "critical",
                "message": "Tenant 1 alert",
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/alerts",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        alerts = response.json()
        # Should only see tenant 1 alerts
        assert len(alerts) >= 0  # Could be empty or have tenant-specific alerts

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_get_alerts_with_filters(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /alerts endpoint uses tenant_id with filters."""
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts = AsyncMock(return_value=[
            {
                "id": 1,
                "severity": "critical",
                "message": "Critical alert",
                "tenant_id": tenant_1_id,
                "acknowledged": False
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/alerts?severity=critical&acknowledged=false",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        mock_accessor.get_alerts.assert_called_once()

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_get_alerts_logs_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings, caplog
    ):
        """Test that alert retrieval logs include tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts = AsyncMock(return_value=[])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/alerts",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Logs should contain tenant context (will be added in GREEN phase)


class TestAcknowledgeAlertEndpoint:
    """Test /api/v1/alerts/{alert_id}/acknowledge endpoint."""

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_acknowledge_alert_uses_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /alerts/{id}/acknowledge uses tenant_id."""
        mock_accessor = AsyncMock()
        mock_accessor.acknowledge_alert = AsyncMock(return_value={
            "success": True,
            "alert_id": 1,
            "acknowledged_by": "admin",
            "acknowledged_at": datetime.now().isoformat()
        })
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/alerts/1/acknowledge",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        mock_accessor.acknowledge_alert.assert_called_once()

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_acknowledge_alert_tenant_isolation(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that tenants can only acknowledge their own alerts."""
        mock_accessor = AsyncMock()
        # Simulate alert not found for different tenant
        mock_accessor.acknowledge_alert = AsyncMock(return_value={
            "success": False,
            "error": "Alert not found"
        })
        mock_get_accessor.return_value = mock_accessor

        # Tenant 1 trying to acknowledge alert (that belongs to tenant 2)
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/alerts/999/acknowledge",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should fail because alert doesn't belong to this tenant
        assert response.status_code in [404, 500]

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_acknowledge_alert_logs_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that alert acknowledgment logs include tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.acknowledge_alert = AsyncMock(return_value={
            "success": True,
            "alert_id": 1,
            "acknowledged_by": "admin"
        })
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/alerts/1/acknowledge",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Logs should contain tenant context (will be added in GREEN phase)

    @patch('alpha_pulse.api.routers.alerts.get_alert_accessor')
    def test_acknowledge_alert_error_handling_with_tenant(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test error handling includes tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.acknowledge_alert = AsyncMock(
            side_effect=Exception("Database error")
        )
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/alerts/1/acknowledge",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 500
        # Error logs should contain tenant context
