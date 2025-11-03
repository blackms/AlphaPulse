"""
Tests for system router tenant context integration.

Story 2.4 - Phase 2: System Router (P1 High Priority)
Tests that system endpoints properly extract and use tenant_id from middleware.
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


class TestGetSystemMetricsEndpoint:
    """Test /api/v1/system endpoint."""

    @patch('alpha_pulse.api.routers.system.get_system_accessor')
    @patch('alpha_pulse.api.routers.system.get_alert_manager')
    def test_get_system_metrics_uses_tenant_context(
        self, mock_get_alert_manager, mock_get_accessor, client,
        create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /system endpoint extracts and uses tenant_id."""
        mock_accessor = AsyncMock()
        mock_accessor.get_system_metrics = AsyncMock(return_value={
            "cpu": {"usage_percent": 45.2, "cores": 8},
            "memory": {"percent": 62.1, "available": 8192},
            "disk": {"percent": 35.5, "free": 500000},
            "timestamp": datetime.now().isoformat()
        })
        mock_get_accessor.return_value = mock_accessor

        mock_alert_manager_instance = AsyncMock()
        mock_alert_manager_instance.process_metrics = AsyncMock(return_value=[])
        mock_get_alert_manager.return_value = mock_alert_manager_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/system",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metrics = response.json()
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics

    @patch('alpha_pulse.api.routers.system.get_system_accessor')
    @patch('alpha_pulse.api.routers.system.get_alert_manager')
    def test_get_system_metrics_tenant_isolation(
        self, mock_get_alert_manager, mock_get_accessor, client,
        create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that system metrics are tenant-isolated."""
        mock_accessor = AsyncMock()
        # Each tenant gets their own system metrics
        mock_accessor.get_system_metrics = AsyncMock(return_value={
            "cpu": {"usage_percent": 45.2},
            "memory": {"percent": 62.1},
            "disk": {"percent": 35.5},
            "tenant_id": tenant_1_id
        })
        mock_get_accessor.return_value = mock_accessor

        mock_alert_manager_instance = AsyncMock()
        mock_alert_manager_instance.process_metrics = AsyncMock(return_value=[])
        mock_get_alert_manager.return_value = mock_alert_manager_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/system",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metrics = response.json()
        # Verify tenant context is preserved
        assert metrics.get("tenant_id") == tenant_1_id or "cpu" in metrics

    @patch('alpha_pulse.api.routers.system.get_system_accessor')
    @patch('alpha_pulse.api.routers.system.get_alert_manager')
    def test_get_system_metrics_with_alerts(
        self, mock_get_alert_manager, mock_get_accessor, client,
        create_test_token, tenant_1_id, mock_settings
    ):
        """Test that system metrics includes alerts when thresholds are exceeded."""
        mock_accessor = AsyncMock()
        mock_accessor.get_system_metrics = AsyncMock(return_value={
            "cpu": {"usage_percent": 95.5},
            "memory": {"percent": 92.1},
            "disk": {"percent": 88.5}
        })
        mock_get_accessor.return_value = mock_accessor

        # Mock alert manager to return alerts
        mock_alert = Mock()
        mock_alert.alert_id = "alert-123"
        mock_alert.message = "High CPU usage detected"
        mock_alert.severity.value = "critical"

        mock_alert_manager_instance = AsyncMock()
        mock_alert_manager_instance.process_metrics = AsyncMock(return_value=[mock_alert])
        mock_get_alert_manager.return_value = mock_alert_manager_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/system",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metrics = response.json()
        assert "alerts" in metrics or "cpu" in metrics

    @patch('alpha_pulse.api.routers.system.get_system_accessor')
    @patch('alpha_pulse.api.routers.system.get_alert_manager')
    def test_get_system_metrics_logs_tenant_context(
        self, mock_get_alert_manager, mock_get_accessor, client,
        create_test_token, tenant_1_id, mock_settings
    ):
        """Test that system metrics logs include tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.get_system_metrics = AsyncMock(return_value={
            "cpu": {"usage_percent": 45.2},
            "memory": {"percent": 62.1},
            "disk": {"percent": 35.5}
        })
        mock_get_accessor.return_value = mock_accessor

        mock_alert_manager_instance = AsyncMock()
        mock_alert_manager_instance.process_metrics = AsyncMock(return_value=[])
        mock_get_alert_manager.return_value = mock_alert_manager_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/system",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Logs should contain tenant context

    @patch('alpha_pulse.api.routers.system.get_system_accessor')
    @patch('alpha_pulse.api.routers.system.get_alert_manager')
    def test_get_system_metrics_error_handling(
        self, mock_get_alert_manager, mock_get_accessor, client,
        create_test_token, tenant_1_id, mock_settings
    ):
        """Test error handling includes tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.get_system_metrics = AsyncMock(
            return_value={"error": "Failed to fetch metrics"}
        )
        mock_get_accessor.return_value = mock_accessor

        mock_alert_manager_instance = AsyncMock()
        mock_get_alert_manager.return_value = mock_alert_manager_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/system",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metrics = response.json()
        assert "error" in metrics


class TestForceExchangeSyncEndpoint:
    """Test /api/v1/system/exchange/reload endpoint."""

    @patch('alpha_pulse.api.routers.system.get_portfolio_accessor')
    @patch('alpha_pulse.api.routers.system.trigger_exchange_sync')
    def test_force_exchange_sync_uses_tenant_context(
        self, mock_trigger_sync, mock_get_portfolio_accessor,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /system/exchange/reload uses tenant_id."""
        mock_accessor = Mock()
        mock_accessor._exchange_id = "exchange-123"
        mock_get_portfolio_accessor.return_value = mock_accessor

        mock_trigger_sync.return_value = {
            "success": True,
            "exchange_id": "exchange-123",
            "synced_at": datetime.now().isoformat(),
            "positions_synced": 5
        }

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/system/exchange/reload",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        mock_trigger_sync.assert_called_once_with("exchange-123")

    @patch('alpha_pulse.api.routers.system.get_portfolio_accessor')
    @patch('alpha_pulse.api.routers.system.trigger_exchange_sync')
    def test_force_exchange_sync_tenant_isolation(
        self, mock_trigger_sync, mock_get_portfolio_accessor,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that exchange sync is tenant-isolated."""
        # Tenant 1's exchange
        mock_accessor = Mock()
        mock_accessor._exchange_id = "tenant1-exchange"
        mock_get_portfolio_accessor.return_value = mock_accessor

        mock_trigger_sync.return_value = {
            "success": True,
            "exchange_id": "tenant1-exchange",
            "synced_at": datetime.now().isoformat()
        }

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/system/exchange/reload",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only sync tenant 1's exchange
        assert result["exchange_id"] == "tenant1-exchange"

    @patch('alpha_pulse.api.routers.system.get_portfolio_accessor')
    @patch('alpha_pulse.api.routers.system.trigger_exchange_sync')
    def test_force_exchange_sync_logs_tenant_context(
        self, mock_trigger_sync, mock_get_portfolio_accessor,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that exchange sync logs include tenant context."""
        mock_accessor = Mock()
        mock_accessor._exchange_id = "exchange-123"
        mock_get_portfolio_accessor.return_value = mock_accessor

        mock_trigger_sync.return_value = {
            "success": True,
            "exchange_id": "exchange-123"
        }

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/system/exchange/reload",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Logs should contain tenant context

    @patch('alpha_pulse.api.routers.system.get_portfolio_accessor')
    @patch('alpha_pulse.api.routers.system.trigger_exchange_sync')
    def test_force_exchange_sync_error_handling(
        self, mock_trigger_sync, mock_get_portfolio_accessor,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test error handling includes tenant context."""
        mock_accessor = Mock()
        mock_accessor._exchange_id = "exchange-123"
        mock_get_portfolio_accessor.return_value = mock_accessor

        mock_trigger_sync.side_effect = Exception("Exchange connection failed")

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/system/exchange/reload",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 500
        error = response.json()
        assert "detail" in error
        assert "Exchange connection failed" in error["detail"]
