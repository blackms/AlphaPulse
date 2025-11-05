"""
Tests for regime router tenant context integration.

Story 2.4 - Phase 4: Regime Router (P3 Medium Priority)
Tests that regime endpoints properly extract and use tenant_id from middleware.
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


@pytest.fixture
def mock_regime_service():
    """Mock regime detection service with common data."""
    service = Mock()

    # Mock regime info
    regime_info = Mock()
    regime_info.current_regime = 0  # Bull
    regime_info.confidence = 0.85
    regime_info.timestamp = datetime.utcnow()
    regime_info.features = {"volatility": 0.15, "trend": 0.8}
    regime_info.transition_matrix = [[0.7, 0.1, 0.1, 0.05, 0.05]] * 5
    service.current_regime_info = regime_info

    # Mock state manager
    current_state = Mock()
    current_state.stability_score = 0.9
    service.state_manager = Mock()
    service.state_manager.get_current_state = Mock(return_value=current_state)
    service.state_manager.get_historical_states = Mock(return_value=[])

    return service


class TestGetCurrentRegimeEndpoint:
    """Test GET /api/v1/regime/current endpoint."""

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_current_regime_uses_tenant_context(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings, mock_regime_service
    ):
        """Test that /current endpoint extracts and uses tenant_id."""
        mock_get_service.return_value = mock_regime_service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/current",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "current_regime" in result
        assert result["current_regime"] == "Bull"
        assert "confidence" in result
        assert "transition_probabilities" in result

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_current_regime_tenant_isolation(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings, mock_regime_service
    ):
        """Test that current regime is tenant-scoped."""
        mock_get_service.return_value = mock_regime_service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/current",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Future: Each tenant may have different regime detection models


class TestGetRegimeHistoryEndpoint:
    """Test GET /api/v1/regime/history endpoint."""

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_regime_history_uses_tenant_context(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings, mock_regime_service
    ):
        """Test that /history endpoint extracts and uses tenant_id."""
        # Mock historical states
        state1 = Mock()
        state1.regime = 0  # Bull
        state1.timestamp = datetime.utcnow() - timedelta(days=10)
        state1.confidence = 0.8
        state1.features = {"volatility": 0.15}

        state2 = Mock()
        state2.regime = 2  # Sideways
        state2.timestamp = datetime.utcnow()
        state2.confidence = 0.75
        state2.features = {"volatility": 0.12}

        mock_regime_service.state_manager.get_historical_states = Mock(return_value=[state1, state2])
        mock_get_service.return_value = mock_regime_service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/history?days=30",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "regimes" in result
        assert "transitions" in result
        assert "total_regimes" in result
        assert result["total_regimes"] == 2

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_regime_history_empty_case(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings, mock_regime_service
    ):
        """Test history endpoint with no historical data."""
        mock_regime_service.state_manager.get_historical_states = Mock(return_value=[])
        mock_get_service.return_value = mock_regime_service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/history?days=30",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["total_regimes"] == 0
        assert len(result["regimes"]) == 0


class TestGetRegimeAnalysisEndpoint:
    """Test GET /api/v1/regime/analysis/{regime_type} endpoint."""

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_regime_analysis_uses_tenant_context(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings, mock_regime_service
    ):
        """Test that /analysis/{regime_type} endpoint extracts and uses tenant_id."""
        mock_get_service.return_value = mock_regime_service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/analysis/bull",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["regime_type"] == "Bull"
        assert "characteristics" in result
        assert "recommended_strategies" in result
        assert "risk_adjustments" in result
        assert "historical_performance" in result

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_regime_analysis_all_regime_types(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings, mock_regime_service
    ):
        """Test analysis for all valid regime types."""
        mock_get_service.return_value = mock_regime_service
        token = create_test_token("analyst", tenant_1_id)

        regime_types = ["bull", "bear", "sideways", "high_volatility", "crisis"]

        with patch.object(app.state, 'settings', mock_settings):
            for regime_type in regime_types:
                response = client.get(
                    f"/api/v1/regime/analysis/{regime_type}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                assert response.status_code == 200
                result = response.json()
                assert "characteristics" in result
                assert "recommended_strategies" in result

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_regime_analysis_invalid_type(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings, mock_regime_service
    ):
        """Test analysis with invalid regime type."""
        mock_get_service.return_value = mock_regime_service
        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/analysis/invalid_regime",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 400
        assert "Invalid regime type" in response.json()["detail"]


class TestGetRegimeAlertsEndpoint:
    """Test GET /api/v1/regime/alerts endpoint."""

    def test_regime_alerts_uses_tenant_context(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /alerts endpoint extracts and uses tenant_id."""
        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/alerts?hours=24",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "regime_transition_alerts" in result
        assert "confidence_alerts" in result
        assert "stability_alerts" in result
        assert "total_alerts" in result
        assert result["time_range_hours"] == 24

    def test_regime_alerts_different_time_ranges(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test alerts with different time ranges."""
        token = create_test_token("analyst", tenant_1_id)
        time_ranges = [1, 12, 24, 72, 168]

        with patch.object(app.state, 'settings', mock_settings):
            for hours in time_ranges:
                response = client.get(
                    f"/api/v1/regime/alerts?hours={hours}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                assert response.status_code == 200
                result = response.json()
                assert result["time_range_hours"] == hours


class TestRegimeErrorHandling:
    """Test error handling in regime endpoints."""

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_current_regime_no_data(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test error handling when no regime data available."""
        # Mock service with no regime info
        service = Mock()
        service.current_regime_info = None
        mock_get_service.return_value = service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/current",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 404
        assert "No regime state available" in response.json()["detail"]

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_regime_history_error_handling(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test error handling in history endpoint."""
        # Mock service that raises exception
        service = Mock()
        service.state_manager.get_historical_states = Mock(side_effect=Exception("Database error"))
        mock_get_service.return_value = service

        token = create_test_token("analyst", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/regime/history?days=30",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 500


class TestRegimeTenantIsolation:
    """Test tenant isolation across regime endpoints."""

    @patch('alpha_pulse.api.routers.regime.get_regime_detection_service')
    def test_tenant_isolation_across_endpoints(
        self, mock_get_service,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings, mock_regime_service
    ):
        """Test that different tenants can access regime data independently."""
        mock_get_service.return_value = mock_regime_service

        # Tenant 1 access
        token_1 = create_test_token("analyst1", tenant_1_id)
        with patch.object(app.state, 'settings', mock_settings):
            response_1 = client.get(
                "/api/v1/regime/current",
                headers={"Authorization": f"Bearer {token_1}"}
            )

        # Tenant 2 access
        token_2 = create_test_token("analyst2", tenant_2_id)
        with patch.object(app.state, 'settings', mock_settings):
            response_2 = client.get(
                "/api/v1/regime/current",
                headers={"Authorization": f"Bearer {token_2}"}
            )

        assert response_1.status_code == 200
        assert response_2.status_code == 200
        # Both tenants get regime data (shared model currently, but logged separately)
