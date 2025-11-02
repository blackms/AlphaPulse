"""
Tests for risk router tenant context integration.

Story 2.4 - Phase 1: Risk Router
Tests that risk endpoints properly extract and use tenant_id from middleware.
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
def mock_exchange():
    """Mock exchange client."""
    exchange = AsyncMock()
    exchange.get_portfolio_value = AsyncMock(return_value=100000.0)
    exchange.fetch_ohlcv = AsyncMock(return_value=[
        Mock(close=50000.0) for _ in range(252)
    ])
    return exchange


class TestRiskExposureEndpoint:
    """Test /api/v1/risk/exposure endpoint."""

    @patch('alpha_pulse.api.routers.risk.RiskManager')
    def test_exposure_endpoint_uses_tenant_context(
        self, mock_risk_manager_class, client, create_test_token,
        tenant_1_id, mock_settings, mock_exchange
    ):
        """Test that /exposure endpoint extracts and uses tenant_id."""
        # Setup mock
        mock_risk_manager = AsyncMock()
        mock_risk_manager.calculate_risk_exposure = AsyncMock(return_value={
            "BTC/USDT": 0.25,
            "ETH/USDT": 0.15,
            "tenant_id": tenant_1_id
        })
        mock_risk_manager_class.return_value = mock_risk_manager

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.risk.get_exchange_client', return_value=mock_exchange):
                response = client.get(
                    "/api/v1/risk/exposure",
                    headers={"Authorization": f"Bearer {token}"}
                )

        # Endpoint should call calculate_risk_exposure with tenant_id
        assert response.status_code == 200
        mock_risk_manager.calculate_risk_exposure.assert_called_once()
        # Verify tenant_id was passed
        call_kwargs = mock_risk_manager.calculate_risk_exposure.call_args.kwargs
        assert "tenant_id" in call_kwargs
        assert call_kwargs["tenant_id"] == tenant_1_id


    def test_exposure_endpoint_rejects_missing_tenant(
        self, client, jwt_secret, mock_settings
    ):
        """Test that /exposure rejects requests without tenant_id in JWT."""
        # Create token WITHOUT tenant_id
        payload = {
            "sub": "admin",
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/risk/exposure",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should return 401 Unauthorized
        assert response.status_code == 401
        assert "tenant_id" in response.json()["detail"].lower()


class TestRiskMetricsEndpoint:
    """Test /api/v1/risk/metrics endpoint."""

    @patch('alpha_pulse.api.routers.risk.RiskAnalyzer')
    def test_metrics_endpoint_uses_tenant_context(
        self, mock_analyzer_class, client, create_test_token,
        tenant_1_id, mock_settings, mock_exchange
    ):
        """Test that /metrics endpoint uses tenant_id for isolation."""
        # Setup mock
        mock_metrics = Mock(
            volatility=0.25,
            var_95=0.05,
            cvar_95=0.07,
            max_drawdown=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2
        )
        mock_analyzer = Mock()
        mock_analyzer.calculate_metrics = Mock(return_value=mock_metrics)
        mock_analyzer_class.return_value = mock_analyzer

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.risk.get_exchange_client', return_value=mock_exchange):
                response = client.get(
                    "/api/v1/risk/metrics",
                    headers={"Authorization": f"Bearer {token}"}
                )

        # Should succeed with tenant context
        assert response.status_code == 200
        data = response.json()
        assert "volatility" in data


class TestRiskLimitsEndpoint:
    """Test /api/v1/risk/limits endpoint."""

    @patch('alpha_pulse.api.routers.risk.RiskManager')
    def test_limits_endpoint_isolates_by_tenant(
        self, mock_risk_manager_class, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings, mock_exchange
    ):
        """Test that /limits endpoint returns tenant-specific limits."""
        # Setup mock
        mock_risk_manager = Mock()
        mock_risk_manager.config = Mock(
            max_portfolio_leverage=1.5,
            max_drawdown=0.25
        )
        mock_risk_manager.get_position_limits = Mock(return_value={"default": 0.2})
        mock_risk_manager.get_risk_report = Mock(return_value={})
        mock_risk_manager_class.return_value = mock_risk_manager

        # Request for tenant 1
        token_1 = create_test_token("user1", tenant_1_id)
        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.risk.get_exchange_client', return_value=mock_exchange):
                response_1 = client.get(
                    "/api/v1/risk/limits",
                    headers={"Authorization": f"Bearer {token_1}"}
                )

        # Request for tenant 2
        token_2 = create_test_token("user2", tenant_2_id)
        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.risk.get_exchange_client', return_value=mock_exchange):
                response_2 = client.get(
                    "/api/v1/risk/limits",
                    headers={"Authorization": f"Bearer {token_2}"}
                )

        # Both should succeed with isolated data
        assert response_1.status_code == 200
        assert response_2.status_code == 200


class TestPositionSizeEndpoint:
    """Test /api/v1/risk/position-size/{asset} endpoint."""

    @patch('alpha_pulse.api.routers.risk.PositionSizer')
    def test_position_size_endpoint_uses_tenant_context(
        self, mock_sizer_class, client, create_test_token,
        tenant_1_id, mock_settings, mock_exchange
    ):
        """Test that /position-size endpoint uses tenant_id."""
        # Setup mock
        mock_recommendation = Mock(
            recommended_size=0.1,
            max_size=0.2,
            risk_per_trade=0.02,
            stop_loss=45000.0,
            take_profit=55000.0
        )
        mock_sizer = AsyncMock()
        mock_sizer.get_position_size_recommendation = AsyncMock(
            return_value=mock_recommendation
        )
        mock_sizer_class.return_value = mock_sizer

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.risk.get_exchange_client', return_value=mock_exchange):
                response = client.get(
                    "/api/v1/risk/position-size/BTC/USDT",
                    headers={"Authorization": f"Bearer {token}"}
                )

        # Should succeed
        assert response.status_code == 200
        data = response.json()
        assert "recommended_size" in data


class TestRiskReportEndpoint:
    """Test /api/v1/risk/report endpoint."""

    @patch('alpha_pulse.api.routers.risk.RiskManager')
    def test_report_endpoint_uses_tenant_context(
        self, mock_risk_manager_class, client, create_test_token,
        tenant_1_id, mock_settings, mock_exchange
    ):
        """Test that /report endpoint uses tenant_id."""
        # Setup mock
        mock_risk_manager = Mock()
        mock_risk_manager.get_risk_report = Mock(return_value={
            "portfolio_value": 100000.0,
            "total_exposure": 0.4,
            "risk_score": 0.3
        })
        mock_risk_manager_class.return_value = mock_risk_manager

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.risk.get_exchange_client', return_value=mock_exchange):
                response = client.get(
                    "/api/v1/risk/report",
                    headers={"Authorization": f"Bearer {token}"}
                )

        # Should succeed
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_value" in data


class TestTenantIsolationAcrossEndpoints:
    """Test that tenant context is consistent across all risk endpoints."""

    def test_all_endpoints_require_authentication(self, client):
        """Test that all risk endpoints reject unauthenticated requests."""
        endpoints = [
            "/api/v1/risk/exposure",
            "/api/v1/risk/metrics",
            "/api/v1/risk/limits",
            "/api/v1/risk/position-size/BTC/USDT",
            "/api/v1/risk/report"
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should return 401 Unauthorized
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
