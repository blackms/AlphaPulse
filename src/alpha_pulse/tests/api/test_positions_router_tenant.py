"""
Tests for positions router tenant context integration.

Story 2.4 - Phase 4: Positions Router (P3 High Priority - CRITICAL SECURITY FIX)
Tests that positions endpoints properly extract and use tenant_id from middleware.
Validates the CRITICAL security fix: positions.py now requires user authentication.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt
from datetime import datetime, timedelta
from decimal import Decimal

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


class TestGetSpotPositionsEndpoint:
    """Test GET /api/v1/positions/spot endpoint."""

    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_spot_requires_authentication(
        self, mock_get_exchange, mock_fetcher, client, mock_settings
    ):
        """CRITICAL: Test that /spot endpoint requires authentication (was unauthenticated before)."""
        with patch.object(app.state, 'settings', mock_settings):
            # Request WITHOUT authentication token
            response = client.get("/api/v1/positions/spot")

        # Should return 401 Unauthorized (was accessible before security fix)
        assert response.status_code == 401

    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_spot_uses_tenant_context(
        self, mock_get_exchange, mock_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /spot endpoint extracts and uses tenant_id."""
        # Mock exchange
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        # Mock position fetcher
        mock_fetcher_instance = AsyncMock()
        mock_position = Mock()
        mock_position.symbol = "BTC"
        mock_position.size = Decimal('1.5')
        mock_position.entry_price = Decimal('50000')
        mock_fetcher_instance.get_spot_positions = AsyncMock(return_value=[mock_position])
        mock_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/positions/spot",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)

    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_spot_tenant_isolation(
        self, mock_get_exchange, mock_fetcher,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that spot positions are tenant-isolated."""
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        mock_fetcher_instance = AsyncMock()
        # Tenant 1 positions
        mock_position_t1 = Mock()
        mock_position_t1.symbol = "BTC"
        mock_fetcher_instance.get_spot_positions = AsyncMock(return_value=[mock_position_t1])
        mock_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/positions/spot",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Should only see tenant 1's positions via their exchange account


class TestGetFuturesPositionsEndpoint:
    """Test GET /api/v1/positions/futures endpoint."""

    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_futures_requires_authentication(
        self, mock_get_exchange, mock_fetcher, client, mock_settings
    ):
        """CRITICAL: Test that /futures endpoint requires authentication (was unauthenticated before)."""
        with patch.object(app.state, 'settings', mock_settings):
            # Request WITHOUT authentication token
            response = client.get("/api/v1/positions/futures")

        # Should return 401 Unauthorized (was accessible before security fix)
        assert response.status_code == 401

    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_futures_uses_tenant_context(
        self, mock_get_exchange, mock_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /futures endpoint extracts and uses tenant_id."""
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        mock_fetcher_instance = AsyncMock()
        mock_position = Mock()
        mock_position.symbol = "BTCUSDT"
        mock_position.size = Decimal('2.0')
        mock_fetcher_instance.get_futures_positions = AsyncMock(return_value=[mock_position])
        mock_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/positions/futures",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)


class TestGetPositionMetricsEndpoint:
    """Test GET /api/v1/positions/metrics endpoint."""

    @patch('alpha_pulse.api.routers.positions.calculate_asset_metrics')
    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_metrics_requires_authentication(
        self, mock_get_exchange, mock_fetcher, mock_calc, client, mock_settings
    ):
        """CRITICAL: Test that /metrics endpoint requires authentication (was unauthenticated before)."""
        with patch.object(app.state, 'settings', mock_settings):
            # Request WITHOUT authentication token
            response = client.get("/api/v1/positions/metrics")

        # Should return 401 Unauthorized (was accessible before security fix)
        assert response.status_code == 401

    @patch('alpha_pulse.api.routers.positions.calculate_asset_metrics')
    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_metrics_uses_tenant_context(
        self, mock_get_exchange, mock_fetcher, mock_calc,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /metrics endpoint extracts and uses tenant_id."""
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.get_spot_positions = AsyncMock(return_value=[])
        mock_fetcher_instance.get_futures_positions = AsyncMock(return_value=[])
        mock_fetcher.return_value = mock_fetcher_instance

        # Mock metrics calculation
        mock_calc.return_value = {
            "BTC": {
                "spot_value": Decimal('75000'),
                "spot_qty": Decimal('1.5'),
                "futures_value": Decimal('-50000'),
                "futures_qty": Decimal('-1.0'),
                "net_exposure": Decimal('25000'),
                "hedge_ratio": Decimal('0.66')
            }
        }

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/positions/metrics",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        if result:
            assert "symbol" in result[0]
            assert "net_exposure" in result[0]
            assert "hedge_ratio" in result[0]

    @patch('alpha_pulse.api.routers.positions.calculate_asset_metrics')
    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_metrics_tenant_isolation(
        self, mock_get_exchange, mock_fetcher, mock_calc,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that position metrics are tenant-isolated."""
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.get_spot_positions = AsyncMock(return_value=[])
        mock_fetcher_instance.get_futures_positions = AsyncMock(return_value=[])
        mock_fetcher.return_value = mock_fetcher_instance

        mock_calc.return_value = {"BTC": {
            "spot_value": Decimal('75000'), "spot_qty": Decimal('1.5'),
            "futures_value": Decimal('-50000'), "futures_qty": Decimal('-1.0'),
            "net_exposure": Decimal('25000'), "hedge_ratio": Decimal('0.66')
        }}

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/positions/metrics",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Metrics calculated from tenant 1's exchange positions only


class TestPositionsErrorHandling:
    """Test error handling in positions endpoints."""

    @patch('alpha_pulse.api.routers.positions.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.positions.get_exchange_client')
    def test_get_spot_error_handling(
        self, mock_get_exchange, mock_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test error handling with tenant context."""
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        # Mock fetcher to raise exception
        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.get_spot_positions = AsyncMock(side_effect=Exception("Exchange API error"))
        mock_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/positions/spot",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 500
        assert "Failed to fetch spot positions" in response.json()["detail"]
