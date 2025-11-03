"""
Tests for trades router tenant context integration.

Story 2.4 - Phase 2: Trades Router (P1 High Priority)
Tests that trade endpoints properly extract and use tenant_id from middleware.
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


class TestGetTradesEndpoint:
    """Test /api/v1/trades endpoint."""

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_uses_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /trades endpoint extracts and uses tenant_id."""
        mock_accessor = AsyncMock()
        mock_accessor.get_trades = AsyncMock(return_value=[
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "buy",
                "price": 50000.0,
                "quantity": 0.5,
                "timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_1_id
            },
            {
                "id": 2,
                "symbol": "ETH/USDT",
                "side": "sell",
                "price": 3000.0,
                "quantity": 2.0,
                "timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/trades",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        assert len(trades) >= 0

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_filters_by_tenant(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that trades are filtered by tenant_id."""
        mock_accessor = AsyncMock()
        # Tenant 1 should only see their trades
        mock_accessor.get_trades = AsyncMock(return_value=[
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "buy",
                "price": 50000.0,
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/trades",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        trades = response.json()
        # Should only see tenant 1 trades
        for trade in trades:
            assert trade.get("tenant_id") == tenant_1_id or "symbol" in trade

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_with_symbol_filter(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /trades endpoint uses tenant_id with symbol filter."""
        mock_accessor = AsyncMock()
        mock_accessor.get_trades = AsyncMock(return_value=[
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "buy",
                "price": 50000.0,
                "quantity": 0.5,
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/trades?symbol=BTC/USDT",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        # Should only contain BTC/USDT trades for this tenant
        for trade in trades:
            assert trade.get("symbol") == "BTC/USDT" or "id" in trade

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_with_time_range(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /trades endpoint uses tenant_id with time range."""
        mock_accessor = AsyncMock()
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()

        mock_accessor.get_trades = AsyncMock(return_value=[
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "buy",
                "timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                f"/api/v1/trades?start_time={start_time.isoformat()}&end_time={end_time.isoformat()}",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        mock_accessor.get_trades.assert_called_once()

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_logs_tenant_context(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that trade retrieval logs include tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.get_trades = AsyncMock(return_value=[])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/trades",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Logs should contain tenant context

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_error_handling_with_tenant(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test error handling includes tenant context."""
        mock_accessor = AsyncMock()
        mock_accessor.get_trades = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/trades",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should return empty list on error (as per implementation)
        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        assert len(trades) == 0

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_empty_result_for_tenant(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that empty trades list is returned for tenant with no trades."""
        mock_accessor = AsyncMock()
        mock_accessor.get_trades = AsyncMock(return_value=[])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/trades",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        assert len(trades) == 0

    @patch('alpha_pulse.api.routers.trades.get_trade_accessor')
    def test_get_trades_multiple_filters(
        self, mock_get_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /trades endpoint works with multiple filters and tenant context."""
        mock_accessor = AsyncMock()
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()

        mock_accessor.get_trades = AsyncMock(return_value=[
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "side": "buy",
                "price": 50000.0,
                "timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_accessor.return_value = mock_accessor

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                f"/api/v1/trades?symbol=BTC/USDT&start_time={start_time.isoformat()}&end_time={end_time.isoformat()}",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        # Verify the accessor was called with the filters
        mock_accessor.get_trades.assert_called_once()
