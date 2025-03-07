"""Tests for trades API endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user
from alpha_pulse.api.data import TradeDataAccessor


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_trade_accessor():
    """Mock the TradeDataAccessor."""
    with patch("alpha_pulse.api.routers.trades.trade_accessor") as mock:
        yield mock


@pytest.fixture
def sample_trades():
    """Generate sample trades data."""
    now = datetime.now()
    return [
        {
            "id": "trade-001",
            "symbol": "BTC-USD",
            "side": "buy",
            "quantity": 0.5,
            "price": 45000.0,
            "timestamp": now.isoformat(),
            "status": "filled",
            "order_type": "market",
            "fees": 22.5
        },
        {
            "id": "trade-002",
            "symbol": "ETH-USD",
            "side": "sell",
            "quantity": 2.0,
            "price": 3000.0,
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "status": "filled",
            "order_type": "limit",
            "fees": 6.0
        },
        {
            "id": "trade-003",
            "symbol": "SOL-USD",
            "side": "buy",
            "quantity": 10.0,
            "price": 125.0,
            "timestamp": (now - timedelta(days=1)).isoformat(),
            "status": "filled",
            "order_type": "market",
            "fees": 12.5
        },
        {
            "id": "trade-004",
            "symbol": "BTC-USD",
            "side": "sell",
            "quantity": 0.2,
            "price": 46000.0,
            "timestamp": (now - timedelta(days=2)).isoformat(),
            "status": "filled",
            "order_type": "limit",
            "fees": 9.2
        }
    ]


@pytest.fixture
def admin_user():
    """Generate admin user authentication."""
    return {
        "username": "admin",
        "role": "admin",
        "permissions": ["view_trades", "view_metrics", "view_alerts"]
    }


@pytest.fixture
def trader_user():
    """Generate trader user authentication."""
    return {
        "username": "trader",
        "role": "trader",
        "permissions": ["view_trades", "execute_trades"]
    }


@pytest.fixture
def viewer_user():
    """Generate viewer user authentication."""
    return {
        "username": "viewer",
        "role": "viewer",
        "permissions": ["view_metrics"]
    }


@pytest.fixture
def auth_override():
    """Override the get_current_user dependency."""
    def _override_dependency(user):
        app.dependency_overrides[get_current_user] = lambda: user
        yield
        app.dependency_overrides = {}
    return _override_dependency


def test_get_trades_success(client, mock_trade_accessor, sample_trades, auth_override, admin_user):
    """Test successful trades retrieval."""
    with auth_override(admin_user):
        # Mock the get_trades method to return sample data
        mock_trade_accessor.get_trades.return_value = sample_trades
        
        # Make request
        response = client.get("/api/v1/trades")
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_trades
        
        # Verify the trade accessor was called with correct parameters
        mock_trade_accessor.get_trades.assert_called_once_with(
            symbol=None,
            start_time=None,
            end_time=None
        )


def test_get_trades_with_symbol(client, mock_trade_accessor, sample_trades, auth_override, admin_user):
    """Test trades retrieval filtered by symbol."""
    with auth_override(admin_user):
        # Filter trades by symbol
        btc_trades = [trade for trade in sample_trades if trade["symbol"] == "BTC-USD"]
        mock_trade_accessor.get_trades.return_value = btc_trades
        
        # Make request with symbol filter
        response = client.get("/api/v1/trades", params={"symbol": "BTC-USD"})
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == btc_trades
        assert all(trade["symbol"] == "BTC-USD" for trade in response.json())
        
        # Verify the trade accessor was called with correct parameters
        mock_trade_accessor.get_trades.assert_called_once_with(
            symbol="BTC-USD",
            start_time=None,
            end_time=None
        )


def test_get_trades_with_time_range(client, mock_trade_accessor, sample_trades, auth_override, admin_user):
    """Test trades retrieval with time range."""
    with auth_override(admin_user):
        # Mock the get_trades method to return sample data
        mock_trade_accessor.get_trades.return_value = sample_trades
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        # Make request with time range
        response = client.get(
            "/api/v1/trades",
            params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        )
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_trades
        
        # Verify the trade accessor was called with correct parameters
        mock_trade_accessor.get_trades.assert_called_once()
        call_args = mock_trade_accessor.get_trades.call_args[1]
        assert isinstance(call_args["start_time"], datetime)
        assert isinstance(call_args["end_time"], datetime)


def test_get_trades_unauthorized(client):
    """Test unauthorized access to trades."""
    # Make request without authentication
    response = client.get("/api/v1/trades")
    
    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


def test_get_trades_forbidden(client, auth_override, viewer_user):
    """Test access with insufficient permissions."""
    with auth_override(viewer_user):
        # Make request (viewer doesn't have view_trades permission)
        response = client.get("/api/v1/trades")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_get_trades_error(client, mock_trade_accessor, auth_override, admin_user):
    """Test error handling in trades endpoint."""
    with auth_override(admin_user):
        # Mock the get_trades method to raise an exception
        mock_trade_accessor.get_trades.side_effect = Exception("Database error")
        
        # Make request
        response = client.get("/api/v1/trades")
        
        # Verify response (should return empty list on error, not fail)
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.performance
def test_trades_endpoint_performance(client, mock_trade_accessor, auth_override, admin_user):
    """Test performance of trades endpoint with large dataset."""
    with auth_override(admin_user):
        # Generate large dataset (e.g., 1000 trades)
        large_dataset = []
        now = datetime.now()
        
        for i in range(1000):
            large_dataset.append({
                "id": f"trade-{i:04d}",
                "symbol": "BTC-USD" if i % 3 == 0 else ("ETH-USD" if i % 3 == 1 else "SOL-USD"),
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 0.1 + (i % 10) * 0.1,
                "price": 45000.0 + i * 10,
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "status": "filled",
                "order_type": "market" if i % 2 == 0 else "limit",
                "fees": 10.0 + i * 0.1
            })
        
        # Mock the get_trades method
        mock_trade_accessor.get_trades.return_value = large_dataset
        
        # Make request and measure time
        import time
        start_time = time.time()
        response = client.get("/api/v1/trades")
        end_time = time.time()
        
        # Verify response
        assert response.status_code == 200
        assert len(response.json()) == 1000
        
        # Check performance (should be under 200ms for processing)
        # This is a guideline and might need adjustment based on actual performance
        assert (end_time - start_time) < 0.2