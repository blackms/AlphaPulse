"""Tests for portfolio API endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user, get_portfolio_accessor
from alpha_pulse.api.data import PortfolioDataAccessor


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_portfolio_accessor():
    """Override the get_portfolio_accessor dependency with a mock."""
    mock_accessor_instance = MagicMock(spec=PortfolioDataAccessor)
    original_override = app.dependency_overrides.get(get_portfolio_accessor)
    app.dependency_overrides[get_portfolio_accessor] = lambda: mock_accessor_instance
    try:
        yield mock_accessor_instance
    finally:
        if original_override is None:
            del app.dependency_overrides[get_portfolio_accessor]
        else:
            app.dependency_overrides[get_portfolio_accessor] = original_override


@pytest.fixture
def sample_portfolio():
    """Generate sample portfolio data."""
    now = datetime.now()
    return {
        "total_value": 1250000.0,
        "cash": 250000.0,
        "positions": [
            {
                "symbol": "BTC-USD",
                "quantity": 5.0,
                "entry_price": 40000.0,
                "current_price": 45000.0,
                "value": 225000.0,
                "pnl": 25000.0,
                "pnl_percentage": 12.5
            },
            {
                "symbol": "ETH-USD",
                "quantity": 50.0,
                "entry_price": 2800.0,
                "current_price": 3000.0,
                "value": 150000.0,
                "pnl": 10000.0,
                "pnl_percentage": 7.14
            },
            {
                "symbol": "SOL-USD",
                "quantity": 1000.0,
                "entry_price": 100.0,
                "current_price": 125.0,
                "value": 125000.0,
                "pnl": 25000.0,
                "pnl_percentage": 25.0
            }
        ],
        "metrics": {
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.2,
            "max_drawdown": 0.15,
            "volatility": 0.25,
            "return_since_inception": 0.35
        }
    }


@pytest.fixture
def sample_portfolio_history():
    """Generate sample portfolio history data."""
    now = datetime.now()
    history = []
    
    for i in range(30):
        timestamp = now - timedelta(days=i)
        history.append({
            "timestamp": timestamp.isoformat(),
            "total_value": 1000000.0 + i * 10000.0,
            "cash": 200000.0 - i * 5000.0,
            "positions_value": 800000.0 + i * 15000.0
        })
    
    return history


@pytest.fixture
def admin_user():
    """Generate admin user authentication."""
    return {
        "username": "admin",
        "role": "admin",
        "permissions": ["view_portfolio", "view_metrics", "view_alerts"]
    }


@pytest.fixture
def viewer_user():
    """Generate viewer user authentication."""
    return {
        "username": "viewer",
        "role": "viewer",
        "permissions": ["view_portfolio"]
    }


@pytest.fixture
def auth_override():
    """Provide a context manager factory for overriding get_current_user."""
    @contextmanager
    def _override_dependency(user):
        original_override = app.dependency_overrides.get(get_current_user)
        app.dependency_overrides[get_current_user] = lambda: user
        try:
            yield
        finally:
            if original_override is None:
                del app.dependency_overrides[get_current_user]
            else:
                app.dependency_overrides[get_current_user] = original_override
    return _override_dependency


def test_get_portfolio_success(client, mock_portfolio_accessor, sample_portfolio, auth_override, admin_user):
    """Test successful portfolio retrieval."""
    with auth_override(admin_user):
        # Mock the get_portfolio method to return sample data
        mock_portfolio_accessor.get_portfolio.return_value = sample_portfolio
        
        # Make request
        response = client.get("/api/v1/portfolio", params={"refresh": "false"})
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_portfolio
        
        # Verify the portfolio accessor was called with correct parameters
        mock_portfolio_accessor.get_portfolio.assert_called_once_with(include_history=False)


def test_get_portfolio_with_history(client, mock_portfolio_accessor, sample_portfolio, sample_portfolio_history, auth_override, admin_user):
    """Test portfolio retrieval with history."""
    with auth_override(admin_user):
        # Create portfolio with history
        portfolio_with_history = sample_portfolio.copy()
        portfolio_with_history["history"] = sample_portfolio_history
        
        # Mock the get_portfolio method
        mock_portfolio_accessor.get_portfolio.return_value = portfolio_with_history
        
        # Make request with include_history parameter
        response = client.get("/api/v1/portfolio", params={"include_history": "true", "refresh": "false"})
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == portfolio_with_history
        assert "history" in response.json()
        assert len(response.json()["history"]) == 30
        
        # Verify the portfolio accessor was called with correct parameters
        mock_portfolio_accessor.get_portfolio.assert_called_once_with(include_history=True)


def test_get_portfolio_unauthorized(client):
    """Test unauthorized access to portfolio."""
    # Make request without authentication
    response = client.get("/api/v1/portfolio")
    
    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


def test_get_portfolio_forbidden(client, auth_override):
    """Test access with insufficient permissions."""
    # User without view_portfolio permission
    user = {
        "username": "no_access",
        "role": "restricted",
        "permissions": []
    }
    
    with auth_override(user):
        # Make request
        response = client.get("/api/v1/portfolio")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_get_portfolio_error(client, mock_portfolio_accessor, auth_override, admin_user):
    """Test error handling in portfolio endpoint."""
    with auth_override(admin_user):
        # Mock the get_portfolio method to raise an exception
        mock_portfolio_accessor.get_portfolio.side_effect = Exception("Database error")
        
        # Make request
        response = client.get("/api/v1/portfolio", params={"refresh": "false"})
        
        # Verify response (should return error object, not fail)
        assert response.status_code == 200
        assert "error" in response.json()
        assert response.json()["total_value"] == 0
        assert response.json()["cash"] == 0
        assert response.json()["positions"] == []


@pytest.mark.performance
def test_portfolio_endpoint_performance(client, mock_portfolio_accessor, sample_portfolio, auth_override, admin_user):
    """Test performance of portfolio endpoint with large position list."""
    with auth_override(admin_user):
        # Create portfolio with many positions
        large_portfolio = sample_portfolio.copy()
        large_portfolio["positions"] = []
        
        # Add 1000 positions
        for i in range(1000):
            large_portfolio["positions"].append({
                "symbol": f"TOKEN-{i}",
                "quantity": 100.0 + i,
                "entry_price": 10.0,
                "current_price": 12.0,
                "value": (100.0 + i) * 12.0,
                "pnl": (100.0 + i) * 2.0,
                "pnl_percentage": 20.0
            })
        
        # Mock the get_portfolio method
        mock_portfolio_accessor.get_portfolio.return_value = large_portfolio
        
        # Make request and measure time
        import time
        start_time = time.time()
        response = client.get("/api/v1/portfolio", params={"refresh": "false"})
        end_time = time.time()
        
        # Verify response
        assert response.status_code == 200
        assert len(response.json()["positions"]) == 1000
        
        # Check performance (should be under 200ms for processing)
        # This is a guideline and might need adjustment based on actual performance
        assert (end_time - start_time) < 0.2