"""Tests for system API endpoints."""
import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user
from alpha_pulse.api.data import SystemDataAccessor


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_system_accessor():
    """Mock the SystemDataAccessor."""
    with patch("alpha_pulse.api.routers.system.system_accessor") as mock:
        yield mock


@pytest.fixture
def sample_system_metrics():
    """Generate sample system metrics data."""
    return {
        "cpu": {
            "usage_percent": 45.2,
            "cores": 8
        },
        "memory": {
            "total_mb": 16384,
            "used_mb": 8192,
            "percent": 50.0
        },
        "disk": {
            "total_gb": 500,
            "used_gb": 250,
            "percent": 50.0
        },
        "process": {
            "pid": 12345,
            "memory_mb": 512,
            "threads": 16,
            "uptime_seconds": 86400
        }
    }


@pytest.fixture
def admin_user():
    """Generate admin user authentication."""
    return {
        "username": "admin",
        "role": "admin",
        "permissions": ["view_system", "view_metrics", "view_alerts"]
    }


@pytest.fixture
def operator_user():
    """Generate operator user authentication."""
    return {
        "username": "operator",
        "role": "operator",
        "permissions": ["view_system"]
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


def test_get_system_metrics_success(client, mock_system_accessor, sample_system_metrics, auth_override, admin_user):
    """Test successful system metrics retrieval."""
    with auth_override(admin_user):
        # Mock the get_system_metrics method to return sample data
        mock_system_accessor.get_system_metrics.return_value = sample_system_metrics
        
        # Make request
        response = client.get("/api/v1/system")
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_system_metrics
        
        # Verify the system accessor was called
        mock_system_accessor.get_system_metrics.assert_called_once()


def test_get_system_metrics_operator(client, mock_system_accessor, sample_system_metrics, auth_override, operator_user):
    """Test system metrics retrieval by operator."""
    with auth_override(operator_user):
        # Mock the get_system_metrics method to return sample data
        mock_system_accessor.get_system_metrics.return_value = sample_system_metrics
        
        # Make request
        response = client.get("/api/v1/system")
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_system_metrics


def test_get_system_metrics_unauthorized(client):
    """Test unauthorized access to system metrics."""
    # Make request without authentication
    response = client.get("/api/v1/system")
    
    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


def test_get_system_metrics_forbidden(client, auth_override, viewer_user):
    """Test access with insufficient permissions."""
    with auth_override(viewer_user):
        # Make request (viewer doesn't have view_system permission)
        response = client.get("/api/v1/system")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_get_system_metrics_error(client, mock_system_accessor, auth_override, admin_user):
    """Test error handling in system metrics endpoint."""
    with auth_override(admin_user):
        # Mock the get_system_metrics method to raise an exception
        mock_system_accessor.get_system_metrics.side_effect = Exception("System error")
        
        # Make request
        response = client.get("/api/v1/system")
        
        # Verify response (should return error object, not fail)
        assert response.status_code == 200
        assert "error" in response.json()