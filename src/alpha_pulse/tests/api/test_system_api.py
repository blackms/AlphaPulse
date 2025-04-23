"""Tests for system API endpoints."""
import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock, AsyncMock # Import AsyncMock

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user, get_user, get_system_accessor # Import get_user and get_system_accessor
from alpha_pulse.api.data import SystemDataAccessor # Import SystemDataAccessor for type hinting


@pytest.fixture
def client(request):
    """Return a TestClient for the API with optional dependency overrides and mocks."""
    config = getattr(request, 'param', {})
    user_fixture_name = config.get('user')
    system_accessor_mock_config = config.get('system_accessor_mock')

    user_data = None
    if user_fixture_name:
        # Get the actual user dictionary from the fixture
        user_data = request.getfixturevalue(user_fixture_name)
        # Temporarily override the get_user dependency
        app.dependency_overrides[get_user] = lambda: user_data

    # Override get_system_accessor dependency if configured
    if system_accessor_mock_config:
        mock_system_accessor_instance = AsyncMock(spec=SystemDataAccessor)

        # Make get_system_metrics an AsyncMock
        mock_system_accessor_instance.get_system_metrics = AsyncMock()

        if 'return_value' in system_accessor_mock_config:
            # Get the actual sample_system_metrics data from the fixture
            metrics_data = request.getfixturevalue(system_accessor_mock_config['return_value'])
            mock_system_accessor_instance.get_system_metrics.return_value = metrics_data
        elif 'side_effect' in system_accessor_mock_config:
            mock_system_accessor_instance.get_system_metrics.side_effect = system_accessor_mock_config['side_effect']

        app.dependency_overrides[get_system_accessor] = lambda: mock_system_accessor_instance


    # Create and yield the TestClient, setting raise_server_exceptions=False
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    # Clean up dependency overrides
    if user_fixture_name:
        app.dependency_overrides.pop(get_user, None)
    if system_accessor_mock_config:
        app.dependency_overrides.pop(get_system_accessor, None)


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


# Removed mock_get_system_accessor fixture


@pytest.mark.parametrize("client", [{"user": "admin_user", "system_accessor_mock": {"return_value": "sample_system_metrics"}}], indirect=True)
def test_get_system_metrics_success(client, sample_system_metrics):
    """Test successful system metrics retrieval."""
    # Make request
    response = client.get("/api/v1/system")

    # Verify response
    assert response.status_code == 200
    assert response.json() == sample_system_metrics


@pytest.mark.parametrize("client", [{"user": "operator_user", "system_accessor_mock": {"return_value": "sample_system_metrics"}}], indirect=True)
def test_get_system_metrics_operator(client, sample_system_metrics):
    """Test system metrics retrieval by operator."""
    # Make request
    response = client.get("/api/v1/system")

    # Verify response
    assert response.status_code == 200
    assert response.json() == sample_system_metrics


def test_get_system_metrics_unauthorized(client):
    """Test unauthorized access to system metrics."""
    # Make request without authentication (get_user will raise HTTPException)
    response = client.get("/api/v1/system")

    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


@pytest.mark.parametrize("client", [{"user": "viewer_user"}], indirect=True)
def test_get_system_metrics_forbidden(client):
    """Test access with insufficient permissions."""
    # Make request (viewer doesn't have view_system permission, check_permission will raise HTTPException)
    response = client.get("/api/v1/system")

    # Verify response
    assert response.status_code == 403
    assert "Not authorized" in response.json().get("detail", "")


@pytest.mark.parametrize("client", [{"user": "admin_user", "system_accessor_mock": {"side_effect": Exception("System error")}}], indirect=True)
def test_get_system_metrics_error(client):
    """Test error handling in system metrics endpoint."""
    # Make request
    response = client.get("/api/v1/system")

    # Verify response (should return 500 Internal Server Error)
    assert response.status_code == 500
    assert response.json().get("detail") == "Internal server error"