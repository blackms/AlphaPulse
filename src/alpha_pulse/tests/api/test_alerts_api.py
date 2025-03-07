"""Tests for alerts API endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user
from alpha_pulse.api.data import AlertDataAccessor


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_alert_accessor():
    """Mock the AlertDataAccessor."""
    with patch("alpha_pulse.api.routers.alerts.alert_accessor") as mock:
        yield mock


@pytest.fixture
def sample_alerts():
    """Generate sample alerts data."""
    now = datetime.now()
    return [
        {
            "alert_id": "alert-001",
            "title": "Portfolio Value Drop",
            "message": "Portfolio value dropped by 5% in the last hour",
            "severity": "warning",
            "timestamp": now.isoformat(),
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_at": None,
            "source": "portfolio_monitor",
            "tags": ["portfolio", "value", "drop"]
        },
        {
            "alert_id": "alert-002",
            "title": "High Volatility Detected",
            "message": "Market volatility exceeds threshold",
            "severity": "info",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "acknowledged": True,
            "acknowledged_by": "system",
            "acknowledged_at": (now - timedelta(hours=1)).isoformat(),
            "source": "market_monitor",
            "tags": ["market", "volatility"]
        },
        {
            "alert_id": "alert-003",
            "title": "API Connection Failed",
            "message": "Failed to connect to exchange API",
            "severity": "critical",
            "timestamp": (now - timedelta(minutes=30)).isoformat(),
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_at": None,
            "source": "system_monitor",
            "tags": ["system", "api", "connection"]
        }
    ]


@pytest.fixture
def admin_user():
    """Generate admin user authentication."""
    return {
        "username": "admin",
        "role": "admin",
        "permissions": ["view_metrics", "view_alerts", "acknowledge_alerts"]
    }


@pytest.fixture
def viewer_user():
    """Generate viewer user authentication."""
    return {
        "username": "viewer",
        "role": "viewer",
        "permissions": ["view_metrics", "view_alerts"]
    }


@pytest.fixture
def auth_override():
    """Override the get_current_user dependency."""
    def _override_dependency(user):
        app.dependency_overrides[get_current_user] = lambda: user
        yield
        app.dependency_overrides = {}
    return _override_dependency


def test_get_alerts_success(client, mock_alert_accessor, sample_alerts, auth_override, admin_user):
    """Test successful alerts retrieval."""
    with auth_override(admin_user):
        # Mock the get_alerts method to return sample data
        mock_alert_accessor.get_alerts.return_value = sample_alerts
        
        # Make request
        response = client.get("/api/v1/alerts")
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_alerts
        
        # Verify the alert accessor was called with correct parameters
        mock_alert_accessor.get_alerts.assert_called_once_with(
            start_time=None,
            end_time=None,
            filters={}
        )


def test_get_alerts_with_filters(client, mock_alert_accessor, sample_alerts, auth_override, admin_user):
    """Test alerts retrieval with filters."""
    with auth_override(admin_user):
        # Filter to only return unacknowledged alerts
        filtered_alerts = [alert for alert in sample_alerts if not alert["acknowledged"]]
        mock_alert_accessor.get_alerts.return_value = filtered_alerts
        
        # Make request with filters
        response = client.get(
            "/api/v1/alerts",
            params={"acknowledged": "false", "severity": "critical"}
        )
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == filtered_alerts
        
        # Verify the alert accessor was called with correct parameters
        mock_alert_accessor.get_alerts.assert_called_once()
        call_args = mock_alert_accessor.get_alerts.call_args[1]
        assert call_args["filters"] == {"severity": "critical", "acknowledged": False}


def test_get_alerts_with_time_range(client, mock_alert_accessor, sample_alerts, auth_override, admin_user):
    """Test alerts retrieval with time range."""
    with auth_override(admin_user):
        # Mock the get_alerts method to return sample data
        mock_alert_accessor.get_alerts.return_value = sample_alerts
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        # Make request with time range
        response = client.get(
            "/api/v1/alerts",
            params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        )
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_alerts
        
        # Verify the alert accessor was called with correct parameters
        mock_alert_accessor.get_alerts.assert_called_once()
        call_args = mock_alert_accessor.get_alerts.call_args[1]
        assert isinstance(call_args["start_time"], datetime)
        assert isinstance(call_args["end_time"], datetime)


def test_get_alerts_unauthorized(client):
    """Test unauthorized access to alerts."""
    # Make request without authentication
    response = client.get("/api/v1/alerts")
    
    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


def test_get_alerts_forbidden(client, auth_override):
    """Test access with insufficient permissions."""
    # User without view_alerts permission
    user = {
        "username": "no_access",
        "role": "restricted",
        "permissions": []
    }
    
    with auth_override(user):
        # Make request
        response = client.get("/api/v1/alerts")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_get_alerts_error(client, mock_alert_accessor, auth_override, admin_user):
    """Test error handling in alerts endpoint."""
    with auth_override(admin_user):
        # Mock the get_alerts method to raise an exception
        mock_alert_accessor.get_alerts.side_effect = Exception("Database error")
        
        # Make request
        response = client.get("/api/v1/alerts")
        
        # Verify response (should return empty list on error, not fail)
        assert response.status_code == 200
        assert response.json() == []


def test_acknowledge_alert_success(client, mock_alert_accessor, auth_override, admin_user):
    """Test successful alert acknowledgment."""
    with auth_override(admin_user):
        # Mock the acknowledge_alert method
        mock_alert_accessor.acknowledge_alert.return_value = {
            "success": True,
            "alert": {
                "alert_id": "alert-001",
                "title": "Portfolio Value Drop",
                "message": "Portfolio value dropped by 5% in the last hour",
                "severity": "warning",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": True,
                "acknowledged_by": "admin",
                "acknowledged_at": datetime.now().isoformat(),
                "source": "portfolio_monitor",
                "tags": ["portfolio", "value", "drop"]
            }
        }
        
        # Make request
        response = client.post("/api/v1/alerts/alert-001/acknowledge")
        
        # Verify response
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["alert"]["acknowledged"] is True
        assert response.json()["alert"]["acknowledged_by"] == "admin"
        
        # Verify the alert accessor was called with correct parameters
        mock_alert_accessor.acknowledge_alert.assert_called_once_with(
            alert_id="alert-001",
            user="admin"
        )


def test_acknowledge_alert_not_found(client, mock_alert_accessor, auth_override, admin_user):
    """Test acknowledgment of non-existent alert."""
    with auth_override(admin_user):
        # Mock the acknowledge_alert method to return failure
        mock_alert_accessor.acknowledge_alert.return_value = {
            "success": False,
            "error": "Alert not found or already acknowledged"
        }
        
        # Make request
        response = client.post("/api/v1/alerts/nonexistent-alert/acknowledge")
        
        # Verify response
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()


def test_acknowledge_alert_unauthorized(client):
    """Test unauthorized access to acknowledge alert."""
    # Make request without authentication
    response = client.post("/api/v1/alerts/alert-001/acknowledge")
    
    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


def test_acknowledge_alert_forbidden(client, auth_override, viewer_user):
    """Test acknowledgment without proper permissions."""
    with auth_override(viewer_user):
        # Make request (viewer has view_alerts but not acknowledge_alerts)
        response = client.post("/api/v1/alerts/alert-001/acknowledge")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_acknowledge_alert_error(client, mock_alert_accessor, auth_override, admin_user):
    """Test error handling in acknowledge endpoint."""
    with auth_override(admin_user):
        # Mock the acknowledge_alert method to raise an exception
        mock_alert_accessor.acknowledge_alert.side_effect = Exception("Database error")
        
        # Make request
        response = client.post("/api/v1/alerts/alert-001/acknowledge")
        
        # Verify response
        assert response.status_code == 500
        assert "error" in response.json()