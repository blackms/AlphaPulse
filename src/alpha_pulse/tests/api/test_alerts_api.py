"""Tests for alerts API endpoints."""
import pytest
from fastapi.testclient import TestClient
import asyncio
from datetime import datetime, timedelta, timezone
import json
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user, get_user, get_alert_accessor
from alpha_pulse.api.data import AlertDataAccessor


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_alert_accessor():
    """Mock the AlertDataAccessor."""
    with patch("alpha_pulse.api.dependencies.get_alert_accessor") as mock:
        yield mock


@pytest.fixture
def sample_alerts():
    """Generate sample alerts data."""
    now = datetime.now(timezone.utc)
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
        "permissions": ["view_metrics", "view_alerts", "acknowledge_alerts", "view_portfolio", "view_trades", "view_system"]
    }


@pytest.fixture
def viewer_user():
    """Generate viewer user authentication."""
    return {
        "username": "viewer",
        "role": "viewer",
        "permissions": ["view_metrics", "view_alerts"]  # No acknowledge_alerts permission
    }


# Using auth_override from conftest.py


def test_get_alerts_success(client, sample_alerts, auth_override, admin_user):
    """Test successful alerts retrieval."""
    with auth_override(admin_user):
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts.return_value = sample_alerts
        
        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
            # Make request
            response = client.get("/api/v1/alerts")

            # Verify response
            print(f"Response content: {response.text}")
            assert response.status_code == 200
            assert response.json() == sample_alerts

            # Verify the alert accessor was called with correct parameters
            mock_accessor.get_alerts.assert_called_once_with(
                start_time=None,
                end_time=None,
                filters={}
            )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)

def test_get_alerts_with_filters(client, sample_alerts, auth_override, admin_user):
    """Test alerts retrieval with filters."""
    with auth_override(admin_user):
        # Filter to only return unacknowledged alerts
        filtered_alerts = [alert for alert in sample_alerts if not alert["acknowledged"]]
        
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts.return_value = filtered_alerts

        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
            # Make request with filters
            response = client.get(
                "/api/v1/alerts",
                params={"acknowledged": "false", "severity": "critical"}
            )

            # Verify response
            assert response.status_code == 200
            # Note: We are not asserting the exact content here as the filtering logic is mocked

            # Verify the alert accessor was called with correct parameters
            mock_accessor.get_alerts.assert_called_once_with(
                start_time=None,
                end_time=None,
                filters={"acknowledged": False, "severity": "critical"}
            )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)


def test_get_alerts_with_time_range(client, sample_alerts, auth_override, admin_user):
    """Test alerts retrieval with time range."""
    with auth_override(admin_user):
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts.return_value = sample_alerts

        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
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
            # Note: We are not asserting the exact content here as the time range logic is mocked

            # Verify the alert accessor was called with correct parameters
            # We need to be careful with datetime comparison due to potential precision differences
            # Let's check if the call was made with datetime objects within a reasonable range
            mock_accessor.get_alerts.assert_called_once()
            call_args, call_kwargs = mock_accessor.get_alerts.call_args
            called_start_time = call_kwargs.get("start_time")
            called_end_time = call_kwargs.get("end_time")

            assert called_start_time is not None and called_end_time is not None
            # Allow for a small tolerance in time comparison
            assert abs((called_start_time - start_time).total_seconds()) < 1
            assert abs((called_end_time - end_time).total_seconds()) < 1
            assert call_kwargs.get("filters") == {}

        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)


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
        "permissions": []  # No permissions at all
    }
    
    with auth_override(user):
        # Make request
        response = client.get("/api/v1/alerts")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_get_alerts_error(client, auth_override, admin_user):
    """Test error handling in alerts endpoint."""
    with auth_override(admin_user):
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.get_alerts.side_effect = Exception("Database error")

        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
            # Make request
            response = client.get("/api/v1/alerts")

            # Verify response (should return empty list on error, not fail)
            assert response.status_code == 200
            assert response.json() == []
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)


def test_acknowledge_alert_success(client, auth_override, admin_user):
    """Test successful alert acknowledgment."""
    with auth_override(admin_user):
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.acknowledge_alert.return_value = {
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

        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
            # Make request
            response = client.post("/api/v1/alerts/1/acknowledge")

            # Verify response
            assert response.status_code == 200
            assert response.json()["success"] is True
            assert response.json()["alert"]["acknowledged"] is True
            assert response.json()["alert"]["acknowledged_by"] == "admin"

            # Verify the alert accessor was called with correct parameters
            mock_accessor.acknowledge_alert.assert_called_once_with(
                alert_id=1,
                user="admin"
            )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)


def test_acknowledge_alert_not_found(client, auth_override, admin_user):
    """Test acknowledgment of non-existent alert."""
    with auth_override(admin_user):
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.acknowledge_alert.return_value = {
            "success": False,
            "error": "Alert not found or already acknowledged"
        }

        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
            # Make request
            response = client.post("/api/v1/alerts/999/acknowledge")

            # Verify response
            assert response.status_code == 404
            assert "not found" in response.json().get("detail", "").lower()
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)


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
        response = client.post("/api/v1/alerts/1/acknowledge")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_acknowledge_alert_error(client, auth_override, admin_user):
    """Test error handling in acknowledge endpoint."""
    with auth_override(admin_user):
        # Mock the AlertDataAccessor dependency
        mock_accessor = AsyncMock()
        mock_accessor.acknowledge_alert.side_effect = Exception("Database error")

        # Override the dependency for this test
        app.dependency_overrides[get_alert_accessor] = lambda: mock_accessor

        try:
            # Make request
            response = client.post("/api/v1/alerts/1/acknowledge")

            # Verify response
            assert response.status_code == 500
            assert "Error: Database error" in response.json()["detail"]
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_alert_accessor, None)