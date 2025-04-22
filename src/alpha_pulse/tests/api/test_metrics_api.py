"""Tests for metrics API endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

# Application imports
from alpha_pulse.api.main import app
from alpha_pulse.api.dependencies import get_current_user
from alpha_pulse.api.data import MetricsDataAccessor


@pytest.fixture
def client():
    """Return a TestClient for the API."""
    return TestClient(app)


@pytest.fixture
def mock_metrics_accessor():
    """Mock the MetricsDataAccessor."""
    from unittest.mock import patch, MagicMock, AsyncMock
    with patch("alpha_pulse.api.routers.metrics.metrics_accessor", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def sample_metrics():
    """Generate sample metrics data."""
    now = datetime.now()
    return [
        {
            "name": "portfolio_value",
            "value": 1000000.0,
            "timestamp": now.isoformat(),
            "labels": {"currency": "USD"}
        },
        {
            "name": "portfolio_return",
            "value": 0.15,
            "timestamp": now.isoformat(),
            "labels": {"period": "daily"}
        },
        {
            "name": "sharpe_ratio",
            "value": 1.8,
            "timestamp": now.isoformat(),
            "labels": {"window": "30d"}
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
        "permissions": ["view_metrics"]
    }


@pytest.fixture
def auth_override():
    """Override the get_current_user dependency."""
    @contextmanager
    def _override_dependency(user):
        app.dependency_overrides[get_current_user] = lambda: user
        try:
            yield
        finally:
            app.dependency_overrides = {}
    return _override_dependency


def test_get_metrics_success(client, mock_metrics_accessor, sample_metrics, auth_override, admin_user):
    """Test successful metrics retrieval."""
    with auth_override(admin_user):
        # Mock the get_metrics method to return sample data
        mock_metrics_accessor.get_metrics.return_value = sample_metrics
        
        # Make request
        response = client.get("/api/v1/metrics/portfolio_value")
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_metrics
        
        # Verify the metrics accessor was called with correct parameters
        mock_metrics_accessor.get_metrics.assert_called_once_with(
            metric_type="portfolio_value",
            start_time=None,
            end_time=None,
            aggregation="avg"
        )


def test_get_metrics_with_params(client, mock_metrics_accessor, sample_metrics, auth_override, admin_user):
    """Test metrics retrieval with parameters."""
    with auth_override(admin_user):
        # Mock the get_metrics method to return sample data
        mock_metrics_accessor.get_metrics.return_value = sample_metrics
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        # Make request with parameters
        response = client.get(
            f"/api/v1/metrics/portfolio_value",
            params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "aggregation": "sum"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == sample_metrics
        
        # Verify the metrics accessor was called with correct parameters
        mock_metrics_accessor.get_metrics.assert_called_once()
        call_args = mock_metrics_accessor.get_metrics.call_args[1]
        assert call_args["metric_type"] == "portfolio_value"
        assert call_args["aggregation"] == "sum"
        # Check that datetime objects were properly parsed
        assert isinstance(call_args["start_time"], datetime)
        assert isinstance(call_args["end_time"], datetime)


def test_get_metrics_unauthorized(client):
    """Test unauthorized access to metrics."""
    # Make request without authentication
    response = client.get("/api/v1/metrics/portfolio_value")
    
    # Verify response
    assert response.status_code == 401
    assert "Not authenticated" in response.json().get("detail", "")


def test_get_metrics_forbidden(client, auth_override):
    """Test access with insufficient permissions."""
    # User without view_metrics permission
    user = {
        "username": "no_access",
        "role": "restricted",
        "permissions": []
    }
    
    with auth_override(user):
        # Make request
        response = client.get("/api/v1/metrics/portfolio_value")
        
        # Verify response
        assert response.status_code == 403
        assert "Not authorized" in response.json().get("detail", "")


def test_get_metrics_error(client, mock_metrics_accessor, auth_override, admin_user):
    """Test error handling in metrics endpoint."""
    with auth_override(admin_user):
        # Mock the get_metrics method to raise an exception
        mock_metrics_accessor.get_metrics.side_effect = Exception("Database error")
        
        # Make request
        response = client.get("/api/v1/metrics/portfolio_value")
        
        # Verify response (should return empty list on error, not fail)
        assert response.status_code == 200
        assert response.json() == []


def test_get_latest_metrics_success(client, mock_metrics_accessor, auth_override, admin_user):
    """Test successful retrieval of latest metrics."""
    with auth_override(admin_user):
        # Mock latest metrics data
        latest_metrics = {
            "portfolio_value": {
                "value": 1050000.0,
                "timestamp": datetime.now().isoformat(),
                "labels": {"currency": "USD"}
            },
            "sharpe_ratio": {
                "value": 1.9,
                "timestamp": datetime.now().isoformat(),
                "labels": {"window": "30d"}
            }
        }
        
        # Mock the get_latest_metrics method
        mock_metrics_accessor.get_latest_metrics.return_value = latest_metrics
        
        # Make request
        response = client.get("/api/v1/metrics/performance/latest")
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == latest_metrics
        
        # Verify the metrics accessor was called correctly
        mock_metrics_accessor.get_latest_metrics.assert_called_once_with("performance")


def test_get_latest_metrics_error(client, mock_metrics_accessor, auth_override, admin_user):
    """Test error handling in latest metrics endpoint."""
    with auth_override(admin_user):
        # Mock the get_latest_metrics method to raise an exception
        mock_metrics_accessor.get_latest_metrics.side_effect = Exception("Database error")
        
        # Make request
        response = client.get("/api/v1/metrics/performance/latest")
        
        # Verify response (should return empty dict on error, not fail)
        assert response.status_code == 200
        assert response.json() == {}


@pytest.mark.integration
def test_metrics_caching(client, mock_metrics_accessor, sample_metrics, auth_override, admin_user):
    """Test that metrics responses are cached."""
    with auth_override(admin_user):
        # Mock the get_metrics method to return sample data
        mock_metrics_accessor.get_metrics.return_value = sample_metrics
        
        # Make first request
        response1 = client.get("/api/v1/metrics/portfolio_value")
        assert response1.status_code == 200
        
        # Make second request (should use cache)
        response2 = client.get("/api/v1/metrics/portfolio_value")
        assert response2.status_code == 200
        
        # Verify metrics accessor was called only once
        assert mock_metrics_accessor.get_metrics.call_count == 1


@pytest.mark.performance
def test_metrics_endpoint_performance(client, mock_metrics_accessor, auth_override, admin_user):
    """Test performance of metrics endpoint with large datasets."""
    with auth_override(admin_user):
        # Generate large dataset (e.g., 1000 data points)
        large_dataset = []
        now = datetime.now()
        for i in range(1000):
            large_dataset.append({
                "name": "portfolio_value",
                "value": 1000000.0 + i * 1000,
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "labels": {"currency": "USD"}
            })
        
        # Mock the get_metrics method
        mock_metrics_accessor.get_metrics.return_value = large_dataset
        
        # Make request and measure time
        import time
        start_time = time.time()
        response = client.get("/api/v1/metrics/portfolio_value")
        end_time = time.time()
        
        # Verify response
        assert response.status_code == 200
        assert len(response.json()) == 1000
        
        # Check performance (should be under 200ms for processing)
        # This is a guideline and might need adjustment based on actual performance
        assert (end_time - start_time) < 0.2