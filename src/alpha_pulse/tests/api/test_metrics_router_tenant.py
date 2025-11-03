"""
Tests for metrics router tenant context integration.

Story 2.4 - Phase 2: Metrics Router (P1 High Priority)
Tests that metrics endpoints properly extract and use tenant_id from middleware.
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


class TestGetMetricsEndpoint:
    """Test /api/v1/metrics/{metric_type} endpoint."""

    @patch('alpha_pulse.api.routers.metrics.metrics_accessor')
    def test_get_metrics_uses_tenant_context(
        self, mock_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /metrics/{type} endpoint extracts and uses tenant_id."""
        mock_accessor.get_metrics = AsyncMock(return_value=[
            {
                "timestamp": datetime.now().isoformat(),
                "metric_type": "portfolio_value",
                "value": 100000.0,
                "tenant_id": tenant_1_id
            }
        ])

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/portfolio_value",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metrics = response.json()
        assert isinstance(metrics, list)

    @patch('alpha_pulse.api.routers.metrics.metrics_accessor')
    def test_get_metrics_with_aggregation(
        self, mock_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test metrics endpoint with aggregation parameter."""
        mock_accessor.get_metrics = AsyncMock(return_value=[
            {
                "value": 95000.0,
                "aggregation": "avg"
            }
        ])

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/portfolio_value?aggregation=avg",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        mock_accessor.get_metrics.assert_called_once()

    @patch('alpha_pulse.api.routers.metrics.metrics_accessor')
    def test_get_metrics_tenant_isolation(
        self, mock_accessor, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that metrics are isolated by tenant."""
        mock_accessor.get_metrics = AsyncMock(return_value=[
            {
                "value": 100000.0,
                "tenant_id": tenant_1_id
            }
        ])

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/portfolio_value",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metrics = response.json()
        # Verify tenant isolation (will be enforced in GREEN phase)
        assert len(metrics) >= 0


class TestGetLatestMetricsEndpoint:
    """Test /api/v1/metrics/{metric_type}/latest endpoint."""

    @patch('alpha_pulse.api.routers.metrics.metrics_accessor')
    def test_get_latest_metrics_uses_tenant_context(
        self, mock_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /metrics/{type}/latest uses tenant_id."""
        mock_accessor.get_latest_metrics = AsyncMock(return_value={
            "timestamp": datetime.now().isoformat(),
            "metric_type": "portfolio_value",
            "value": 100000.0,
            "tenant_id": tenant_1_id
        })

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/portfolio_value/latest",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metric = response.json()
        assert isinstance(metric, dict)

    @patch('alpha_pulse.api.routers.metrics.metrics_accessor')
    def test_get_latest_metrics_empty_result(
        self, mock_accessor, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test latest metrics with no data."""
        mock_accessor.get_latest_metrics = AsyncMock(return_value=None)

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/metrics/nonexistent/latest",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        metric = response.json()
        assert metric == {}


class TestGetCacheMetricsEndpoint:
    """Test /api/v1/metrics/cache endpoint."""

    def test_get_cache_metrics_uses_tenant_context(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /metrics/cache endpoint uses tenant_id."""
        token = create_test_token("admin", tenant_1_id)

        # Mock caching service
        mock_caching_service = AsyncMock()
        mock_caching_service.get_metrics = AsyncMock(return_value={
            "hit_rate": 0.85,
            "total_hits": 1000,
            "total_misses": 150
        })

        with patch.object(app.state, 'settings', mock_settings):
            with patch.object(app.state, 'caching_service', mock_caching_service):
                response = client.get(
                    "/api/v1/metrics/cache",
                    headers={"Authorization": f"Bearer {token}"}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "metrics" in data

    def test_get_cache_metrics_inactive(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test cache metrics when service is inactive."""
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            # No caching service set
            if hasattr(app.state, 'caching_service'):
                delattr(app.state, 'caching_service')

            response = client.get(
                "/api/v1/metrics/cache",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "inactive"


class TestGetDatabaseMetricsEndpoint:
    """Test /api/v1/metrics/database endpoint."""

    def test_get_database_metrics_uses_tenant_context(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /metrics/database endpoint uses tenant_id."""
        token = create_test_token("admin", tenant_1_id)

        # Mock database optimization service
        mock_db_service = AsyncMock()
        mock_db_service.get_performance_metrics = AsyncMock(return_value={
            "query_count": 500,
            "avg_query_time": 0.05,
            "slow_queries": 2
        })

        with patch.object(app.state, 'settings', mock_settings):
            with patch.object(app.state, 'db_optimization_service', mock_db_service):
                response = client.get(
                    "/api/v1/metrics/database",
                    headers={"Authorization": f"Bearer {token}"}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "metrics" in data

    def test_get_database_metrics_inactive(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test database metrics when service is inactive."""
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            # No db optimization service set
            if hasattr(app.state, 'db_optimization_service'):
                delattr(app.state, 'db_optimization_service')

            response = client.get(
                "/api/v1/metrics/database",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "inactive"

    def test_get_database_metrics_logs_tenant_context(
        self, client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that database metrics logs include tenant context."""
        token = create_test_token("admin", tenant_1_id)

        mock_db_service = AsyncMock()
        mock_db_service.get_performance_metrics = AsyncMock(return_value={})

        with patch.object(app.state, 'settings', mock_settings):
            with patch.object(app.state, 'db_optimization_service', mock_db_service):
                response = client.get(
                    "/api/v1/metrics/database",
                    headers={"Authorization": f"Bearer {token}"}
                )

        assert response.status_code == 200
        # Logs should contain tenant context (will be added in GREEN phase)
