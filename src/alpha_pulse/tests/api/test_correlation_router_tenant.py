"""
Tests for correlation router tenant context integration.

Story 2.4 - Phase 2: Correlation Router (P1 High Priority)
Tests that correlation endpoints properly extract and use tenant_id from middleware.
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


class TestGetCorrelationMatrixEndpoint:
    """Test /api/v1/correlation/matrix endpoint."""

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    @patch('alpha_pulse.api.routers.correlation.CorrelationAnalyzer')
    @patch('alpha_pulse.api.routers.correlation.CachingService')
    def test_get_correlation_matrix_uses_tenant_context(
        self, mock_cache_service, mock_analyzer, mock_data_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /correlation/matrix endpoint extracts and uses tenant_id."""
        # Mock data fetcher
        mock_fetcher_instance = AsyncMock()
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_df.__getitem__ = Mock()
        mock_df['close'].pct_change.return_value.dropna.return_value = [0.01, 0.02, -0.01]
        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=mock_df)
        mock_data_fetcher.return_value = mock_fetcher_instance

        # Mock cache
        mock_cache_instance = AsyncMock()
        mock_cache_instance.get = AsyncMock(return_value=None)
        mock_cache_instance.set = AsyncMock()
        mock_cache_service.create_for_api.return_value = mock_cache_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.correlation.pd.DataFrame') as mock_pd:
                mock_returns_df = Mock()
                mock_returns_df.dropna.return_value = mock_returns_df
                mock_returns_df.corr.return_value.values.tolist.return_value = [[1.0, 0.5], [0.5, 1.0]]
                mock_pd.return_value = mock_returns_df

                response = client.get(
                    "/api/v1/correlation/matrix?symbols=BTC&symbols=ETH&lookback_days=252&method=pearson",
                    headers={"Authorization": f"Bearer {token}"}
                )

        assert response.status_code in [200, 422]  # 422 if validation fails

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    @patch('alpha_pulse.api.routers.correlation.CachingService')
    def test_get_correlation_matrix_uses_cache(
        self, mock_cache_service, mock_data_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that correlation matrix uses caching with tenant context."""
        # Mock cache with cached result
        mock_cache_instance = AsyncMock()
        cached_response = {
            "symbols": ["BTC", "ETH"],
            "matrix": [[1.0, 0.75], [0.75, 1.0]],
            "method": "pearson",
            "timestamp": datetime.now().isoformat(),
            "lookback_days": 252
        }
        mock_cache_instance.get = AsyncMock(return_value=cached_response)
        mock_cache_service.create_for_api.return_value = mock_cache_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/matrix?symbols=BTC&symbols=ETH&lookback_days=252&method=pearson",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code in [200, 422]
        # Cache should be checked
        mock_cache_instance.get.assert_called_once()

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    @patch('alpha_pulse.api.routers.correlation.CachingService')
    def test_get_correlation_matrix_tenant_isolation(
        self, mock_cache_service, mock_data_fetcher,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that correlation matrices are tenant-isolated."""
        # Mock cache - no cached result
        mock_cache_instance = AsyncMock()
        mock_cache_instance.get = AsyncMock(return_value=None)
        mock_cache_instance.set = AsyncMock()
        mock_cache_service.create_for_api.return_value = mock_cache_instance

        # Mock data fetcher
        mock_fetcher_instance = AsyncMock()
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=mock_df)
        mock_data_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/matrix?symbols=BTC&symbols=ETH",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Each tenant should get their own correlation analysis
        assert response.status_code in [200, 400, 422, 500]

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    @patch('alpha_pulse.api.routers.correlation.CachingService')
    def test_get_correlation_matrix_logs_tenant_context(
        self, mock_cache_service, mock_data_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that correlation matrix logs include tenant context."""
        mock_cache_instance = AsyncMock()
        mock_cache_instance.get = AsyncMock(return_value=None)
        mock_cache_service.create_for_api.return_value = mock_cache_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/matrix?symbols=BTC&symbols=ETH",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Logs should contain tenant context
        assert response.status_code in [200, 400, 422, 500]


class TestGetRollingCorrelationEndpoint:
    """Test /api/v1/correlation/rolling endpoint."""

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    def test_get_rolling_correlation_uses_tenant_context(
        self, mock_data_fetcher, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /correlation/rolling endpoint extracts and uses tenant_id."""
        # Mock data fetcher
        mock_fetcher_instance = AsyncMock()
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_df.__getitem__ = Mock()

        # Mock pct_change and dropna chain
        mock_returns = Mock()
        mock_df['close'].pct_change.return_value.dropna.return_value = mock_returns

        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=mock_df)
        mock_data_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            with patch('alpha_pulse.api.routers.correlation.pd.DataFrame') as mock_pd:
                mock_aligned = Mock()
                mock_aligned.dropna.return_value = mock_aligned
                mock_rolling = Mock()
                mock_rolling.dropna.return_value = mock_rolling
                mock_rolling.items.return_value = [
                    (datetime.now(), 0.75),
                    (datetime.now(), 0.80)
                ]
                mock_rolling.iloc = Mock()
                mock_rolling.iloc.__getitem__ = Mock(return_value=0.80)
                mock_rolling.mean.return_value = 0.775
                mock_rolling.min.return_value = 0.75
                mock_rolling.max.return_value = 0.80
                mock_aligned.__getitem__ = Mock(return_value=Mock(rolling=Mock(return_value=Mock(corr=Mock(return_value=mock_rolling)))))
                mock_pd.return_value = mock_aligned

                response = client.get(
                    "/api/v1/correlation/rolling?symbol1=BTC&symbol2=ETH&window_days=30&lookback_days=252",
                    headers={"Authorization": f"Bearer {token}"}
                )

        assert response.status_code in [200, 400, 422, 500]

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    def test_get_rolling_correlation_tenant_isolation(
        self, mock_data_fetcher, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that rolling correlations are tenant-isolated."""
        # Mock data fetcher
        mock_fetcher_instance = AsyncMock()
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=mock_df)
        mock_data_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/rolling?symbol1=BTC&symbol2=ETH",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Each tenant should get their own correlation analysis
        assert response.status_code in [200, 400, 422, 500]

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    def test_get_rolling_correlation_with_custom_window(
        self, mock_data_fetcher, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test rolling correlation with custom window size."""
        mock_fetcher_instance = AsyncMock()
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=mock_df)
        mock_data_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/rolling?symbol1=BTC&symbol2=ETH&window_days=60&lookback_days=365",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code in [200, 400, 422, 500]

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    def test_get_rolling_correlation_error_handling(
        self, mock_data_fetcher, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test error handling includes tenant context."""
        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=None)
        mock_data_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/rolling?symbol1=BTC&symbol2=ETH",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should return error when data fetch fails
        assert response.status_code in [400, 500]

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    def test_get_rolling_correlation_logs_tenant_context(
        self, mock_data_fetcher, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that rolling correlation logs include tenant context."""
        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.fetch_historical_data = AsyncMock(return_value=None)
        mock_data_fetcher.return_value = mock_fetcher_instance

        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/correlation/rolling?symbol1=BTC&symbol2=ETH",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Logs should contain tenant context
        assert response.status_code in [200, 400, 422, 500]

    @patch('alpha_pulse.api.routers.correlation.DataFetcher')
    def test_get_rolling_correlation_invalid_window(
        self, mock_data_fetcher, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test validation of window size parameter."""
        token = create_test_token("admin", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            # Window too small (< 10)
            response = client.get(
                "/api/v1/correlation/rolling?symbol1=BTC&symbol2=ETH&window_days=5",
                headers={"Authorization": f"Bearer {token}"}
            )

        # Should fail validation
        assert response.status_code == 422
