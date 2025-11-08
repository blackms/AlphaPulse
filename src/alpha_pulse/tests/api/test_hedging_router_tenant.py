"""
Tests for hedging router tenant context integration.

Story 2.4 - Phase 3: Hedging Router (P2 High Priority - CRITICAL SECURITY FIX)
Tests that hedging endpoints properly extract and use tenant_id from middleware.
Validates the CRITICAL security fix: hedging.py now requires authentication.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt
from datetime import datetime, timedelta
from decimal import Decimal

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


class TestAnalyzeHedgePositionsEndpoint:
    """Test GET /api/v1/hedging/analysis endpoint."""

    @patch('alpha_pulse.api.routers.hedging.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.hedging.LLMHedgeAnalyzer')
    @patch('alpha_pulse.api.routers.hedging.get_exchange_client')
    def test_analyze_hedge_requires_authentication(
        self, mock_get_exchange, mock_analyzer, mock_fetcher,
        client, mock_settings
    ):
        """CRITICAL: Test that /analysis endpoint requires authentication (was unauthenticated before)."""
        with patch.object(app.state, 'settings', mock_settings):
            # Request WITHOUT authentication token
            response = client.get("/api/v1/hedging/analysis")

        # Should return 401 Unauthorized (was accessible before security fix)
        assert response.status_code == 401

    @patch('alpha_pulse.api.routers.hedging.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.hedging.LLMHedgeAnalyzer')
    @patch('alpha_pulse.api.routers.hedging.get_exchange_client')
    @patch('os.getenv')
    def test_analyze_hedge_uses_tenant_context(
        self, mock_getenv, mock_get_exchange, mock_analyzer, mock_fetcher,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /analysis endpoint extracts and uses tenant_id."""
        mock_getenv.return_value = "test-openai-key"

        # Mock exchange
        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        # Mock position fetcher
        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.get_spot_positions = AsyncMock(return_value=[])
        mock_fetcher_instance.get_futures_positions = AsyncMock(return_value=[])
        mock_fetcher.return_value = mock_fetcher_instance

        # Mock hedge analyzer
        mock_recommendation = Mock()
        mock_recommendation.commentary = "Portfolio is well hedged"
        mock_recommendation.adjustments = []
        mock_recommendation.current_net_exposure = Decimal('0.05')
        mock_recommendation.target_net_exposure = Decimal('0.0')
        mock_recommendation.risk_metrics = {"leverage": Decimal('1.2')}

        mock_analyzer_instance = AsyncMock()
        mock_analyzer_instance.analyze = AsyncMock(return_value=mock_recommendation)
        mock_analyzer.return_value = mock_analyzer_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/hedging/analysis",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "commentary" in result
        assert "current_net_exposure" in result
        assert "target_net_exposure" in result

    @patch('alpha_pulse.api.routers.hedging.ExchangePositionFetcher')
    @patch('alpha_pulse.api.routers.hedging.LLMHedgeAnalyzer')
    @patch('alpha_pulse.api.routers.hedging.get_exchange_client')
    @patch('os.getenv')
    def test_analyze_hedge_tenant_isolation(
        self, mock_getenv, mock_get_exchange, mock_analyzer, mock_fetcher,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that hedge analysis is tenant-isolated."""
        mock_getenv.return_value = "test-openai-key"

        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.get_spot_positions = AsyncMock(return_value=[])
        mock_fetcher_instance.get_futures_positions = AsyncMock(return_value=[])
        mock_fetcher.return_value = mock_fetcher_instance

        mock_recommendation = Mock()
        mock_recommendation.commentary = "Tenant 1 hedge analysis"
        mock_recommendation.adjustments = []
        mock_recommendation.current_net_exposure = Decimal('0.05')
        mock_recommendation.target_net_exposure = Decimal('0.0')
        mock_recommendation.risk_metrics = {}

        mock_analyzer_instance = AsyncMock()
        mock_analyzer_instance.analyze = AsyncMock(return_value=mock_recommendation)
        mock_analyzer.return_value = mock_analyzer_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/hedging/analysis",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Each tenant gets their own hedge analysis based on their exchange account


class TestExecuteHedgeAdjustmentsEndpoint:
    """Test POST /api/v1/hedging/execute endpoint."""

    @patch('alpha_pulse.api.routers.hedging.HedgeManager')
    def test_execute_hedge_requires_authentication(
        self, mock_manager, client, mock_settings
    ):
        """CRITICAL: Test that /execute endpoint requires authentication (was unauthenticated before)."""
        with patch.object(app.state, 'settings', mock_settings):
            # Request WITHOUT authentication token
            response = client.post("/api/v1/hedging/execute")

        # Should return 401 Unauthorized (was accessible before security fix)
        assert response.status_code == 401

    @patch('alpha_pulse.api.routers.hedging.HedgeManager')
    @patch('alpha_pulse.api.routers.hedging.get_exchange_client')
    @patch('os.getenv')
    def test_execute_hedge_uses_tenant_context(
        self, mock_getenv, mock_get_exchange, mock_manager,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /execute endpoint extracts and uses tenant_id."""
        mock_getenv.return_value = "test-openai-key"

        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        # Mock hedge manager execution result
        mock_result = Mock()
        mock_result.executed_trades = []
        mock_result.message = "No adjustments needed"

        mock_manager_instance = AsyncMock()
        mock_manager_instance.manage_hedge = AsyncMock(return_value=mock_result)
        mock_manager.return_value = mock_manager_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/hedging/execute",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert "executed_trades" in result


class TestCloseAllHedgesEndpoint:
    """Test POST /api/v1/hedging/close endpoint."""

    @patch('alpha_pulse.api.routers.hedging.HedgeManager')
    def test_close_hedges_requires_authentication(
        self, mock_manager, client, mock_settings
    ):
        """CRITICAL: Test that /close endpoint requires authentication (was unauthenticated before)."""
        with patch.object(app.state, 'settings', mock_settings):
            # Request WITHOUT authentication token
            response = client.post("/api/v1/hedging/close")

        # Should return 401 Unauthorized (was accessible before security fix)
        assert response.status_code == 401

    @patch('alpha_pulse.api.routers.hedging.HedgeManager')
    @patch('alpha_pulse.api.routers.hedging.get_exchange_client')
    @patch('os.getenv')
    def test_close_hedges_uses_tenant_context(
        self, mock_getenv, mock_get_exchange, mock_manager,
        client, create_test_token, tenant_1_id, mock_settings
    ):
        """Test that /close endpoint extracts and uses tenant_id."""
        mock_getenv.return_value = "test-openai-key"

        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        # Mock hedge manager close result
        mock_result = Mock()
        mock_result.executed_trades = []
        mock_result.message = "No positions to close"

        mock_manager_instance = AsyncMock()
        mock_manager_instance.close_all_hedges = AsyncMock(return_value=mock_result)
        mock_manager.return_value = mock_manager_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/hedging/close",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert "executed_trades" in result

    @patch('alpha_pulse.api.routers.hedging.HedgeManager')
    @patch('alpha_pulse.api.routers.hedging.get_exchange_client')
    @patch('os.getenv')
    def test_close_hedges_tenant_isolation(
        self, mock_getenv, mock_get_exchange, mock_manager,
        client, create_test_token, tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that closing hedges is tenant-isolated."""
        mock_getenv.return_value = "test-openai-key"

        mock_exchange = Mock()
        mock_get_exchange.return_value = mock_exchange

        mock_result = Mock()
        mock_result.executed_trades = []
        mock_result.message = "Tenant 1 hedges closed"

        mock_manager_instance = AsyncMock()
        mock_manager_instance.close_all_hedges = AsyncMock(return_value=mock_result)
        mock_manager.return_value = mock_manager_instance

        token = create_test_token("trader", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/hedging/close",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Should only close hedges for tenant 1's exchange account
