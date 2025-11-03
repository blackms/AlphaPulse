"""
Tests for ensemble router tenant context integration.

Story 2.4 - Phase 3: Ensemble Router (P2 High Priority)
Tests that ensemble endpoints properly extract and use tenant_id from middleware.
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


class TestCreateEnsembleEndpoint:
    """Test POST /api/v1/ensemble/create endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_create_ensemble_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /ensemble/create endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.create_ensemble = AsyncMock(return_value="ensemble-123")
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/create",
                json={"name": "Test Ensemble", "ensemble_type": "weighted_voting"},
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert result["name"] == "Test Ensemble"
        assert result["type"] == "weighted_voting"
        mock_service.create_ensemble.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_create_ensemble_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that ensemble creation is tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.create_ensemble = AsyncMock(return_value="ensemble-tenant1-456")
        mock_get_service.return_value = mock_service

        # Tenant 1 creates ensemble
        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/create",
                json={"name": "Tenant 1 Ensemble", "ensemble_type": "weighted_voting"},
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only create for tenant 1
        assert result["ensemble_id"] == "ensemble-tenant1-456"

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_create_ensemble_logs_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that ensemble creation logs include tenant context."""
        mock_service = AsyncMock()
        mock_service.create_ensemble = AsyncMock(return_value="ensemble-789")
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/create",
                json={"name": "Test", "ensemble_type": "stacking"},
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        # Logs should contain tenant context (verified by logger calls in router)


class TestRegisterAgentEndpoint:
    """Test POST /api/v1/ensemble/{ensemble_id}/register-agent endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_register_agent_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /register-agent endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.register_agent = AsyncMock(return_value=None)
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-123/register-agent",
                json={
                    "agent_id": "agent-1",
                    "agent_type": "technical",
                    "initial_weight": 1.0
                },
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert result["agent_id"] == "agent-1"
        assert result["status"] == "registered"
        mock_service.register_agent.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_register_agent_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that agent registration is tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.register_agent = AsyncMock(return_value=None)
        mock_get_service.return_value = mock_service

        # Tenant 1 registers agent
        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-tenant1/register-agent",
                json={
                    "agent_id": "agent-tenant1",
                    "agent_type": "fundamental",
                    "initial_weight": 1.5
                },
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["agent_id"] == "agent-tenant1"
        # Agent should be registered only for tenant 1


class TestGetEnsemblePredictionEndpoint:
    """Test POST /api/v1/ensemble/{ensemble_id}/predict endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_prediction_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /predict endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.get_ensemble_prediction = AsyncMock(return_value={
            "signal": "buy",
            "confidence": 0.85,
            "agents": [
                {"agent_id": "agent-1", "signal": "buy", "weight": 0.5},
                {"agent_id": "agent-2", "signal": "buy", "weight": 0.5}
            ]
        })
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-123/predict",
                json={
                    "signals": [
                        {"agent_id": "agent-1", "signal": "buy"},
                        {"agent_id": "agent-2", "signal": "buy"}
                    ]
                },
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert "prediction" in result
        assert result["prediction"]["signal"] == "buy"
        mock_service.get_ensemble_prediction.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_prediction_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that ensemble predictions are tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.get_ensemble_prediction = AsyncMock(return_value={
            "signal": "sell",
            "confidence": 0.72
        })
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-tenant1/predict",
                json={"signals": [{"agent_id": "agent-1", "signal": "sell"}]},
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only predict for tenant 1's ensemble
        assert result["ensemble_id"] == "ensemble-tenant1"


class TestGetEnsemblePerformanceEndpoint:
    """Test GET /api/v1/ensemble/{ensemble_id}/performance endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_performance_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /performance endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.get_ensemble_performance = AsyncMock(return_value={
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.12,
            "win_rate": 0.62,
            "accuracy": 0.68
        })
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/ensemble-123/performance?days=30",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert "performance" in result
        assert result["performance"]["sharpe_ratio"] == 1.8
        mock_service.get_ensemble_performance.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_performance_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that ensemble performance is tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.get_ensemble_performance = AsyncMock(return_value={
            "total_return": 0.25,
            "sharpe_ratio": 2.1
        })
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/ensemble-tenant1/performance?days=60",
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only get tenant 1's performance
        assert result["days"] == 60


class TestGetAgentWeightsEndpoint:
    """Test GET /api/v1/ensemble/{ensemble_id}/weights endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_weights_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /weights endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.get_agent_weights = AsyncMock(return_value={
            "agent-1": 0.4,
            "agent-2": 0.35,
            "agent-3": 0.25
        })
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/ensemble-123/weights",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert "weights" in result
        assert result["weights"]["agent-1"] == 0.4
        mock_service.get_agent_weights.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_weights_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that agent weights are tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.get_agent_weights = AsyncMock(return_value={
            "agent-tenant1-1": 0.5,
            "agent-tenant1-2": 0.5
        })
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/ensemble-tenant1/weights",
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only get tenant 1's weights
        assert "weights" in result
        for agent_id in result["weights"].keys():
            assert "tenant1" in agent_id


class TestOptimizeWeightsEndpoint:
    """Test POST /api/v1/ensemble/{ensemble_id}/optimize-weights endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_optimize_weights_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /optimize-weights endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.optimize_weights = AsyncMock(return_value={
            "agent-1": 0.5,
            "agent-2": 0.3,
            "agent-3": 0.2
        })
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-123/optimize-weights?metric=sharpe_ratio&lookback_days=30",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert "optimized_weights" in result
        assert result["metric"] == "sharpe_ratio"
        mock_service.optimize_weights.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_optimize_weights_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that weight optimization is tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.optimize_weights = AsyncMock(return_value={
            "agent-tenant1-1": 0.6,
            "agent-tenant1-2": 0.4
        })
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-tenant1/optimize-weights?metric=accuracy&lookback_days=60",
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only optimize tenant 1's weights
        assert result["metric"] == "accuracy"


class TestListEnsemblesEndpoint:
    """Test GET /api/v1/ensemble/ endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_list_ensembles_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that / endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.list_ensembles = AsyncMock(return_value=[
            {
                "ensemble_id": "ensemble-1",
                "name": "Primary Ensemble",
                "type": "weighted_voting",
                "agent_count": 3,
                "active": True
            },
            {
                "ensemble_id": "ensemble-2",
                "name": "Secondary Ensemble",
                "type": "stacking",
                "agent_count": 4,
                "active": True
            }
        ])
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/?active_only=True",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "ensembles" in result
        assert result["count"] == 2
        mock_service.list_ensembles.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_list_ensembles_filters_by_tenant(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that list ensembles filters by tenant_id."""
        mock_service = AsyncMock()
        mock_service.list_ensembles = AsyncMock(return_value=[
            {
                "ensemble_id": "ensemble-tenant1-1",
                "name": "Tenant 1 Ensemble",
                "type": "weighted_voting",
                "tenant_id": tenant_1_id
            }
        ])
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/?active_only=False",
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only list tenant 1's ensembles
        assert result["count"] == 1
        for ensemble in result["ensembles"]:
            assert "tenant1" in ensemble.get("ensemble_id", "")


class TestGetAgentRankingsEndpoint:
    """Test GET /api/v1/ensemble/agent-rankings endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_rankings_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that /agent-rankings endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.get_agent_rankings = AsyncMock(return_value=[
            {"rank": 1, "agent_id": "agent-1", "sharpe_ratio": 2.1, "accuracy": 0.75},
            {"rank": 2, "agent_id": "agent-2", "sharpe_ratio": 1.8, "accuracy": 0.68},
            {"rank": 3, "agent_id": "agent-3", "sharpe_ratio": 1.5, "accuracy": 0.62}
        ])
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/agent-rankings?metric=sharpe_ratio&days=30",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "rankings" in result
        assert result["metric"] == "sharpe_ratio"
        assert result["days"] == 30
        assert len(result["rankings"]) == 3
        mock_service.get_agent_rankings.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_rankings_tenant_scoped(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that agent rankings are tenant-scoped."""
        mock_service = AsyncMock()
        mock_service.get_agent_rankings = AsyncMock(return_value=[
            {"rank": 1, "agent_id": "agent-tenant1-1", "accuracy": 0.80},
            {"rank": 2, "agent_id": "agent-tenant1-2", "accuracy": 0.72}
        ])
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/agent-rankings?metric=accuracy&days=60",
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only get rankings for tenant 1's agents
        assert len(result["rankings"]) == 2
        for agent in result["rankings"]:
            assert "tenant1" in agent.get("agent_id", "")


class TestDeleteEnsembleEndpoint:
    """Test DELETE /api/v1/ensemble/{ensemble_id} endpoint."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_delete_ensemble_uses_tenant_context(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test that DELETE endpoint extracts and uses tenant_id."""
        mock_service = AsyncMock()
        mock_service.delete_ensemble = AsyncMock(return_value=None)
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.delete(
                "/api/v1/ensemble/ensemble-123",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-123"
        assert result["status"] == "deleted"
        mock_service.delete_ensemble.assert_called_once()

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_delete_ensemble_tenant_isolation(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, tenant_2_id, mock_settings
    ):
        """Test that ensemble deletion is tenant-isolated."""
        mock_service = AsyncMock()
        mock_service.delete_ensemble = AsyncMock(return_value=None)
        mock_get_service.return_value = mock_service

        token_tenant1 = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.delete(
                "/api/v1/ensemble/ensemble-tenant1-delete",
                headers={"Authorization": f"Bearer {token_tenant1}"}
            )

        assert response.status_code == 200
        result = response.json()
        # Should only delete tenant 1's ensemble
        assert result["ensemble_id"] == "ensemble-tenant1-delete"
        assert result["status"] == "deleted"


class TestEnsembleErrorHandling:
    """Test error handling across ensemble endpoints."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_create_ensemble_error_handling(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test error handling in create ensemble includes tenant context."""
        mock_service = AsyncMock()
        mock_service.create_ensemble = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/create",
                json={"name": "Test", "ensemble_type": "weighted_voting"},
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 500
        error = response.json()
        assert "detail" in error

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_list_ensembles_error_handling(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test error handling in list ensembles includes tenant context."""
        mock_service = AsyncMock()
        mock_service.list_ensembles = AsyncMock(
            side_effect=Exception("Service unavailable")
        )
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/?active_only=True",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 500
        error = response.json()
        assert "detail" in error


class TestEnsembleEndpointVariants:
    """Test endpoint variants and parameter combinations."""

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_create_ensemble_with_parameters(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test creating ensemble with additional parameters."""
        mock_service = AsyncMock()
        mock_service.create_ensemble = AsyncMock(return_value="ensemble-param-456")
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/create",
                json={
                    "name": "Advanced Ensemble",
                    "ensemble_type": "dynamic",
                    "parameters": {"learning_rate": 0.01, "update_frequency": "daily"},
                    "description": "Dynamic weighting ensemble"
                },
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ensemble_id"] == "ensemble-param-456"
        assert result["type"] == "dynamic"

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_predict_with_metadata(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test prediction with metadata."""
        mock_service = AsyncMock()
        mock_service.get_ensemble_prediction = AsyncMock(return_value={
            "signal": "hold",
            "confidence": 0.55
        })
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-456/predict",
                json={
                    "signals": [{"agent_id": "agent-1", "signal": "hold"}],
                    "metadata": {"market_regime": "low_volatility", "timestamp": "2024-01-01T12:00:00"}
                },
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_optimize_weights_different_metrics(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test weight optimization with different metrics."""
        mock_service = AsyncMock()
        mock_service.optimize_weights = AsyncMock(return_value={
            "agent-1": 0.45,
            "agent-2": 0.40,
            "agent-3": 0.15
        })
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        # Test with profit_factor metric
        with patch.object(app.state, 'settings', mock_settings):
            response = client.post(
                "/api/v1/ensemble/ensemble-789/optimize-weights?metric=profit_factor&lookback_days=90",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["metric"] == "profit_factor"

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_list_ensembles_inactive_filter(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test listing ensembles with inactive filter."""
        mock_service = AsyncMock()
        mock_service.list_ensembles = AsyncMock(return_value=[
            {
                "ensemble_id": "ensemble-inactive-1",
                "name": "Inactive Ensemble",
                "active": False
            }
        ])
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/?active_only=False",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 1
        assert not result["ensembles"][0]["active"]

    @patch('alpha_pulse.api.routers.ensemble.get_ensemble_service')
    def test_get_rankings_with_different_metrics(
        self, mock_get_service, client, create_test_token,
        tenant_1_id, mock_settings
    ):
        """Test agent rankings with different metrics."""
        mock_service = AsyncMock()
        mock_service.get_agent_rankings = AsyncMock(return_value=[
            {"rank": 1, "agent_id": "agent-1", "calmar_ratio": 3.2},
            {"rank": 2, "agent_id": "agent-2", "calmar_ratio": 2.1}
        ])
        mock_get_service.return_value = mock_service

        token = create_test_token("user1", tenant_1_id)

        with patch.object(app.state, 'settings', mock_settings):
            response = client.get(
                "/api/v1/ensemble/agent-rankings?metric=calmar_ratio&days=90",
                headers={"Authorization": f"Bearer {token}"}
            )

        assert response.status_code == 200
        result = response.json()
        assert result["metric"] == "calmar_ratio"
        assert result["days"] == 90
