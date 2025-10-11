"""
Integration tests for API startup services.
Tests the actual initialization and shutdown of performance services.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.mark.asyncio
class TestAPIStartupServices:
    """Test suite for API startup service initialization."""

    async def test_initialize_ensemble_service_uses_db_session(self):
        """Ensure ensemble service initialization obtains a DB session."""
        from pathlib import Path
        import ast

        module_path = Path("src/alpha_pulse/api/main.py")
        module_ast = ast.parse(module_path.read_text())
        function_node = next(
            node
            for node in module_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "initialize_ensemble_service"
        )

        @dataclass
        class _StubSession:
            close_called: bool = False

            def close(self) -> None:
                self.close_called = True

        @dataclass
        class _StubState:
            ensemble_service: Any | None = None
            ensemble_db_session: Any | None = None

        stub_session = _StubSession()

        mock_get_db_session = MagicMock(return_value=stub_session)
        mock_service_instance = MagicMock()
        MockEnsembleService = MagicMock(return_value=mock_service_instance)

        exec_globals = {
            "Any": __import__("typing").Any,
            "get_db_session": mock_get_db_session,
            "EnsembleService": MockEnsembleService,
        }

        exec(
            compile(
                ast.Module(body=[function_node], type_ignores=[]),
                str(module_path),
                "exec",
            ),
            exec_globals,
        )

        initialize_ensemble_service = exec_globals["initialize_ensemble_service"]

        app_state = _StubState()

        initialize_ensemble_service(app_state)

        MockEnsembleService.assert_called_once_with(stub_session)
        assert app_state.ensemble_service is mock_service_instance
        assert app_state.ensemble_db_session is stub_session
        assert stub_session.close_called is False

    async def test_caching_service_initialization(self):
        """Test CachingService initialization during startup."""
        with patch("alpha_pulse.api.main.CachingService") as MockCachingService:
            # Setup mock
            mock_service = MagicMock()
            mock_service.initialize = AsyncMock()
            mock_service.close = AsyncMock()
            mock_service.get_metrics = AsyncMock(
                return_value={
                    "hit_rate": 0.85,
                    "total_requests": 1000,
                    "cache_hits": 850,
                }
            )
            MockCachingService.create_for_trading.return_value = mock_service

            # Import and trigger startup
            from alpha_pulse.api import main

            # Simulate startup
            app = main.app

            # Create a mock startup context
            async def simulate_startup():
                # Manually call the startup event
                await main.startup_event()

                # Verify service was created and initialized
                assert MockCachingService.create_for_trading.called
                assert mock_service.initialize.called

                # Verify service is stored in app.state
                assert hasattr(app.state, "caching_service")

                # Test shutdown cleanup
                await main.shutdown_event()
                assert mock_service.close.called

            await simulate_startup()

    async def test_database_optimization_service_initialization(self):
        """Test DatabaseOptimizationService initialization during startup."""
        with patch(
            "alpha_pulse.api.main.DatabaseOptimizationService"
        ) as MockDBService, patch(
            "alpha_pulse.api.main.get_database_config"
        ) as mock_get_config:

            # Setup mocks
            mock_config = MagicMock()
            mock_config.get_connection_string.return_value = "postgresql://test"
            mock_get_config.return_value = mock_config

            mock_service = MagicMock()
            mock_service.initialize = AsyncMock()
            mock_service.start_monitoring = AsyncMock()
            mock_service.stop_monitoring = AsyncMock()
            mock_service.close = AsyncMock()
            mock_service.get_performance_metrics = AsyncMock(
                return_value={
                    "avg_query_time": 0.05,
                    "slow_queries": 2,
                    "optimizations_applied": 10,
                }
            )
            MockDBService.return_value = mock_service

            from alpha_pulse.api import main

            app = main.app

            async def simulate_startup():
                await main.startup_event()

                # Verify service was created with correct config
                MockDBService.assert_called_with(
                    connection_string="postgresql://test",
                    monitoring_interval=300,
                    enable_auto_optimization=True,
                )

                # Verify initialization and monitoring started
                assert mock_service.initialize.called
                assert mock_service.start_monitoring.called

                # Verify service is stored in app.state
                assert hasattr(app.state, "db_optimization_service")

                # Test shutdown cleanup
                await main.shutdown_event()
                assert mock_service.stop_monitoring.called
                assert mock_service.close.called

            await simulate_startup()

    async def test_data_aggregation_service_initialization(self):
        """Test DataAggregationService initialization during startup."""
        with patch("alpha_pulse.api.main.DataAggregationService") as MockDataService:
            # Setup mock
            mock_service = MagicMock()
            mock_service.initialize = AsyncMock()
            mock_service.close = AsyncMock()
            MockDataService.return_value = mock_service

            from alpha_pulse.api import main

            app = main.app

            async def simulate_startup():
                await main.startup_event()

                # Verify service was created and initialized
                assert MockDataService.called
                assert mock_service.initialize.called

                # Verify service is stored in app.state
                assert hasattr(app.state, "data_aggregation_service")

                # Test shutdown cleanup
                await main.shutdown_event()
                assert mock_service.close.called

            await simulate_startup()

    async def test_services_graceful_failure(self):
        """Test that API continues if services fail to initialize."""
        with patch("alpha_pulse.api.main.CachingService") as MockCachingService, patch(
            "alpha_pulse.api.main.logger"
        ) as mock_logger:

            # Make initialization fail
            MockCachingService.create_for_trading.side_effect = Exception(
                "Redis connection failed"
            )

            from alpha_pulse.api import main

            app = main.app

            async def simulate_startup():
                # Should not raise exception
                await main.startup_event()

                # Verify error was logged
                mock_logger.error.assert_any_call(
                    "Error initializing CachingService: Redis connection failed"
                )

                # Verify service is None
                assert app.state.caching_service is None

            await simulate_startup()

    async def test_all_services_initialization_order(self):
        """Test that all services initialize in correct order."""
        initialization_order = []

        with patch("alpha_pulse.api.main.CachingService") as MockCaching, patch(
            "alpha_pulse.api.main.DatabaseOptimizationService"
        ) as MockDB, patch(
            "alpha_pulse.api.main.DataAggregationService"
        ) as MockData, patch(
            "alpha_pulse.api.main.get_database_config"
        ):

            # Setup mocks to track initialization order
            async def cache_init():
                initialization_order.append("cache")

            async def db_init():
                initialization_order.append("db")

            async def data_init():
                initialization_order.append("data")

            mock_cache = MagicMock()
            mock_cache.initialize = AsyncMock(side_effect=cache_init)
            mock_cache.close = AsyncMock()
            MockCaching.create_for_trading.return_value = mock_cache

            mock_db = MagicMock()
            mock_db.initialize = AsyncMock(side_effect=db_init)
            mock_db.start_monitoring = AsyncMock()
            mock_db.stop_monitoring = AsyncMock()
            mock_db.close = AsyncMock()
            MockDB.return_value = mock_db

            mock_data = MagicMock()
            mock_data.initialize = AsyncMock(side_effect=data_init)
            mock_data.close = AsyncMock()
            MockData.return_value = mock_data

            from alpha_pulse.api import main

            async def simulate_startup():
                await main.startup_event()

                # Verify initialization order
                assert initialization_order == ["cache", "db", "data"]

                # Verify all services are initialized
                assert mock_cache.initialize.called
                assert mock_db.initialize.called
                assert mock_data.initialize.called

            await simulate_startup()


class TestPerformanceEndpoints:
    """Test suite for performance monitoring endpoints."""

    def test_cache_metrics_endpoint_active(self):
        """Test cache metrics endpoint when service is active."""
        from fastapi.testclient import TestClient

        with patch("alpha_pulse.api.main.app") as mock_app:
            # Setup mock app with caching service
            mock_app.state.caching_service = MagicMock()
            mock_app.state.caching_service.get_metrics = AsyncMock(
                return_value={
                    "hit_rate": 0.90,
                    "total_requests": 5000,
                    "cache_hits": 4500,
                    "cache_misses": 500,
                    "avg_latency_ms": 0.5,
                }
            )

            # Import router after patching
            from alpha_pulse.api.routers import metrics
            from fastapi import FastAPI

            test_app = FastAPI()
            test_app.state = mock_app.state
            test_app.include_router(metrics.router, prefix="/api/v1")

            client = TestClient(test_app)

            # Make request to cache metrics endpoint
            response = client.get("/api/v1/metrics/cache")

            # Should return metrics
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "active"
            assert "metrics" in data
            assert data["metrics"]["hit_rate"] == 0.90

    def test_cache_metrics_endpoint_inactive(self):
        """Test cache metrics endpoint when service is not initialized."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        # Create test app without caching service
        test_app = FastAPI()
        test_app.state = MagicMock()

        from alpha_pulse.api.routers import metrics

        test_app.include_router(metrics.router, prefix="/api/v1")

        client = TestClient(test_app)

        # Make request to cache metrics endpoint
        response = client.get("/api/v1/metrics/cache")

        # Should return inactive status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "inactive"
        assert data["message"] == "CachingService is not initialized"

    def test_database_metrics_endpoint_active(self):
        """Test database metrics endpoint when service is active."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        # Setup mock app with db optimization service
        test_app = FastAPI()
        test_app.state = MagicMock()
        test_app.state.db_optimization_service = MagicMock()
        test_app.state.db_optimization_service.get_performance_metrics = AsyncMock(
            return_value={
                "avg_query_time_ms": 2.5,
                "slow_queries_count": 5,
                "total_queries": 10000,
                "cache_hit_rate": 0.75,
                "active_connections": 25,
            }
        )

        from alpha_pulse.api.routers import metrics

        test_app.include_router(metrics.router, prefix="/api/v1")

        client = TestClient(test_app)

        # Make request to database metrics endpoint
        response = client.get("/api/v1/metrics/database")

        # Should return metrics
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "metrics" in data
        assert data["metrics"]["avg_query_time_ms"] == 2.5

    def test_database_metrics_endpoint_error_handling(self):
        """Test database metrics endpoint error handling."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        # Setup mock app with failing service
        test_app = FastAPI()
        test_app.state = MagicMock()
        test_app.state.db_optimization_service = MagicMock()
        test_app.state.db_optimization_service.get_performance_metrics = AsyncMock(
            side_effect=Exception("Database connection lost")
        )

        from alpha_pulse.api.routers import metrics

        test_app.include_router(metrics.router, prefix="/api/v1")

        client = TestClient(test_app)

        # Make request to database metrics endpoint
        response = client.get("/api/v1/metrics/database")

        # Should return error
        assert response.status_code == 500
        assert "Database connection lost" in response.json()["detail"]
