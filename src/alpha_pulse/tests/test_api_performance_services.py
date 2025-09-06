"""
Tests for performance services initialization in API.
Following TDD approach - RED phase.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import asyncio


class TestPerformanceServicesInitialization:
    """Test suite for CachingService and DatabaseOptimizationService initialization."""
    
    @pytest.mark.asyncio
    async def test_caching_service_initializes_on_startup(self):
        """Test that CachingService is initialized during API startup."""
        with patch('alpha_pulse.services.caching_service.CachingService') as MockCachingService:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockCachingService.create_for_trading.return_value = mock_instance
            
            # Import after patching
            from alpha_pulse.api.main import app
            
            # Trigger startup event
            async with app.router.lifespan_context(app):
                # Verify CachingService was initialized
                assert MockCachingService.create_for_trading.called, "CachingService.create_for_trading was not called"
                assert mock_instance.initialize.called, "CachingService.initialize was not called"
    
    @pytest.mark.asyncio
    async def test_database_optimization_service_initializes_on_startup(self):
        """Test that DatabaseOptimizationService is initialized during API startup."""
        with patch('alpha_pulse.services.database_optimization_service.DatabaseOptimizationService') as MockDBService:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockDBService.return_value = mock_instance
            
            # Import after patching
            from alpha_pulse.api.main import app
            
            # Trigger startup event
            async with app.router.lifespan_context(app):
                # Verify DatabaseOptimizationService was initialized
                assert MockDBService.called, "DatabaseOptimizationService was not instantiated"
                assert mock_instance.initialize.called, "DatabaseOptimizationService.initialize was not called"
    
    @pytest.mark.asyncio
    async def test_services_are_accessible_after_startup(self):
        """Test that services are accessible via app.state after initialization."""
        from alpha_pulse.api.main import app
        
        # Trigger startup event
        async with app.router.lifespan_context(app):
            # Check services are in app.state
            assert hasattr(app.state, 'caching_service'), "CachingService not in app.state"
            assert hasattr(app.state, 'db_optimization_service'), "DatabaseOptimizationService not in app.state"
            assert app.state.caching_service is not None, "CachingService is None"
            assert app.state.db_optimization_service is not None, "DatabaseOptimizationService is None"
    
    @pytest.mark.asyncio
    async def test_caching_service_configuration(self):
        """Test that CachingService is configured with correct parameters."""
        with patch('alpha_pulse.services.caching_service.CachingService') as MockCachingService:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockCachingService.create_for_trading.return_value = mock_instance
            
            from alpha_pulse.api.main import app
            
            async with app.router.lifespan_context(app):
                # Verify correct configuration
                MockCachingService.create_for_trading.assert_called_once()
                # Check if Redis connection is established
                assert mock_instance.initialize.called
    
    @pytest.mark.asyncio
    async def test_database_optimization_configuration(self):
        """Test that DatabaseOptimizationService is configured correctly."""
        with patch('alpha_pulse.services.database_optimization_service.DatabaseOptimizationService') as MockDBService:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.start_monitoring = AsyncMock()
            MockDBService.return_value = mock_instance
            
            from alpha_pulse.api.main import app
            
            async with app.router.lifespan_context(app):
                # Verify monitoring starts
                assert mock_instance.start_monitoring.called, "Database monitoring not started"
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_cleanup(self):
        """Test that services are properly cleaned up on shutdown."""
        with patch('alpha_pulse.services.caching_service.CachingService') as MockCachingService, \
             patch('alpha_pulse.services.database_optimization_service.DatabaseOptimizationService') as MockDBService:
            
            cache_mock = MagicMock()
            cache_mock.initialize = AsyncMock()
            cache_mock.close = AsyncMock()
            MockCachingService.create_for_trading.return_value = cache_mock
            
            db_mock = MagicMock()
            db_mock.initialize = AsyncMock()
            db_mock.stop_monitoring = AsyncMock()
            db_mock.close = AsyncMock()
            MockDBService.return_value = db_mock
            
            from alpha_pulse.api.main import app
            
            # Start and stop the app
            async with app.router.lifespan_context(app):
                pass  # Startup happens here
            
            # After context exit, shutdown should have occurred
            assert cache_mock.close.called, "CachingService not closed on shutdown"
            assert db_mock.stop_monitoring.called, "Database monitoring not stopped"
            assert db_mock.close.called, "DatabaseOptimizationService not closed"
    
    @pytest.mark.asyncio
    async def test_service_initialization_error_handling(self):
        """Test that initialization errors are handled gracefully."""
        with patch('alpha_pulse.services.caching_service.CachingService') as MockCachingService:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock(side_effect=Exception("Redis connection failed"))
            MockCachingService.create_for_trading.return_value = mock_instance
            
            from alpha_pulse.api.main import app
            
            # Should log error but not crash
            try:
                async with app.router.lifespan_context(app):
                    pass
            except Exception as e:
                # Should handle gracefully with fallback
                assert "Redis connection failed" not in str(e), "Service initialization error not handled"
    
    def test_performance_endpoints_available(self):
        """Test that performance monitoring endpoints are available."""
        from alpha_pulse.api.main import app
        
        client = TestClient(app)
        
        # Test cache metrics endpoint
        response = client.get("/api/v1/metrics/cache")
        assert response.status_code in [200, 401], "Cache metrics endpoint not available"
        
        # Test database performance endpoint
        response = client.get("/api/v1/metrics/database")
        assert response.status_code in [200, 401], "Database metrics endpoint not available"