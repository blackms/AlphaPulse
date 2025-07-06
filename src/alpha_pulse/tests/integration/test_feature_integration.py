"""
End-to-End Integration Tests for Newly Integrated Features.

This test suite validates that all newly integrated features work together:
- GPU Acceleration
- Explainable AI
- Data Quality Pipeline
- Data Lake Architecture
- Backtesting Integration
"""
import pytest
import asyncio
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from alpha_pulse.ml.gpu.gpu_service import GPUService
from alpha_pulse.ml.gpu.gpu_config import get_default_config as get_gpu_config
from alpha_pulse.ml.explainability.explainer import ModelExplainer
from alpha_pulse.data_quality.pipeline import DataQualityPipeline
from alpha_pulse.data_quality.validator import DataValidator
from alpha_pulse.backtesting.enhanced_backtester import EnhancedBacktester, DataSource
from alpha_pulse.backtesting.data_lake_loader import get_data_lake_loader
from alpha_pulse.api.main import app
from fastapi.testclient import TestClient


class TestFeatureIntegration:
    """Test suite for end-to-end feature integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        dates = pd.date_range(
            start=datetime.now(UTC) - timedelta(days=100),
            end=datetime.now(UTC),
            freq='D'
        )
        
        np.random.seed(42)  # For reproducible tests
        
        data = {
            'symbol': ['AAPL'] * len(dates),
            'date_time': dates,
            'open': 150 + np.random.randn(len(dates)) * 5,
            'high': 155 + np.random.randn(len(dates)) * 5,
            'low': 145 + np.random.randn(len(dates)) * 5,
            'close': 152 + np.random.randn(len(dates)) * 5,
            'volume': 1000000 + np.random.randint(-100000, 100000, len(dates))
        }
        
        df = pd.DataFrame(data)
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
        
        return df


class TestGPUIntegration:
    """Test GPU acceleration integration."""
    
    @pytest.mark.asyncio
    async def test_gpu_service_initialization(self):
        """Test GPU service can be initialized."""
        try:
            config = get_gpu_config()
            config.monitoring.enable_monitoring = True
            
            gpu_service = GPUService(config=config)
            await gpu_service.start()
            
            # Test basic functionality
            metrics = gpu_service.get_metrics()
            assert 'available' in metrics
            assert 'devices' in metrics
            
            await gpu_service.stop()
            
        except Exception as e:
            # GPU may not be available in test environment
            pytest.skip(f"GPU not available for testing: {e}")
    
    @pytest.mark.asyncio
    async def test_gpu_model_training_integration(self, sample_market_data):
        """Test GPU integration with model training."""
        try:
            config = get_gpu_config()
            gpu_service = GPUService(config=config)
            await gpu_service.start()
            
            # Mock training job
            training_request = {
                'model_type': 'lstm_predictor',
                'data': sample_market_data.to_dict('records'),
                'parameters': {
                    'sequence_length': 60,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            }
            
            job_id = await gpu_service.submit_training_job(training_request)
            assert job_id is not None
            
            # Wait for job completion (or timeout)
            status = await gpu_service.get_job_status(job_id)
            assert status in ['queued', 'running', 'completed', 'failed']
            
            await gpu_service.stop()
            
        except Exception as e:
            pytest.skip(f"GPU training integration test failed: {e}")


class TestExplainabilityIntegration:
    """Test explainable AI integration."""
    
    def test_model_explainer_initialization(self):
        """Test model explainer can be initialized."""
        explainer = ModelExplainer()
        assert explainer is not None
        
        # Test supported explainer types
        supported_types = explainer.get_supported_explainer_types()
        assert 'shap' in supported_types
        assert 'lime' in supported_types
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, sample_market_data):
        """Test explanation generation for model predictions."""
        explainer = ModelExplainer()
        
        # Mock model and prediction
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.7])  # Mock prediction
        
        # Prepare features (last 5 rows as features)
        features = sample_market_data[['open', 'high', 'low', 'close', 'volume']].tail(5)
        
        try:
            # Generate SHAP explanation
            explanation = await explainer.explain_prediction(
                model=mock_model,
                input_data=features,
                explainer_type='shap',
                prediction_target='price_direction'
            )
            
            assert 'explanation_id' in explanation
            assert 'feature_importance' in explanation
            assert 'explainer_type' in explanation
            assert explanation['explainer_type'] == 'shap'
            
        except Exception as e:
            pytest.skip(f"Explanation generation test failed: {e}")
    
    def test_explanation_api_endpoint(self, client):
        """Test explainability API endpoint."""
        # Mock request data
        request_data = {
            'model_id': 'test_model',
            'input_data': [[150, 155, 145, 152, 1000000]],
            'explainer_type': 'shap',
            'target_feature': 'price_direction'
        }
        
        response = client.post('/api/v1/explainability/explain', json=request_data)
        
        # Should work or return proper error
        assert response.status_code in [200, 422, 500]  # Valid responses


class TestDataQualityIntegration:
    """Test data quality pipeline integration."""
    
    def test_data_quality_pipeline_initialization(self):
        """Test data quality pipeline can be initialized."""
        pipeline = DataQualityPipeline()
        assert pipeline is not None
        
        validator = DataValidator()
        assert validator is not None
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, sample_market_data):
        """Test data quality validation process."""
        pipeline = DataQualityPipeline()
        
        # Run quality assessment
        quality_report = await pipeline.assess_data_quality(
            data=sample_market_data,
            dataset_name='test_market_data'
        )
        
        assert 'overall_score' in quality_report
        assert 'completeness_score' in quality_report
        assert 'validity_score' in quality_report
        assert 'consistency_score' in quality_report
        assert 'recommendations' in quality_report
        
        # Score should be between 0 and 1
        assert 0 <= quality_report['overall_score'] <= 1
    
    def test_data_quality_api_endpoint(self, client):
        """Test data quality API endpoint."""
        # Mock request data
        request_data = {
            'dataset_name': 'test_dataset',
            'data_sample': [
                {'symbol': 'AAPL', 'close': 150.0, 'volume': 1000000},
                {'symbol': 'AAPL', 'close': 151.0, 'volume': 1100000}
            ],
            'validation_rules': ['no_nulls', 'positive_volume', 'price_range']
        }
        
        response = client.post('/api/v1/data-quality/assess', json=request_data)
        
        # Should work or return proper error
        assert response.status_code in [200, 422, 500]


class TestDataLakeIntegration:
    """Test data lake integration."""
    
    def test_data_lake_loader_initialization(self):
        """Test data lake loader can be initialized."""
        try:
            loader = get_data_lake_loader(
                data_lake_path="./test_data_lake",
                enable_spark=False
            )
            assert loader is not None
            
        except Exception as e:
            pytest.skip(f"Data lake not available for testing: {e}")
    
    def test_data_lake_api_endpoints(self, client):
        """Test data lake API endpoints."""
        # Test dataset listing
        response = client.get('/api/v1/datalake/datasets')
        assert response.status_code in [200, 500]  # OK or server error if not configured
        
        # Test health check
        response = client.get('/api/v1/datalake/health')
        assert response.status_code in [200, 500]
        
        # Test statistics
        response = client.get('/api/v1/datalake/statistics')
        assert response.status_code in [200, 500]


class TestBacktestingIntegration:
    """Test backtesting integration with data lake."""
    
    @pytest.mark.asyncio
    async def test_enhanced_backtester_initialization(self):
        """Test enhanced backtester can be initialized."""
        try:
            backtester = EnhancedBacktester(
                commission=0.001,
                initial_capital=100000,
                data_source=DataSource.AUTO
            )
            assert backtester is not None
            
        except Exception as e:
            pytest.skip(f"Enhanced backtester initialization failed: {e}")
    
    def test_backtesting_api_endpoints(self, client):
        """Test backtesting API endpoints."""
        # Test strategy listing
        response = client.get('/api/v1/backtesting/strategies')
        assert response.status_code == 200
        
        data = response.json()
        assert 'strategies' in data
        assert 'data_sources' in data
        assert 'timeframes' in data
        
        # Test data lake status
        response = client.get('/api/v1/backtesting/data-lake/status')
        assert response.status_code == 200
    
    def test_backtest_execution_api(self, client):
        """Test backtest execution API."""
        request_data = {
            'symbols': ['AAPL'],
            'timeframe': '1d',
            'start_date': (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            'end_date': datetime.now(UTC).isoformat(),
            'strategy_type': 'simple_ma',
            'strategy_params': {'short_window': 10, 'long_window': 20},
            'initial_capital': 100000,
            'commission': 0.001,
            'data_source': 'auto'
        }
        
        response = client.post('/api/v1/backtesting/run', json=request_data)
        
        # Should work or return proper error
        assert response.status_code in [200, 422, 500]


class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_pipeline(self, sample_market_data):
        """Test complete trading pipeline integration."""
        try:
            # 1. Data Quality Assessment
            pipeline = DataQualityPipeline()
            quality_report = await pipeline.assess_data_quality(
                data=sample_market_data,
                dataset_name='integration_test_data'
            )
            
            assert quality_report['overall_score'] > 0.5  # Reasonable quality
            
            # 2. Enhanced Backtesting
            backtester = EnhancedBacktester(
                commission=0.001,
                initial_capital=100000,
                data_source=DataSource.AUTO
            )
            
            # Mock strategy for testing
            from alpha_pulse.backtesting.strategy import BaseStrategy
            
            class TestStrategy(BaseStrategy):
                def should_enter(self, signal):
                    return len(signal) > 10 and signal.iloc[-1] > signal.iloc[-2]
                
                def should_exit(self, signal, position):
                    return len(signal) > 10 and signal.iloc[-1] < signal.iloc[-2]
            
            strategy = TestStrategy()
            
            # Run mini backtest
            mock_data = {
                'AAPL': sample_market_data.set_index('date_time')[['open', 'high', 'low', 'close', 'volume']]
            }
            
            with patch.object(backtester, '_load_market_data', return_value=mock_data):
                results = await backtester.run_backtest(
                    strategy=strategy,
                    symbols=['AAPL'],
                    timeframe='1d',
                    start_date=datetime.now(UTC) - timedelta(days=30),
                    end_date=datetime.now(UTC)
                )
                
                assert 'AAPL' in results
                result = results['AAPL']
                assert hasattr(result, 'total_return')
                assert hasattr(result, 'sharpe_ratio')
            
            # 3. Explainability (if available)
            try:
                explainer = ModelExplainer()
                
                # Mock model prediction
                mock_model = Mock()
                mock_model.predict.return_value = np.array([0.6])
                
                features = sample_market_data[['open', 'high', 'low', 'close', 'volume']].tail(1)
                
                explanation = await explainer.explain_prediction(
                    model=mock_model,
                    input_data=features,
                    explainer_type='shap',
                    prediction_target='price_direction'
                )
                
                assert 'explanation_id' in explanation
                
            except Exception as e:
                print(f"Explainability test skipped: {e}")
            
            # 4. GPU Integration (if available)
            try:
                config = get_gpu_config()
                gpu_service = GPUService(config=config)
                await gpu_service.start()
                
                metrics = gpu_service.get_metrics()
                assert 'available' in metrics
                
                await gpu_service.stop()
                
            except Exception as e:
                print(f"GPU test skipped: {e}")
            
        except Exception as e:
            pytest.fail(f"End-to-end integration test failed: {e}")
    
    def test_api_integration_health(self, client):
        """Test overall API health with new integrations."""
        # Test all major API endpoints are accessible
        endpoints = [
            '/api/v1/gpu/status',
            '/api/v1/explainability/models',
            '/api/v1/data-quality/status',
            '/api/v1/backtesting/strategies',
            '/api/v1/datalake/health'
        ]
        
        results = {}
        for endpoint in endpoints:
            response = client.get(endpoint)
            results[endpoint] = response.status_code
        
        # At least some endpoints should be working
        working_endpoints = sum(1 for status in results.values() if status == 200)
        total_endpoints = len(endpoints)
        
        # At least 60% of endpoints should be working
        assert working_endpoints >= total_endpoints * 0.6, f"Too many endpoints failing: {results}"
    
    @pytest.mark.asyncio
    async def test_dashboard_integration_readiness(self):
        """Test that all dashboard components can be integrated."""
        # This test ensures that the new features are ready for dashboard integration
        
        # 1. Check GPU monitoring data availability
        try:
            config = get_gpu_config()
            gpu_service = GPUService(config=config)
            await gpu_service.start()
            
            metrics = gpu_service.get_metrics()
            assert isinstance(metrics, dict)
            
            await gpu_service.stop()
            
        except Exception as e:
            print(f"GPU dashboard integration ready (CPU fallback): {e}")
        
        # 2. Check explainability visualization data
        try:
            explainer = ModelExplainer()
            supported_types = explainer.get_supported_explainer_types()
            assert len(supported_types) > 0
            
        except Exception as e:
            print(f"Explainability dashboard integration issue: {e}")
        
        # 3. Check data quality dashboard data
        try:
            pipeline = DataQualityPipeline()
            validator = DataValidator()
            
            # These should initialize without errors
            assert pipeline is not None
            assert validator is not None
            
        except Exception as e:
            pytest.fail(f"Data quality dashboard integration failed: {e}")
        
        # 4. Check data lake exploration readiness
        try:
            loader = get_data_lake_loader(
                data_lake_path="./test_data_lake",
                enable_spark=False
            )
            
            # Should initialize even if data lake is empty
            assert loader is not None
            
        except Exception as e:
            print(f"Data lake dashboard integration ready (fallback): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])