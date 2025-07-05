"""
Tests for GPU Acceleration Service.
"""

import pytest
import numpy as np
import torch
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import json

from alpha_pulse.ml.gpu.gpu_service import GPUService
from alpha_pulse.ml.gpu.gpu_config import (
    GPUConfig, get_default_config, get_inference_config
)


@pytest.fixture
def gpu_config():
    """Create test GPU configuration."""
    config = get_default_config()
    config.monitoring.enable_monitoring = False  # Disable for tests
    return config


@pytest.fixture
def gpu_service(gpu_config):
    """Create GPU service instance."""
    with patch('alpha_pulse.ml.gpu.gpu_service.optimize_cuda_settings'):
        with patch('alpha_pulse.ml.gpu.gpu_service.GPUManager'):
            with patch('alpha_pulse.ml.gpu.gpu_service.GPUMemoryManager'):
                with patch('alpha_pulse.ml.gpu.gpu_service.GPUBatchProcessor'):
                    service = GPUService(config=gpu_config)
                    yield service


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    return X, y


class TestGPUService:
    """Test GPU service functionality."""
    
    def test_initialization(self, gpu_service):
        """Test service initialization."""
        assert gpu_service.config is not None
        assert not gpu_service.is_running
        assert hasattr(gpu_service, 'gpu_manager')
        assert hasattr(gpu_service, 'memory_manager')
        assert hasattr(gpu_service, 'batch_processor')
        assert hasattr(gpu_service, 'cuda_ops')
    
    @pytest.mark.asyncio
    async def test_start_stop(self, gpu_service):
        """Test starting and stopping service."""
        # Start service
        await gpu_service.start()
        assert gpu_service.is_running
        
        # Stop service
        await gpu_service.stop()
        assert not gpu_service.is_running
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, gpu_config):
        """Test monitoring loop with enabled monitoring."""
        gpu_config.monitoring.enable_monitoring = True
        gpu_config.monitoring.monitor_interval_sec = 0.01
        
        with patch('alpha_pulse.ml.gpu.gpu_service.optimize_cuda_settings'):
            with patch('alpha_pulse.ml.gpu.gpu_service.GPUManager'):
                with patch('alpha_pulse.ml.gpu.gpu_service.GPUMemoryManager'):
                    with patch('alpha_pulse.ml.gpu.gpu_service.GPUBatchProcessor'):
                        service = GPUService(config=gpu_config)
                        
                        # Mock get_metrics
                        service.get_metrics = Mock(return_value={
                            'devices': {
                                0: {
                                    'memory_usage': 0.95,
                                    'utilization': 0.98
                                }
                            }
                        })
                        
                        # Start service
                        await service.start()
                        
                        # Let monitoring run briefly
                        await asyncio.sleep(0.02)
                        
                        # Should have called get_metrics
                        assert service.get_metrics.called
                        
                        # Stop service
                        await service.stop()
    
    def test_create_model_linear(self, gpu_service):
        """Test creating linear regression model."""
        with patch('alpha_pulse.ml.gpu.gpu_service.select_best_gpu', return_value=0):
            with patch('alpha_pulse.ml.gpu.gpu_service.GPULinearRegression') as MockModel:
                mock_model = Mock()
                mock_model.to_gpu = Mock(return_value=mock_model)
                MockModel.return_value = mock_model
                
                model = gpu_service.create_model(
                    model_type='linear',
                    model_name='test_linear',
                    n_features=10
                )
                
                assert model is mock_model
                assert 'test_linear' in gpu_service._models
                MockModel.assert_called_once_with(n_features=10)
    
    def test_create_model_neural_network(self, gpu_service):
        """Test creating neural network model."""
        with patch('alpha_pulse.ml.gpu.gpu_service.select_best_gpu', return_value=0):
            with patch('alpha_pulse.ml.gpu.gpu_service.GPUNeuralNetwork') as MockModel:
                mock_model = Mock()
                mock_model.to_gpu = Mock(return_value=mock_model)
                mock_model.eval = Mock()
                MockModel.return_value = mock_model
                
                model = gpu_service.create_model(
                    model_type='neural_network',
                    model_name='test_nn',
                    input_size=10,
                    hidden_sizes=[64, 32],
                    output_size=1
                )
                
                assert model is mock_model
                assert 'test_nn' in gpu_service._models
    
    def test_create_model_invalid_type(self, gpu_service):
        """Test creating model with invalid type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            gpu_service.create_model(
                model_type='invalid',
                model_name='test'
            )
    
    def test_create_model_no_gpu(self, gpu_service):
        """Test model creation when no GPU available."""
        gpu_service.config.fallback_to_cpu = False
        
        with patch('alpha_pulse.ml.gpu.gpu_service.select_best_gpu', return_value=None):
            with pytest.raises(RuntimeError, match="No suitable GPU"):
                gpu_service.create_model(
                    model_type='linear',
                    model_name='test',
                    n_features=10
                )
    
    @pytest.mark.asyncio
    async def test_train_model(self, gpu_service, sample_data):
        """Test model training."""
        X, y = sample_data
        
        # Create mock model
        mock_model = Mock()
        mock_model.fit = Mock()
        
        gpu_service._models['test_model'] = {
            'model': mock_model,
            'type': 'neural_network',
            'device_id': 0,
            'created_at': datetime.now()
        }
        
        # Mock GPU allocation
        gpu_service.gpu_manager.allocate_gpu = Mock(return_value=0)
        gpu_service.gpu_manager.release_gpu = Mock()
        
        # Train model
        results = await gpu_service.train_model(
            'test_model',
            X, y,
            epochs=10,
            batch_size=32
        )
        
        # Check results
        assert results['model_name'] == 'test_model'
        assert results['samples_trained'] == len(X)
        assert 'training_time' in results
        
        # Check model was trained
        mock_model.fit.assert_called_once()
        
        # Check GPU was allocated and released
        gpu_service.gpu_manager.allocate_gpu.assert_called_once()
        gpu_service.gpu_manager.release_gpu.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_train_model_validation(self, gpu_service, sample_data):
        """Test model training with validation data."""
        X, y = sample_data
        X_val, y_val = X[:20], y[:20]
        
        # Create mock model
        mock_model = Mock()
        mock_model.fit = Mock()
        
        gpu_service._models['test_model'] = {
            'model': mock_model,
            'type': 'neural_network',
            'device_id': 0,
            'created_at': datetime.now()
        }
        
        gpu_service.gpu_manager.allocate_gpu = Mock(return_value=0)
        gpu_service.gpu_manager.release_gpu = Mock()
        
        # Train with validation
        results = await gpu_service.train_model(
            'test_model',
            X, y,
            validation_data=(X_val, y_val),
            epochs=5
        )
        
        # Check fit was called with validation data
        call_kwargs = mock_model.fit.call_args[1]
        assert 'validation_data' in call_kwargs
    
    @pytest.mark.asyncio
    async def test_predict(self, gpu_service):
        """Test model prediction."""
        X = np.random.randn(10, 5).astype(np.float32)
        
        # Test with batch processor model
        gpu_service.batch_processor.models = {'test_model': Mock()}
        gpu_service.batch_processor.process_async = AsyncMock(
            return_value=np.random.randn(10, 1)
        )
        
        predictions = await gpu_service.predict('test_model', X)
        
        assert predictions.shape == (10, 1)
        gpu_service.batch_processor.process_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_direct(self, gpu_service):
        """Test direct model prediction."""
        X = np.random.randn(10, 5).astype(np.float32)
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.random.randn(10, 1))
        
        gpu_service._models['test_model'] = {
            'model': mock_model,
            'type': 'linear',
            'device_id': 0,
            'created_at': datetime.now()
        }
        
        predictions = await gpu_service.predict('test_model', X)
        
        assert predictions.shape == (10, 1)
        mock_model.predict.assert_called_once()
    
    def test_calculate_technical_indicators(self, gpu_service):
        """Test technical indicator calculation."""
        prices = np.random.randn(100).cumsum() + 100
        
        # Mock CUDA operations
        gpu_service.cuda_ops.calculate_returns = Mock(
            return_value=np.random.randn(100)
        )
        gpu_service.cuda_ops.calculate_rsi = Mock(
            return_value=np.random.uniform(0, 100, 100)
        )
        gpu_service.cuda_ops.calculate_macd = Mock(
            return_value=(
                np.random.randn(100),
                np.random.randn(100),
                np.random.randn(100)
            )
        )
        
        indicators = gpu_service.calculate_technical_indicators(
            prices,
            ['returns', 'rsi', 'macd']
        )
        
        assert 'returns' in indicators
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'macd_signal' in indicators
        assert 'macd_histogram' in indicators
    
    def test_run_monte_carlo(self, gpu_service):
        """Test Monte Carlo simulation."""
        mock_paths = np.random.randn(253, 1000).cumsum(axis=0) + 100
        
        gpu_service.cuda_ops.monte_carlo_simulation = Mock(
            return_value=torch.tensor(mock_paths)
        )
        
        paths = gpu_service.run_monte_carlo(
            initial_price=100.0,
            drift=0.05,
            volatility=0.2,
            time_horizon=252,
            n_simulations=1000
        )
        
        assert paths.shape == (253, 1000)
        gpu_service.cuda_ops.monte_carlo_simulation.assert_called_once()
    
    def test_optimize_portfolio(self, gpu_service):
        """Test portfolio optimization."""
        returns = np.random.randn(252, 5)
        
        # Mock CUDA operations
        gpu_service.cuda_ops.covariance_matrix = Mock(
            return_value=np.eye(5)
        )
        
        weights = gpu_service.optimize_portfolio(
            returns,
            method='mean_variance'
        )
        
        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
    
    def test_optimize_portfolio_constraints(self, gpu_service):
        """Test portfolio optimization with constraints."""
        returns = np.random.randn(252, 5)
        
        gpu_service.cuda_ops.covariance_matrix = Mock(
            return_value=np.eye(5)
        )
        
        constraints = {
            'min_weight': 0.1,
            'max_weight': 0.3
        }
        
        weights = gpu_service.optimize_portfolio(
            returns,
            method='mean_variance',
            constraints=constraints
        )
        
        assert (weights >= 0.1).all()
        assert (weights <= 0.3).all()
    
    def test_get_metrics(self, gpu_service):
        """Test getting service metrics."""
        # Add a model
        gpu_service._models['test_model'] = {
            'type': 'linear',
            'device_id': 0,
            'created_at': datetime.now()
        }
        
        # Mock component metrics
        gpu_service.batch_processor.get_performance_stats = Mock(
            return_value={'batching': 'stats'}
        )
        gpu_service.memory_manager.get_memory_info = Mock(
            return_value={
                'allocated_memory': 1024**3,
                'reserved_memory': 2 * 1024**3,
                'free_memory': 10 * 1024**3
            }
        )
        
        with patch('alpha_pulse.ml.gpu.gpu_service.get_gpu_info') as mock_info:
            mock_info.return_value = [{
                'device_id': 0,
                'name': 'Test GPU',
                'total_memory_gb': 24,
                'used_memory_gb': 4,
                'gpu_utilization': 20
            }]
            
            metrics = gpu_service.get_metrics()
        
        assert metrics['service_running'] == False
        assert 'test_model' in metrics['models']
        assert 0 in metrics['devices']
        assert 'batching' in metrics['batch_processing']
        assert 0 in metrics['memory']
    
    def test_save_load_model(self, gpu_service, tmp_path):
        """Test saving and loading models."""
        # Create mock PyTorch model
        mock_model = Mock()
        mock_model.state_dict = Mock(return_value={'weight': torch.randn(10, 5)})
        mock_model.load_state_dict = Mock()
        
        gpu_service._models['test_model'] = {
            'model': mock_model,
            'type': 'neural_network',
            'device_id': 0,
            'created_at': datetime.now()
        }
        
        # Save model
        save_path = tmp_path / "test_model.pt"
        
        with patch('torch.save') as mock_save:
            gpu_service.save_model('test_model', str(save_path))
            mock_save.assert_called_once()
        
        # Load model
        with patch('torch.load') as mock_load:
            mock_load.return_value = {
                'model_state_dict': {'weight': torch.randn(10, 5)},
                'model_type': 'neural_network',
                'model_config': {'input_size': 10}
            }
            
            with patch.object(gpu_service, 'create_model') as mock_create:
                mock_create.return_value = mock_model
                
                gpu_service.load_model(str(save_path), 'loaded_model')
                
                mock_create.assert_called_once_with(
                    'neural_network',
                    'loaded_model',
                    input_size=10
                )
    
    def test_error_handling(self, gpu_service):
        """Test error handling in service methods."""
        # Test with non-existent model
        with pytest.raises(ValueError, match="Model .* not found"):
            asyncio.run(gpu_service.train_model('non_existent', np.array([]), np.array([])))
        
        with pytest.raises(ValueError, match="Model .* not found"):
            asyncio.run(gpu_service.predict('non_existent', np.array([])))
        
        with pytest.raises(ValueError, match="Model .* not found"):
            gpu_service.save_model('non_existent', 'path.pt')
    
    def test_integration_workflow(self, gpu_service, sample_data):
        """Test complete workflow integration."""
        X, y = sample_data
        
        with patch('alpha_pulse.ml.gpu.gpu_service.select_best_gpu', return_value=0):
            with patch('alpha_pulse.ml.gpu.gpu_service.GPUNeuralNetwork') as MockModel:
                # Setup mock model
                mock_model = Mock()
                mock_model.to_gpu = Mock(return_value=mock_model)
                mock_model.eval = Mock()
                mock_model.fit = Mock()
                mock_model.predict = Mock(return_value=np.random.randn(10, 1))
                MockModel.return_value = mock_model
                
                # Create model
                model = gpu_service.create_model(
                    model_type='neural_network',
                    model_name='workflow_model',
                    input_size=10,
                    hidden_sizes=[32],
                    output_size=1
                )
                
                # Mock GPU manager
                gpu_service.gpu_manager.allocate_gpu = Mock(return_value=0)
                gpu_service.gpu_manager.release_gpu = Mock()
                
                # Train model
                async def train_async():
                    return await gpu_service.train_model(
                        'workflow_model',
                        X, y,
                        epochs=5
                    )
                
                results = asyncio.run(train_async())
                assert results['model_name'] == 'workflow_model'
                
                # Make predictions
                async def predict_async():
                    return await gpu_service.predict(
                        'workflow_model',
                        X[:10]
                    )
                
                predictions = asyncio.run(predict_async())
                assert predictions.shape == (10, 1)


class TestGPUServiceConfig:
    """Test GPU service with different configurations."""
    
    def test_inference_config(self):
        """Test service with inference configuration."""
        config = get_inference_config()
        
        with patch('alpha_pulse.ml.gpu.gpu_service.optimize_cuda_settings'):
            with patch('alpha_pulse.ml.gpu.gpu_service.GPUManager'):
                with patch('alpha_pulse.ml.gpu.gpu_service.GPUMemoryManager'):
                    with patch('alpha_pulse.ml.gpu.gpu_service.GPUBatchProcessor'):
                        service = GPUService(config=config)
                        
                        assert service.config.optimization.optimize_for_inference
                        assert service.config.batching.max_batch_size == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])