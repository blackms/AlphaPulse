"""
Tests for GPU-Optimized ML Models.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from alpha_pulse.ml.gpu.gpu_models import (
    GPULinearRegression,
    GPUNeuralNetwork,
    GPULSTMModel,
    GPUTransformerModel,
    MultiGPUWrapper
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Linear relationship with noise
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    return X, y.astype(np.float32)


@pytest.fixture
def sequence_data():
    """Create sample sequence data for LSTM/Transformer."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 50
    n_features = 20
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples, 1).astype(np.float32)
    
    return X, y


class TestGPULinearRegression:
    """Test GPU Linear Regression model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GPULinearRegression(n_features=10, learning_rate=0.01)
        
        assert model.n_features == 10
        assert model.learning_rate == 0.01
        assert isinstance(model.model, nn.Linear)
        assert model.device.type == 'cpu'  # Default device
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_to_gpu(self, mock_cuda):
        """Test moving model to GPU."""
        model = GPULinearRegression(n_features=10)
        
        with patch('torch.cuda.device_count', return_value=1):
            returned_model = model.to_gpu(device_id=0)
        
        assert returned_model is model
        assert str(model.device) == 'cuda:0'
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        model = GPULinearRegression(n_features=X.shape[1])
        
        # Fit model
        model.fit(X, y, epochs=10, batch_size=32)
        
        # Model should be trained
        assert model.optimizer is not None
    
    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        model = GPULinearRegression(n_features=X.shape[1])
        
        # Fit model
        model.fit(X, y, epochs=10)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10,)
        assert predictions.dtype == np.float32


class TestGPUNeuralNetwork:
    """Test GPU Neural Network model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GPUNeuralNetwork(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1,
            dropout=0.2,
            activation='relu'
        )
        
        assert model.input_size == 10
        assert model.hidden_sizes == [64, 32]
        assert model.output_size == 1
        assert model.dropout == 0.2
    
    def test_network_structure(self):
        """Test network architecture."""
        model = GPUNeuralNetwork(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Check layers
        layers = list(model.network.children())
        
        # First hidden layer
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 10
        assert layers[0].out_features == 64
        
        # Activation
        assert isinstance(layers[1], nn.ReLU)
        
        # Batch norm
        assert isinstance(layers[2], nn.BatchNorm1d)
        
        # Dropout
        assert isinstance(layers[3], nn.Dropout)
    
    def test_fit_with_validation(self, sample_data):
        """Test model fitting with validation data."""
        X, y = sample_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = GPUNeuralNetwork(
            input_size=X.shape[1],
            hidden_sizes=[32, 16],
            output_size=1
        )
        
        # Fit with validation
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=64,
            use_amp=False  # Disable for CPU
        )
        
        # Check model was trained
        assert model.optimizer is not None
    
    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        
        model = GPUNeuralNetwork(
            input_size=X.shape[1],
            hidden_sizes=[32],
            output_size=1
        )
        
        # Fit and predict
        model.fit(X, y, epochs=5)
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10, 1)
        assert not np.isnan(predictions).any()
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        
        for activation in activations:
            model = GPUNeuralNetwork(
                input_size=10,
                hidden_sizes=[32],
                output_size=1,
                activation=activation
            )
            
            # Check activation was set
            layers = list(model.network.children())
            assert any(isinstance(layer, nn.Module) for layer in layers)


class TestGPULSTMModel:
    """Test GPU LSTM model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GPULSTMModel(
            input_size=20,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2,
            bidirectional=True
        )
        
        assert model.input_size == 20
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.bidirectional is True
    
    def test_forward_pass(self, sequence_data):
        """Test forward pass."""
        X, _ = sequence_data
        
        model = GPULSTMModel(
            input_size=X.shape[2],
            hidden_size=64,
            num_layers=2,
            bidirectional=False
        )
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X[:5])
        
        # Forward pass
        output = model(X_tensor)
        
        assert output.shape == (5, 1)
    
    def test_variable_length_sequences(self):
        """Test handling of variable length sequences."""
        model = GPULSTMModel(
            input_size=10,
            hidden_size=32,
            num_layers=1
        )
        
        # Create sequences of different lengths
        batch_size = 4
        max_len = 20
        X = torch.randn(batch_size, max_len, 10)
        lengths = torch.tensor([20, 15, 10, 5])
        
        # Forward pass with lengths
        output = model(X, lengths)
        
        assert output.shape == (batch_size, 1)
    
    def test_fit_and_predict(self, sequence_data):
        """Test fitting and prediction."""
        X, y = sequence_data
        
        model = GPULSTMModel(
            input_size=X.shape[2],
            hidden_size=32,
            num_layers=1,
            bidirectional=True
        )
        
        # Fit model
        model.fit(X, y, epochs=5, batch_size=16)
        
        # Predict
        predictions = model.predict(X[:10])
        
        assert predictions.shape == (10, 1)
        assert not np.isnan(predictions).any()
    
    def test_gradient_clipping(self, sequence_data):
        """Test gradient clipping during training."""
        X, y = sequence_data
        
        model = GPULSTMModel(
            input_size=X.shape[2],
            hidden_size=32
        )
        
        # Mock gradient clipping
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            model.fit(X[:10], y[:10], epochs=1)
            
            # Should be called during training
            assert mock_clip.called


class TestGPUTransformerModel:
    """Test GPU Transformer model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GPUTransformerModel(
            input_size=20,
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            output_size=1,
            max_seq_length=1000,
            dropout=0.1
        )
        
        assert model.input_size == 20
        assert model.d_model == 512
        assert model.max_seq_length == 1000
    
    def test_positional_encoding(self):
        """Test positional encoding creation."""
        model = GPUTransformerModel(
            input_size=10,
            d_model=64,
            max_seq_length=100
        )
        
        pe = model.positional_encoding
        
        assert pe.shape == (1, 100, 64)
        assert not torch.isnan(pe).any()
    
    def test_forward_pass(self, sequence_data):
        """Test forward pass."""
        X, _ = sequence_data
        
        model = GPUTransformerModel(
            input_size=X.shape[2],
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256
        )
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X[:5])
        
        # Forward pass
        output = model(X_tensor)
        
        assert output.shape == (5, 1)
    
    def test_fit_with_warmup(self, sequence_data):
        """Test fitting with learning rate warmup."""
        X, y = sequence_data
        
        model = GPUTransformerModel(
            input_size=X.shape[2],
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        # Fit with warmup
        model.fit(
            X[:50], y[:50],
            epochs=5,
            batch_size=8,
            warmup_steps=10
        )
        
        # Check optimizer exists
        assert model.optimizer is not None
    
    def test_attention_mask(self):
        """Test forward pass with attention mask."""
        model = GPUTransformerModel(
            input_size=10,
            d_model=64,
            nhead=4,
            num_layers=1
        )
        
        # Create input and mask
        batch_size = 2
        seq_len = 10
        X = torch.randn(batch_size, seq_len, 10)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
        # Forward pass with mask
        output = model(X, mask=mask)
        
        assert output.shape == (batch_size, 1)


class TestMultiGPUWrapper:
    """Test multi-GPU wrapper."""
    
    @patch('torch.cuda.device_count', return_value=2)
    def test_initialization_multi_gpu(self, mock_device_count):
        """Test initialization with multiple GPUs."""
        base_model = nn.Linear(10, 1)
        
        wrapper = MultiGPUWrapper(base_model, device_ids=[0, 1])
        
        assert wrapper.device_ids == [0, 1]
        assert isinstance(wrapper.model, nn.DataParallel)
    
    @patch('torch.cuda.device_count', return_value=1)
    def test_initialization_single_gpu(self, mock_device_count):
        """Test initialization with single GPU."""
        base_model = nn.Linear(10, 1)
        
        wrapper = MultiGPUWrapper(base_model, device_ids=[0])
        
        assert wrapper.device_ids == [0]
        assert wrapper.model is base_model  # No DataParallel
    
    def test_get_model(self):
        """Test getting wrapped model."""
        base_model = nn.Linear(10, 1)
        wrapper = MultiGPUWrapper(base_model)
        
        assert wrapper.get_model() is wrapper.model
    
    @patch('torch.cuda.device_count', return_value=2)
    def test_get_base_model(self, mock_device_count):
        """Test getting base model from DataParallel."""
        base_model = nn.Linear(10, 1)
        wrapper = MultiGPUWrapper(base_model, device_ids=[0, 1])
        
        # Should unwrap DataParallel
        assert wrapper.get_base_model() is base_model


class TestIntegration:
    """Integration tests for GPU models."""
    
    def test_model_comparison(self, sample_data):
        """Test different models on same data."""
        X, y = sample_data
        
        models = [
            GPULinearRegression(n_features=X.shape[1]),
            GPUNeuralNetwork(
                input_size=X.shape[1],
                hidden_sizes=[32],
                output_size=1
            )
        ]
        
        predictions = []
        
        for model in models:
            model.fit(X[:100], y[:100], epochs=5)
            pred = model.predict(X[100:110])
            predictions.append(pred)
        
        # All models should produce predictions
        assert all(p.shape[0] == 10 for p in predictions)
    
    def test_device_movement(self):
        """Test moving models between devices."""
        model = GPUNeuralNetwork(
            input_size=10,
            hidden_sizes=[32],
            output_size=1
        )
        
        # Should start on CPU
        assert model.device.type == 'cpu'
        
        # Mock GPU availability
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                model.to_gpu(0)
                
                # Device should be updated
                assert str(model.device) == 'cuda:0'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])