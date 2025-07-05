"""
Tests for CUDA Operations.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from alpha_pulse.ml.gpu.cuda_operations import (
    CUDAOperations, gpu_timer
)


@pytest.fixture
def cuda_ops():
    """Create CUDA operations instance."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.set_device'):
            ops = CUDAOperations(device_id=0, precision='float32')
            yield ops
            ops.cleanup()


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    np.random.seed(42)
    # Simulate realistic price movement
    returns = np.random.normal(0.0001, 0.01, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    # Multiple assets
    n_samples, n_assets = 252, 5
    returns = np.random.normal(0.0001, 0.01, (n_samples, n_assets))
    return returns


class TestCUDAOperations:
    """Test CUDA operations functionality."""
    
    def test_initialization(self, cuda_ops):
        """Test CUDA operations initialization."""
        assert cuda_ops.device_id == 0
        assert cuda_ops.precision == 'float32'
        assert cuda_ops.dtype == torch.float32
    
    def test_rolling_window_statistics(self, cuda_ops, sample_prices):
        """Test rolling window statistics calculation."""
        window_size = 20
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            mean, std = cuda_ops.rolling_window_statistics(
                sample_prices, window_size
            )
        
        # Check output shape
        assert len(mean) == len(sample_prices)
        assert len(std) == len(sample_prices)
        
        # Check NaN padding
        assert np.isnan(mean[:window_size-1]).all()
        assert np.isnan(std[:window_size-1]).all()
        
        # Check calculations
        assert not np.isnan(mean[window_size:]).any()
        assert not np.isnan(std[window_size:]).any()
        assert (std[window_size:] >= 0).all()
    
    def test_calculate_returns(self, cuda_ops, sample_prices):
        """Test returns calculation."""
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            # Simple returns
            simple_returns = cuda_ops.calculate_returns(
                sample_prices, method='simple'
            )
            
            # Log returns
            log_returns = cuda_ops.calculate_returns(
                sample_prices, method='log'
            )
        
        # Check shape
        assert len(simple_returns) == len(sample_prices)
        assert len(log_returns) == len(sample_prices)
        
        # First return should be NaN
        assert np.isnan(simple_returns[0])
        assert np.isnan(log_returns[0])
        
        # Check return values are reasonable
        assert np.abs(simple_returns[1:]).max() < 1.0
        assert np.abs(log_returns[1:]).max() < 1.0
    
    def test_exponential_moving_average(self, cuda_ops, sample_prices):
        """Test EMA calculation."""
        alpha = 0.1
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            ema = cuda_ops.exponential_moving_average(sample_prices, alpha)
        
        # Check shape
        assert len(ema) == len(sample_prices)
        
        # First value should equal first price
        assert ema[0] == sample_prices[0]
        
        # EMA should be smoother than prices
        price_vol = np.std(np.diff(sample_prices))
        ema_vol = np.std(np.diff(ema))
        assert ema_vol < price_vol
    
    def test_calculate_rsi(self, cuda_ops, sample_prices):
        """Test RSI calculation."""
        period = 14
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            rsi = cuda_ops.calculate_rsi(sample_prices, period)
        
        # Check shape
        assert len(rsi) == len(sample_prices)
        
        # First 'period' values should be NaN
        assert np.isnan(rsi[:period]).all()
        
        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_calculate_bollinger_bands(self, cuda_ops, sample_prices):
        """Test Bollinger Bands calculation."""
        period = 20
        num_std = 2.0
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            upper, middle, lower = cuda_ops.calculate_bollinger_bands(
                sample_prices, period, num_std
            )
        
        # Check shapes
        assert len(upper) == len(sample_prices)
        assert len(middle) == len(sample_prices)
        assert len(lower) == len(sample_prices)
        
        # Check band relationships
        valid_idx = ~np.isnan(upper)
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()
    
    def test_calculate_macd(self, cuda_ops, sample_prices):
        """Test MACD calculation."""
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            macd, signal, histogram = cuda_ops.calculate_macd(
                sample_prices,
                fast_period=12,
                slow_period=26,
                signal_period=9
            )
        
        # Check shapes
        assert len(macd) == len(sample_prices)
        assert len(signal) == len(sample_prices)
        assert len(histogram) == len(sample_prices)
        
        # Check histogram calculation
        valid_idx = ~np.isnan(histogram)
        np.testing.assert_allclose(
            histogram[valid_idx],
            macd[valid_idx] - signal[valid_idx],
            rtol=1e-5
        )
    
    def test_correlation_matrix(self, cuda_ops, sample_returns):
        """Test correlation matrix calculation."""
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            corr = cuda_ops.correlation_matrix(sample_returns)
        
        n_assets = sample_returns.shape[1]
        
        # Check shape
        assert corr.shape == (n_assets, n_assets)
        
        # Check diagonal is 1
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-5)
        
        # Check symmetry
        np.testing.assert_allclose(corr, corr.T, rtol=1e-5)
        
        # Check range
        assert (corr >= -1).all()
        assert (corr <= 1).all()
    
    def test_covariance_matrix(self, cuda_ops, sample_returns):
        """Test covariance matrix calculation."""
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            cov = cuda_ops.covariance_matrix(sample_returns)
        
        n_assets = sample_returns.shape[1]
        
        # Check shape
        assert cov.shape == (n_assets, n_assets)
        
        # Check symmetry
        np.testing.assert_allclose(cov, cov.T, rtol=1e-5)
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(cov.numpy() if hasattr(cov, 'numpy') else cov)
        assert (eigenvalues >= -1e-10).all()  # Allow small numerical errors
    
    def test_parallel_portfolio_metrics(self, cuda_ops, sample_returns):
        """Test parallel portfolio metrics calculation."""
        n_portfolios = 10
        n_assets = sample_returns.shape[1]
        
        # Random portfolio weights
        weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            metrics = cuda_ops.parallel_portfolio_metrics(
                sample_returns, weights
            )
        
        # Check all metrics present
        assert 'mean_returns' in metrics
        assert 'std_returns' in metrics
        assert 'sharpe_ratios' in metrics
        assert 'max_drawdowns' in metrics
        
        # Check shapes
        assert len(metrics['mean_returns']) == n_portfolios
        assert len(metrics['std_returns']) == n_portfolios
        
        # Check values are reasonable
        assert (metrics['std_returns'] >= 0).all()
        assert (metrics['max_drawdowns'] <= 0).all()
    
    def test_monte_carlo_simulation(self, cuda_ops):
        """Test Monte Carlo simulation."""
        initial_price = 100.0
        drift = 0.05  # 5% annual
        volatility = 0.2  # 20% annual
        time_horizon = 252
        n_simulations = 1000
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            paths = cuda_ops.monte_carlo_simulation(
                initial_price, drift, volatility,
                time_horizon, n_simulations
            )
        
        # Check shape
        assert paths.shape == (time_horizon + 1, n_simulations)
        
        # Check initial price
        assert (paths[0] == initial_price).all()
        
        # Check all prices positive
        assert (paths > 0).all()
        
        # Check average drift approximately matches expected
        final_returns = np.log(paths[-1] / paths[0])
        avg_return = final_returns.mean()
        expected_return = drift - 0.5 * volatility**2
        
        # Should be close with large sample
        assert abs(avg_return - expected_return) < 0.05
    
    def test_gpu_timer_decorator(self):
        """Test GPU timer decorator."""
        @gpu_timer
        def dummy_function(x):
            return x * 2
        
        with patch('time.time', side_effect=[0, 0.1]):
            with patch('torch.cuda.synchronize'):
                result = dummy_function(5)
        
        assert result == 10
    
    def test_to_numpy_conversion(self, cuda_ops):
        """Test tensor to numpy conversion."""
        # Test with torch tensor
        tensor = torch.randn(10, 5)
        numpy_array = cuda_ops.to_numpy(tensor)
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (10, 5)
        
        # Test with numpy array (should return as-is)
        array = np.random.randn(10, 5)
        result = cuda_ops.to_numpy(array)
        assert result is array
    
    def test_cleanup(self, cuda_ops):
        """Test cleanup functionality."""
        with patch('torch.cuda.empty_cache') as mock_empty:
            with patch('torch.cuda.synchronize') as mock_sync:
                cuda_ops.cleanup()
                
                mock_empty.assert_called_once()
                mock_sync.assert_called_once()
    
    def test_invalid_method(self, cuda_ops, sample_prices):
        """Test invalid method handling."""
        with pytest.raises(ValueError):
            cuda_ops.calculate_returns(sample_prices, method='invalid')
    
    def test_edge_cases(self, cuda_ops):
        """Test edge cases."""
        # Empty data
        empty_data = np.array([])
        
        # Single value
        single_value = np.array([100.0])
        
        with patch.object(cuda_ops, 'device', torch.device('cpu')):
            # Rolling stats with window larger than data
            mean, std = cuda_ops.rolling_window_statistics(
                single_value, window_size=10
            )
            assert len(mean) == 1
            
            # Returns on single value
            returns = cuda_ops.calculate_returns(single_value)
            assert len(returns) == 1
            assert np.isnan(returns[0])


class TestNumbaKernels:
    """Test Numba CUDA kernels."""
    
    @pytest.mark.skipif(not hasattr(pytest, 'numba_available'), 
                       reason="Numba not available")
    def test_matrix_multiply_kernel(self):
        """Test matrix multiplication kernel."""
        from alpha_pulse.ml.gpu.cuda_operations import matrix_multiply_kernel
        
        # This would require actual CUDA device
        # Just verify the kernel is defined
        assert callable(matrix_multiply_kernel)
    
    @pytest.mark.skipif(not hasattr(pytest, 'numba_available'),
                       reason="Numba not available")
    def test_vectorized_operations_kernel(self):
        """Test vectorized operations kernel."""
        from alpha_pulse.ml.gpu.cuda_operations import vectorized_operations_kernel
        
        # Verify kernel is defined
        assert callable(vectorized_operations_kernel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])