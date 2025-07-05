"""
CUDA Operations for GPU-Accelerated Computing.

This module provides CUDA kernels and operations for efficient
GPU computation in trading algorithms and ML models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from functools import wraps
import time

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    from cupy import RawKernel
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import numba
    from numba import cuda, float32, float64, int32, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


def gpu_timer(func: Callable) -> Callable:
    """Decorator to time GPU operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            logger.debug(f"{func.__name__} GPU execution time: {end_time - start_time:.4f}s")
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} CPU execution time: {end_time - start_time:.4f}s")
        return result
    return wrapper


class CUDAOperations:
    """
    CUDA operations for GPU-accelerated computing.
    
    Provides optimized implementations of common operations
    used in trading algorithms and ML models.
    """
    
    def __init__(self, device_id: int = 0, precision: str = 'float32'):
        """
        Initialize CUDA operations.
        
        Args:
            device_id: GPU device ID
            precision: Computation precision ('float32' or 'float64')
        """
        self.device_id = device_id
        self.precision = precision
        self.dtype = torch.float32 if precision == 'float32' else torch.float64
        
        # Initialize device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
        
        # Compile custom kernels
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile custom CUDA kernels."""
        if CUPY_AVAILABLE and cp.cuda.is_available():
            # Custom kernel for rolling window statistics
            self.rolling_stats_kernel = RawKernel(r'''
            extern "C" __global__
            void rolling_stats(const float* data, float* mean, float* std, 
                             int n_samples, int window_size) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (idx >= n_samples - window_size + 1) return;
                
                float sum = 0.0f;
                float sum_sq = 0.0f;
                
                for (int i = 0; i < window_size; i++) {
                    float val = data[idx + i];
                    sum += val;
                    sum_sq += val * val;
                }
                
                float avg = sum / window_size;
                mean[idx] = avg;
                std[idx] = sqrtf(sum_sq / window_size - avg * avg);
            }
            ''', 'rolling_stats')
            
            # Custom kernel for technical indicators
            self.ema_kernel = RawKernel(r'''
            extern "C" __global__
            void ema_kernel(const float* data, float* ema, 
                           int n_samples, float alpha) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (idx >= n_samples) return;
                
                if (idx == 0) {
                    ema[idx] = data[idx];
                } else {
                    ema[idx] = alpha * data[idx] + (1 - alpha) * ema[idx - 1];
                }
            }
            ''', 'ema_kernel')
    
    @gpu_timer
    def rolling_window_statistics(self, 
                                data: Union[np.ndarray, torch.Tensor],
                                window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate rolling window mean and standard deviation on GPU.
        
        Args:
            data: Input data
            window_size: Window size
            
        Returns:
            Tuple of (mean, std) tensors
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, dtype=self.dtype)
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device, dtype=self.dtype)
        
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        n_samples, n_features = data.shape
        
        # Use unfold for efficient rolling window
        windows = data.unfold(0, window_size, 1)  # (n_windows, n_features, window_size)
        
        # Calculate statistics
        mean = windows.mean(dim=2)
        std = windows.std(dim=2)
        
        # Pad to match original length
        pad_size = window_size - 1
        mean = F.pad(mean, (0, 0, pad_size, 0), value=float('nan'))
        std = F.pad(std, (0, 0, pad_size, 0), value=float('nan'))
        
        return mean, std
    
    @gpu_timer
    def calculate_returns(self,
                         prices: Union[np.ndarray, torch.Tensor],
                         method: str = 'simple') -> torch.Tensor:
        """
        Calculate returns on GPU.
        
        Args:
            prices: Price data
            method: 'simple' or 'log' returns
            
        Returns:
            Returns tensor
        """
        if isinstance(prices, np.ndarray):
            prices = torch.from_numpy(prices).to(self.device, dtype=self.dtype)
        else:
            prices = prices.to(self.device, dtype=self.dtype)
        
        if method == 'simple':
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
        elif method == 'log':
            returns = torch.log(prices[1:] / prices[:-1])
        else:
            raise ValueError(f"Unknown return method: {method}")
        
        # Pad with NaN to maintain length
        returns = F.pad(returns, (0, 0, 1, 0), value=float('nan'))
        
        return returns
    
    @gpu_timer
    def exponential_moving_average(self,
                                 data: Union[np.ndarray, torch.Tensor],
                                 alpha: float) -> torch.Tensor:
        """
        Calculate exponential moving average on GPU.
        
        Args:
            data: Input data
            alpha: Smoothing factor (0 < alpha < 1)
            
        Returns:
            EMA tensor
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, dtype=self.dtype)
        else:
            data = data.to(self.device, dtype=self.dtype)
        
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        n_samples, n_features = data.shape
        
        # Initialize EMA
        ema = torch.zeros_like(data)
        ema[0] = data[0]
        
        # Calculate EMA using cumulative product trick
        # EMA[t] = alpha * X[t] + (1-alpha) * EMA[t-1]
        beta = 1 - alpha
        
        for i in range(1, n_samples):
            ema[i] = alpha * data[i] + beta * ema[i-1]
        
        return ema
    
    @gpu_timer
    def calculate_rsi(self,
                     prices: Union[np.ndarray, torch.Tensor],
                     period: int = 14) -> torch.Tensor:
        """
        Calculate Relative Strength Index on GPU.
        
        Args:
            prices: Price data
            period: RSI period
            
        Returns:
            RSI tensor
        """
        if isinstance(prices, np.ndarray):
            prices = torch.from_numpy(prices).to(self.device, dtype=self.dtype)
        else:
            prices = prices.to(self.device, dtype=self.dtype)
        
        # Calculate price changes
        deltas = prices[1:] - prices[:-1]
        deltas = F.pad(deltas, (0, 0, 1, 0), value=0)
        
        # Separate gains and losses
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        # Calculate average gains and losses
        alpha = 1.0 / period
        avg_gains = self.exponential_moving_average(gains, alpha)
        avg_losses = self.exponential_moving_average(losses, alpha)
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Set initial values to NaN
        rsi[:period] = float('nan')
        
        return rsi.squeeze()
    
    @gpu_timer
    def calculate_bollinger_bands(self,
                                prices: Union[np.ndarray, torch.Tensor],
                                period: int = 20,
                                num_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate Bollinger Bands on GPU.
        
        Args:
            prices: Price data
            period: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if isinstance(prices, np.ndarray):
            prices = torch.from_numpy(prices).to(self.device, dtype=self.dtype)
        else:
            prices = prices.to(self.device, dtype=self.dtype)
        
        # Calculate rolling mean and std
        mean, std = self.rolling_window_statistics(prices, period)
        
        # Calculate bands
        upper_band = mean + num_std * std
        lower_band = mean - num_std * std
        
        return upper_band.squeeze(), mean.squeeze(), lower_band.squeeze()
    
    @gpu_timer
    def calculate_macd(self,
                      prices: Union[np.ndarray, torch.Tensor],
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate MACD on GPU.
        
        Args:
            prices: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        if isinstance(prices, np.ndarray):
            prices = torch.from_numpy(prices).to(self.device, dtype=self.dtype)
        else:
            prices = prices.to(self.device, dtype=self.dtype)
        
        # Calculate EMAs
        ema_fast = self.exponential_moving_average(prices, 2.0 / (fast_period + 1))
        ema_slow = self.exponential_moving_average(prices, 2.0 / (slow_period + 1))
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate signal line
        signal = self.exponential_moving_average(macd, 2.0 / (signal_period + 1))
        
        # Calculate histogram
        histogram = macd - signal
        
        return macd.squeeze(), signal.squeeze(), histogram.squeeze()
    
    @gpu_timer
    def correlation_matrix(self,
                         data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Calculate correlation matrix on GPU.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Correlation matrix
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, dtype=self.dtype)
        else:
            data = data.to(self.device, dtype=self.dtype)
        
        # Standardize data
        data_mean = data.mean(dim=0, keepdim=True)
        data_std = data.std(dim=0, keepdim=True)
        data_normalized = (data - data_mean) / (data_std + 1e-10)
        
        # Calculate correlation matrix
        n_samples = data.shape[0]
        corr = torch.matmul(data_normalized.T, data_normalized) / (n_samples - 1)
        
        return corr
    
    @gpu_timer
    def covariance_matrix(self,
                         data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Calculate covariance matrix on GPU.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Covariance matrix
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, dtype=self.dtype)
        else:
            data = data.to(self.device, dtype=self.dtype)
        
        # Center data
        data_mean = data.mean(dim=0, keepdim=True)
        data_centered = data - data_mean
        
        # Calculate covariance matrix
        n_samples = data.shape[0]
        cov = torch.matmul(data_centered.T, data_centered) / (n_samples - 1)
        
        return cov
    
    @gpu_timer
    def parallel_portfolio_metrics(self,
                                 returns: Union[np.ndarray, torch.Tensor],
                                 weights: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate portfolio metrics in parallel on GPU.
        
        Args:
            returns: Asset returns (n_samples, n_assets)
            weights: Portfolio weights (n_portfolios, n_assets)
            
        Returns:
            Dictionary of metrics for each portfolio
        """
        if isinstance(returns, np.ndarray):
            returns = torch.from_numpy(returns).to(self.device, dtype=self.dtype)
        else:
            returns = returns.to(self.device, dtype=self.dtype)
        
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).to(self.device, dtype=self.dtype)
        else:
            weights = weights.to(self.device, dtype=self.dtype)
        
        # Ensure weights is 2D
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        
        # Calculate portfolio returns
        portfolio_returns = torch.matmul(returns, weights.T)  # (n_samples, n_portfolios)
        
        # Calculate metrics
        mean_returns = portfolio_returns.mean(dim=0)
        std_returns = portfolio_returns.std(dim=0)
        sharpe_ratios = mean_returns / (std_returns + 1e-10) * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod(dim=0)
        running_max = cumulative_returns.cummax(dim=0)[0]
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdowns = drawdowns.min(dim=0)[0]
        
        # Convert to CPU for output
        metrics = {
            'mean_returns': mean_returns.cpu().numpy(),
            'std_returns': std_returns.cpu().numpy(),
            'sharpe_ratios': sharpe_ratios.cpu().numpy(),
            'max_drawdowns': max_drawdowns.cpu().numpy()
        }
        
        return metrics
    
    @gpu_timer
    def monte_carlo_simulation(self,
                             initial_price: float,
                             drift: float,
                             volatility: float,
                             time_horizon: int,
                             n_simulations: int,
                             dt: float = 1/252) -> torch.Tensor:
        """
        Run Monte Carlo simulation on GPU.
        
        Args:
            initial_price: Initial asset price
            drift: Expected return (annualized)
            volatility: Volatility (annualized)
            time_horizon: Number of time steps
            n_simulations: Number of simulation paths
            dt: Time step size
            
        Returns:
            Simulated price paths (time_horizon, n_simulations)
        """
        # Generate random numbers on GPU
        torch.manual_seed(42)  # For reproducibility
        
        # Brownian motion increments
        dW = torch.randn(time_horizon, n_simulations, device=self.device, dtype=self.dtype) * np.sqrt(dt)
        
        # Calculate price paths using geometric Brownian motion
        # S(t+dt) = S(t) * exp((drift - 0.5*vol^2)*dt + vol*sqrt(dt)*dW)
        returns = (drift - 0.5 * volatility**2) * dt + volatility * dW
        
        # Cumulative product to get price paths
        price_paths = torch.zeros(time_horizon + 1, n_simulations, device=self.device, dtype=self.dtype)
        price_paths[0] = initial_price
        price_paths[1:] = initial_price * torch.exp(returns.cumsum(dim=0))
        
        return price_paths
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def cleanup(self):
        """Cleanup GPU resources."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Numba CUDA kernels for additional operations
if NUMBA_AVAILABLE:
    @cuda.jit
    def matrix_multiply_kernel(A, B, C):
        """CUDA kernel for matrix multiplication."""
        row, col = cuda.grid(2)
        
        if row < C.shape[0] and col < C.shape[1]:
            temp = 0.0
            for k in range(A.shape[1]):
                temp += A[row, k] * B[k, col]
            C[row, col] = temp
    
    @cuda.jit
    def vectorized_operations_kernel(a, b, c, operation):
        """CUDA kernel for vectorized operations."""
        idx = cuda.grid(1)
        
        if idx < a.shape[0]:
            if operation == 0:  # Addition
                c[idx] = a[idx] + b[idx]
            elif operation == 1:  # Subtraction
                c[idx] = a[idx] - b[idx]
            elif operation == 2:  # Multiplication
                c[idx] = a[idx] * b[idx]
            elif operation == 3:  # Division
                c[idx] = a[idx] / (b[idx] + 1e-10)