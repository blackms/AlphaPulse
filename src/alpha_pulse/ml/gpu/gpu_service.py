"""
GPU Acceleration Service.

This module provides a high-level service interface for GPU-accelerated
machine learning operations in the AlphaPulse trading system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from .gpu_manager import GPUManager
from .gpu_config import GPUConfig, get_default_config
from .batch_processor import GPUBatchProcessor, BatchingStrategy
from .memory_manager import GPUMemoryManager
from .cuda_operations import CUDAOperations
from .gpu_models import (
    GPULinearRegression,
    GPUNeuralNetwork,
    GPULSTMModel,
    GPUTransformerModel,
    MultiGPUWrapper
)
from .gpu_utilities import (
    get_gpu_info,
    select_best_gpu,
    optimize_cuda_settings,
    gpu_profiler
)

logger = logging.getLogger(__name__)


class GPUService:
    """
    Unified GPU acceleration service for AlphaPulse.
    
    This service provides:
    - GPU resource management
    - Model training and inference
    - Batch processing optimization
    - Memory management
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        """
        Initialize GPU service.
        
        Args:
            config: GPU configuration (uses default if None)
        """
        self.config = config or get_default_config()
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            logger.warning(f"Configuration issues: {issues}")
        
        # Initialize components
        self._initialize_components()
        
        # Service state
        self.is_running = False
        self._models = {}
        self._training_tasks = {}
        
        logger.info("GPU service initialized")
    
    def _initialize_components(self):
        """Initialize GPU service components."""
        # Apply global settings
        if self.config.visible_devices:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                map(str, self.config.visible_devices)
            )
        
        # Optimize CUDA settings
        optimize_cuda_settings()
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager(
            max_memory_fraction=self.config.memory.gc_threshold,
            allow_growth=self.config.memory.growth_policy.value != "fixed"
        )
        
        # Initialize memory manager
        self.memory_manager = GPUMemoryManager(
            max_memory_fraction=self.config.memory.gc_threshold,
            enable_memory_pool=self.config.memory.enable_pooling,
            gc_threshold=self.config.memory.gc_threshold,
            defrag_threshold=self.config.memory.defrag_threshold
        )
        
        # Initialize batch processor
        self.batch_processor = GPUBatchProcessor(
            gpu_manager=self.gpu_manager,
            max_batch_size=self.config.batching.max_batch_size,
            max_queue_size=self.config.batching.queue_size,
            num_workers=self.config.batching.num_workers,
            batching_strategy=BatchingStrategy(self.config.batching.strategy)
        )
        
        # Initialize CUDA operations
        self.cuda_ops = CUDAOperations(
            device_id=0,  # Primary device
            precision=self.config.compute.precision.value
        )
    
    async def start(self):
        """Start GPU service."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring if enabled
        if self.config.monitoring.enable_monitoring:
            asyncio.create_task(self._monitor_loop())
        
        logger.info("GPU service started")
    
    async def stop(self):
        """Stop GPU service."""
        self.is_running = False
        
        # Cleanup components
        self.batch_processor.shutdown()
        self.memory_manager.cleanup()
        self.gpu_manager.cleanup()
        
        logger.info("GPU service stopped")
    
    async def _monitor_loop(self):
        """Monitoring loop for GPU metrics."""
        while self.is_running:
            try:
                # Collect metrics
                metrics = self.get_metrics()
                
                # Check for alerts
                for device_id, device_metrics in metrics['devices'].items():
                    # Memory alert
                    if (self.config.monitoring.alert_on_oom and
                        device_metrics['memory_usage'] > self.config.monitoring.alert_threshold_memory):
                        logger.warning(f"GPU {device_id} memory usage high: "
                                     f"{device_metrics['memory_usage']:.1%}")
                    
                    # Utilization alert
                    if device_metrics['utilization'] > self.config.monitoring.alert_threshold_util:
                        logger.warning(f"GPU {device_id} utilization high: "
                                     f"{device_metrics['utilization']:.1%}")
                
                await asyncio.sleep(self.config.monitoring.monitor_interval_sec)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def create_model(self,
                    model_type: str,
                    model_name: str,
                    **kwargs) -> Any:
        """
        Create a GPU-optimized model.
        
        Args:
            model_type: Type of model (linear, neural_network, lstm, transformer)
            model_name: Unique model name
            **kwargs: Model-specific parameters
            
        Returns:
            Created model
        """
        # Select GPU
        device_id = select_best_gpu(min_memory_gb=2.0)
        if device_id is None and not self.config.fallback_to_cpu:
            raise RuntimeError("No suitable GPU available")
        
        # Create model based on type
        if model_type == 'linear':
            model = GPULinearRegression(**kwargs)
        elif model_type == 'neural_network':
            model = GPUNeuralNetwork(**kwargs)
        elif model_type == 'lstm':
            model = GPULSTMModel(**kwargs)
        elif model_type == 'transformer':
            model = GPUTransformerModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to GPU if available
        if device_id is not None:
            model.to_gpu(device_id)
        
        # Register for batch processing
        if hasattr(model, 'eval'):  # PyTorch model
            self.batch_processor.register_model(model_name, model, device_id)
        
        # Store model
        self._models[model_name] = {
            'model': model,
            'type': model_type,
            'device_id': device_id,
            'created_at': datetime.now()
        }
        
        logger.info(f"Created {model_type} model '{model_name}' on GPU {device_id}")
        
        return model
    
    async def train_model(self,
                         model_name: str,
                         X: np.ndarray,
                         y: np.ndarray,
                         validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                         **training_params) -> Dict[str, Any]:
        """
        Train a model on GPU.
        
        Args:
            model_name: Model name
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            **training_params: Training parameters
            
        Returns:
            Training results
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self._models[model_name]
        model = model_info['model']
        
        # Profile training
        gpu_profiler.start_profile(f"train_{model_name}")
        
        try:
            # Allocate memory for training data
            data_size = X.nbytes + y.nbytes
            if validation_data:
                data_size += validation_data[0].nbytes + validation_data[1].nbytes
            
            device_id = self.gpu_manager.allocate_gpu(
                memory_required=data_size * 2,  # 2x for gradients
                task_type='training',
                priority=10
            )
            
            if device_id is None:
                raise RuntimeError("Insufficient GPU memory for training")
            
            # Train model
            start_time = datetime.now()
            
            if hasattr(model, 'fit'):
                # Use model's fit method
                model.fit(X, y, validation_data=validation_data, **training_params)
            else:
                raise NotImplementedError(f"Model {model_name} does not support training")
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Get training results
            results = {
                'model_name': model_name,
                'training_time': training_time,
                'samples_trained': len(X),
                'device_id': device_id,
                'parameters': training_params
            }
            
            # Save training task
            task_id = f"train_{model_name}_{int(start_time.timestamp())}"
            self._training_tasks[task_id] = {
                'model_name': model_name,
                'status': 'completed',
                'results': results,
                'timestamp': datetime.now()
            }
            
            return results
            
        finally:
            gpu_profiler.end_profile()
            
            # Release GPU
            if 'device_id' in locals():
                self.gpu_manager.release_gpu(device_id)
    
    async def predict(self,
                     model_name: str,
                     X: np.ndarray,
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using GPU model.
        
        Args:
            model_name: Model name
            X: Input features
            batch_size: Batch size for processing
            
        Returns:
            Predictions
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        # Use batch processor for registered models
        if model_name in self.batch_processor.models:
            # Process through batch processor
            predictions = await self.batch_processor.process_async(
                X, model_name, priority=5
            )
            return predictions
        
        # Direct prediction for non-registered models
        model_info = self._models[model_name]
        model = model_info['model']
        
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            return predictions
        else:
            raise NotImplementedError(f"Model {model_name} does not support prediction")
    
    def calculate_technical_indicators(self,
                                     prices: np.ndarray,
                                     indicators: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate technical indicators on GPU.
        
        Args:
            prices: Price data
            indicators: List of indicators to calculate
            
        Returns:
            Dictionary of indicator values
        """
        results = {}
        
        for indicator in indicators:
            if indicator == 'returns':
                results['returns'] = self.cuda_ops.calculate_returns(prices)
            elif indicator == 'rsi':
                results['rsi'] = self.cuda_ops.calculate_rsi(prices)
            elif indicator == 'macd':
                macd, signal, hist = self.cuda_ops.calculate_macd(prices)
                results['macd'] = macd
                results['macd_signal'] = signal
                results['macd_histogram'] = hist
            elif indicator == 'bollinger':
                upper, middle, lower = self.cuda_ops.calculate_bollinger_bands(prices)
                results['bb_upper'] = upper
                results['bb_middle'] = middle
                results['bb_lower'] = lower
            elif indicator.startswith('ema_'):
                period = int(indicator.split('_')[1])
                alpha = 2.0 / (period + 1)
                results[indicator] = self.cuda_ops.exponential_moving_average(prices, alpha)
            else:
                logger.warning(f"Unknown indicator: {indicator}")
        
        return results
    
    def run_monte_carlo(self,
                       initial_price: float,
                       drift: float,
                       volatility: float,
                       time_horizon: int,
                       n_simulations: int = 10000) -> np.ndarray:
        """
        Run Monte Carlo simulation on GPU.
        
        Args:
            initial_price: Initial asset price
            drift: Expected return
            volatility: Volatility
            time_horizon: Number of time steps
            n_simulations: Number of simulations
            
        Returns:
            Simulated price paths
        """
        paths = self.cuda_ops.monte_carlo_simulation(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            time_horizon=time_horizon,
            n_simulations=n_simulations
        )
        
        # Convert to numpy if needed
        if hasattr(paths, 'cpu'):
            paths = paths.cpu().numpy()
        
        return paths
    
    def optimize_portfolio(self,
                          returns: np.ndarray,
                          method: str = 'mean_variance',
                          constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Optimize portfolio weights on GPU.
        
        Args:
            returns: Asset returns
            method: Optimization method
            constraints: Portfolio constraints
            
        Returns:
            Optimal weights
        """
        # Calculate statistics on GPU
        cov_matrix = self.cuda_ops.covariance_matrix(returns)
        mean_returns = returns.mean(axis=0)
        
        # Simple mean-variance optimization
        if method == 'mean_variance':
            # This is a simplified implementation
            # In practice, you'd use a proper optimizer
            n_assets = returns.shape[1]
            weights = np.ones(n_assets) / n_assets  # Equal weight baseline
            
            # Apply constraints if provided
            if constraints:
                if 'min_weight' in constraints:
                    weights = np.maximum(weights, constraints['min_weight'])
                if 'max_weight' in constraints:
                    weights = np.minimum(weights, constraints['max_weight'])
                
                # Normalize
                weights = weights / weights.sum()
            
            return weights
        else:
            raise NotImplementedError(f"Method {method} not implemented")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive GPU metrics."""
        metrics = {
            'service_running': self.is_running,
            'models': {},
            'devices': {},
            'batch_processing': self.batch_processor.get_performance_stats(),
            'memory': {}
        }
        
        # Model information
        for name, info in self._models.items():
            metrics['models'][name] = {
                'type': info['type'],
                'device_id': info['device_id'],
                'created_at': info['created_at'].isoformat()
            }
        
        # Device metrics
        gpu_info = get_gpu_info()
        for gpu in gpu_info:
            device_id = gpu['device_id']
            metrics['devices'][device_id] = {
                'name': gpu['name'],
                'memory_usage': gpu.get('used_memory_gb', 0) / gpu['total_memory_gb'],
                'utilization': gpu.get('gpu_utilization', 0) / 100,
                'temperature': gpu.get('temperature', 0)
            }
        
        # Memory metrics
        for device_id in range(len(gpu_info)):
            memory_info = self.memory_manager.get_memory_info(device_id)
            metrics['memory'][device_id] = {
                'allocated_mb': memory_info['allocated_memory'] / 1024**2,
                'reserved_mb': memory_info['reserved_memory'] / 1024**2,
                'free_mb': memory_info['free_memory'] / 1024**2
            }
        
        return metrics
    
    def save_model(self, model_name: str, filepath: str):
        """Save GPU model to file."""
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self._models[model_name]
        model = model_info['model']
        
        # Save based on model type
        import torch
        if hasattr(model, 'state_dict'):
            # PyTorch model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_info['type'],
                'model_config': getattr(model, 'config', {})
            }, filepath)
        else:
            # Other model types
            import joblib
            joblib.dump(model, filepath)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str):
        """Load GPU model from file."""
        import torch
        
        if filepath.endswith('.pt') or filepath.endswith('.pth'):
            # PyTorch model
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Recreate model based on type
            model_type = checkpoint.get('model_type', 'neural_network')
            model_config = checkpoint.get('model_config', {})
            
            model = self.create_model(model_type, model_name, **model_config)
            
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Other model types
            import joblib
            model = joblib.load(filepath)
            
            # Register model
            self._models[model_name] = {
                'model': model,
                'type': 'unknown',
                'device_id': 0,
                'created_at': datetime.now()
            }
        
        logger.info(f"Model loaded from {filepath} as {model_name}")