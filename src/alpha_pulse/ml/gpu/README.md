# GPU Acceleration Module

This module provides GPU-accelerated computing capabilities for AlphaPulse's machine learning operations.

## Installation

```bash
# Install with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x
pip install tensorflow[and-cuda]
pip install numba
```

## Quick Start

```python
from alpha_pulse.ml.gpu import GPUService, get_default_config

# Initialize and use GPU service
async def main():
    config = get_default_config()
    gpu_service = GPUService(config)
    
    await gpu_service.start()
    
    # Create and train model
    model = gpu_service.create_model(
        model_type='neural_network',
        model_name='trading_model',
        input_size=100,
        hidden_sizes=[256, 128],
        output_size=1
    )
    
    results = await gpu_service.train_model(
        'trading_model',
        X_train, y_train,
        epochs=100
    )
    
    # Make predictions
    predictions = await gpu_service.predict(
        'trading_model',
        X_test
    )
    
    await gpu_service.stop()

# Run
import asyncio
asyncio.run(main())
```

## Components

- **gpu_config.py**: Configuration management
- **gpu_manager.py**: GPU resource allocation and monitoring
- **cuda_operations.py**: CUDA kernels for financial computations
- **gpu_models.py**: GPU-optimized ML models
- **memory_manager.py**: Advanced memory management
- **batch_processor.py**: Dynamic batching for inference
- **gpu_utilities.py**: Helper functions and profiling
- **gpu_service.py**: High-level service interface

## Features

- Multi-GPU support with automatic load balancing
- Dynamic memory management with pooling
- Mixed precision training (FP16/FP32)
- Optimized batch processing for inference
- Real-time performance monitoring
- Technical indicator calculations on GPU
- Monte Carlo simulations
- Portfolio optimization

## Configuration

### Predefined Configurations

```python
from alpha_pulse.ml.gpu import (
    get_default_config,      # Balanced configuration
    get_inference_config,    # Optimized for low-latency inference
    get_training_config,     # Optimized for model training
    get_limited_memory_config  # For GPUs with limited memory
)
```

### Custom Configuration

```python
from alpha_pulse.ml.gpu import GPUConfigBuilder, PrecisionMode

config = (GPUConfigBuilder()
    .auto_detect_devices()
    .set_precision(PrecisionMode.MIXED)
    .set_batching(max_batch_size=128)
    .optimize_for_inference()
    .build())
```

## Performance Tips

1. **Use Mixed Precision**: Enables faster training with minimal accuracy loss
2. **Dynamic Batching**: Groups requests for efficient processing
3. **Memory Pooling**: Reduces allocation overhead
4. **Multi-GPU**: Utilize all available GPUs for training
5. **Profile Your Code**: Use built-in profiler to identify bottlenecks

## Testing

```bash
# Run GPU tests
pytest src/alpha_pulse/tests/ml/gpu/ -v

# Run specific test
pytest src/alpha_pulse/tests/ml/gpu/test_gpu_service.py -v
```

## Requirements

- NVIDIA GPU with compute capability >= 3.5
- CUDA Toolkit >= 11.0
- cuDNN >= 8.0
- Python >= 3.8

## License

See AlphaPulse main LICENSE file.