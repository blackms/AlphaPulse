# GPU Acceleration Module Documentation

## Overview

The GPU Acceleration module provides high-performance computing capabilities for AlphaPulse's machine learning operations. It enables efficient training and inference of ML models using NVIDIA GPUs, with support for multiple frameworks including PyTorch, TensorFlow, CuPy, and Numba.

## Key Features

- **Multi-GPU Support**: Automatic GPU discovery and resource allocation
- **Memory Management**: Advanced memory pooling and optimization
- **Dynamic Batching**: Intelligent request batching for optimal throughput
- **Mixed Precision**: Support for FP16/FP32 mixed precision training
- **Model Optimization**: GPU-optimized implementations of common ML models
- **Real-time Monitoring**: Comprehensive GPU metrics and performance tracking
- **Framework Agnostic**: Support for PyTorch, TensorFlow, CuPy, and Numba

## Architecture

```
gpu/
├── gpu_config.py       # Configuration management
├── gpu_manager.py      # GPU resource allocation
├── cuda_operations.py  # CUDA kernels and operations
├── gpu_models.py       # GPU-optimized ML models
├── memory_manager.py   # Memory pooling and optimization
├── batch_processor.py  # Dynamic batching for inference
├── gpu_utilities.py    # Helper functions and profiling
└── gpu_service.py      # High-level service interface
```

## Quick Start

### Basic Usage

```python
from alpha_pulse.ml.gpu import GPUService, get_default_config

# Initialize GPU service
config = get_default_config()
gpu_service = GPUService(config)

# Start service
await gpu_service.start()

# Create a GPU-optimized model
model = gpu_service.create_model(
    model_type='neural_network',
    model_name='price_predictor',
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=1
)

# Train model
results = await gpu_service.train_model(
    'price_predictor',
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64
)

# Make predictions
predictions = await gpu_service.predict(
    'price_predictor',
    X_test
)

# Stop service
await gpu_service.stop()
```

### Configuration

#### Default Configuration

```python
from alpha_pulse.ml.gpu import get_default_config

config = get_default_config()
```

#### Inference-Optimized Configuration

```python
from alpha_pulse.ml.gpu import get_inference_config

config = get_inference_config()
# Optimized for low-latency inference
# - Dynamic batching
# - FP16 precision
# - Larger batch sizes
```

#### Training-Optimized Configuration

```python
from alpha_pulse.ml.gpu import get_training_config

config = get_training_config()
# Optimized for model training
# - Mixed precision training
# - Memory growth enabled
# - Multi-GPU support
```

#### Custom Configuration

```python
from alpha_pulse.ml.gpu import GPUConfigBuilder

config = (GPUConfigBuilder()
    .auto_detect_devices()
    .set_precision(PrecisionMode.MIXED)
    .set_memory_policy(MemoryGrowthPolicy.GROW_AS_NEEDED)
    .set_batching(max_batch_size=128, strategy="adaptive")
    .enable_multi_gpu("data_parallel")
    .build())
```

## GPU Models

### Linear Regression

```python
model = gpu_service.create_model(
    model_type='linear',
    model_name='linear_model',
    n_features=50,
    learning_rate=0.01
)
```

### Neural Network

```python
model = gpu_service.create_model(
    model_type='neural_network',
    model_name='nn_model',
    input_size=100,
    hidden_sizes=[512, 256, 128],
    output_size=10,
    dropout=0.3,
    activation='relu'
)
```

### LSTM

```python
model = gpu_service.create_model(
    model_type='lstm',
    model_name='lstm_model',
    input_size=50,
    hidden_size=256,
    num_layers=3,
    output_size=1,
    bidirectional=True
)
```

### Transformer

```python
model = gpu_service.create_model(
    model_type='transformer',
    model_name='transformer_model',
    input_size=100,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    output_size=1
)
```

## CUDA Operations

### Technical Indicators

```python
# Calculate technical indicators on GPU
indicators = gpu_service.calculate_technical_indicators(
    prices,
    indicators=['returns', 'rsi', 'macd', 'bollinger', 'ema_20']
)

# Access results
returns = indicators['returns']
rsi_values = indicators['rsi']
macd_line = indicators['macd']
```

### Monte Carlo Simulation

```python
# Run Monte Carlo simulation on GPU
price_paths = gpu_service.run_monte_carlo(
    initial_price=100.0,
    drift=0.05,        # 5% annual return
    volatility=0.20,   # 20% annual volatility
    time_horizon=252,  # 1 year
    n_simulations=10000
)
```

### Portfolio Optimization

```python
# Optimize portfolio weights on GPU
optimal_weights = gpu_service.optimize_portfolio(
    returns,
    method='mean_variance',
    constraints={
        'min_weight': 0.0,
        'max_weight': 0.3,
        'target_return': 0.10
    }
)
```

## Batch Processing

### Dynamic Batching

```python
from alpha_pulse.ml.gpu import GPUBatchProcessor, BatchingStrategy

# Create batch processor
batch_processor = GPUBatchProcessor(
    gpu_manager,
    max_batch_size=64,
    max_queue_size=1000,
    num_workers=4,
    batching_strategy=BatchingStrategy.ADAPTIVE
)

# Register model for batch processing
batch_processor.register_model('model_name', model, device_id=0)

# Process requests (automatically batched)
result = await batch_processor.process_async(
    data,
    model_name='model_name',
    priority=5  # Higher priority = processed sooner
)
```

### Streaming Processing

```python
from alpha_pulse.ml.gpu import StreamingBatchProcessor

# Create streaming processor
streaming = StreamingBatchProcessor(
    batch_processor,
    window_size=100,
    update_interval=0.1
)

# Add data stream
streaming.add_stream('btc_stream', 'price_model')

# Add real-time data
streaming.add_data('btc_stream', new_price_data)

# Get predictions
predictions = await streaming.get_predictions('btc_stream')
```

## Memory Management

### Memory Pool Configuration

```python
from alpha_pulse.ml.gpu import MemoryConfig, MemoryGrowthPolicy

memory_config = MemoryConfig(
    growth_policy=MemoryGrowthPolicy.GROW_AS_NEEDED,
    pool_size_mb=2048,
    gc_threshold=0.8,      # Trigger GC at 80% usage
    defrag_threshold=0.7,  # Defragment at 70% fragmentation
    enable_pooling=True
)
```

### Memory Monitoring

```python
# Get memory usage
memory_info = gpu_service.memory_manager.get_memory_info(device_id=0)
print(f"Allocated: {memory_info['allocated_memory'] / 1e9:.2f} GB")
print(f"Free: {memory_info['free_memory'] / 1e9:.2f} GB")

# Get optimization recommendations
recommendations = gpu_service.memory_manager.get_optimization_recommendations()
```

## Performance Optimization

### Profiling

```python
from alpha_pulse.ml.gpu import gpu_profiler

# Profile a function
@gpu_profiler.profile("matrix_multiply")
def compute_correlation(data):
    return gpu_service.cuda_ops.correlation_matrix(data)

# Get profiling results
summary = gpu_profiler.get_summary("matrix_multiply")
print(f"Average time: {summary['avg_time']:.4f}s")
print(f"Total calls: {summary['count']}")
```

### Benchmarking

```python
from alpha_pulse.ml.gpu import benchmark_operation

# Benchmark an operation
results = benchmark_operation(
    func=model.predict,
    args=(test_data,),
    num_runs=100,
    warmup_runs=10,
    device_id=0
)

print(f"Mean inference time: {results['mean_time']*1000:.2f}ms")
print(f"Throughput: {1/results['mean_time']:.0f} samples/sec")
```

### Multi-GPU Training

```python
from alpha_pulse.ml.gpu import MultiGPUWrapper

# Wrap model for multi-GPU
multi_gpu_model = MultiGPUWrapper(
    base_model,
    device_ids=[0, 1, 2, 3]  # Use 4 GPUs
)

# Training automatically uses all GPUs
wrapped_model = multi_gpu_model.get_model()
```

## Monitoring and Metrics

### Real-time Monitoring

```python
# Enable monitoring in configuration
config.monitoring.enable_monitoring = True
config.monitoring.monitor_interval_sec = 5.0
config.monitoring.alert_on_oom = True

# Get comprehensive metrics
metrics = gpu_service.get_metrics()

# Device metrics
for device_id, device_metrics in metrics['devices'].items():
    print(f"GPU {device_id}:")
    print(f"  Memory Usage: {device_metrics['memory_usage']:.1%}")
    print(f"  Utilization: {device_metrics['utilization']:.1%}")
    print(f"  Temperature: {device_metrics['temperature']}°C")

# Batch processing metrics
batch_stats = metrics['batch_processing']
print(f"Average batch size: {batch_stats['batching']['avg_batch_size']:.1f}")
print(f"Average wait time: {batch_stats['batching']['avg_wait_time']*1000:.1f}ms")
```

### Performance Statistics

```python
# Get model-specific performance stats
perf_stats = gpu_service.batch_processor.get_performance_stats()

for model_name, stats in perf_stats['models'].items():
    print(f"{model_name}:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Average latency: {stats['avg_latency']*1000:.2f}ms")
    print(f"  Throughput: {stats['avg_throughput']:.0f} req/sec")
    print(f"  Errors: {stats['errors']}")
```

## Error Handling

### GPU Diagnostics

```python
from alpha_pulse.ml.gpu import diagnose_cuda_errors

# Diagnose CUDA issues
diagnostics = diagnose_cuda_errors()

if diagnostics['errors']:
    print("CUDA Errors found:")
    for error in diagnostics['errors']:
        print(f"  - {error}")
    
    print("\nSuggestions:")
    for suggestion in diagnostics['suggestions']:
        print(f"  - {suggestion}")
```

### Fallback to CPU

```python
# Configure CPU fallback
config.fallback_to_cpu = True

# Service automatically uses CPU if no GPU available
model = gpu_service.create_model(
    model_type='neural_network',
    model_name='cpu_fallback_model',
    input_size=100,
    hidden_sizes=[64, 32],
    output_size=1
)
```

## Best Practices

### 1. Memory Management

- Use memory pooling for frequent allocations
- Enable garbage collection for long-running processes
- Monitor memory fragmentation
- Set appropriate memory limits per GPU

### 2. Batch Processing

- Use adaptive batching for variable workloads
- Set appropriate queue sizes to prevent OOM
- Prioritize time-sensitive requests
- Monitor queue depths and adjust workers

### 3. Model Optimization

- Use mixed precision for training when possible
- Enable JIT compilation for inference
- Profile models to identify bottlenecks
- Use appropriate batch sizes for your GPU

### 4. Multi-GPU Usage

- Use data parallelism for large batches
- Consider model parallelism for very large models
- Balance load across GPUs
- Monitor individual GPU utilization

### 5. Production Deployment

- Enable comprehensive monitoring
- Set up alerts for critical metrics
- Implement proper error handling
- Use configuration files for different environments
- Regular profiling and optimization

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Enable memory pooling
   - Use gradient checkpointing
   - Clear cache regularly

2. **Low GPU Utilization**
   - Increase batch size
   - Use multiple workers
   - Check data loading bottlenecks
   - Enable CUDA optimization

3. **High Latency**
   - Use dynamic batching
   - Optimize model architecture
   - Enable mixed precision
   - Profile and identify bottlenecks

4. **Multi-GPU Scaling Issues**
   - Check interconnect bandwidth
   - Verify load balancing
   - Monitor individual GPU metrics
   - Consider different parallelism strategies

## Configuration Examples

### High-Frequency Trading Configuration

```yaml
devices:
  - device_id: 0
    memory_fraction: 0.95
    priority: 0

compute:
  precision: fp16
  enable_tf32: true
  enable_cudnn: true
  cudnn_benchmark: true

batching:
  max_batch_size: 1
  max_wait_time_ms: 0.1
  strategy: fixed_size

optimization:
  optimize_for_inference: true
  enable_jit: true
```

### Deep Learning Training Configuration

```yaml
devices:
  - device_id: 0
    memory_fraction: 0.9
  - device_id: 1
    memory_fraction: 0.9

memory:
  growth_policy: grow_as_needed
  pool_size_mb: 4096
  gc_threshold: 0.85

compute:
  precision: mixed
  enable_amp: true

optimization:
  gradient_checkpointing: true
  gradient_accumulation_steps: 4

multi_gpu_strategy: distributed
```

## API Reference

For detailed API documentation, see the individual module docstrings:

- `gpu_config.py`: Configuration classes and builders
- `gpu_manager.py`: GPU resource management
- `cuda_operations.py`: CUDA kernel operations
- `gpu_models.py`: GPU-optimized model implementations
- `memory_manager.py`: Memory management and pooling
- `batch_processor.py`: Dynamic batching system
- `gpu_utilities.py`: Utility functions and helpers
- `gpu_service.py`: High-level service interface