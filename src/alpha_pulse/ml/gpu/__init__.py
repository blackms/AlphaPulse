"""
GPU Acceleration Module for AlphaPulse ML Operations.

This module provides GPU-accelerated computing capabilities for
machine learning model training and inference, optimized for
real-time trading applications.
"""

from .gpu_config import (
    GPUConfig,
    GPUDeviceConfig,
    MemoryConfig,
    ComputeConfig,
    BatchingConfig,
    OptimizationConfig,
    MonitoringConfig,
    GPUBackend,
    PrecisionMode,
    MemoryGrowthPolicy,
    GPUConfigBuilder,
    get_default_config,
    get_inference_config,
    get_training_config,
    get_limited_memory_config
)

from .gpu_manager import (
    GPUManager,
    GPUInfo,
    GPUAllocation
)

from .cuda_operations import (
    CUDAOperations,
    gpu_timer
)

from .gpu_models import (
    GPUModel,
    GPULinearRegression,
    GPUNeuralNetwork,
    GPULSTMModel,
    GPUTransformerModel,
    MultiGPUWrapper
)

from .memory_manager import (
    GPUMemoryManager,
    GPUMemoryPool,
    MemoryBlock
)

from .batch_processor import (
    GPUBatchProcessor,
    DynamicBatcher,
    StreamingBatchProcessor,
    BatchingStrategy,
    InferenceRequest,
    BatchedRequest
)

from .gpu_utilities import (
    GPUProfiler,
    gpu_profiler,
    get_gpu_info,
    select_best_gpu,
    optimize_cuda_settings,
    memory_efficient_load,
    estimate_model_memory,
    parallel_execute,
    benchmark_operation,
    convert_precision,
    diagnose_cuda_errors,
    save_gpu_profile,
    load_gpu_profile
)

from .gpu_service import GPUService

__all__ = [
    # Configuration
    'GPUConfig',
    'GPUDeviceConfig',
    'MemoryConfig',
    'ComputeConfig',
    'BatchingConfig',
    'OptimizationConfig',
    'MonitoringConfig',
    'GPUBackend',
    'PrecisionMode',
    'MemoryGrowthPolicy',
    'GPUConfigBuilder',
    'get_default_config',
    'get_inference_config',
    'get_training_config',
    'get_limited_memory_config',
    
    # Manager
    'GPUManager',
    'GPUInfo',
    'GPUAllocation',
    
    # Operations
    'CUDAOperations',
    'gpu_timer',
    
    # Models
    'GPUModel',
    'GPULinearRegression',
    'GPUNeuralNetwork',
    'GPULSTMModel',
    'GPUTransformerModel',
    'MultiGPUWrapper',
    
    # Memory
    'GPUMemoryManager',
    'GPUMemoryPool',
    'MemoryBlock',
    
    # Batch Processing
    'GPUBatchProcessor',
    'DynamicBatcher',
    'StreamingBatchProcessor',
    'BatchingStrategy',
    'InferenceRequest',
    'BatchedRequest',
    
    # Utilities
    'GPUProfiler',
    'gpu_profiler',
    'get_gpu_info',
    'select_best_gpu',
    'optimize_cuda_settings',
    'memory_efficient_load',
    'estimate_model_memory',
    'parallel_execute',
    'benchmark_operation',
    'convert_precision',
    'diagnose_cuda_errors',
    'save_gpu_profile',
    'load_gpu_profile',
    
    # Service
    'GPUService'
]

__version__ = '1.0.0'