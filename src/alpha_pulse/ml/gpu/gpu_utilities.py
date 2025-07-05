"""
GPU Utilities and Helper Functions.

This module provides utility functions for GPU operations,
performance profiling, and optimization helpers.
"""

import os
import logging
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from functools import wraps
import time
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUProfiler:
    """GPU performance profiler for optimization."""
    
    def __init__(self):
        """Initialize GPU profiler."""
        self.profiles = {}
        self._active_profile = None
        
        if TORCH_AVAILABLE and cuda.is_available():
            self.cuda_available = True
        else:
            self.cuda_available = False
            logger.warning("CUDA not available for profiling")
    
    def profile(self, name: str):
        """
        Decorator for profiling GPU functions.
        
        Args:
            name: Profile name
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cuda_available:
                    return func(*args, **kwargs)
                
                # Start profiling
                cuda.synchronize()
                start_time = time.time()
                start_memory = cuda.memory_allocated()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # End profiling
                cuda.synchronize()
                end_time = time.time()
                end_memory = cuda.memory_allocated()
                
                # Record profile
                profile_data = {
                    'execution_time': end_time - start_time,
                    'memory_used': end_memory - start_memory,
                    'timestamp': datetime.now()
                }
                
                if name not in self.profiles:
                    self.profiles[name] = []
                self.profiles[name].append(profile_data)
                
                return result
            
            return wrapper
        return decorator
    
    def start_profile(self, name: str):
        """Start manual profiling."""
        if not self.cuda_available:
            return
        
        cuda.synchronize()
        self._active_profile = {
            'name': name,
            'start_time': time.time(),
            'start_memory': cuda.memory_allocated()
        }
    
    def end_profile(self):
        """End manual profiling."""
        if not self.cuda_available or not self._active_profile:
            return
        
        cuda.synchronize()
        
        profile_data = {
            'execution_time': time.time() - self._active_profile['start_time'],
            'memory_used': cuda.memory_allocated() - self._active_profile['start_memory'],
            'timestamp': datetime.now()
        }
        
        name = self._active_profile['name']
        if name not in self.profiles:
            self.profiles[name] = []
        self.profiles[name].append(profile_data)
        
        self._active_profile = None
    
    def get_summary(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling summary."""
        if name:
            if name not in self.profiles:
                return {}
            
            profiles = self.profiles[name]
            return {
                'name': name,
                'count': len(profiles),
                'avg_time': np.mean([p['execution_time'] for p in profiles]),
                'total_time': sum(p['execution_time'] for p in profiles),
                'avg_memory_mb': np.mean([p['memory_used'] / 1024**2 for p in profiles]),
                'max_memory_mb': max([p['memory_used'] / 1024**2 for p in profiles])
            }
        
        # Summary for all profiles
        summaries = {}
        for name in self.profiles:
            summaries[name] = self.get_summary(name)
        return summaries


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get information about available GPUs."""
    gpu_info = []
    
    if TORCH_AVAILABLE and cuda.is_available():
        for device_id in range(cuda.device_count()):
            props = cuda.get_device_properties(device_id)
            
            info = {
                'device_id': device_id,
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory_gb': props.total_memory / 1024**3,
                'multiprocessor_count': props.multi_processor_count,
                'cuda_cores': props.multi_processor_count * 64,  # Approximate
                'memory_bandwidth_gb': 0  # Not available in PyTorch
            }
            
            # Get current usage if NVML available
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    info.update({
                        'used_memory_gb': mem_info.used / 1024**3,
                        'free_memory_gb': mem_info.free / 1024**3,
                        'gpu_utilization': util.gpu,
                        'memory_utilization': util.memory
                    })
                except:
                    pass
            
            gpu_info.append(info)
    
    return gpu_info


def select_best_gpu(min_memory_gb: float = 2.0) -> Optional[int]:
    """
    Select the best available GPU based on free memory.
    
    Args:
        min_memory_gb: Minimum required memory in GB
        
    Returns:
        GPU device ID or None
    """
    gpus = get_gpu_info()
    
    if not gpus:
        return None
    
    # Filter by minimum memory
    suitable_gpus = [
        gpu for gpu in gpus
        if gpu.get('free_memory_gb', gpu['total_memory_gb']) >= min_memory_gb
    ]
    
    if not suitable_gpus:
        return None
    
    # Sort by free memory (descending)
    suitable_gpus.sort(
        key=lambda g: g.get('free_memory_gb', g['total_memory_gb']),
        reverse=True
    )
    
    return suitable_gpus[0]['device_id']


def optimize_cuda_settings():
    """Optimize CUDA settings for performance."""
    if not TORCH_AVAILABLE or not cuda.is_available():
        return
    
    # Enable TF32 for better performance on Ampere GPUs
    if hasattr(cuda, 'set_float32_matmul_precision'):
        cuda.set_float32_matmul_precision('high')
    
    # Enable cudnn benchmarking
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Set optimal number of threads
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(os.cpu_count())
    
    logger.info("CUDA settings optimized for performance")


def memory_efficient_load(model_path: str, 
                         device_id: int = 0,
                         map_location: Optional[str] = None) -> Any:
    """
    Load model with memory-efficient strategy.
    
    Args:
        model_path: Path to model file
        device_id: Target GPU device ID
        map_location: Custom map location
        
    Returns:
        Loaded model
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    # Clear cache before loading
    cuda.empty_cache()
    
    # Load to CPU first
    if map_location is None:
        map_location = 'cpu'
    
    model = torch.load(model_path, map_location=map_location)
    
    # Move to GPU
    device = torch.device(f'cuda:{device_id}')
    model = model.to(device)
    
    # Clear cache again
    cuda.empty_cache()
    
    return model


def estimate_model_memory(model: torch.nn.Module,
                         input_shape: Tuple[int, ...],
                         batch_size: int = 1,
                         dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        batch_size: Batch size
        dtype: Data type
        
    Returns:
        Memory estimates in MB
    """
    if not TORCH_AVAILABLE:
        return {}
    
    # Calculate parameter memory
    param_memory = 0
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    # Estimate activation memory (rough approximation)
    # This is a simplified estimation
    total_neurons = 0
    for module in model.modules():
        if hasattr(module, 'out_features'):
            total_neurons += module.out_features
        elif hasattr(module, 'out_channels'):
            if hasattr(module, 'kernel_size'):
                # Conv layer - estimate output size
                total_neurons += module.out_channels * 100  # Rough estimate
    
    element_size = torch.tensor([], dtype=dtype).element_size()
    activation_memory = total_neurons * batch_size * element_size * 2  # x2 for forward/backward
    
    # Input/output memory
    input_memory = np.prod(input_shape) * batch_size * element_size
    output_memory = input_memory  # Assume similar size
    
    # Optimizer memory (approximate)
    optimizer_memory = param_memory * 2  # Adam uses 2x parameter memory
    
    return {
        'parameters_mb': param_memory / 1024**2,
        'activations_mb': activation_memory / 1024**2,
        'input_output_mb': (input_memory + output_memory) / 1024**2,
        'optimizer_mb': optimizer_memory / 1024**2,
        'total_mb': (param_memory + activation_memory + input_memory + 
                    output_memory + optimizer_memory) / 1024**2
    }


def parallel_execute(func: Callable,
                    args_list: List[Tuple],
                    device_ids: Optional[List[int]] = None,
                    max_workers: Optional[int] = None) -> List[Any]:
    """
    Execute function in parallel across multiple GPUs.
    
    Args:
        func: Function to execute
        args_list: List of argument tuples
        device_ids: GPU device IDs to use
        max_workers: Maximum parallel workers
        
    Returns:
        List of results
    """
    import concurrent.futures
    
    if device_ids is None:
        device_ids = list(range(cuda.device_count() if TORCH_AVAILABLE else 1))
    
    if max_workers is None:
        max_workers = len(device_ids)
    
    results = []
    
    def worker(args, device_id):
        """Worker function."""
        if TORCH_AVAILABLE:
            cuda.set_device(device_id)
        return func(*args)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = []
        for i, args in enumerate(args_list):
            device_id = device_ids[i % len(device_ids)]
            future = executor.submit(worker, args, device_id)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    return results


def benchmark_operation(func: Callable,
                       args: Tuple,
                       num_runs: int = 100,
                       warmup_runs: int = 10,
                       device_id: int = 0) -> Dict[str, float]:
    """
    Benchmark a GPU operation.
    
    Args:
        func: Function to benchmark
        args: Function arguments
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device_id: GPU device ID
        
    Returns:
        Benchmark results
    """
    if TORCH_AVAILABLE:
        cuda.set_device(device_id)
        
        # Warmup
        for _ in range(warmup_runs):
            func(*args)
        
        cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            func(*args)
            cuda.synchronize()
            times.append(time.time() - start)
    else:
        # CPU benchmark
        for _ in range(warmup_runs):
            func(*args)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            func(*args)
            times.append(time.time() - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'total_runs': num_runs
    }


def convert_precision(tensor: Union[torch.Tensor, np.ndarray],
                     target_dtype: Union[torch.dtype, np.dtype]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert tensor precision for memory optimization.
    
    Args:
        tensor: Input tensor
        target_dtype: Target data type
        
    Returns:
        Converted tensor
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(target_dtype)
    elif isinstance(tensor, np.ndarray):
        return tensor.astype(target_dtype)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def diagnose_cuda_errors():
    """Diagnose common CUDA errors and provide solutions."""
    diagnostics = {
        'cuda_available': False,
        'device_count': 0,
        'errors': [],
        'suggestions': []
    }
    
    # Check CUDA availability
    if not TORCH_AVAILABLE:
        diagnostics['errors'].append("PyTorch not installed")
        diagnostics['suggestions'].append("Install PyTorch with CUDA support")
        return diagnostics
    
    if not cuda.is_available():
        diagnostics['errors'].append("CUDA not available")
        diagnostics['suggestions'].append("Check NVIDIA driver installation")
        diagnostics['suggestions'].append("Verify CUDA toolkit installation")
        return diagnostics
    
    diagnostics['cuda_available'] = True
    diagnostics['device_count'] = cuda.device_count()
    
    # Check for common issues
    try:
        # Test basic operation
        test_tensor = torch.randn(10, 10).cuda()
        result = test_tensor @ test_tensor.T
        del test_tensor, result
        cuda.empty_cache()
    except RuntimeError as e:
        diagnostics['errors'].append(f"Runtime error: {str(e)}")
        if "out of memory" in str(e):
            diagnostics['suggestions'].append("Reduce batch size")
            diagnostics['suggestions'].append("Clear GPU memory cache")
            diagnostics['suggestions'].append("Use gradient checkpointing")
    
    # Check driver version
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            diagnostics['errors'].append("nvidia-smi not accessible")
            diagnostics['suggestions'].append("Check NVIDIA driver installation")
    except:
        diagnostics['errors'].append("Could not run nvidia-smi")
    
    return diagnostics


def save_gpu_profile(profile_data: Dict[str, Any], 
                    filepath: str):
    """Save GPU profiling data to file."""
    with open(filepath, 'w') as f:
        json.dump(profile_data, f, indent=2, default=str)
    logger.info(f"GPU profile saved to {filepath}")


def load_gpu_profile(filepath: str) -> Dict[str, Any]:
    """Load GPU profiling data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# Global profiler instance
gpu_profiler = GPUProfiler()