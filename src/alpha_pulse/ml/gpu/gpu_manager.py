"""
GPU Resource Manager for ML Operations.

This module provides GPU resource allocation, monitoring, and management
for efficient multi-GPU training and inference operations.
"""

import os
import threading
import psutil
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

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


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    total_memory: int  # bytes
    free_memory: int   # bytes
    used_memory: int   # bytes
    temperature: float # celsius
    utilization: float # percentage
    power_draw: float  # watts
    compute_capability: Tuple[int, int]
    available: bool = True
    assigned_processes: List[int] = field(default_factory=list)


@dataclass
class GPUAllocation:
    """GPU allocation record."""
    process_id: int
    gpu_id: int
    memory_allocated: int
    timestamp: datetime
    task_type: str  # 'training', 'inference', 'preprocessing'
    priority: int = 0


class GPUManager:
    """
    Manages GPU resources for training and inference operations.
    
    This class provides:
    - GPU discovery and monitoring
    - Resource allocation and scheduling
    - Multi-GPU coordination
    - Memory management
    - Performance monitoring
    """
    
    def __init__(self, 
                 max_memory_fraction: float = 0.9,
                 allow_growth: bool = True,
                 visible_devices: Optional[List[int]] = None):
        """
        Initialize GPU Manager.
        
        Args:
            max_memory_fraction: Maximum fraction of GPU memory to use
            allow_growth: Allow dynamic memory growth
            visible_devices: List of GPU indices to use
        """
        self.max_memory_fraction = max_memory_fraction
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices
        
        # GPU information
        self.gpu_info: Dict[int, GPUInfo] = {}
        self.allocations: List[GPUAllocation] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize GPUs
        self._initialize_gpus()
        
        # Start monitoring
        self.monitoring_active = True
        self._start_monitoring()
    
    def _initialize_gpus(self):
        """Initialize and configure GPUs."""
        # Set visible devices if specified
        if self.visible_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.visible_devices))
        
        # Initialize based on available frameworks
        if TORCH_AVAILABLE and cuda.is_available():
            self._initialize_pytorch_gpus()
        
        if TF_AVAILABLE:
            self._initialize_tensorflow_gpus()
        
        if not self.gpu_info:
            logger.warning("No GPUs detected or available")
    
    def _initialize_pytorch_gpus(self):
        """Initialize PyTorch CUDA devices."""
        device_count = cuda.device_count()
        
        for device_id in range(device_count):
            try:
                cuda.set_device(device_id)
                props = cuda.get_device_properties(device_id)
                
                gpu_info = GPUInfo(
                    device_id=device_id,
                    name=props.name,
                    total_memory=props.total_memory,
                    free_memory=props.total_memory,  # Will be updated
                    used_memory=0,
                    temperature=0.0,  # Will be updated
                    utilization=0.0,  # Will be updated
                    power_draw=0.0,   # Will be updated
                    compute_capability=(props.major, props.minor)
                )
                
                self.gpu_info[device_id] = gpu_info
                
                # Configure memory settings
                if self.allow_growth:
                    cuda.set_per_process_memory_fraction(
                        self.max_memory_fraction, device_id
                    )
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU {device_id}: {e}")
    
    def _initialize_tensorflow_gpus(self):
        """Initialize TensorFlow GPU devices."""
        gpus = tf.config.list_physical_devices('GPU')
        
        for gpu in gpus:
            try:
                if self.allow_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(self.max_memory_fraction * 10000)
                        )]
                    )
            except Exception as e:
                logger.error(f"Failed to configure TensorFlow GPU: {e}")
    
    def _start_monitoring(self):
        """Start GPU monitoring thread."""
        if NVML_AVAILABLE:
            import threading
            self._monitor_thread = threading.Thread(target=self._monitor_gpus, daemon=True)
            self._monitor_thread.start()
    
    def _monitor_gpus(self):
        """Monitor GPU metrics using NVML."""
        import time
        
        while self.monitoring_active:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                
                for device_id in range(device_count):
                    if device_id not in self.gpu_info:
                        continue
                    
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_info[device_id].total_memory = mem_info.total
                    self.gpu_info[device_id].free_memory = mem_info.free
                    self.gpu_info[device_id].used_memory = mem_info.used
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_info[device_id].utilization = util.gpu
                    
                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.gpu_info[device_id].temperature = temp
                    except:
                        pass
                    
                    # Power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        self.gpu_info[device_id].power_draw = power
                    except:
                        pass
                
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
            
            time.sleep(1)  # Update every second
    
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        with self._lock:
            return [gpu_id for gpu_id, info in self.gpu_info.items() 
                   if info.available and info.free_memory > 100 * 1024 * 1024]  # >100MB free
    
    def allocate_gpu(self, 
                     memory_required: int,
                     task_type: str = 'training',
                     priority: int = 0) -> Optional[int]:
        """
        Allocate a GPU for a task.
        
        Args:
            memory_required: Memory required in bytes
            task_type: Type of task
            priority: Task priority
            
        Returns:
            GPU ID if allocated, None otherwise
        """
        with self._lock:
            available_gpus = self.get_available_gpus()
            
            if not available_gpus:
                logger.warning("No GPUs available for allocation")
                return None
            
            # Sort by free memory (descending)
            available_gpus.sort(
                key=lambda gpu_id: self.gpu_info[gpu_id].free_memory,
                reverse=True
            )
            
            # Find suitable GPU
            for gpu_id in available_gpus:
                if self.gpu_info[gpu_id].free_memory >= memory_required:
                    # Create allocation
                    allocation = GPUAllocation(
                        process_id=os.getpid(),
                        gpu_id=gpu_id,
                        memory_allocated=memory_required,
                        timestamp=datetime.now(),
                        task_type=task_type,
                        priority=priority
                    )
                    
                    self.allocations.append(allocation)
                    self.gpu_info[gpu_id].assigned_processes.append(os.getpid())
                    
                    logger.info(f"Allocated GPU {gpu_id} for {task_type} "
                               f"(PID: {os.getpid()}, Memory: {memory_required/1e9:.2f}GB)")
                    
                    return gpu_id
            
            logger.warning(f"No GPU with sufficient memory ({memory_required/1e9:.2f}GB required)")
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release GPU allocation."""
        with self._lock:
            process_id = os.getpid()
            
            # Remove allocations
            self.allocations = [
                alloc for alloc in self.allocations
                if not (alloc.gpu_id == gpu_id and alloc.process_id == process_id)
            ]
            
            # Update GPU info
            if gpu_id in self.gpu_info:
                self.gpu_info[gpu_id].assigned_processes = [
                    pid for pid in self.gpu_info[gpu_id].assigned_processes
                    if pid != process_id
                ]
                
                logger.info(f"Released GPU {gpu_id} (PID: {process_id})")
    
    def get_optimal_batch_size(self, 
                              model_memory: int,
                              sample_memory: int,
                              gpu_id: Optional[int] = None) -> int:
        """
        Calculate optimal batch size for available GPU memory.
        
        Args:
            model_memory: Memory required for model in bytes
            sample_memory: Memory per sample in bytes
            gpu_id: Specific GPU ID (None for automatic selection)
            
        Returns:
            Optimal batch size
        """
        if gpu_id is None:
            available_gpus = self.get_available_gpus()
            if not available_gpus:
                return 1
            gpu_id = available_gpus[0]
        
        if gpu_id not in self.gpu_info:
            return 1
        
        # Calculate available memory
        free_memory = self.gpu_info[gpu_id].free_memory
        available_memory = int(free_memory * self.max_memory_fraction)
        
        # Reserve memory for model and overhead
        overhead = 500 * 1024 * 1024  # 500MB overhead
        memory_for_data = available_memory - model_memory - overhead
        
        if memory_for_data <= 0:
            logger.warning("Insufficient GPU memory for model")
            return 1
        
        # Calculate batch size
        batch_size = max(1, memory_for_data // sample_memory)
        
        # Round down to power of 2 for efficiency
        batch_size = 2 ** int(np.log2(batch_size))
        
        return batch_size
    
    def get_memory_usage(self, gpu_id: Optional[int] = None) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Args:
            gpu_id: Specific GPU ID (None for all GPUs)
            
        Returns:
            Dictionary with memory statistics
        """
        if gpu_id is not None:
            if gpu_id not in self.gpu_info:
                return {}
            
            info = self.gpu_info[gpu_id]
            return {
                'total_gb': info.total_memory / 1e9,
                'used_gb': info.used_memory / 1e9,
                'free_gb': info.free_memory / 1e9,
                'utilization': info.used_memory / info.total_memory * 100
            }
        
        # Aggregate stats for all GPUs
        total_memory = sum(info.total_memory for info in self.gpu_info.values())
        used_memory = sum(info.used_memory for info in self.gpu_info.values())
        free_memory = sum(info.free_memory for info in self.gpu_info.values())
        
        return {
            'total_gb': total_memory / 1e9,
            'used_gb': used_memory / 1e9,
            'free_gb': free_memory / 1e9,
            'utilization': used_memory / total_memory * 100 if total_memory > 0 else 0
        }
    
    def get_gpu_stats(self) -> List[Dict[str, Any]]:
        """Get comprehensive statistics for all GPUs."""
        stats = []
        
        for gpu_id, info in self.gpu_info.items():
            stats.append({
                'gpu_id': gpu_id,
                'name': info.name,
                'compute_capability': f"{info.compute_capability[0]}.{info.compute_capability[1]}",
                'memory': self.get_memory_usage(gpu_id),
                'utilization': info.utilization,
                'temperature': info.temperature,
                'power_draw': info.power_draw,
                'processes': len(info.assigned_processes),
                'available': info.available
            })
        
        return stats
    
    def set_device(self, gpu_id: int):
        """Set current device for operations."""
        if TORCH_AVAILABLE:
            cuda.set_device(gpu_id)
        
        if TF_AVAILABLE:
            # TensorFlow device setting is handled differently
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    def synchronize(self, gpu_id: Optional[int] = None):
        """Synchronize GPU operations."""
        if TORCH_AVAILABLE:
            if gpu_id is not None:
                cuda.synchronize(gpu_id)
            else:
                cuda.synchronize()
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if TORCH_AVAILABLE:
            cuda.empty_cache()
        
        if TF_AVAILABLE:
            # TensorFlow doesn't have direct cache clearing
            import gc
            gc.collect()
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current GPU allocations."""
        summary = {
            'total_allocations': len(self.allocations),
            'by_task_type': defaultdict(int),
            'by_gpu': defaultdict(int),
            'total_memory_allocated': 0
        }
        
        for alloc in self.allocations:
            summary['by_task_type'][alloc.task_type] += 1
            summary['by_gpu'][alloc.gpu_id] += 1
            summary['total_memory_allocated'] += alloc.memory_allocated
        
        summary['by_task_type'] = dict(summary['by_task_type'])
        summary['by_gpu'] = dict(summary['by_gpu'])
        summary['total_memory_allocated_gb'] = summary['total_memory_allocated'] / 1e9
        
        return summary
    
    def cleanup(self):
        """Cleanup resources."""
        self.monitoring_active = False
        
        # Release all allocations for this process
        process_id = os.getpid()
        with self._lock:
            gpu_ids = set(alloc.gpu_id for alloc in self.allocations 
                         if alloc.process_id == process_id)
            
            for gpu_id in gpu_ids:
                self.release_gpu(gpu_id)
        
        # Clear cache
        self.clear_cache()
        
        # Shutdown NVML
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()