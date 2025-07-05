"""
GPU Memory Manager for Efficient Resource Utilization.

This module provides advanced memory management for GPU operations,
including memory pooling, garbage collection, and optimization strategies.
"""

import gc
import logging
import threading
import weakref
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict, OrderedDict
import psutil

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    from cupy import cuda as cupy_cuda
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a memory block allocation."""
    block_id: str
    size: int  # bytes
    device_id: int
    allocated_time: datetime
    last_accessed: datetime
    tensor_ref: Any = None  # Weak reference to tensor
    is_free: bool = False
    priority: int = 0  # Higher priority blocks are kept longer


class GPUMemoryPool:
    """Memory pool for efficient GPU memory reuse."""
    
    def __init__(self, device_id: int, max_pool_size: int = 2 * 1024**3):  # 2GB default
        self.device_id = device_id
        self.max_pool_size = max_pool_size
        self.pool: Dict[int, List[MemoryBlock]] = defaultdict(list)  # size -> blocks
        self.allocated_blocks: Dict[str, MemoryBlock] = {}
        self.total_pooled_memory = 0
        self._lock = threading.Lock()
        self._block_counter = 0
    
    def allocate(self, size: int, priority: int = 0) -> Optional[MemoryBlock]:
        """Allocate memory from pool or create new block."""
        with self._lock:
            # Round up to nearest power of 2 for better reuse
            rounded_size = 2 ** int(np.ceil(np.log2(size)))
            
            # Check pool for available block
            if rounded_size in self.pool and self.pool[rounded_size]:
                block = self.pool[rounded_size].pop()
                block.is_free = False
                block.last_accessed = datetime.now()
                block.priority = priority
                logger.debug(f"Reused pooled block {block.block_id} of size {rounded_size}")
                return block
            
            # Create new block
            block_id = f"block_{self.device_id}_{self._block_counter}"
            self._block_counter += 1
            
            block = MemoryBlock(
                block_id=block_id,
                size=rounded_size,
                device_id=self.device_id,
                allocated_time=datetime.now(),
                last_accessed=datetime.now(),
                priority=priority
            )
            
            self.allocated_blocks[block_id] = block
            return block
    
    def release(self, block: MemoryBlock):
        """Release memory block back to pool."""
        with self._lock:
            if block.block_id not in self.allocated_blocks:
                return
            
            # Check if we should pool this block
            if self.total_pooled_memory + block.size <= self.max_pool_size:
                block.is_free = True
                self.pool[block.size].append(block)
                self.total_pooled_memory += block.size
                logger.debug(f"Released block {block.block_id} to pool")
            else:
                # Actually free the memory
                del self.allocated_blocks[block.block_id]
                logger.debug(f"Freed block {block.block_id}")
    
    def clear_pool(self):
        """Clear all pooled memory."""
        with self._lock:
            self.pool.clear()
            self.total_pooled_memory = 0
            logger.info(f"Cleared memory pool for device {self.device_id}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            stats = {
                'device_id': self.device_id,
                'total_pooled_memory_mb': self.total_pooled_memory / 1024**2,
                'max_pool_size_mb': self.max_pool_size / 1024**2,
                'pool_utilization': self.total_pooled_memory / self.max_pool_size * 100,
                'pooled_blocks': sum(len(blocks) for blocks in self.pool.values()),
                'allocated_blocks': len(self.allocated_blocks)
            }
            
            # Size distribution
            size_distribution = {}
            for size, blocks in self.pool.items():
                size_distribution[f"{size/1024**2:.1f}MB"] = len(blocks)
            stats['size_distribution'] = size_distribution
            
            return stats


class GPUMemoryManager:
    """
    Advanced GPU memory manager with optimization strategies.
    
    Features:
    - Memory pooling for efficient reuse
    - Automatic garbage collection
    - Memory defragmentation
    - Out-of-memory handling
    - Memory usage tracking and optimization
    """
    
    def __init__(self,
                 max_memory_fraction: float = 0.9,
                 enable_memory_pool: bool = True,
                 gc_threshold: float = 0.8,
                 defrag_threshold: float = 0.7):
        """
        Initialize GPU memory manager.
        
        Args:
            max_memory_fraction: Maximum fraction of GPU memory to use
            enable_memory_pool: Enable memory pooling
            gc_threshold: Trigger GC when memory usage exceeds this fraction
            defrag_threshold: Trigger defragmentation at this fragmentation level
        """
        self.max_memory_fraction = max_memory_fraction
        self.enable_memory_pool = enable_memory_pool
        self.gc_threshold = gc_threshold
        self.defrag_threshold = defrag_threshold
        
        # Memory pools per device
        self.memory_pools: Dict[int, GPUMemoryPool] = {}
        
        # Tracking
        self.allocated_tensors: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.memory_usage_history: List[Dict[str, Any]] = []
        self._allocation_counter = 0
        
        # Initialize for available devices
        self._initialize_devices()
        
        # Start monitoring thread
        self._monitoring_active = True
        self._start_monitoring()
    
    def _initialize_devices(self):
        """Initialize memory management for available GPU devices."""
        if TORCH_AVAILABLE and cuda.is_available():
            for device_id in range(cuda.device_count()):
                if self.enable_memory_pool:
                    # Calculate max pool size based on GPU memory
                    props = cuda.get_device_properties(device_id)
                    max_pool_size = int(props.total_memory * 0.2)  # 20% of GPU memory
                    self.memory_pools[device_id] = GPUMemoryPool(device_id, max_pool_size)
                
                # Set memory fraction
                cuda.set_per_process_memory_fraction(self.max_memory_fraction, device_id)
    
    def _start_monitoring(self):
        """Start memory monitoring thread."""
        import threading
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_memory(self):
        """Monitor GPU memory usage and trigger optimization."""
        import time
        
        while self._monitoring_active:
            try:
                for device_id in range(cuda.device_count() if TORCH_AVAILABLE else 0):
                    usage = self.get_memory_info(device_id)
                    
                    # Check if GC needed
                    if usage['usage_fraction'] > self.gc_threshold:
                        logger.info(f"GPU {device_id} memory usage {usage['usage_fraction']:.1%}, "
                                   f"triggering garbage collection")
                        self.garbage_collect(device_id)
                    
                    # Check fragmentation
                    fragmentation = self._estimate_fragmentation(device_id)
                    if fragmentation > self.defrag_threshold:
                        logger.info(f"GPU {device_id} fragmentation {fragmentation:.1%}, "
                                   f"triggering defragmentation")
                        self.defragment(device_id)
                    
                    # Record usage
                    self.memory_usage_history.append({
                        'timestamp': datetime.now(),
                        'device_id': device_id,
                        'usage': usage,
                        'fragmentation': fragmentation
                    })
                    
                    # Keep only recent history
                    if len(self.memory_usage_history) > 1000:
                        self.memory_usage_history = self.memory_usage_history[-1000:]
                
            except Exception as e:
                logger.debug(f"Memory monitoring error: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def allocate_tensor(self,
                       shape: Tuple[int, ...],
                       dtype: torch.dtype = torch.float32,
                       device_id: int = 0,
                       priority: int = 0) -> torch.Tensor:
        """
        Allocate tensor with memory management.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device_id: GPU device ID
            priority: Allocation priority (higher = keep longer)
            
        Returns:
            Allocated tensor
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Calculate memory requirement
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_size = np.prod(shape) * element_size
        
        # Try to allocate
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Check available memory
                if not self._check_memory_available(device_id, total_size):
                    self._free_memory(device_id, total_size)
                
                # Allocate tensor
                device = torch.device(f'cuda:{device_id}')
                tensor = torch.empty(shape, dtype=dtype, device=device)
                
                # Track allocation
                allocation_id = f"tensor_{self._allocation_counter}"
                self._allocation_counter += 1
                self.allocated_tensors[allocation_id] = tensor
                
                logger.debug(f"Allocated tensor {allocation_id} of size {total_size/1024**2:.1f}MB "
                           f"on GPU {device_id}")
                
                return tensor
                
            except cuda.OutOfMemoryError as e:
                logger.warning(f"OOM on GPU {device_id}, attempt {retry + 1}/{max_retries}")
                
                if retry < max_retries - 1:
                    # Try to free memory
                    self.garbage_collect(device_id)
                    self.defragment(device_id)
                    
                    if self.enable_memory_pool and device_id in self.memory_pools:
                        self.memory_pools[device_id].clear_pool()
                else:
                    raise
    
    def _check_memory_available(self, device_id: int, required_size: int) -> bool:
        """Check if enough memory is available."""
        info = self.get_memory_info(device_id)
        return info['free_memory'] >= required_size * 1.1  # 10% buffer
    
    def _free_memory(self, device_id: int, required_size: int):
        """Free memory to make room for allocation."""
        logger.info(f"Attempting to free {required_size/1024**2:.1f}MB on GPU {device_id}")
        
        # First try garbage collection
        self.garbage_collect(device_id)
        
        # Clear caches
        if TORCH_AVAILABLE:
            cuda.empty_cache()
        
        # Clear memory pool if needed
        if self.enable_memory_pool and device_id in self.memory_pools:
            pool_stats = self.memory_pools[device_id].get_pool_stats()
            if pool_stats['total_pooled_memory_mb'] * 1024**2 >= required_size:
                self.memory_pools[device_id].clear_pool()
    
    def garbage_collect(self, device_id: Optional[int] = None):
        """Run garbage collection on GPU memory."""
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache clearing
        if TORCH_AVAILABLE:
            if device_id is not None:
                with cuda.device(device_id):
                    cuda.empty_cache()
                    cuda.synchronize()
            else:
                cuda.empty_cache()
                cuda.synchronize()
        
        # CuPy memory pool clearing
        if CUPY_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        
        logger.debug(f"Garbage collection completed for GPU {device_id}")
    
    def defragment(self, device_id: int):
        """Defragment GPU memory by consolidating allocations."""
        if not TORCH_AVAILABLE:
            return
        
        logger.info(f"Starting defragmentation for GPU {device_id}")
        
        # This is a simplified defragmentation strategy
        # In practice, this would require more sophisticated memory management
        
        # 1. Snapshot current tensors
        device = torch.device(f'cuda:{device_id}')
        current_tensors = []
        
        for tensor_ref in list(self.allocated_tensors.values()):
            if tensor_ref is not None and tensor_ref.device == device:
                current_tensors.append({
                    'data': tensor_ref.cpu(),  # Move to CPU temporarily
                    'shape': tensor_ref.shape,
                    'dtype': tensor_ref.dtype
                })
        
        # 2. Clear GPU memory
        self.garbage_collect(device_id)
        
        # 3. Reallocate tensors
        for tensor_info in current_tensors:
            new_tensor = torch.empty(
                tensor_info['shape'],
                dtype=tensor_info['dtype'],
                device=device
            )
            new_tensor.copy_(tensor_info['data'].to(device))
        
        logger.info(f"Defragmentation completed for GPU {device_id}")
    
    def _estimate_fragmentation(self, device_id: int) -> float:
        """Estimate memory fragmentation level."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        try:
            # Get memory info
            info = self.get_memory_info(device_id)
            
            # Simple fragmentation estimate based on allocation efficiency
            if info['allocated_memory'] > 0:
                efficiency = info['reserved_memory'] / info['allocated_memory']
                fragmentation = max(0, efficiency - 1.0)
            else:
                fragmentation = 0.0
            
            return min(fragmentation, 1.0)
            
        except Exception:
            return 0.0
    
    def get_memory_info(self, device_id: int) -> Dict[str, Any]:
        """Get detailed memory information for a device."""
        info = {
            'device_id': device_id,
            'total_memory': 0,
            'allocated_memory': 0,
            'reserved_memory': 0,
            'free_memory': 0,
            'usage_fraction': 0.0
        }
        
        if TORCH_AVAILABLE and cuda.is_available():
            try:
                # Get memory stats
                stats = cuda.memory_stats(device_id)
                reserved = cuda.memory_reserved(device_id)
                allocated = cuda.memory_allocated(device_id)
                
                # Get total memory
                props = cuda.get_device_properties(device_id)
                total = props.total_memory
                
                info.update({
                    'total_memory': total,
                    'allocated_memory': allocated,
                    'reserved_memory': reserved,
                    'free_memory': total - reserved,
                    'usage_fraction': allocated / total if total > 0 else 0,
                    'allocation_count': stats.get('allocation.all.current', 0),
                    'reserved_count': stats.get('reserved_bytes.all.current', 0)
                })
            except Exception as e:
                logger.debug(f"Error getting memory info: {e}")
        
        return info
    
    def optimize_batch_size(self,
                           model_size: int,
                           sample_size: int,
                           device_id: int = 0,
                           safety_factor: float = 0.9) -> int:
        """
        Calculate optimal batch size for available memory.
        
        Args:
            model_size: Model memory footprint in bytes
            sample_size: Memory per sample in bytes
            device_id: GPU device ID
            safety_factor: Safety margin (0-1)
            
        Returns:
            Optimal batch size
        """
        info = self.get_memory_info(device_id)
        
        # Available memory for data
        available = info['free_memory'] * safety_factor - model_size
        
        if available <= 0:
            logger.warning("Insufficient memory for model")
            return 1
        
        # Calculate batch size
        batch_size = int(available / sample_size)
        
        # Round down to power of 2 for efficiency
        if batch_size > 1:
            batch_size = 2 ** int(np.log2(batch_size))
        
        return max(1, batch_size)
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on usage patterns."""
        recommendations = []
        
        if not self.memory_usage_history:
            return recommendations
        
        # Analyze recent usage
        recent_usage = self.memory_usage_history[-100:]
        
        # Check average usage
        avg_usage = np.mean([u['usage']['usage_fraction'] for u in recent_usage])
        if avg_usage > 0.9:
            recommendations.append("High memory usage detected. Consider reducing batch size.")
        elif avg_usage < 0.3:
            recommendations.append("Low memory usage. Can increase batch size for better efficiency.")
        
        # Check fragmentation
        avg_fragmentation = np.mean([u['fragmentation'] for u in recent_usage])
        if avg_fragmentation > 0.5:
            recommendations.append("High memory fragmentation. Enable periodic defragmentation.")
        
        # Check pool usage
        if self.enable_memory_pool:
            for device_id, pool in self.memory_pools.items():
                stats = pool.get_pool_stats()
                if stats['pool_utilization'] > 90:
                    recommendations.append(f"GPU {device_id} memory pool nearly full. "
                                        f"Consider increasing pool size.")
        
        return recommendations
    
    def save_memory_profile(self, filepath: str):
        """Save memory usage profile for analysis."""
        import json
        
        profile = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'max_memory_fraction': self.max_memory_fraction,
                'enable_memory_pool': self.enable_memory_pool,
                'gc_threshold': self.gc_threshold,
                'defrag_threshold': self.defrag_threshold
            },
            'device_info': {},
            'usage_history': [],
            'recommendations': self.get_optimization_recommendations()
        }
        
        # Add device information
        if TORCH_AVAILABLE and cuda.is_available():
            for device_id in range(cuda.device_count()):
                props = cuda.get_device_properties(device_id)
                profile['device_info'][device_id] = {
                    'name': props.name,
                    'total_memory_gb': props.total_memory / 1024**3,
                    'compute_capability': f"{props.major}.{props.minor}"
                }
        
        # Add usage history summary
        if self.memory_usage_history:
            # Group by device
            by_device = defaultdict(list)
            for entry in self.memory_usage_history:
                by_device[entry['device_id']].append(entry)
            
            for device_id, entries in by_device.items():
                usage_values = [e['usage']['usage_fraction'] for e in entries]
                profile['usage_history'].append({
                    'device_id': device_id,
                    'samples': len(entries),
                    'avg_usage': np.mean(usage_values),
                    'max_usage': np.max(usage_values),
                    'min_usage': np.min(usage_values)
                })
        
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Memory profile saved to {filepath}")
    
    def cleanup(self):
        """Cleanup memory manager resources."""
        self._monitoring_active = False
        
        # Clear all memory pools
        for pool in self.memory_pools.values():
            pool.clear_pool()
        
        # Final garbage collection
        self.garbage_collect()
        
        logger.info("GPU memory manager cleaned up")