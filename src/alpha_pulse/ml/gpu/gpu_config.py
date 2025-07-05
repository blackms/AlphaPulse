"""
GPU Configuration Management.

This module provides configuration classes and utilities for GPU-accelerated
computing, including device settings, memory limits, and optimization parameters.
"""

import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
import logging

logger = logging.getLogger(__name__)


class GPUBackend(Enum):
    """Supported GPU backends."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUPY = "cupy"
    NUMBA = "numba"
    AUTO = "auto"


class PrecisionMode(Enum):
    """Computation precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    MIXED = "mixed"
    INT8 = "int8"
    AUTO = "auto"


class MemoryGrowthPolicy(Enum):
    """GPU memory growth policy."""
    FIXED = "fixed"
    GROW_AS_NEEDED = "grow_as_needed"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


@dataclass
class GPUDeviceConfig:
    """Configuration for a single GPU device."""
    device_id: int
    enabled: bool = True
    memory_fraction: float = 0.9  # Maximum memory fraction to use
    memory_limit_mb: Optional[int] = None  # Absolute memory limit
    allow_growth: bool = True
    compute_mode: str = "DEFAULT"  # DEFAULT, EXCLUSIVE_PROCESS, PROHIBITED
    priority: int = 0  # Higher priority for primary devices
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'device_id': self.device_id,
            'enabled': self.enabled,
            'memory_fraction': self.memory_fraction,
            'memory_limit_mb': self.memory_limit_mb,
            'allow_growth': self.allow_growth,
            'compute_mode': self.compute_mode,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPUDeviceConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MemoryConfig:
    """GPU memory management configuration."""
    growth_policy: MemoryGrowthPolicy = MemoryGrowthPolicy.GROW_AS_NEEDED
    pool_size_mb: int = 2048  # Memory pool size
    gc_threshold: float = 0.8  # Trigger GC at this usage
    defrag_threshold: float = 0.7  # Trigger defragmentation
    enable_pooling: bool = True
    cache_limit_mb: int = 1024  # PyTorch cache limit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'growth_policy': self.growth_policy.value,
            'pool_size_mb': self.pool_size_mb,
            'gc_threshold': self.gc_threshold,
            'defrag_threshold': self.defrag_threshold,
            'enable_pooling': self.enable_pooling,
            'cache_limit_mb': self.cache_limit_mb
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryConfig':
        """Create from dictionary."""
        data['growth_policy'] = MemoryGrowthPolicy(data['growth_policy'])
        return cls(**data)


@dataclass
class ComputeConfig:
    """GPU computation configuration."""
    backend: GPUBackend = GPUBackend.AUTO
    precision: PrecisionMode = PrecisionMode.MIXED
    enable_tf32: bool = True  # For Ampere GPUs
    enable_cudnn: bool = True
    cudnn_benchmark: bool = True
    num_threads: Optional[int] = None  # CPU threads for data loading
    compile_mode: str = "default"  # torch.compile mode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backend': self.backend.value,
            'precision': self.precision.value,
            'enable_tf32': self.enable_tf32,
            'enable_cudnn': self.enable_cudnn,
            'cudnn_benchmark': self.cudnn_benchmark,
            'num_threads': self.num_threads,
            'compile_mode': self.compile_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputeConfig':
        """Create from dictionary."""
        data['backend'] = GPUBackend(data['backend'])
        data['precision'] = PrecisionMode(data['precision'])
        return cls(**data)


@dataclass
class BatchingConfig:
    """Batch processing configuration."""
    max_batch_size: int = 64
    min_batch_size: int = 1
    max_wait_time_ms: float = 10.0
    queue_size: int = 1000
    num_workers: int = 2
    strategy: str = "adaptive"  # fixed_size, dynamic, time_based, adaptive
    priority_levels: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size,
            'max_wait_time_ms': self.max_wait_time_ms,
            'queue_size': self.queue_size,
            'num_workers': self.num_workers,
            'strategy': self.strategy,
            'priority_levels': self.priority_levels
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchingConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OptimizationConfig:
    """GPU optimization configuration."""
    enable_amp: bool = True  # Automatic Mixed Precision
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    enable_jit: bool = True  # JIT compilation
    fusion_enabled: bool = True  # Kernel fusion
    profile_enabled: bool = False
    optimize_for_inference: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_amp': self.enable_amp,
            'gradient_checkpointing': self.gradient_checkpointing,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'enable_jit': self.enable_jit,
            'fusion_enabled': self.fusion_enabled,
            'profile_enabled': self.profile_enabled,
            'optimize_for_inference': self.optimize_for_inference
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MonitoringConfig:
    """GPU monitoring configuration."""
    enable_monitoring: bool = True
    monitor_interval_sec: float = 5.0
    log_memory_usage: bool = True
    log_utilization: bool = True
    alert_on_oom: bool = True
    alert_threshold_memory: float = 0.95
    alert_threshold_util: float = 0.95
    metrics_history_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_monitoring': self.enable_monitoring,
            'monitor_interval_sec': self.monitor_interval_sec,
            'log_memory_usage': self.log_memory_usage,
            'log_utilization': self.log_utilization,
            'alert_on_oom': self.alert_on_oom,
            'alert_threshold_memory': self.alert_threshold_memory,
            'alert_threshold_util': self.alert_threshold_util,
            'metrics_history_size': self.metrics_history_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GPUConfig:
    """Complete GPU configuration."""
    devices: List[GPUDeviceConfig] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    visible_devices: Optional[List[int]] = None  # CUDA_VISIBLE_DEVICES
    multi_gpu_strategy: str = "data_parallel"  # data_parallel, distributed, horovod
    fallback_to_cpu: bool = True
    error_handling: str = "log"  # log, raise, ignore
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'devices': [d.to_dict() for d in self.devices],
            'memory': self.memory.to_dict(),
            'compute': self.compute.to_dict(),
            'batching': self.batching.to_dict(),
            'optimization': self.optimization.to_dict(),
            'monitoring': self.monitoring.to_dict(),
            'visible_devices': self.visible_devices,
            'multi_gpu_strategy': self.multi_gpu_strategy,
            'fallback_to_cpu': self.fallback_to_cpu,
            'error_handling': self.error_handling
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPUConfig':
        """Create from dictionary."""
        config = cls()
        
        if 'devices' in data:
            config.devices = [GPUDeviceConfig.from_dict(d) for d in data['devices']]
        if 'memory' in data:
            config.memory = MemoryConfig.from_dict(data['memory'])
        if 'compute' in data:
            config.compute = ComputeConfig.from_dict(data['compute'])
        if 'batching' in data:
            config.batching = BatchingConfig.from_dict(data['batching'])
        if 'optimization' in data:
            config.optimization = OptimizationConfig.from_dict(data['optimization'])
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig.from_dict(data['monitoring'])
        
        # Global settings
        for key in ['visible_devices', 'multi_gpu_strategy', 'fallback_to_cpu', 'error_handling']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def save(self, filepath: str, format: str = 'yaml'):
        """Save configuration to file."""
        data = self.to_dict()
        
        if format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"GPU configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GPUConfig':
        """Load configuration from file."""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check device configurations
        device_ids = set()
        for device in self.devices:
            if device.device_id in device_ids:
                issues.append(f"Duplicate device ID: {device.device_id}")
            device_ids.add(device.device_id)
            
            if device.memory_fraction <= 0 or device.memory_fraction > 1:
                issues.append(f"Invalid memory fraction for device {device.device_id}: {device.memory_fraction}")
        
        # Check memory configuration
        if self.memory.gc_threshold <= 0 or self.memory.gc_threshold > 1:
            issues.append(f"Invalid GC threshold: {self.memory.gc_threshold}")
        
        if self.memory.defrag_threshold <= 0 or self.memory.defrag_threshold > 1:
            issues.append(f"Invalid defragmentation threshold: {self.memory.defrag_threshold}")
        
        # Check batching configuration
        if self.batching.max_batch_size < self.batching.min_batch_size:
            issues.append("Max batch size must be >= min batch size")
        
        if self.batching.num_workers < 1:
            issues.append("Number of workers must be >= 1")
        
        return issues


class GPUConfigBuilder:
    """Builder for creating GPU configurations."""
    
    def __init__(self):
        """Initialize builder."""
        self.config = GPUConfig()
    
    def auto_detect_devices(self) -> 'GPUConfigBuilder':
        """Auto-detect and configure available GPUs."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_config = GPUDeviceConfig(
                        device_id=i,
                        enabled=True,
                        priority=0 if i == 0 else 1
                    )
                    self.config.devices.append(device_config)
        except ImportError:
            logger.warning("PyTorch not available for auto-detection")
        
        return self
    
    def set_memory_policy(self, 
                         policy: MemoryGrowthPolicy,
                         pool_size_mb: int = 2048) -> 'GPUConfigBuilder':
        """Set memory management policy."""
        self.config.memory.growth_policy = policy
        self.config.memory.pool_size_mb = pool_size_mb
        return self
    
    def set_precision(self, precision: PrecisionMode) -> 'GPUConfigBuilder':
        """Set computation precision."""
        self.config.compute.precision = precision
        
        # Auto-configure related settings
        if precision == PrecisionMode.FP16 or precision == PrecisionMode.MIXED:
            self.config.optimization.enable_amp = True
        
        return self
    
    def enable_multi_gpu(self, strategy: str = "data_parallel") -> 'GPUConfigBuilder':
        """Enable multi-GPU support."""
        self.config.multi_gpu_strategy = strategy
        return self
    
    def set_batching(self,
                    max_batch_size: int,
                    strategy: str = "adaptive") -> 'GPUConfigBuilder':
        """Configure batching."""
        self.config.batching.max_batch_size = max_batch_size
        self.config.batching.strategy = strategy
        return self
    
    def optimize_for_inference(self) -> 'GPUConfigBuilder':
        """Optimize configuration for inference."""
        self.config.optimization.optimize_for_inference = True
        self.config.optimization.gradient_checkpointing = False
        self.config.compute.cudnn_benchmark = True
        self.config.batching.strategy = "dynamic"
        return self
    
    def optimize_for_training(self) -> 'GPUConfigBuilder':
        """Optimize configuration for training."""
        self.config.optimization.optimize_for_inference = False
        self.config.optimization.enable_amp = True
        self.config.compute.precision = PrecisionMode.MIXED
        return self
    
    def build(self) -> GPUConfig:
        """Build and validate configuration."""
        issues = self.config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
        
        return self.config


# Predefined configurations
def get_default_config() -> GPUConfig:
    """Get default GPU configuration."""
    return GPUConfigBuilder().auto_detect_devices().build()


def get_inference_config() -> GPUConfig:
    """Get configuration optimized for inference."""
    return (GPUConfigBuilder()
            .auto_detect_devices()
            .optimize_for_inference()
            .set_precision(PrecisionMode.FP16)
            .set_batching(max_batch_size=128, strategy="dynamic")
            .build())


def get_training_config() -> GPUConfig:
    """Get configuration optimized for training."""
    return (GPUConfigBuilder()
            .auto_detect_devices()
            .optimize_for_training()
            .set_memory_policy(MemoryGrowthPolicy.GROW_AS_NEEDED)
            .enable_multi_gpu()
            .build())


def get_limited_memory_config(memory_limit_mb: int = 4096) -> GPUConfig:
    """Get configuration for limited GPU memory."""
    config = GPUConfigBuilder().auto_detect_devices().build()
    
    # Set memory limits
    for device in config.devices:
        device.memory_limit_mb = memory_limit_mb
        device.memory_fraction = 0.8
    
    # Conservative settings
    config.memory.growth_policy = MemoryGrowthPolicy.CONSERVATIVE
    config.memory.gc_threshold = 0.7
    config.batching.max_batch_size = 32
    config.optimization.gradient_checkpointing = True
    
    return config