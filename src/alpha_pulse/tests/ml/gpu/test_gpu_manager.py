"""
Tests for GPU Resource Manager.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from alpha_pulse.ml.gpu.gpu_manager import (
    GPUManager, GPUInfo, GPUAllocation
)


@pytest.fixture
def gpu_manager():
    """Create GPU manager instance."""
    with patch('alpha_pulse.ml.gpu.gpu_manager.pynvml'):
        manager = GPUManager(
            max_memory_fraction=0.9,
            allow_growth=True,
            visible_devices=[0, 1]
        )
        yield manager
        manager.cleanup()


@pytest.fixture
def mock_gpu_info():
    """Create mock GPU info."""
    return GPUInfo(
        device_id=0,
        name="NVIDIA GeForce RTX 3090",
        total_memory=24 * 1024**3,  # 24GB
        free_memory=20 * 1024**3,   # 20GB
        used_memory=4 * 1024**3,     # 4GB
        temperature=50.0,
        utilization=20.0,
        power_draw=150.0,
        compute_capability=(8, 6),
        available=True
    )


class TestGPUManager:
    """Test GPU manager functionality."""
    
    def test_initialization(self, gpu_manager):
        """Test GPU manager initialization."""
        assert gpu_manager.max_memory_fraction == 0.9
        assert gpu_manager.allow_growth is True
        assert gpu_manager.visible_devices == [0, 1]
        assert gpu_manager.monitoring_active is True
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_gpu_discovery(self, mock_device_count, mock_is_available):
        """Test GPU discovery."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value = MagicMock(
                name="NVIDIA GeForce RTX 3090",
                total_memory=24 * 1024**3,
                major=8,
                minor=6
            )
            
            manager = GPUManager()
            assert len(manager.gpu_info) == 2
            assert all(isinstance(info, GPUInfo) for info in manager.gpu_info.values())
    
    def test_get_available_gpus(self, gpu_manager, mock_gpu_info):
        """Test getting available GPUs."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        gpu_manager.gpu_info[1] = GPUInfo(
            device_id=1,
            name="NVIDIA GeForce RTX 3090",
            total_memory=24 * 1024**3,
            free_memory=50 * 1024**2,  # Only 50MB free
            used_memory=24 * 1024**3 - 50 * 1024**2,
            temperature=70.0,
            utilization=95.0,
            power_draw=300.0,
            compute_capability=(8, 6),
            available=True
        )
        
        available = gpu_manager.get_available_gpus()
        assert len(available) == 1
        assert 0 in available
        assert 1 not in available  # Not enough free memory
    
    def test_allocate_gpu(self, gpu_manager, mock_gpu_info):
        """Test GPU allocation."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        # Successful allocation
        gpu_id = gpu_manager.allocate_gpu(
            memory_required=2 * 1024**3,  # 2GB
            task_type='training',
            priority=10
        )
        
        assert gpu_id == 0
        assert len(gpu_manager.allocations) == 1
        assert gpu_manager.allocations[0].gpu_id == 0
        assert gpu_manager.allocations[0].memory_allocated == 2 * 1024**3
    
    def test_allocate_gpu_insufficient_memory(self, gpu_manager, mock_gpu_info):
        """Test GPU allocation with insufficient memory."""
        mock_gpu_info.free_memory = 1 * 1024**3  # Only 1GB free
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        gpu_id = gpu_manager.allocate_gpu(
            memory_required=2 * 1024**3,  # Request 2GB
            task_type='training'
        )
        
        assert gpu_id is None
    
    def test_release_gpu(self, gpu_manager, mock_gpu_info):
        """Test GPU release."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        # Allocate GPU
        gpu_id = gpu_manager.allocate_gpu(
            memory_required=2 * 1024**3,
            task_type='training'
        )
        
        assert len(gpu_manager.allocations) == 1
        
        # Release GPU
        gpu_manager.release_gpu(gpu_id)
        
        # Check allocation removed
        assert len(gpu_manager.allocations) == 0
        assert len(mock_gpu_info.assigned_processes) == 0
    
    def test_optimal_batch_size(self, gpu_manager, mock_gpu_info):
        """Test optimal batch size calculation."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        batch_size = gpu_manager.get_optimal_batch_size(
            model_memory=2 * 1024**3,    # 2GB model
            sample_memory=10 * 1024**2,  # 10MB per sample
            gpu_id=0
        )
        
        # Should fit many samples with 20GB free
        assert batch_size > 0
        assert batch_size <= 2048  # Reasonable upper limit
        
        # Test power of 2 optimization
        assert batch_size & (batch_size - 1) == 0  # Is power of 2
    
    def test_memory_usage_stats(self, gpu_manager, mock_gpu_info):
        """Test memory usage statistics."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        # Single GPU stats
        stats = gpu_manager.get_memory_usage(gpu_id=0)
        assert stats['total_gb'] == 24.0
        assert stats['used_gb'] == 4.0
        assert stats['free_gb'] == 20.0
        assert abs(stats['utilization'] - 16.67) < 0.1
        
        # All GPUs stats
        all_stats = gpu_manager.get_memory_usage()
        assert all_stats['total_gb'] == 24.0
    
    def test_gpu_stats(self, gpu_manager, mock_gpu_info):
        """Test comprehensive GPU statistics."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        stats = gpu_manager.get_gpu_stats()
        assert len(stats) == 1
        
        gpu_stat = stats[0]
        assert gpu_stat['gpu_id'] == 0
        assert gpu_stat['name'] == "NVIDIA GeForce RTX 3090"
        assert gpu_stat['compute_capability'] == "8.6"
        assert gpu_stat['temperature'] == 50.0
        assert gpu_stat['power_draw'] == 150.0
    
    @patch('torch.cuda.synchronize')
    def test_synchronize(self, mock_sync, gpu_manager):
        """Test GPU synchronization."""
        gpu_manager.synchronize(0)
        mock_sync.assert_called_once()
        
        mock_sync.reset_mock()
        gpu_manager.synchronize()
        mock_sync.assert_called_once()
    
    @patch('torch.cuda.empty_cache')
    def test_clear_cache(self, mock_clear, gpu_manager):
        """Test cache clearing."""
        gpu_manager.clear_cache()
        mock_clear.assert_called_once()
    
    def test_allocation_summary(self, gpu_manager, mock_gpu_info):
        """Test allocation summary."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        
        # Create multiple allocations
        for i in range(3):
            gpu_manager.allocate_gpu(
                memory_required=1 * 1024**3,
                task_type='training' if i < 2 else 'inference',
                priority=i
            )
        
        summary = gpu_manager.get_allocation_summary()
        assert summary['total_allocations'] == 3
        assert summary['by_task_type']['training'] == 2
        assert summary['by_task_type']['inference'] == 1
        assert summary['by_gpu'][0] == 3
        assert summary['total_memory_allocated_gb'] == 3.0
    
    def test_context_manager(self, mock_gpu_info):
        """Test GPU manager as context manager."""
        with patch('alpha_pulse.ml.gpu.gpu_manager.pynvml'):
            with GPUManager() as manager:
                manager.gpu_info[0] = mock_gpu_info
                gpu_id = manager.allocate_gpu(
                    memory_required=1 * 1024**3,
                    task_type='test'
                )
                assert gpu_id is not None
            
            # Cleanup should have been called
            assert not manager.monitoring_active
    
    def test_concurrent_allocation(self, gpu_manager, mock_gpu_info):
        """Test concurrent GPU allocation."""
        gpu_manager.gpu_info[0] = mock_gpu_info
        results = []
        
        def allocate_gpu():
            gpu_id = gpu_manager.allocate_gpu(
                memory_required=1 * 1024**3,
                task_type='concurrent'
            )
            results.append(gpu_id)
            if gpu_id is not None:
                time.sleep(0.1)
                gpu_manager.release_gpu(gpu_id)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=allocate_gpu)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed with enough memory
        assert all(r is not None for r in results)
    
    @patch('pynvml.nvmlDeviceGetMemoryInfo')
    @patch('pynvml.nvmlDeviceGetUtilizationRates')
    def test_monitoring(self, mock_util, mock_mem, gpu_manager):
        """Test GPU monitoring."""
        # Mock memory info
        mock_mem.return_value = MagicMock(
            total=24 * 1024**3,
            free=20 * 1024**3,
            used=4 * 1024**3
        )
        
        # Mock utilization
        mock_util.return_value = MagicMock(gpu=50)
        
        # Let monitoring run briefly
        time.sleep(0.1)
        
        # Check if monitoring is active
        assert gpu_manager.monitoring_active
    
    def test_error_handling(self, gpu_manager):
        """Test error handling in GPU manager."""
        # Test with invalid GPU ID
        gpu_manager.release_gpu(999)  # Should not raise
        
        # Test allocation with no GPUs
        gpu_manager.gpu_info.clear()
        gpu_id = gpu_manager.allocate_gpu(
            memory_required=1 * 1024**3,
            task_type='test'
        )
        assert gpu_id is None


class TestGPUAllocation:
    """Test GPU allocation record."""
    
    def test_allocation_creation(self):
        """Test creating GPU allocation."""
        import os
        from datetime import datetime
        
        alloc = GPUAllocation(
            process_id=os.getpid(),
            gpu_id=0,
            memory_allocated=2 * 1024**3,
            timestamp=datetime.now(),
            task_type='training',
            priority=10
        )
        
        assert alloc.process_id == os.getpid()
        assert alloc.gpu_id == 0
        assert alloc.memory_allocated == 2 * 1024**3
        assert alloc.task_type == 'training'
        assert alloc.priority == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])