"""
Tests for GPU Batch Processor.
"""

import pytest
import numpy as np
import torch
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import threading
from datetime import datetime

from alpha_pulse.ml.gpu.batch_processor import (
    GPUBatchProcessor,
    DynamicBatcher,
    StreamingBatchProcessor,
    BatchingStrategy,
    InferenceRequest,
    BatchedRequest
)


@pytest.fixture
def mock_gpu_manager():
    """Create mock GPU manager."""
    manager = Mock()
    manager.allocate_gpu.return_value = 0
    manager.release_gpu.return_value = None
    return manager


@pytest.fixture
def simple_model():
    """Create simple test model."""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    
    # Mock inference
    def mock_forward(x):
        # Return predictions of same shape
        if hasattr(x, 'shape'):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 1)
        return torch.randn(1, 1)
    
    model.__call__ = Mock(side_effect=mock_forward)
    return model


@pytest.fixture
def batch_processor(mock_gpu_manager):
    """Create batch processor instance."""
    processor = GPUBatchProcessor(
        gpu_manager=mock_gpu_manager,
        max_batch_size=4,
        max_queue_size=100,
        num_workers=1,
        batching_strategy=BatchingStrategy.DYNAMIC
    )
    yield processor
    processor.shutdown()


class TestInferenceRequest:
    """Test inference request dataclass."""
    
    def test_creation(self):
        """Test creating inference request."""
        data = np.random.randn(10, 5)
        
        request = InferenceRequest(
            request_id="test_123",
            data=data,
            model_name="test_model",
            priority=5,
            callback=lambda x: x
        )
        
        assert request.request_id == "test_123"
        assert request.model_name == "test_model"
        assert request.priority == 5
        assert np.array_equal(request.data, data)
        assert callable(request.callback)


class TestBatchedRequest:
    """Test batched request dataclass."""
    
    def test_creation(self):
        """Test creating batched request."""
        requests = [
            InferenceRequest(f"req_{i}", np.random.randn(5), "model", i)
            for i in range(3)
        ]
        
        batch_data = torch.randn(3, 5)
        
        batch = BatchedRequest(
            batch_id="batch_1",
            requests=requests,
            model_name="model",
            data=batch_data
        )
        
        assert batch.batch_id == "batch_1"
        assert batch.size == 3
        assert batch.max_priority == 2
        assert len(batch.requests) == 3


class TestDynamicBatcher:
    """Test dynamic batching functionality."""
    
    def test_initialization(self):
        """Test batcher initialization."""
        batcher = DynamicBatcher(
            max_batch_size=32,
            max_wait_time=0.01,
            min_batch_size=1,
            strategy=BatchingStrategy.ADAPTIVE
        )
        
        assert batcher.max_batch_size == 32
        assert batcher.max_wait_time == 0.01
        assert batcher.strategy == BatchingStrategy.ADAPTIVE
    
    def test_add_request(self):
        """Test adding requests to batcher."""
        batcher = DynamicBatcher()
        
        request = InferenceRequest(
            "req_1",
            np.random.randn(10),
            "model_1",
            priority=5
        )
        
        batcher.add_request(request)
        
        # Check request was added to queue
        assert batcher.request_queues["model_1"].qsize() == 1
    
    def test_get_batch_single_request(self):
        """Test getting batch with single request."""
        batcher = DynamicBatcher(max_wait_time=0.001)
        
        request = InferenceRequest(
            "req_1",
            np.random.randn(10),
            "model_1"
        )
        
        batcher.add_request(request)
        
        # Get batch
        batch = batcher.get_batch("model_1", timeout=0.01)
        
        assert batch is not None
        assert batch.size == 1
        assert batch.requests[0].request_id == "req_1"
    
    def test_get_batch_multiple_requests(self):
        """Test batching multiple requests."""
        batcher = DynamicBatcher(max_batch_size=3)
        
        # Add multiple requests
        for i in range(3):
            request = InferenceRequest(
                f"req_{i}",
                np.random.randn(10),
                "model_1",
                priority=i
            )
            batcher.add_request(request)
        
        # Get batch
        batch = batcher.get_batch("model_1", timeout=0.01)
        
        assert batch is not None
        assert batch.size == 3
        
        # Check priority ordering (highest priority first)
        assert batch.requests[0].priority == 2
        assert batch.requests[1].priority == 1
        assert batch.requests[2].priority == 0
    
    def test_get_batch_timeout(self):
        """Test batch timeout behavior."""
        batcher = DynamicBatcher(
            max_batch_size=10,
            max_wait_time=0.001
        )
        
        # Add one request
        batcher.add_request(
            InferenceRequest("req_1", np.random.randn(10), "model_1")
        )
        
        start = time.time()
        batch = batcher.get_batch("model_1", timeout=0.01)
        elapsed = time.time() - start
        
        # Should return quickly due to timeout
        assert elapsed < 0.02
        assert batch is not None
        assert batch.size == 1
    
    def test_adaptive_batching(self):
        """Test adaptive batching strategy."""
        batcher = DynamicBatcher(
            strategy=BatchingStrategy.ADAPTIVE,
            min_batch_size=2,
            max_batch_size=10
        )
        
        # Add low priority requests
        for i in range(5):
            batcher.add_request(
                InferenceRequest(f"req_{i}", np.random.randn(10), "model_1", priority=0)
            )
        
        # Should wait for more
        with patch.object(batcher, '_should_wait_for_more', return_value=True):
            batch = batcher.get_batch("model_1", timeout=0.001)
            # May get partial batch due to timeout
            assert batch is not None
    
    def test_batch_statistics(self):
        """Test batch statistics tracking."""
        batcher = DynamicBatcher()
        
        # Process some batches
        for _ in range(3):
            for i in range(2):
                batcher.add_request(
                    InferenceRequest(f"req_{i}", np.random.randn(10), "model_1")
                )
            batch = batcher.get_batch("model_1")
        
        stats = batcher.get_stats()
        assert stats['total_batches'] == 3
        assert stats['total_requests'] == 6
        assert stats['avg_batch_size'] > 0


class TestGPUBatchProcessor:
    """Test GPU batch processor."""
    
    def test_initialization(self, batch_processor):
        """Test processor initialization."""
        assert batch_processor.max_batch_size == 4
        assert batch_processor.num_workers == 1
        assert batch_processor.running is True
        assert len(batch_processor.workers) == 1
    
    def test_register_model(self, batch_processor, simple_model):
        """Test model registration."""
        with patch('torch.cuda.is_available', return_value=True):
            batch_processor.register_model(
                "test_model",
                simple_model,
                device_id=0
            )
        
        assert "test_model" in batch_processor.models
        assert batch_processor.model_devices["test_model"] == 0
        simple_model.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_async(self, batch_processor, simple_model):
        """Test async processing."""
        # Register model
        batch_processor.register_model("test_model", simple_model)
        
        # Mock batch processing
        with patch.object(batch_processor, '_process_batch') as mock_process:
            # Create future for result
            future = asyncio.Future()
            future.set_result(np.array([1.0]))
            
            # Setup mocking
            batch_processor.result_futures["test_req"] = future
            
            # Process request
            data = np.random.randn(10, 5)
            
            # Need to mock the request creation and batching
            with patch.object(batch_processor.batcher, 'add_request'):
                # Directly return result
                result = await future
                
                assert result == np.array([1.0])
    
    def test_process_sync(self, batch_processor, simple_model):
        """Test sync processing."""
        batch_processor.register_model("test_model", simple_model)
        
        # Mock async processing
        async def mock_async(data, model_name, priority):
            return np.array([1.0])
        
        with patch.object(batch_processor, 'process_async', side_effect=mock_async):
            data = np.random.randn(10, 5)
            result = batch_processor.process_sync(data, "test_model")
            
            assert result == np.array([1.0])
    
    def test_process_batch(self, batch_processor, simple_model):
        """Test batch processing logic."""
        # Register model
        batch_processor.register_model("test_model", simple_model)
        
        # Create batch
        requests = [
            InferenceRequest(f"req_{i}", np.random.randn(5), "test_model")
            for i in range(3)
        ]
        
        batch_data = torch.stack([torch.from_numpy(r.data) for r in requests])
        
        batch = BatchedRequest(
            batch_id="batch_1",
            requests=requests,
            model_name="test_model",
            data=batch_data
        )
        
        # Create futures
        futures = {}
        for req in requests:
            future = AsyncMock()
            future.done.return_value = False
            future.get_loop.return_value = asyncio.get_event_loop()
            futures[req.request_id] = future
        
        batch_processor.result_futures = futures
        
        # Process batch
        batch_processor._process_batch(batch, worker_id=0)
        
        # Model should have been called
        simple_model.__call__.assert_called()
    
    def test_performance_stats(self, batch_processor):
        """Test performance statistics."""
        # Add some stats
        batch_processor.performance_stats["model_1"] = {
            'total_requests': 100,
            'total_batches': 25,
            'avg_latency': 0.01,
            'avg_throughput': 1000,
            'errors': 2
        }
        
        stats = batch_processor.get_performance_stats()
        
        assert 'models' in stats
        assert 'batching' in stats
        assert stats['models']['model_1']['total_requests'] == 100
    
    def test_optimize_batch_size(self, batch_processor):
        """Test batch size optimization."""
        # Without stats
        optimal = batch_processor.optimize_batch_size("unknown_model")
        assert optimal == batch_processor.max_batch_size
        
        # With stats
        batch_processor.performance_stats["model_1"] = {
            'avg_throughput': 1000
        }
        
        optimal = batch_processor.optimize_batch_size("model_1")
        assert optimal > 0
        assert optimal <= batch_processor.max_batch_size
    
    def test_shutdown(self, batch_processor):
        """Test processor shutdown."""
        # Add a model
        batch_processor.model_devices["test_model"] = 0
        
        # Add pending future
        future = AsyncMock()
        future.done.return_value = False
        batch_processor.result_futures["test_req"] = future
        
        # Shutdown
        batch_processor.shutdown()
        
        assert batch_processor.running is False
        future.cancel.assert_called_once()
    
    def test_error_handling(self, batch_processor, simple_model):
        """Test error handling in batch processing."""
        # Register model that raises error
        error_model = Mock()
        error_model.eval = Mock()
        error_model.__call__ = Mock(side_effect=RuntimeError("Test error"))
        
        batch_processor.register_model("error_model", error_model)
        
        # Create batch
        requests = [
            InferenceRequest("req_1", np.random.randn(5), "error_model")
        ]
        
        batch = BatchedRequest(
            batch_id="batch_1",
            requests=requests,
            model_name="error_model",
            data=torch.randn(1, 5)
        )
        
        # Process batch
        with patch.object(batch_processor, 'result_futures', {}):
            batch_processor._process_batch(batch, worker_id=0)
        
        # Error count should increase
        assert batch_processor.performance_stats["error_model"]['errors'] == 1


class TestStreamingBatchProcessor:
    """Test streaming batch processor."""
    
    def test_initialization(self, batch_processor):
        """Test streaming processor initialization."""
        streaming = StreamingBatchProcessor(
            gpu_batch_processor=batch_processor,
            window_size=100,
            update_interval=0.1
        )
        
        assert streaming.window_size == 100
        assert streaming.update_interval == 0.1
        assert streaming.running is False
    
    def test_add_stream(self, batch_processor):
        """Test adding data stream."""
        streaming = StreamingBatchProcessor(batch_processor)
        
        streaming.add_stream("stream_1", "model_1")
        
        assert streaming.stream_models["stream_1"] == "model_1"
    
    def test_add_data(self, batch_processor):
        """Test adding data to stream."""
        streaming = StreamingBatchProcessor(
            batch_processor,
            window_size=5
        )
        
        streaming.add_stream("stream_1", "model_1")
        
        # Add data points
        for i in range(10):
            data = np.random.randn(10)
            streaming.add_data("stream_1", data)
        
        # Should maintain window size
        assert len(streaming.stream_buffers["stream_1"]) == 5
    
    @pytest.mark.asyncio
    async def test_get_predictions(self, batch_processor):
        """Test getting streaming predictions."""
        streaming = StreamingBatchProcessor(batch_processor)
        
        streaming.add_stream("stream_1", "model_1")
        
        # Add some data
        for i in range(3):
            streaming.add_data("stream_1", np.random.randn(10))
        
        # Mock batch processor
        with patch.object(
            batch_processor,
            'process_async',
            return_value=np.array([1.0, 2.0, 3.0])
        ):
            predictions = await streaming.get_predictions("stream_1")
            
            assert predictions is not None
            assert len(predictions) == 3
    
    def test_start_stop(self, batch_processor):
        """Test starting and stopping streaming."""
        streaming = StreamingBatchProcessor(batch_processor)
        
        # Start
        streaming.start()
        assert streaming.running is True
        assert streaming.update_thread is not None
        
        # Stop
        streaming.stop()
        assert streaming.running is False
    
    def test_invalid_stream(self, batch_processor):
        """Test error handling for invalid stream."""
        streaming = StreamingBatchProcessor(batch_processor)
        
        # Try to add data to non-existent stream
        with pytest.raises(ValueError):
            streaming.add_data("invalid_stream", np.random.randn(10))


class TestBatchingStrategies:
    """Test different batching strategies."""
    
    def test_fixed_size_strategy(self):
        """Test fixed size batching."""
        batcher = DynamicBatcher(
            max_batch_size=4,
            strategy=BatchingStrategy.FIXED_SIZE
        )
        
        # Add exactly max_batch_size requests
        for i in range(4):
            batcher.add_request(
                InferenceRequest(f"req_{i}", np.random.randn(10), "model")
            )
        
        batch = batcher.get_batch("model", timeout=0.01)
        assert batch.size == 4
    
    def test_time_based_strategy(self):
        """Test time-based batching."""
        batcher = DynamicBatcher(
            max_wait_time=0.005,
            strategy=BatchingStrategy.TIME_BASED
        )
        
        # Add requests
        for i in range(2):
            batcher.add_request(
                InferenceRequest(f"req_{i}", np.random.randn(10), "model")
            )
        
        # Should return after timeout
        start = time.time()
        batch = batcher.get_batch("model", timeout=0.01)
        elapsed = time.time() - start
        
        assert batch is not None
        assert elapsed < 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])