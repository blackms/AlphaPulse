"""
GPU Batch Processing for High-Throughput Inference.

This module provides optimized batch processing capabilities for
GPU-accelerated model inference with dynamic batching and queuing.
"""

import asyncio
import logging
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """Batching strategy for request aggregation."""
    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"


@dataclass
class InferenceRequest:
    """Individual inference request."""
    request_id: str
    data: np.ndarray
    model_name: str
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchedRequest:
    """Batched inference request."""
    batch_id: str
    requests: List[InferenceRequest]
    model_name: str
    data: torch.Tensor
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.requests)
    
    @property
    def max_priority(self) -> int:
        """Get maximum priority in batch."""
        return max(req.priority for req in self.requests)


class DynamicBatcher:
    """Dynamic batching for optimal GPU utilization."""
    
    def __init__(self,
                 max_batch_size: int = 64,
                 max_wait_time: float = 0.01,  # 10ms
                 min_batch_size: int = 1,
                 strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE):
        """
        Initialize dynamic batcher.
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time in seconds
            min_batch_size: Minimum batch size
            strategy: Batching strategy
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        self.strategy = strategy
        
        # Request queues per model
        self.request_queues: Dict[str, queue.PriorityQueue] = defaultdict(
            lambda: queue.PriorityQueue()
        )
        
        # Batch statistics
        self.batch_stats = {
            'total_batches': 0,
            'total_requests': 0,
            'avg_batch_size': 0,
            'avg_wait_time': 0
        }
        
        self._batch_counter = 0
        self._lock = threading.Lock()
    
    def add_request(self, request: InferenceRequest):
        """Add request to batching queue."""
        # Priority queue uses negative priority for max heap
        priority = -request.priority
        self.request_queues[request.model_name].put((priority, request))
    
    def get_batch(self, model_name: str, timeout: Optional[float] = None) -> Optional[BatchedRequest]:
        """
        Get a batch of requests for processing.
        
        Args:
            model_name: Model name to get batch for
            timeout: Maximum wait time
            
        Returns:
            Batched request or None
        """
        if model_name not in self.request_queues:
            return None
        
        queue = self.request_queues[model_name]
        requests = []
        start_time = time.time()
        timeout = timeout or self.max_wait_time
        
        while len(requests) < self.max_batch_size:
            remaining_time = timeout - (time.time() - start_time)
            
            if remaining_time <= 0:
                break
            
            try:
                _, request = queue.get(timeout=remaining_time)
                requests.append(request)
                
                # Check if we should wait for more requests
                if self.strategy == BatchingStrategy.ADAPTIVE:
                    if self._should_wait_for_more(requests, queue.qsize()):
                        continue
                    else:
                        break
                elif self.strategy == BatchingStrategy.FIXED_SIZE:
                    if len(requests) < self.max_batch_size:
                        continue
                    else:
                        break
                
            except queue.Empty:
                break
        
        if not requests:
            return None
        
        # Create batch
        batch = self._create_batch(requests, model_name)
        
        # Update statistics
        self._update_stats(batch, time.time() - start_time)
        
        return batch
    
    def _should_wait_for_more(self, 
                             current_requests: List[InferenceRequest],
                             queue_size: int) -> bool:
        """Determine if we should wait for more requests."""
        # Adaptive strategy based on queue size and priorities
        if len(current_requests) >= self.min_batch_size:
            # Check if high priority requests are waiting
            if queue_size > 0:
                avg_priority = np.mean([req.priority for req in current_requests])
                if avg_priority < 5:  # Low priority batch
                    return queue_size > self.max_batch_size // 2
            return False
        return True
    
    def _create_batch(self, 
                     requests: List[InferenceRequest],
                     model_name: str) -> BatchedRequest:
        """Create batched request from individual requests."""
        with self._lock:
            batch_id = f"batch_{model_name}_{self._batch_counter}"
            self._batch_counter += 1
        
        # Stack data into batch tensor
        if TORCH_AVAILABLE:
            data_list = [torch.from_numpy(req.data) for req in requests]
            batch_data = torch.stack(data_list)
        else:
            batch_data = np.stack([req.data for req in requests])
        
        return BatchedRequest(
            batch_id=batch_id,
            requests=requests,
            model_name=model_name,
            data=batch_data
        )
    
    def _update_stats(self, batch: BatchedRequest, wait_time: float):
        """Update batching statistics."""
        with self._lock:
            self.batch_stats['total_batches'] += 1
            self.batch_stats['total_requests'] += batch.size
            
            # Update moving averages
            alpha = 0.1
            self.batch_stats['avg_batch_size'] = (
                (1 - alpha) * self.batch_stats['avg_batch_size'] +
                alpha * batch.size
            )
            self.batch_stats['avg_wait_time'] = (
                (1 - alpha) * self.batch_stats['avg_wait_time'] +
                alpha * wait_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        with self._lock:
            return self.batch_stats.copy()


class GPUBatchProcessor:
    """
    High-performance batch processor for GPU inference.
    
    Features:
    - Dynamic batching with multiple strategies
    - Priority-based request handling
    - Concurrent model execution
    - Automatic result distribution
    - Performance monitoring
    """
    
    def __init__(self,
                 gpu_manager: Any,
                 max_batch_size: int = 64,
                 max_queue_size: int = 1000,
                 num_workers: int = 2,
                 batching_strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE):
        """
        Initialize GPU batch processor.
        
        Args:
            gpu_manager: GPU resource manager
            max_batch_size: Maximum batch size
            max_queue_size: Maximum queue size per model
            num_workers: Number of processing workers
            batching_strategy: Batching strategy
        """
        self.gpu_manager = gpu_manager
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Dynamic batcher
        self.batcher = DynamicBatcher(
            max_batch_size=max_batch_size,
            strategy=batching_strategy
        )
        
        # Model registry
        self.models: Dict[str, nn.Module] = {}
        self.model_devices: Dict[str, int] = {}
        
        # Result queues
        self.result_futures: Dict[str, asyncio.Future] = {}
        
        # Processing threads
        self.workers = []
        self.running = False
        
        # Performance monitoring
        self.performance_stats = defaultdict(lambda: {
            'total_requests': 0,
            'total_batches': 0,
            'avg_latency': 0,
            'avg_throughput': 0,
            'errors': 0
        })
        
        # Start workers
        self._start_workers()
    
    def register_model(self, 
                      model_name: str,
                      model: nn.Module,
                      device_id: Optional[int] = None):
        """
        Register model for batch processing.
        
        Args:
            model_name: Unique model identifier
            model: PyTorch model
            device_id: GPU device ID (None for auto-allocation)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Allocate GPU if needed
        if device_id is None:
            model_size = sum(p.numel() * p.element_size() 
                           for p in model.parameters())
            device_id = self.gpu_manager.allocate_gpu(
                memory_required=model_size * 2,  # 2x for gradients
                task_type='inference'
            )
        
        if device_id is None:
            raise RuntimeError(f"No GPU available for model {model_name}")
        
        # Move model to GPU
        device = torch.device(f'cuda:{device_id}')
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        self.models[model_name] = model
        self.model_devices[model_name] = device_id
        
        logger.info(f"Registered model {model_name} on GPU {device_id}")
    
    async def process_async(self,
                           data: np.ndarray,
                           model_name: str,
                           priority: int = 0,
                           metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Process inference request asynchronously.
        
        Args:
            data: Input data
            model_name: Model to use
            priority: Request priority
            metadata: Optional metadata
            
        Returns:
            Model predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        # Create request
        request_id = f"req_{model_name}_{int(time.time() * 1000000)}"
        future = asyncio.Future()
        
        request = InferenceRequest(
            request_id=request_id,
            data=data,
            model_name=model_name,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store future for result delivery
        self.result_futures[request_id] = future
        
        # Add to batching queue
        self.batcher.add_request(request)
        
        # Wait for result
        try:
            result = await future
            return result
        finally:
            # Cleanup
            self.result_futures.pop(request_id, None)
    
    def process_sync(self,
                    data: np.ndarray,
                    model_name: str,
                    priority: int = 0) -> np.ndarray:
        """
        Process inference request synchronously.
        
        Args:
            data: Input data
            model_name: Model to use
            priority: Request priority
            
        Returns:
            Model predictions
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.process_async(data, model_name, priority)
            )
        finally:
            loop.close()
    
    def _start_workers(self):
        """Start processing workers."""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} processing workers")
    
    def _worker_loop(self, worker_id: int):
        """Worker processing loop."""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Process each model's queue
                for model_name in list(self.models.keys()):
                    batch = self.batcher.get_batch(model_name, timeout=0.001)
                    
                    if batch:
                        self._process_batch(batch, worker_id)
                
                # Brief sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def _process_batch(self, batch: BatchedRequest, worker_id: int):
        """Process a batch of requests."""
        start_time = time.time()
        model_name = batch.model_name
        
        try:
            # Get model and device
            model = self.models[model_name]
            device_id = self.model_devices[model_name]
            
            # Process batch
            with torch.cuda.device(device_id):
                with torch.no_grad():
                    # Move data to GPU
                    if not isinstance(batch.data, torch.Tensor):
                        batch_data = torch.from_numpy(batch.data)
                    else:
                        batch_data = batch.data
                    
                    batch_data = batch_data.to(f'cuda:{device_id}')
                    
                    # Run inference
                    predictions = model(batch_data)
                    
                    # Move results back to CPU
                    predictions = predictions.cpu().numpy()
            
            # Distribute results
            for i, request in enumerate(batch.requests):
                result = predictions[i]
                
                # Deliver result
                if request.request_id in self.result_futures:
                    future = self.result_futures[request.request_id]
                    if not future.done():
                        # Set result in event loop
                        loop = future.get_loop()
                        loop.call_soon_threadsafe(future.set_result, result)
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(result, request.metadata)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            # Update performance stats
            self._update_performance_stats(
                model_name, batch, time.time() - start_time
            )
            
            logger.debug(f"Worker {worker_id} processed batch {batch.batch_id} "
                        f"({batch.size} requests) in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
            # Set error for all requests
            for request in batch.requests:
                if request.request_id in self.result_futures:
                    future = self.result_futures[request.request_id]
                    if not future.done():
                        loop = future.get_loop()
                        loop.call_soon_threadsafe(future.set_exception, e)
            
            # Update error stats
            self.performance_stats[model_name]['errors'] += 1
    
    def _update_performance_stats(self,
                                 model_name: str,
                                 batch: BatchedRequest,
                                 processing_time: float):
        """Update performance statistics."""
        stats = self.performance_stats[model_name]
        
        # Update counters
        stats['total_requests'] += batch.size
        stats['total_batches'] += 1
        
        # Update latency (moving average)
        alpha = 0.1
        avg_latency = processing_time / batch.size
        stats['avg_latency'] = (
            (1 - alpha) * stats['avg_latency'] +
            alpha * avg_latency
        )
        
        # Update throughput
        throughput = batch.size / processing_time
        stats['avg_throughput'] = (
            (1 - alpha) * stats['avg_throughput'] +
            alpha * throughput
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'models': dict(self.performance_stats),
            'batching': self.batcher.get_stats(),
            'queue_sizes': {
                model: self.batcher.request_queues[model].qsize()
                for model in self.models
            }
        }
        return stats
    
    def optimize_batch_size(self, model_name: str) -> int:
        """
        Dynamically optimize batch size based on performance.
        
        Args:
            model_name: Model to optimize
            
        Returns:
            Optimal batch size
        """
        if model_name not in self.performance_stats:
            return self.max_batch_size
        
        stats = self.performance_stats[model_name]
        
        # Simple optimization based on throughput
        current_throughput = stats['avg_throughput']
        current_batch_size = self.batcher.max_batch_size
        
        # Try to find optimal batch size
        if current_throughput > 0:
            # Estimate based on latency and throughput
            optimal_size = int(current_batch_size * 1.1)  # Try 10% increase
            
            # Constrain to limits
            optimal_size = max(1, min(optimal_size, self.max_batch_size))
            
            return optimal_size
        
        return current_batch_size
    
    def shutdown(self):
        """Shutdown batch processor."""
        logger.info("Shutting down batch processor")
        
        self.running = False
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Cancel pending futures
        for future in self.result_futures.values():
            if not future.done():
                future.cancel()
        
        # Release GPU resources
        for model_name, device_id in self.model_devices.items():
            self.gpu_manager.release_gpu(device_id)
        
        logger.info("Batch processor shutdown complete")


class StreamingBatchProcessor:
    """
    Streaming batch processor for continuous data streams.
    
    Optimized for real-time trading data processing with
    minimal latency and high throughput.
    """
    
    def __init__(self,
                 gpu_batch_processor: GPUBatchProcessor,
                 window_size: int = 100,
                 update_interval: float = 0.1):
        """
        Initialize streaming processor.
        
        Args:
            gpu_batch_processor: GPU batch processor
            window_size: Sliding window size
            update_interval: Update interval in seconds
        """
        self.batch_processor = gpu_batch_processor
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Data buffers per stream
        self.stream_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.stream_models: Dict[str, str] = {}
        
        # Update thread
        self.running = False
        self.update_thread = None
    
    def add_stream(self, stream_id: str, model_name: str):
        """Add a data stream for processing."""
        self.stream_models[stream_id] = model_name
        logger.info(f"Added stream {stream_id} with model {model_name}")
    
    def add_data(self, stream_id: str, data: np.ndarray):
        """Add data to stream buffer."""
        if stream_id not in self.stream_models:
            raise ValueError(f"Stream {stream_id} not registered")
        
        self.stream_buffers[stream_id].append(data)
        
        # Maintain window size
        if len(self.stream_buffers[stream_id]) > self.window_size:
            self.stream_buffers[stream_id].pop(0)
    
    async def get_predictions(self, stream_id: str) -> Optional[np.ndarray]:
        """Get latest predictions for stream."""
        if stream_id not in self.stream_models:
            return None
        
        if not self.stream_buffers[stream_id]:
            return None
        
        # Get latest window
        window_data = np.stack(self.stream_buffers[stream_id])
        
        # Process through batch processor
        model_name = self.stream_models[stream_id]
        predictions = await self.batch_processor.process_async(
            window_data,
            model_name,
            priority=5  # Medium priority for streaming
        )
        
        return predictions
    
    def start(self):
        """Start streaming processor."""
        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
    
    def stop(self):
        """Stop streaming processor."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Continuous update loop for streaming data."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # Process each stream
                for stream_id in list(self.stream_models.keys()):
                    if self.stream_buffers[stream_id]:
                        loop.run_until_complete(
                            self.get_predictions(stream_id)
                        )
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Streaming update error: {e}")
        
        loop.close()