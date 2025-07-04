"""Memory management for efficient streaming data processing."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import heapq
import sys
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Item stored in memory with metadata."""
    data: Any
    timestamp: datetime
    importance: float = 1.0
    access_count: int = 0
    last_access: Optional[datetime] = None
    size_bytes: int = 0


class MemoryManager:
    """Manages memory efficiently for streaming data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_memory_mb = config.get('max_memory_mb', 1024)
        self.eviction_policy = config.get('eviction_policy', 'lru')
        self.time_window = timedelta(seconds=config.get('time_window_seconds', 3600))
        self.importance_decay = config.get('importance_decay', 0.95)
        
        self._memory: OrderedDict[str, MemoryItem] = OrderedDict()
        self._current_memory_usage = 0
        self._access_history = deque(maxlen=1000)
        self._gc_threshold = config.get('gc_threshold', 0.9)
        
    def store(self, key: str, data: Any, importance: float = 1.0) -> bool:
        """Store data in memory with importance score."""
        # Estimate memory size
        size_bytes = self._estimate_size(data)
        
        # Check if we need to evict items
        if self._should_evict(size_bytes):
            self._evict_items(size_bytes)
            
        # Create memory item
        item = MemoryItem(
            data=data,
            timestamp=datetime.now(),
            importance=importance,
            size_bytes=size_bytes
        )
        
        # Store item
        self._memory[key] = item
        self._current_memory_usage += size_bytes
        
        # Trigger garbage collection if needed
        if self._current_memory_usage / (self.max_memory_mb * 1024 * 1024) > self._gc_threshold:
            gc.collect()
            
        return True
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from memory."""
        if key not in self._memory:
            return None
            
        item = self._memory[key]
        item.access_count += 1
        item.last_access = datetime.now()
        
        # Move to end for LRU
        if self.eviction_policy == 'lru':
            self._memory.move_to_end(key)
            
        # Record access
        self._access_history.append((key, datetime.now()))
        
        return item.data
        
    def batch_retrieve(self, keys: List[str]) -> Dict[str, Any]:
        """Retrieve multiple items efficiently."""
        result = {}
        for key in keys:
            data = self.retrieve(key)
            if data is not None:
                result[key] = data
        return result
        
    def remove(self, key: str) -> bool:
        """Remove item from memory."""
        if key not in self._memory:
            return False
            
        item = self._memory.pop(key)
        self._current_memory_usage -= item.size_bytes
        return True
        
    def _should_evict(self, required_bytes: int) -> bool:
        """Check if eviction is needed."""
        max_bytes = self.max_memory_mb * 1024 * 1024
        return self._current_memory_usage + required_bytes > max_bytes
        
    def _evict_items(self, required_bytes: int) -> None:
        """Evict items based on policy."""
        max_bytes = self.max_memory_mb * 1024 * 1024
        target_usage = max_bytes * 0.8  # Target 80% usage after eviction
        
        bytes_to_free = self._current_memory_usage + required_bytes - target_usage
        
        if bytes_to_free <= 0:
            return
            
        if self.eviction_policy == 'lru':
            self._evict_lru(bytes_to_free)
        elif self.eviction_policy == 'lfu':
            self._evict_lfu(bytes_to_free)
        elif self.eviction_policy == 'importance':
            self._evict_by_importance(bytes_to_free)
        elif self.eviction_policy == 'time':
            self._evict_by_time(bytes_to_free)
        elif self.eviction_policy == 'adaptive':
            self._evict_adaptive(bytes_to_free)
            
    def _evict_lru(self, bytes_to_free: int) -> None:
        """Evict least recently used items."""
        freed_bytes = 0
        keys_to_remove = []
        
        for key, item in self._memory.items():
            if freed_bytes >= bytes_to_free:
                break
            keys_to_remove.append(key)
            freed_bytes += item.size_bytes
            
        for key in keys_to_remove:
            self.remove(key)
            
        logger.info(f"Evicted {len(keys_to_remove)} items using LRU policy")
        
    def _evict_lfu(self, bytes_to_free: int) -> None:
        """Evict least frequently used items."""
        # Sort by access count
        items = [(item.access_count, key, item) 
                 for key, item in self._memory.items()]
        heapq.heapify(items)
        
        freed_bytes = 0
        keys_to_remove = []
        
        while items and freed_bytes < bytes_to_free:
            _, key, item = heapq.heappop(items)
            keys_to_remove.append(key)
            freed_bytes += item.size_bytes
            
        for key in keys_to_remove:
            self.remove(key)
            
        logger.info(f"Evicted {len(keys_to_remove)} items using LFU policy")
        
    def _evict_by_importance(self, bytes_to_free: int) -> None:
        """Evict items with lowest importance scores."""
        # Apply time decay to importance scores
        current_time = datetime.now()
        
        scored_items = []
        for key, item in self._memory.items():
            age_seconds = (current_time - item.timestamp).total_seconds()
            decayed_importance = item.importance * (self.importance_decay ** (age_seconds / 3600))
            scored_items.append((decayed_importance, key, item))
            
        # Sort by importance (ascending)
        scored_items.sort(key=lambda x: x[0])
        
        freed_bytes = 0
        keys_to_remove = []
        
        for _, key, item in scored_items:
            if freed_bytes >= bytes_to_free:
                break
            keys_to_remove.append(key)
            freed_bytes += item.size_bytes
            
        for key in keys_to_remove:
            self.remove(key)
            
        logger.info(f"Evicted {len(keys_to_remove)} items by importance")
        
    def _evict_by_time(self, bytes_to_free: int) -> None:
        """Evict oldest items outside time window."""
        current_time = datetime.now()
        cutoff_time = current_time - self.time_window
        
        freed_bytes = 0
        keys_to_remove = []
        
        for key, item in self._memory.items():
            if item.timestamp < cutoff_time:
                keys_to_remove.append(key)
                freed_bytes += item.size_bytes
                if freed_bytes >= bytes_to_free:
                    break
                    
        for key in keys_to_remove:
            self.remove(key)
            
        logger.info(f"Evicted {len(keys_to_remove)} items by time")
        
    def _evict_adaptive(self, bytes_to_free: int) -> None:
        """Adaptive eviction combining multiple strategies."""
        # Calculate composite score for each item
        current_time = datetime.now()
        scored_items = []
        
        for key, item in self._memory.items():
            # Time factor
            age_seconds = (current_time - item.timestamp).total_seconds()
            time_score = 1.0 / (1.0 + age_seconds / 3600)
            
            # Access factor
            access_score = item.access_count / (1.0 + age_seconds / 3600)
            
            # Importance factor with decay
            importance_score = item.importance * (self.importance_decay ** (age_seconds / 3600))
            
            # Recency factor
            recency_score = 0.0
            if item.last_access:
                recency_seconds = (current_time - item.last_access).total_seconds()
                recency_score = 1.0 / (1.0 + recency_seconds / 300)  # 5-minute half-life
                
            # Composite score (higher is better)
            composite_score = (
                0.3 * time_score +
                0.3 * access_score +
                0.2 * importance_score +
                0.2 * recency_score
            )
            
            scored_items.append((composite_score, key, item))
            
        # Sort by composite score (ascending = worst first)
        scored_items.sort(key=lambda x: x[0])
        
        freed_bytes = 0
        keys_to_remove = []
        
        for _, key, item in scored_items:
            if freed_bytes >= bytes_to_free:
                break
            keys_to_remove.append(key)
            freed_bytes += item.size_bytes
            
        for key in keys_to_remove:
            self.remove(key)
            
        logger.info(f"Evicted {len(keys_to_remove)} items using adaptive policy")
        
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        return sys.getsizeof(obj)
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        system_memory = process.memory_info()
        
        return {
            'managed_bytes': self._current_memory_usage,
            'managed_mb': self._current_memory_usage / (1024 * 1024),
            'managed_items': len(self._memory),
            'system_rss_mb': system_memory.rss / (1024 * 1024),
            'system_vms_mb': system_memory.vms / (1024 * 1024),
            'utilization': self._current_memory_usage / (self.max_memory_mb * 1024 * 1024),
            'eviction_policy': self.eviction_policy
        }
        
    def optimize_memory(self) -> None:
        """Optimize memory usage by compacting and garbage collecting."""
        # Remove expired items
        if self.time_window:
            current_time = datetime.now()
            cutoff_time = current_time - self.time_window
            
            keys_to_remove = [
                key for key, item in self._memory.items()
                if item.timestamp < cutoff_time
            ]
            
            for key in keys_to_remove:
                self.remove(key)
                
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Memory optimization complete. Current usage: {self._current_memory_usage / (1024 * 1024):.2f} MB")


class SlidingWindowBuffer:
    """Efficient sliding window buffer for streaming data."""
    
    def __init__(self, window_size: int, dtype: type = float):
        self.window_size = window_size
        self.dtype = dtype
        self._buffer = np.zeros(window_size, dtype=dtype)
        self._timestamps = deque(maxlen=window_size)
        self._position = 0
        self._is_full = False
        
    def append(self, value: Union[float, np.ndarray], timestamp: datetime) -> None:
        """Add value to buffer."""
        self._buffer[self._position] = value
        self._timestamps.append(timestamp)
        
        self._position = (self._position + 1) % self.window_size
        if self._position == 0:
            self._is_full = True
            
    def get_window(self) -> np.ndarray:
        """Get current window of data."""
        if self._is_full:
            return self._buffer
        else:
            return self._buffer[:self._position]
            
    def get_recent(self, n: int) -> np.ndarray:
        """Get n most recent values."""
        if n >= self.window_size:
            return self.get_window()
            
        if self._is_full:
            start = (self._position - n) % self.window_size
            if start < self._position:
                return self._buffer[start:self._position]
            else:
                return np.concatenate([
                    self._buffer[start:],
                    self._buffer[:self._position]
                ])
        else:
            return self._buffer[max(0, self._position - n):self._position]
            
    def get_by_time(self, start_time: datetime) -> Tuple[np.ndarray, List[datetime]]:
        """Get values since start_time."""
        indices = [
            i for i, ts in enumerate(self._timestamps)
            if ts >= start_time
        ]
        
        if not indices:
            return np.array([]), []
            
        values = self.get_window()
        return values[indices], [self._timestamps[i] for i in indices]
        
    @property
    def size(self) -> int:
        """Current number of elements in buffer."""
        return self.window_size if self._is_full else self._position
        
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._is_full
        
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.fill(0)
        self._timestamps.clear()
        self._position = 0
        self._is_full = False


class ReservoirSampler:
    """Reservoir sampling for maintaining representative sample of stream."""
    
    def __init__(self, reservoir_size: int):
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.n_seen = 0
        
    def add(self, item: Any) -> None:
        """Add item to reservoir."""
        self.n_seen += 1
        
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        else:
            # Randomly replace with decreasing probability
            j = np.random.randint(0, self.n_seen)
            if j < self.reservoir_size:
                self.reservoir[j] = item
                
    def get_sample(self) -> List[Any]:
        """Get current reservoir sample."""
        return self.reservoir.copy()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            'reservoir_size': self.reservoir_size,
            'current_size': len(self.reservoir),
            'total_seen': self.n_seen,
            'sampling_rate': len(self.reservoir) / self.n_seen if self.n_seen > 0 else 0
        }