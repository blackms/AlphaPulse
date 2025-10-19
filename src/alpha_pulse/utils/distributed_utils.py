"""Utilities for distributed computing operations."""

import logging
import os
import sys
import time
import pickle
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from functools import wraps
import psutil
import ray
import dask
from dask.distributed import Client, get_client
import socket
import threading
from collections import deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ResourceMonitor:
    """Monitor system resources for distributed computing."""
    cpu_threshold: float = 90.0  # Percentage
    memory_threshold: float = 85.0  # Percentage
    disk_threshold: float = 90.0  # Percentage
    network_threshold: float = 80.0  # Mbps
    check_interval: int = 5  # seconds
    
    def __post_init__(self):
        """Initialize monitoring state."""
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._resource_history = {
            "cpu": deque(maxlen=100),
            "memory": deque(maxlen=100),
            "disk": deque(maxlen=100),
            "network": deque(maxlen=100),
        }
        self._alerts: List[Dict[str, Any]] = []
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
        
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_mbps": self._get_network_usage(),
        }
        
    def get_resource_history(self) -> Dict[str, List[float]]:
        """Get resource usage history."""
        return {
            key: list(values)
            for key, values in self._resource_history.items()
        }
        
    def check_resource_availability(self) -> Tuple[bool, List[str]]:
        """Check if resources are available for computation.
        
        Returns:
            Tuple of (available, list of warnings)
        """
        resources = self.get_current_resources()
        warnings = []
        
        if resources["cpu_percent"] > self.cpu_threshold:
            warnings.append(f"CPU usage high: {resources['cpu_percent']:.1f}%")
            
        if resources["memory_percent"] > self.memory_threshold:
            warnings.append(f"Memory usage high: {resources['memory_percent']:.1f}%")
            
        if resources["disk_percent"] > self.disk_threshold:
            warnings.append(f"Disk usage high: {resources['disk_percent']:.1f}%")
            
        return len(warnings) == 0, warnings
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                resources = self.get_current_resources()
                
                # Update history
                self._resource_history["cpu"].append(resources["cpu_percent"])
                self._resource_history["memory"].append(resources["memory_percent"])
                self._resource_history["disk"].append(resources["disk_percent"])
                self._resource_history["network"].append(resources["network_mbps"])
                
                # Check thresholds
                self._check_thresholds(resources)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                
            time.sleep(self.check_interval)
            
    def _check_thresholds(self, resources: Dict[str, float]) -> None:
        """Check resource thresholds and generate alerts."""
        timestamp = datetime.now()
        
        if resources["cpu_percent"] > self.cpu_threshold:
            self._alerts.append({
                "timestamp": timestamp,
                "type": "cpu",
                "value": resources["cpu_percent"],
                "threshold": self.cpu_threshold,
            })
            
        if resources["memory_percent"] > self.memory_threshold:
            self._alerts.append({
                "timestamp": timestamp,
                "type": "memory",
                "value": resources["memory_percent"],
                "threshold": self.memory_threshold,
            })
            
    def _get_network_usage(self) -> float:
        """Get current network usage in Mbps."""
        try:
            stats1 = psutil.net_io_counters()
            time.sleep(1)
            stats2 = psutil.net_io_counters()
            
            bytes_sent = stats2.bytes_sent - stats1.bytes_sent
            bytes_recv = stats2.bytes_recv - stats1.bytes_recv
            
            mbps = (bytes_sent + bytes_recv) * 8 / 1024 / 1024
            return mbps
        except Exception:
            return 0.0


class DataPartitioner:
    """Utilities for partitioning data for distributed processing."""
    
    @staticmethod
    def partition_by_time(data: pd.DataFrame,
                         n_partitions: int,
                         overlap: int = 0) -> List[pd.DataFrame]:
        """Partition data by time periods.
        
        Args:
            data: DataFrame with time index
            n_partitions: Number of partitions
            overlap: Number of overlapping periods
            
        Returns:
            List of DataFrame partitions
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
            
        total_rows = len(data)
        base_size = total_rows // n_partitions
        
        partitions = []
        for i in range(n_partitions):
            start_idx = max(0, i * base_size - overlap)
            
            if i == n_partitions - 1:
                # Last partition gets remaining data
                end_idx = total_rows
            else:
                end_idx = min((i + 1) * base_size + overlap, total_rows)
                
            partition = data.iloc[start_idx:end_idx]
            partitions.append(partition)
            
        return partitions
        
    @staticmethod
    def partition_by_symbol(data: Dict[str, pd.DataFrame],
                          n_partitions: int) -> List[Dict[str, pd.DataFrame]]:
        """Partition data by symbols.
        
        Args:
            data: Dictionary of symbol DataFrames
            n_partitions: Number of partitions
            
        Returns:
            List of symbol dictionaries
        """
        symbols = list(data.keys())
        n_symbols = len(symbols)
        symbols_per_partition = max(1, n_symbols // n_partitions)
        
        partitions = []
        for i in range(n_partitions):
            start_idx = i * symbols_per_partition
            if i == n_partitions - 1:
                # Last partition gets remaining symbols
                partition_symbols = symbols[start_idx:]
            else:
                end_idx = (i + 1) * symbols_per_partition
                partition_symbols = symbols[start_idx:end_idx]
                
            partition = {s: data[s] for s in partition_symbols}
            partitions.append(partition)
            
        return partitions
        
    @staticmethod
    def partition_by_size(data: pd.DataFrame,
                         max_size_mb: float = 128) -> List[pd.DataFrame]:
        """Partition data by memory size.
        
        Args:
            data: DataFrame to partition
            max_size_mb: Maximum partition size in MB
            
        Returns:
            List of DataFrame partitions
        """
        # Estimate DataFrame size
        size_bytes = data.memory_usage(deep=True).sum()
        size_mb = size_bytes / 1024 / 1024
        
        if size_mb <= max_size_mb:
            return [data]
            
        # Calculate number of partitions needed
        n_partitions = int(np.ceil(size_mb / max_size_mb))
        rows_per_partition = len(data) // n_partitions
        
        partitions = []
        for i in range(n_partitions):
            start_idx = i * rows_per_partition
            if i == n_partitions - 1:
                partition = data.iloc[start_idx:]
            else:
                end_idx = (i + 1) * rows_per_partition
                partition = data.iloc[start_idx:end_idx]
                
            partitions.append(partition)
            
        return partitions


class CacheManager:
    """Manage distributed caching for computation results."""
    
    def __init__(self, cache_dir: str = "./distributed_cache",
                 max_size_gb: float = 10.0):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._cache_index: Dict[str, Dict[str, Any]] = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache_index()
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key not in self._cache_index:
            return None
            
        cache_info = self._cache_index[key]
        cache_path = os.path.join(self.cache_dir, cache_info["filename"])
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache {key}: {e}")
            return None
            
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store result in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        try:
            # Check cache size
            self._enforce_size_limit()
            
            # Generate filename
            filename = f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.pkl"
            cache_path = os.path.join(self.cache_dir, filename)
            
            # Store value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
                
            # Update index
            self._cache_index[key] = {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "size": os.path.getsize(cache_path),
                "metadata": metadata or {},
            }
            
            self._save_cache_index()
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
            
    def clear(self) -> None:
        """Clear all cached data."""
        for cache_info in self._cache_index.values():
            cache_path = os.path.join(self.cache_dir, cache_info["filename"])
            try:
                os.remove(cache_path)
            except Exception:
                pass
                
        self._cache_index.clear()
        self._save_cache_index()
        
    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        index_path = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    self._cache_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
                self._cache_index = {}
                
    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        index_path = os.path.join(self.cache_dir, "cache_index.json")
        try:
            with open(index_path, 'w') as f:
                json.dump(self._cache_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
            
    def _enforce_size_limit(self) -> None:
        """Remove old entries if cache size exceeds limit."""
        total_size = sum(info["size"] for info in self._cache_index.values())
        
        if total_size <= self.max_size_bytes:
            return
            
        # Sort by timestamp and remove oldest
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        while total_size > self.max_size_bytes and sorted_entries:
            key, info = sorted_entries.pop(0)
            cache_path = os.path.join(self.cache_dir, info["filename"])
            
            try:
                os.remove(cache_path)
                total_size -= info["size"]
                del self._cache_index[key]
            except Exception:
                pass


def retry_on_failure(max_retries: int = 3,
                    delay: float = 1.0,
                    backoff: float = 2.0,
                    exceptions: Tuple[type, ...] = (Exception,)):
    """Decorator for retrying failed operations.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} retries failed")
                        
            raise last_exception
            
        return wrapper
    return decorator


def distribute_dataframe(df: pd.DataFrame,
                        framework: str = "auto") -> Union["ray.ObjectRef", "dask.dataframe.DataFrame"]:
    """Distribute a pandas DataFrame across cluster.
    
    Args:
        df: Pandas DataFrame
        framework: Distribution framework ("ray", "dask", or "auto")
        
    Returns:
        Distributed DataFrame reference
    """
    if framework == "ray" or (framework == "auto" and ray.is_initialized()):
        # Put DataFrame in Ray object store
        return ray.put(df)
    elif framework == "dask" or framework == "auto":
        try:
            client = get_client()
            # Convert to Dask DataFrame
            n_partitions = max(1, len(df) // 10000)
            return dask.dataframe.from_pandas(df, npartitions=n_partitions)
        except ValueError:
            # No Dask client, return original
            return df
    else:
        return df


def gather_results(futures: List[Any],
                  framework: str = "auto") -> List[Any]:
    """Gather results from distributed futures.
    
    Args:
        futures: List of futures
        framework: Distribution framework
        
    Returns:
        List of results
    """
    if not futures:
        return []
        
    if framework == "ray" or (framework == "auto" and ray.is_initialized()):
        return ray.get(futures)
    elif framework == "dask" or framework == "auto":
        try:
            client = get_client()
            return client.gather(futures)
        except ValueError:
            # No Dask client, assume results are direct values
            return futures
    else:
        return futures


def estimate_task_duration(data_size: int,
                         complexity: str = "medium") -> float:
    """Estimate task duration based on data size.
    
    Args:
        data_size: Size of data (e.g., number of rows)
        complexity: Task complexity ("low", "medium", "high")
        
    Returns:
        Estimated duration in seconds
    """
    # Base rates (rows per second)
    rates = {
        "low": 100000,
        "medium": 10000,
        "high": 1000,
    }
    
    rate = rates.get(complexity, rates["medium"])
    duration = data_size / rate
    
    # Add overhead
    duration += 0.5  # Fixed overhead
    
    return duration


def optimize_partition_size(total_size: int,
                          n_workers: int,
                          min_chunk_size: int = 1000,
                          max_chunk_size: Optional[int] = None) -> int:
    """Calculate optimal partition size for parallel processing.
    
    Args:
        total_size: Total data size
        n_workers: Number of workers
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        
    Returns:
        Optimal chunk size
    """
    # Basic calculation
    chunk_size = max(min_chunk_size, total_size // (n_workers * 4))
    
    # Apply maximum if specified
    if max_chunk_size:
        chunk_size = min(chunk_size, max_chunk_size)
        
    return chunk_size


def get_cluster_info(framework: str = "auto") -> Dict[str, Any]:
    """Get information about the active cluster.
    
    Args:
        framework: Distribution framework
        
    Returns:
        Cluster information
    """
    info = {
        "framework": "none",
        "n_workers": 1,
        "total_cores": psutil.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
    }
    
    if framework == "ray" or (framework == "auto" and ray.is_initialized()):
        info["framework"] = "ray"
        info["ray_resources"] = ray.cluster_resources()
        info["n_workers"] = len(ray.nodes())
        
    elif framework == "dask" or framework == "auto":
        try:
            client = get_client()
            info["framework"] = "dask"
            info["dask_scheduler"] = client.scheduler.address
            info["n_workers"] = len(client.nthreads())
            info["total_threads"] = sum(client.nthreads().values())
        except ValueError:
            pass
            
    return info


def validate_cluster_resources(required_memory_gb: float,
                             required_cores: int,
                             framework: str = "auto") -> Tuple[bool, str]:
    """Validate that cluster has required resources.
    
    Args:
        required_memory_gb: Required memory in GB
        required_cores: Required CPU cores
        framework: Distribution framework
        
    Returns:
        Tuple of (valid, message)
    """
    cluster_info = get_cluster_info(framework)
    
    if cluster_info["framework"] == "none":
        # Check local resources
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
        available_cores = psutil.cpu_count()
        
        if available_memory < required_memory_gb:
            return False, f"Insufficient memory: {available_memory:.1f}GB < {required_memory_gb}GB"
            
        if available_cores < required_cores:
            return False, f"Insufficient cores: {available_cores} < {required_cores}"
            
    elif cluster_info["framework"] == "ray":
        resources = cluster_info.get("ray_resources", {})
        available_memory = resources.get("memory", 0) / 1024 / 1024 / 1024
        available_cores = resources.get("CPU", 0)
        
        if available_memory < required_memory_gb:
            return False, f"Insufficient Ray memory: {available_memory:.1f}GB < {required_memory_gb}GB"
            
        if available_cores < required_cores:
            return False, f"Insufficient Ray CPUs: {available_cores} < {required_cores}"
            
    return True, "Sufficient resources available"