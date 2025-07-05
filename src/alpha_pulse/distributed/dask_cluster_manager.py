"""Dask cluster manager for distributed computing."""

import logging
import os
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
import pandas as pd

from dask.distributed import Client, as_completed, wait, LocalCluster
from dask.distributed import Worker, Scheduler
from dask import delayed, dataframe as dd, array as da
from dask.diagnostics import ProgressBar
import dask

logger = logging.getLogger(__name__)


class DaskClusterStatus(Enum):
    """Dask cluster status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SCALING = "scaling"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class DaskClusterConfig:
    """Configuration for Dask cluster."""
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "4GB"
    processes: bool = True
    dashboard_address: str = ":8787"
    scheduler_port: int = 0
    silence_logs: int = logging.WARNING
    local_directory: Optional[str] = None
    host: Optional[str] = None
    protocol: str = "tcp://"
    security: Optional[Dict[str, Any]] = None
    interface: Optional[str] = None
    death_timeout: str = "60s"
    preload: Optional[List[str]] = None
    preload_argv: Optional[List[str]] = None


class DaskClusterManager:
    """Manages Dask cluster for distributed computing."""
    
    def __init__(self, config: Optional[DaskClusterConfig] = None):
        """Initialize Dask cluster manager.
        
        Args:
            config: Dask cluster configuration
        """
        self.config = config or DaskClusterConfig()
        self.status = DaskClusterStatus.UNINITIALIZED
        self._client: Optional[Client] = None
        self._cluster: Optional[LocalCluster] = None
        self._shutdown_callbacks: List[Callable] = []
        
    def initialize_cluster(self, address: Optional[str] = None) -> bool:
        """Initialize Dask cluster.
        
        Args:
            address: Dask scheduler address (None for local cluster)
            
        Returns:
            Success status
        """
        try:
            self.status = DaskClusterStatus.INITIALIZING
            logger.info("Initializing Dask cluster...")
            
            if self._client is not None:
                logger.warning("Dask client already initialized")
                self.status = DaskClusterStatus.READY
                return True
            
            if address:
                # Connect to existing cluster
                self._client = Client(address)
            else:
                # Create local cluster
                cluster_kwargs = {
                    "n_workers": self.config.n_workers,
                    "threads_per_worker": self.config.threads_per_worker,
                    "memory_limit": self.config.memory_limit,
                    "processes": self.config.processes,
                    "dashboard_address": self.config.dashboard_address,
                    "scheduler_port": self.config.scheduler_port,
                    "silence_logs": self.config.silence_logs,
                    "death_timeout": self.config.death_timeout,
                }
                
                if self.config.local_directory:
                    cluster_kwargs["local_directory"] = self.config.local_directory
                if self.config.host:
                    cluster_kwargs["host"] = self.config.host
                if self.config.protocol:
                    cluster_kwargs["protocol"] = self.config.protocol
                if self.config.security:
                    cluster_kwargs["security"] = self.config.security
                if self.config.interface:
                    cluster_kwargs["interface"] = self.config.interface
                if self.config.preload:
                    cluster_kwargs["preload"] = self.config.preload
                if self.config.preload_argv:
                    cluster_kwargs["preload_argv"] = self.config.preload_argv
                    
                self._cluster = LocalCluster(**cluster_kwargs)
                self._client = Client(self._cluster)
            
            # Wait for workers to connect
            time.sleep(2)
            
            self.status = DaskClusterStatus.READY
            logger.info(f"Dask cluster initialized. Dashboard: {self._client.dashboard_link}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Dask cluster: {e}")
            self.status = DaskClusterStatus.ERROR
            return False
    
    def shutdown_cluster(self) -> bool:
        """Shutdown Dask cluster.
        
        Returns:
            Success status
        """
        try:
            self.status = DaskClusterStatus.SHUTTING_DOWN
            logger.info("Shutting down Dask cluster...")
            
            # Execute shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Shutdown callback failed: {e}")
            
            if self._client:
                self._client.close()
                self._client = None
                
            if self._cluster:
                self._cluster.close()
                self._cluster = None
                
            self.status = DaskClusterStatus.UNINITIALIZED
            logger.info("Dask cluster shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown Dask cluster: {e}")
            self.status = DaskClusterStatus.ERROR
            return False
    
    def scale_cluster(self, n_workers: int) -> bool:
        """Scale cluster to specified number of workers.
        
        Args:
            n_workers: Desired number of workers
            
        Returns:
            Success status
        """
        try:
            if not self._cluster:
                logger.warning("Cannot scale non-local cluster")
                return False
                
            self.status = DaskClusterStatus.SCALING
            logger.info(f"Scaling cluster to {n_workers} workers...")
            
            self._cluster.scale(n_workers)
            
            # Wait for scaling to complete
            time.sleep(2)
            
            self.status = DaskClusterStatus.READY
            logger.info(f"Cluster scaled to {len(self._client.nthreads())} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale cluster: {e}")
            self.status = DaskClusterStatus.ERROR
            return False
    
    def adapt_cluster(self, minimum: int = 0, maximum: int = 10) -> bool:
        """Enable adaptive scaling for the cluster.
        
        Args:
            minimum: Minimum number of workers
            maximum: Maximum number of workers
            
        Returns:
            Success status
        """
        try:
            if not self._cluster:
                logger.warning("Cannot adapt non-local cluster")
                return False
                
            self._cluster.adapt(minimum=minimum, maximum=maximum)
            logger.info(f"Adaptive scaling enabled: {minimum}-{maximum} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable adaptive scaling: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get detailed cluster information.
        
        Returns:
            Cluster information dictionary
        """
        if not self._client:
            return {}
            
        try:
            info = self._client.scheduler_info()
            return {
                "status": self.status.value,
                "n_workers": len(info["workers"]),
                "workers": list(info["workers"].keys()),
                "total_threads": sum(self._client.nthreads().values()),
                "memory": {
                    worker: info["workers"][worker]["memory_limit"]
                    for worker in info["workers"]
                },
                "dashboard_link": self._client.dashboard_link,
                "scheduler_address": self._client.scheduler.address,
            }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {}
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task to the cluster.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dask future
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return self._client.submit(func, *args, **kwargs)
    
    def map_tasks(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """Map a function over an iterable.
        
        Args:
            func: Function to map
            iterable: Iterable of arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            List of Dask futures
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return self._client.map(func, iterable, **kwargs)
    
    def compute_delayed(self, *args, **kwargs) -> tuple:
        """Compute delayed objects.
        
        Args:
            *args: Delayed objects
            **kwargs: Compute options
            
        Returns:
            Computed results
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return dask.compute(*args, scheduler=self._client, **kwargs)
    
    def create_dask_dataframe(self, df: pd.DataFrame, 
                            npartitions: Optional[int] = None) -> dd.DataFrame:
        """Create a Dask DataFrame from pandas DataFrame.
        
        Args:
            df: Pandas DataFrame
            npartitions: Number of partitions
            
        Returns:
            Dask DataFrame
        """
        if npartitions is None:
            # Auto-calculate partitions based on size
            size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            npartitions = max(1, int(size_mb / 128))  # ~128MB per partition
            
        return dd.from_pandas(df, npartitions=npartitions)
    
    def create_dask_array(self, arr: np.ndarray, 
                         chunks: Optional[Union[int, tuple]] = None) -> da.Array:
        """Create a Dask Array from numpy array.
        
        Args:
            arr: Numpy array
            chunks: Chunk size specification
            
        Returns:
            Dask Array
        """
        if chunks is None:
            # Auto-calculate chunks based on size
            size_mb = arr.nbytes / 1024 / 1024
            if size_mb > 128:
                # Aim for ~128MB chunks
                chunk_size = int(np.sqrt(128 * 1024 * 1024 / arr.itemsize))
                if arr.ndim == 1:
                    chunks = chunk_size
                elif arr.ndim == 2:
                    chunks = (chunk_size, chunk_size)
                else:
                    chunks = tuple(chunk_size for _ in range(arr.ndim))
            else:
                chunks = arr.shape
                
        return da.from_array(arr, chunks=chunks)
    
    def persist(self, *collections) -> tuple:
        """Persist collections in distributed memory.
        
        Args:
            *collections: Dask collections to persist
            
        Returns:
            Persisted collections
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return self._client.persist(*collections)
    
    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to workers.
        
        Args:
            data: Data to scatter
            broadcast: Whether to broadcast to all workers
            
        Returns:
            Scattered data future
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return self._client.scatter(data, broadcast=broadcast)
    
    def gather(self, futures: List[Any]) -> List[Any]:
        """Gather results from futures.
        
        Args:
            futures: List of futures
            
        Returns:
            List of results
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return self._client.gather(futures)
    
    def run_on_workers(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Run a function on all workers.
        
        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary of worker results
        """
        if not self._client:
            raise RuntimeError("Dask cluster not initialized")
            
        return self._client.run(func, *args, **kwargs)
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to be executed on cluster shutdown.
        
        Args:
            callback: Callback function
        """
        self._shutdown_callbacks.append(callback)
    
    @property
    def is_ready(self) -> bool:
        """Check if cluster is ready."""
        return self.status == DaskClusterStatus.READY and self._client is not None
    
    @property
    def client(self) -> Optional[Client]:
        """Get Dask client."""
        return self._client
    
    @property
    def dashboard_link(self) -> Optional[str]:
        """Get dashboard link."""
        return self._client.dashboard_link if self._client else None


# Utility functions for distributed operations
def distributed_portfolio_optimization(returns_df: pd.DataFrame,
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Distributed portfolio optimization using Dask.
    
    Args:
        returns_df: Returns DataFrame
        constraints: Optimization constraints
        
    Returns:
        Optimization results
    """
    # Convert to Dask DataFrame for distributed processing
    ddf = dd.from_pandas(returns_df, npartitions=4)
    
    # Calculate statistics in parallel
    mean_returns = ddf.mean().compute()
    cov_matrix = ddf.cov().compute()
    
    # Simple mean-variance optimization
    n_assets = len(mean_returns)
    weights = np.ones(n_assets) / n_assets  # Equal weight for now
    
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    return {
        "weights": weights.tolist(),
        "expected_return": portfolio_return,
        "volatility": portfolio_std,
        "sharpe_ratio": portfolio_return / portfolio_std if portfolio_std > 0 else 0,
    }


@delayed
def backtest_chunk(data_chunk: pd.DataFrame, 
                  strategy_params: Dict[str, Any]) -> Dict[str, Any]:
    """Backtest a chunk of data.
    
    Args:
        data_chunk: Data chunk to backtest
        strategy_params: Strategy parameters
        
    Returns:
        Backtesting results
    """
    # Placeholder for actual backtesting logic
    returns = data_chunk['close'].pct_change().dropna()
    
    return {
        "total_return": (1 + returns).prod() - 1,
        "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
        "max_drawdown": (returns.cumsum() - returns.cumsum().cummax()).min(),
        "n_trades": len(returns),
        "start_date": data_chunk.index[0],
        "end_date": data_chunk.index[-1],
    }


def distributed_monte_carlo(n_simulations: int, 
                          n_steps: int,
                          initial_price: float,
                          drift: float,
                          volatility: float,
                          client: Client) -> da.Array:
    """Run distributed Monte Carlo simulation.
    
    Args:
        n_simulations: Number of simulations
        n_steps: Number of time steps
        initial_price: Initial asset price
        drift: Drift parameter
        volatility: Volatility parameter
        client: Dask client
        
    Returns:
        Dask array of simulation results
    """
    # Create chunks for parallel processing
    chunk_size = max(1, n_simulations // len(client.nthreads()))
    
    # Define simulation function
    def simulate_chunk(n_sims, n_steps, S0, mu, sigma):
        dt = 1/252  # Daily steps
        prices = np.zeros((n_sims, n_steps))
        prices[:, 0] = S0
        
        for t in range(1, n_steps):
            Z = np.random.standard_normal(n_sims)
            prices[:, t] = prices[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
        return prices
    
    # Create delayed tasks
    tasks = []
    for i in range(0, n_simulations, chunk_size):
        n_sims_chunk = min(chunk_size, n_simulations - i)
        task = delayed(simulate_chunk)(n_sims_chunk, n_steps, initial_price, drift, volatility)
        tasks.append(task)
    
    # Combine results
    arrays = [da.from_delayed(task, shape=(chunk_size, n_steps), dtype=float) for task in tasks]
    return da.vstack(arrays)