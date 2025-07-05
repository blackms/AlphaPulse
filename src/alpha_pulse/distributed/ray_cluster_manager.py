"""Ray cluster manager for distributed backtesting."""

import logging
import os
from typing import Dict, Any, Optional, List, Callable
import ray
from ray import tune
from ray.tune import schedulers
from ray.util.state import list_nodes
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClusterStatus(Enum):
    """Cluster status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SCALING = "scaling"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class RayClusterConfig:
    """Configuration for Ray cluster."""
    head_node_cpu: int = 4
    head_node_memory: int = 8  # GB
    worker_node_cpu: int = 4
    worker_node_memory: int = 8  # GB
    min_workers: int = 0
    max_workers: int = 10
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265
    namespace: str = "alphapulse"
    object_store_memory: Optional[int] = None
    runtime_env: Optional[Dict[str, Any]] = None
    enable_autoscaling: bool = True
    autoscale_idle_timeout: int = 300  # seconds


class RayClusterManager:
    """Manages Ray cluster for distributed computing."""
    
    def __init__(self, config: Optional[RayClusterConfig] = None):
        """Initialize Ray cluster manager.
        
        Args:
            config: Ray cluster configuration
        """
        self.config = config or RayClusterConfig()
        self.status = ClusterStatus.UNINITIALIZED
        self._cluster_info: Optional[Dict[str, Any]] = None
        self._shutdown_callbacks: List[Callable] = []
        
    def initialize_cluster(self, address: Optional[str] = None) -> bool:
        """Initialize Ray cluster.
        
        Args:
            address: Ray cluster address (None for local cluster)
            
        Returns:
            Success status
        """
        try:
            self.status = ClusterStatus.INITIALIZING
            logger.info("Initializing Ray cluster...")
            
            if ray.is_initialized():
                logger.warning("Ray is already initialized")
                self.status = ClusterStatus.READY
                return True
            
            init_kwargs = {
                "dashboard_host": self.config.dashboard_host,
                "dashboard_port": self.config.dashboard_port,
                "namespace": self.config.namespace,
                "logging_level": logging.INFO,
                "include_dashboard": True,
            }
            
            if address:
                init_kwargs["address"] = address
            else:
                # Local cluster configuration
                init_kwargs["num_cpus"] = self.config.head_node_cpu
                init_kwargs["num_gpus"] = 0
                
                if self.config.object_store_memory:
                    init_kwargs["object_store_memory"] = self.config.object_store_memory
                    
            if self.config.runtime_env:
                init_kwargs["runtime_env"] = self.config.runtime_env
                
            ray.init(**init_kwargs)
            
            # Wait for cluster to be ready
            time.sleep(2)
            
            self._cluster_info = self._get_cluster_info()
            self.status = ClusterStatus.READY
            
            logger.info(f"Ray cluster initialized successfully. Dashboard: http://{self.config.dashboard_host}:{self.config.dashboard_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}")
            self.status = ClusterStatus.ERROR
            return False
    
    def shutdown_cluster(self) -> bool:
        """Shutdown Ray cluster.
        
        Returns:
            Success status
        """
        try:
            self.status = ClusterStatus.SHUTTING_DOWN
            logger.info("Shutting down Ray cluster...")
            
            # Execute shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Shutdown callback failed: {e}")
            
            if ray.is_initialized():
                ray.shutdown()
                
            self.status = ClusterStatus.UNINITIALIZED
            self._cluster_info = None
            logger.info("Ray cluster shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown Ray cluster: {e}")
            self.status = ClusterStatus.ERROR
            return False
    
    def scale_cluster(self, num_workers: int) -> bool:
        """Scale cluster to specified number of workers.
        
        Args:
            num_workers: Desired number of workers
            
        Returns:
            Success status
        """
        try:
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster not initialized")
                
            self.status = ClusterStatus.SCALING
            logger.info(f"Scaling cluster to {num_workers} workers...")
            
            # For autoscaling clusters, this would trigger scaling
            # For now, we'll just update the status
            if self.config.enable_autoscaling:
                # Ray autoscaler will handle this based on resource demands
                logger.info("Autoscaling enabled - cluster will scale based on demand")
            else:
                logger.warning("Manual scaling not implemented for local clusters")
                
            self.status = ClusterStatus.READY
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale cluster: {e}")
            self.status = ClusterStatus.ERROR
            return False
    
    def get_cluster_resources(self) -> Dict[str, Any]:
        """Get current cluster resources.
        
        Returns:
            Dictionary of available resources
        """
        if not ray.is_initialized():
            return {}
            
        try:
            resources = ray.cluster_resources()
            available = ray.available_resources()
            
            return {
                "total": resources,
                "available": available,
                "used": {
                    key: resources.get(key, 0) - available.get(key, 0)
                    for key in resources
                }
            }
        except Exception as e:
            logger.error(f"Failed to get cluster resources: {e}")
            return {}
    
    def get_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Get information about cluster nodes.
        
        Returns:
            List of node information
        """
        if not ray.is_initialized():
            return []
            
        try:
            nodes = list_nodes()
            return [
                {
                    "node_id": node.node_id,
                    "node_ip": node.node_ip,
                    "state": node.state,
                    "resources": node.resources_total,
                    "labels": node.labels,
                }
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Failed to get cluster nodes: {e}")
            return []
    
    def submit_task(self, func: Callable, *args, **kwargs) -> ray.ObjectRef:
        """Submit a task to the cluster.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Ray object reference
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray cluster not initialized")
            
        remote_func = ray.remote(func)
        return remote_func.remote(*args, **kwargs)
    
    def submit_actor(self, actor_class: type, *args, **kwargs) -> ray.actor.ActorHandle:
        """Submit an actor to the cluster.
        
        Args:
            actor_class: Actor class
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Ray actor handle
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray cluster not initialized")
            
        remote_actor = ray.remote(actor_class)
        return remote_actor.remote(*args, **kwargs)
    
    def wait_for_tasks(self, tasks: List[ray.ObjectRef], 
                      timeout: Optional[float] = None) -> List[Any]:
        """Wait for tasks to complete.
        
        Args:
            tasks: List of Ray object references
            timeout: Timeout in seconds
            
        Returns:
            List of task results
        """
        if not tasks:
            return []
            
        ready, not_ready = ray.wait(tasks, num_returns=len(tasks), timeout=timeout)
        
        if not_ready:
            logger.warning(f"{len(not_ready)} tasks did not complete within timeout")
            
        return ray.get(ready)
    
    def run_hyperparameter_optimization(self,
                                      trainable: Callable,
                                      config: Dict[str, Any],
                                      num_samples: int = 10,
                                      metric: str = "loss",
                                      mode: str = "min",
                                      scheduler_type: str = "asha") -> tune.ExperimentAnalysis:
        """Run distributed hyperparameter optimization using Ray Tune.
        
        Args:
            trainable: Training function
            config: Hyperparameter search space
            num_samples: Number of trials
            metric: Metric to optimize
            mode: Optimization mode ('min' or 'max')
            scheduler_type: Scheduler type ('asha', 'pbt', 'fifo')
            
        Returns:
            Ray Tune analysis object
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray cluster not initialized")
            
        # Select scheduler
        if scheduler_type == "asha":
            scheduler = schedulers.ASHAScheduler(
                metric=metric,
                mode=mode,
                max_t=100,
                grace_period=10,
                reduction_factor=3
            )
        elif scheduler_type == "pbt":
            scheduler = schedulers.PopulationBasedTraining(
                metric=metric,
                mode=mode,
                perturbation_interval=4,
                hyperparam_mutations=config
            )
        else:
            scheduler = schedulers.FIFOScheduler()
            
        # Run optimization
        analysis = tune.run(
            trainable,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=tune.CLIReporter(
                metric_columns=[metric, "training_iteration"]
            ),
            local_dir="./ray_results",
            name=f"alphapulse_opt_{int(time.time())}"
        )
        
        return analysis
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to be executed on cluster shutdown.
        
        Args:
            callback: Callback function
        """
        self._shutdown_callbacks.append(callback)
    
    def _get_cluster_info(self) -> Dict[str, Any]:
        """Get detailed cluster information.
        
        Returns:
            Cluster information dictionary
        """
        if not ray.is_initialized():
            return {}
            
        try:
            nodes = self.get_cluster_nodes()
            resources = self.get_cluster_resources()
            
            return {
                "status": self.status.value,
                "num_nodes": len(nodes),
                "nodes": nodes,
                "resources": resources,
                "dashboard_url": f"http://{self.config.dashboard_host}:{self.config.dashboard_port}",
                "namespace": self.config.namespace,
            }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {}
    
    @property
    def is_ready(self) -> bool:
        """Check if cluster is ready."""
        return self.status == ClusterStatus.READY and ray.is_initialized()
    
    @property
    def cluster_info(self) -> Dict[str, Any]:
        """Get current cluster information."""
        if self.is_ready:
            self._cluster_info = self._get_cluster_info()
        return self._cluster_info or {}


# Utility functions for distributed operations
@ray.remote
def distributed_backtest_task(strategy_config: Dict[str, Any],
                            data: np.ndarray,
                            start_idx: int,
                            end_idx: int) -> Dict[str, Any]:
    """Execute backtesting task on a subset of data.
    
    Args:
        strategy_config: Strategy configuration
        data: Market data array
        start_idx: Start index for data subset
        end_idx: End index for data subset
        
    Returns:
        Backtesting results
    """
    # This is a placeholder for the actual backtesting logic
    # In practice, this would call the real backtesting engine
    subset_data = data[start_idx:end_idx]
    
    # Simulate some computation
    returns = np.random.randn(len(subset_data)) * 0.01
    
    return {
        "returns": returns.tolist(),
        "sharpe_ratio": returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
        "max_drawdown": np.min(np.minimum.accumulate(returns)),
        "total_return": np.prod(1 + returns) - 1,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


@ray.remote
class DistributedPortfolioOptimizer:
    """Actor for stateful portfolio optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize optimizer with configuration."""
        self.config = config
        self.optimization_history = []
        
    def optimize(self, returns_matrix: np.ndarray,
                constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize portfolio weights.
        
        Args:
            returns_matrix: Historical returns matrix
            constraints: Optimization constraints
            
        Returns:
            Optimization results
        """
        # Placeholder for actual optimization logic
        num_assets = returns_matrix.shape[1]
        
        # Simple equal weight for now
        weights = np.ones(num_assets) / num_assets
        
        portfolio_return = np.mean(returns_matrix @ weights)
        portfolio_risk = np.std(returns_matrix @ weights)
        
        result = {
            "weights": weights.tolist(),
            "expected_return": portfolio_return,
            "risk": portfolio_risk,
            "sharpe_ratio": portfolio_return / (portfolio_risk + 1e-8),
        }
        
        self.optimization_history.append(result)
        return result
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history