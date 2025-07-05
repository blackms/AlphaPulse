"""Unified distributed computing service for AlphaPulse."""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os

from ..distributed.ray_cluster_manager import RayClusterManager
from ..distributed.dask_cluster_manager import DaskClusterManager
from ..backtesting.distributed_backtester import (
    DistributedBacktester, 
    DistributedBacktestConfig,
    DistributionFramework
)
from ..backtesting.parallel_strategy_runner import (
    ParallelStrategyRunner,
    StrategyExecutionConfig,
    ExecutionMode
)
from ..backtesting.result_aggregator import (
    ResultAggregator,
    AggregationConfig,
    MergeStrategy
)
from ..config.cluster_config import (
    ClusterConfig,
    get_local_development_config,
    get_aws_production_config,
    validate_cluster_config
)
from ..utils.distributed_utils import (
    ResourceMonitor,
    DataPartitioner,
    CacheManager,
    get_cluster_info,
    validate_cluster_resources
)

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class DistributedJobResult:
    """Result from distributed job execution."""
    job_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    result_data: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]
    resource_usage: Dict[str, float]


class DistributedComputingService:
    """Main service for distributed computing operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize distributed computing service.
        
        Args:
            config_path: Path to cluster configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.cluster_config = ClusterConfig.load(config_path)
        else:
            self.cluster_config = get_local_development_config()
            
        # Validate configuration
        errors = validate_cluster_config(self.cluster_config)
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")
            
        # Initialize components
        self.ray_manager: Optional[RayClusterManager] = None
        self.dask_manager: Optional[DaskClusterManager] = None
        self.distributed_backtester: Optional[DistributedBacktester] = None
        self.strategy_runner: Optional[ParallelStrategyRunner] = None
        self.result_aggregator = ResultAggregator()
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager()
        
        # Service state
        self.status = ServiceStatus.UNINITIALIZED
        self._active_jobs: Dict[str, DistributedJobResult] = {}
        self._job_counter = 0
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self, framework: str = "auto") -> bool:
        """Initialize the distributed computing service.
        
        Args:
            framework: Framework to use ("ray", "dask", or "auto")
            
        Returns:
            Success status
        """
        try:
            self.status = ServiceStatus.INITIALIZING
            logger.info("Initializing distributed computing service...")
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Initialize distributed framework
            if framework == "ray" or framework == "auto":
                success = await self._initialize_ray()
                if success:
                    self.status = ServiceStatus.READY
                    return True
                    
            if framework == "dask" or framework == "auto":
                success = await self._initialize_dask()
                if success:
                    self.status = ServiceStatus.READY
                    return True
                    
            logger.error("Failed to initialize any distributed framework")
            self.status = ServiceStatus.ERROR
            return False
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            self.status = ServiceStatus.ERROR
            return False
            
    async def shutdown(self) -> bool:
        """Shutdown the distributed computing service.
        
        Returns:
            Success status
        """
        try:
            self.status = ServiceStatus.SHUTTING_DOWN
            logger.info("Shutting down distributed computing service...")
            
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            
            # Wait for active jobs
            await self._wait_for_active_jobs()
            
            # Shutdown frameworks
            if self.ray_manager:
                self.ray_manager.shutdown_cluster()
                self.ray_manager = None
                
            if self.dask_manager:
                self.dask_manager.shutdown_cluster()
                self.dask_manager = None
                
            # Cleanup
            self._executor.shutdown(wait=True)
            self.cache_manager.clear()
            
            self.status = ServiceStatus.UNINITIALIZED
            logger.info("Service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Service shutdown failed: {e}")
            return False
            
    async def run_distributed_backtest(self,
                                     strategy: Any,
                                     data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                     config: Optional[DistributedBacktestConfig] = None,
                                     **kwargs) -> DistributedJobResult:
        """Run a distributed backtest.
        
        Args:
            strategy: Trading strategy
            data: Market data
            config: Backtest configuration
            **kwargs: Additional parameters
            
        Returns:
            Job result
        """
        # Create job
        job_id = self._generate_job_id()
        job_result = DistributedJobResult(
            job_id=job_id,
            status="running",
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            result_data=None,
            error=None,
            metadata={"type": "backtest", "strategy": str(strategy)},
            resource_usage={}
        )
        
        self._active_jobs[job_id] = job_result
        
        # Run in executor
        future = self._executor.submit(
            self._run_backtest_job,
            job_result,
            strategy,
            data,
            config,
            **kwargs
        )
        
        # Return job result (will be updated asynchronously)
        return job_result
        
    async def run_parameter_optimization(self,
                                       strategy_class: type,
                                       data: pd.DataFrame,
                                       param_grid: Dict[str, List[Any]],
                                       objective: str = "sharpe_ratio",
                                       n_trials: Optional[int] = None) -> DistributedJobResult:
        """Run distributed parameter optimization.
        
        Args:
            strategy_class: Strategy class
            data: Market data
            param_grid: Parameter search space
            objective: Optimization objective
            n_trials: Number of trials
            
        Returns:
            Job result
        """
        # Create job
        job_id = self._generate_job_id()
        job_result = DistributedJobResult(
            job_id=job_id,
            status="running",
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            result_data=None,
            error=None,
            metadata={
                "type": "optimization",
                "strategy_class": strategy_class.__name__,
                "objective": objective,
                "n_params": len(param_grid)
            },
            resource_usage={}
        )
        
        self._active_jobs[job_id] = job_result
        
        # Run in executor
        future = self._executor.submit(
            self._run_optimization_job,
            job_result,
            strategy_class,
            data,
            param_grid,
            objective,
            n_trials
        )
        
        return job_result
        
    async def run_monte_carlo_simulation(self,
                                       n_simulations: int,
                                       n_steps: int,
                                       parameters: Dict[str, Any]) -> DistributedJobResult:
        """Run distributed Monte Carlo simulation.
        
        Args:
            n_simulations: Number of simulations
            n_steps: Number of time steps
            parameters: Simulation parameters
            
        Returns:
            Job result
        """
        # Create job
        job_id = self._generate_job_id()
        job_result = DistributedJobResult(
            job_id=job_id,
            status="running",
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            result_data=None,
            error=None,
            metadata={
                "type": "monte_carlo",
                "n_simulations": n_simulations,
                "n_steps": n_steps
            },
            resource_usage={}
        )
        
        self._active_jobs[job_id] = job_result
        
        # Run in executor
        future = self._executor.submit(
            self._run_monte_carlo_job,
            job_result,
            n_simulations,
            n_steps,
            parameters
        )
        
        return job_result
        
    def get_job_status(self, job_id: str) -> Optional[DistributedJobResult]:
        """Get status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job result or None
        """
        return self._active_jobs.get(job_id)
        
    def get_all_jobs(self) -> Dict[str, DistributedJobResult]:
        """Get all jobs.
        
        Returns:
            Dictionary of all jobs
        """
        return self._active_jobs.copy()
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Success status
        """
        if job_id not in self._active_jobs:
            return False
            
        job = self._active_jobs[job_id]
        if job.status == "running":
            job.status = "cancelled"
            job.end_time = datetime.now()
            job.duration_seconds = (job.end_time - job.start_time).total_seconds()
            return True
            
        return False
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status.
        
        Returns:
            Cluster status information
        """
        status = {
            "service_status": self.status.value,
            "active_jobs": len([j for j in self._active_jobs.values() if j.status == "running"]),
            "total_jobs": len(self._active_jobs),
            "cluster_info": get_cluster_info(),
            "resource_usage": self.resource_monitor.get_current_resources(),
        }
        
        if self.ray_manager and self.ray_manager.is_ready:
            status["ray_cluster"] = self.ray_manager.cluster_info
            
        if self.dask_manager and self.dask_manager.is_ready:
            status["dask_cluster"] = self.dask_manager.get_cluster_info()
            
        return status
        
    def scale_cluster(self, n_workers: int) -> bool:
        """Scale the cluster to specified number of workers.
        
        Args:
            n_workers: Target number of workers
            
        Returns:
            Success status
        """
        try:
            if self.ray_manager and self.ray_manager.is_ready:
                return self.ray_manager.scale_cluster(n_workers)
            elif self.dask_manager and self.dask_manager.is_ready:
                return self.dask_manager.scale_cluster(n_workers)
            else:
                logger.error("No active cluster to scale")
                return False
        except Exception as e:
            logger.error(f"Failed to scale cluster: {e}")
            return False
            
    async def _initialize_ray(self) -> bool:
        """Initialize Ray framework."""
        try:
            from ..distributed.ray_cluster_manager import RayClusterConfig
            
            ray_config = RayClusterConfig(
                head_node_cpu=self.cluster_config.head_node.cpu_cores,
                head_node_memory=self.cluster_config.head_node.memory_gb,
                min_workers=self.cluster_config.autoscaling.min_nodes,
                max_workers=self.cluster_config.autoscaling.max_nodes,
                enable_autoscaling=self.cluster_config.autoscaling.enabled,
            )
            
            # Apply custom Ray config
            for key, value in self.cluster_config.ray_config.items():
                setattr(ray_config, key, value)
                
            self.ray_manager = RayClusterManager(ray_config)
            
            # Initialize cluster
            cluster_address = os.environ.get("RAY_ADDRESS")
            if self.ray_manager.initialize_cluster(cluster_address):
                logger.info("Ray cluster initialized successfully")
                
                # Initialize backtester
                backtest_config = DistributedBacktestConfig(
                    framework=DistributionFramework.RAY
                )
                self.distributed_backtester = DistributedBacktester(backtest_config)
                self.distributed_backtester.ray_manager = self.ray_manager
                
                # Initialize strategy runner
                runner_config = StrategyExecutionConfig(
                    execution_mode=ExecutionMode.DISTRIBUTED_RAY
                )
                self.strategy_runner = ParallelStrategyRunner(runner_config)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            
        return False
        
    async def _initialize_dask(self) -> bool:
        """Initialize Dask framework."""
        try:
            from ..distributed.dask_cluster_manager import DaskClusterConfig
            
            dask_config = DaskClusterConfig(
                n_workers=self.cluster_config.autoscaling.min_nodes,
                threads_per_worker=2,
                memory_limit=f"{self.cluster_config.worker_node.memory_gb}GB",
            )
            
            # Apply custom Dask config
            for key, value in self.cluster_config.dask_config.items():
                setattr(dask_config, key, value)
                
            self.dask_manager = DaskClusterManager(dask_config)
            
            # Initialize cluster
            cluster_address = os.environ.get("DASK_SCHEDULER_ADDRESS")
            if self.dask_manager.initialize_cluster(cluster_address):
                logger.info("Dask cluster initialized successfully")
                
                # Initialize backtester
                backtest_config = DistributedBacktestConfig(
                    framework=DistributionFramework.DASK
                )
                self.distributed_backtester = DistributedBacktester(backtest_config)
                self.distributed_backtester.dask_manager = self.dask_manager
                
                # Initialize strategy runner
                runner_config = StrategyExecutionConfig(
                    execution_mode=ExecutionMode.DISTRIBUTED_DASK
                )
                self.strategy_runner = ParallelStrategyRunner(runner_config)
                
                # Enable adaptive scaling
                if self.cluster_config.autoscaling.enabled:
                    self.dask_manager.adapt_cluster(
                        minimum=self.cluster_config.autoscaling.min_nodes,
                        maximum=self.cluster_config.autoscaling.max_nodes
                    )
                    
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Dask: {e}")
            
        return False
        
    async def _wait_for_active_jobs(self, timeout: int = 300) -> None:
        """Wait for active jobs to complete."""
        start_time = datetime.now()
        
        while True:
            active_jobs = [j for j in self._active_jobs.values() if j.status == "running"]
            if not active_jobs:
                break
                
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning(f"Timeout waiting for {len(active_jobs)} active jobs")
                break
                
            await asyncio.sleep(1)
            
    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        self._job_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"job_{timestamp}_{self._job_counter}"
        
    def _run_backtest_job(self,
                         job_result: DistributedJobResult,
                         strategy: Any,
                         data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                         config: Optional[DistributedBacktestConfig],
                         **kwargs) -> None:
        """Run backtest job."""
        try:
            # Check resources
            available, message = validate_cluster_resources(
                required_memory_gb=4,
                required_cores=2
            )
            if not available:
                raise RuntimeError(f"Insufficient resources: {message}")
                
            # Run backtest
            if self.distributed_backtester:
                result = self.distributed_backtester.run_backtest(
                    strategy, data, **kwargs
                )
                job_result.result_data = result
                job_result.status = "completed"
            else:
                raise RuntimeError("Distributed backtester not initialized")
                
        except Exception as e:
            logger.error(f"Backtest job {job_result.job_id} failed: {e}")
            job_result.error = str(e)
            job_result.status = "failed"
            
        finally:
            job_result.end_time = datetime.now()
            job_result.duration_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            job_result.resource_usage = self.resource_monitor.get_current_resources()
            
    def _run_optimization_job(self,
                            job_result: DistributedJobResult,
                            strategy_class: type,
                            data: pd.DataFrame,
                            param_grid: Dict[str, List[Any]],
                            objective: str,
                            n_trials: Optional[int]) -> None:
        """Run optimization job."""
        try:
            if self.distributed_backtester:
                result = self.distributed_backtester.run_parameter_optimization(
                    strategy_class, data, param_grid, objective, n_trials
                )
                job_result.result_data = result
                job_result.status = "completed"
            else:
                raise RuntimeError("Distributed backtester not initialized")
                
        except Exception as e:
            logger.error(f"Optimization job {job_result.job_id} failed: {e}")
            job_result.error = str(e)
            job_result.status = "failed"
            
        finally:
            job_result.end_time = datetime.now()
            job_result.duration_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            job_result.resource_usage = self.resource_monitor.get_current_resources()
            
    def _run_monte_carlo_job(self,
                           job_result: DistributedJobResult,
                           n_simulations: int,
                           n_steps: int,
                           parameters: Dict[str, Any]) -> None:
        """Run Monte Carlo simulation job."""
        try:
            # Extract parameters
            initial_price = parameters.get("initial_price", 100.0)
            drift = parameters.get("drift", 0.05)
            volatility = parameters.get("volatility", 0.2)
            
            if self.dask_manager and self.dask_manager.is_ready:
                # Use Dask for Monte Carlo
                from ..distributed.dask_cluster_manager import distributed_monte_carlo
                
                result_array = distributed_monte_carlo(
                    n_simulations, n_steps, initial_price, drift, volatility,
                    self.dask_manager.client
                )
                
                # Compute statistics
                final_prices = result_array[:, -1].compute()
                
                job_result.result_data = {
                    "mean_final_price": float(np.mean(final_prices)),
                    "std_final_price": float(np.std(final_prices)),
                    "percentiles": {
                        "5": float(np.percentile(final_prices, 5)),
                        "25": float(np.percentile(final_prices, 25)),
                        "50": float(np.percentile(final_prices, 50)),
                        "75": float(np.percentile(final_prices, 75)),
                        "95": float(np.percentile(final_prices, 95)),
                    },
                    "n_simulations": n_simulations,
                    "n_steps": n_steps,
                }
                job_result.status = "completed"
            else:
                raise RuntimeError("No active distributed framework for Monte Carlo")
                
        except Exception as e:
            logger.error(f"Monte Carlo job {job_result.job_id} failed: {e}")
            job_result.error = str(e)
            job_result.status = "failed"
            
        finally:
            job_result.end_time = datetime.now()
            job_result.duration_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            job_result.resource_usage = self.resource_monitor.get_current_resources()


# Singleton instance
_service_instance: Optional[DistributedComputingService] = None


def get_distributed_service(config_path: Optional[str] = None) -> DistributedComputingService:
    """Get or create the distributed computing service instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Service instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = DistributedComputingService(config_path)
        
    return _service_instance