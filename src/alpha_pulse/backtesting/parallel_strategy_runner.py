"""Parallel strategy execution framework for distributed backtesting."""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import ray
from dask import delayed
from dask.distributed import Client, as_completed, Future
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
import json

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Strategy execution mode."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREAD = "parallel_thread"
    PARALLEL_PROCESS = "parallel_process"
    DISTRIBUTED_RAY = "distributed_ray"
    DISTRIBUTED_DASK = "distributed_dask"


@dataclass
class StrategyExecutionConfig:
    """Configuration for strategy execution."""
    execution_mode: ExecutionMode = ExecutionMode.DISTRIBUTED_RAY
    max_workers: Optional[int] = None
    timeout: Optional[int] = None  # seconds
    retry_failed: bool = True
    max_retries: int = 3
    cache_results: bool = True
    checkpoint_interval: int = 100  # iterations
    memory_limit: Optional[str] = None  # e.g., "2GB"
    batch_size: Optional[int] = None
    progress_callback: Optional[Callable] = None


@dataclass
class StrategyTask:
    """Represents a strategy execution task."""
    task_id: str
    strategy_class: type
    strategy_params: Dict[str, Any]
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate task ID if not provided."""
        if not self.task_id:
            # Generate deterministic ID based on content
            content = f"{self.strategy_class.__name__}_{json.dumps(self.strategy_params, sort_keys=True)}"
            self.task_id = hashlib.md5(content.encode()).hexdigest()[:8]


class ParallelStrategyRunner:
    """Executes multiple strategies in parallel."""
    
    def __init__(self, config: Optional[StrategyExecutionConfig] = None):
        """Initialize parallel strategy runner.
        
        Args:
            config: Execution configuration
        """
        self.config = config or StrategyExecutionConfig()
        self._results_cache: Dict[str, Any] = {}
        self._task_queue: List[StrategyTask] = []
        self._completed_tasks: Dict[str, Any] = {}
        self._failed_tasks: Dict[str, Exception] = {}
        self._ray_actors: List[ray.actor.ActorHandle] = []
        self._dask_futures: List[Future] = []
        
    def add_strategy(self,
                    strategy_class: type,
                    strategy_params: Dict[str, Any],
                    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    task_id: Optional[str] = None,
                    priority: int = 0,
                    dependencies: Optional[List[str]] = None) -> str:
        """Add a strategy to the execution queue.
        
        Args:
            strategy_class: Strategy class to execute
            strategy_params: Strategy parameters
            data: Market data
            task_id: Optional task identifier
            priority: Task priority (higher = more important)
            dependencies: List of task IDs that must complete first
            
        Returns:
            Task ID
        """
        task = StrategyTask(
            task_id=task_id or "",
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            data=data,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self._task_queue.append(task)
        logger.info(f"Added strategy task: {task.task_id}")
        
        return task.task_id
    
    def run_all(self) -> Dict[str, Any]:
        """Execute all queued strategies.
        
        Returns:
            Dictionary mapping task IDs to results
        """
        if not self._task_queue:
            logger.warning("No strategies in queue")
            return {}
            
        # Sort by priority
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        # Execute based on mode
        if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
            return self._run_sequential()
        elif self.config.execution_mode == ExecutionMode.PARALLEL_THREAD:
            return self._run_parallel_thread()
        elif self.config.execution_mode == ExecutionMode.PARALLEL_PROCESS:
            return self._run_parallel_process()
        elif self.config.execution_mode == ExecutionMode.DISTRIBUTED_RAY:
            return self._run_distributed_ray()
        else:  # DISTRIBUTED_DASK
            return self._run_distributed_dask()
    
    def run_single(self, task_id: str) -> Any:
        """Execute a single strategy by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Execution result
        """
        task = next((t for t in self._task_queue if t.task_id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        return self._execute_strategy_task(task)
    
    def get_results(self) -> Dict[str, Any]:
        """Get all completed results.
        
        Returns:
            Dictionary of results
        """
        return self._completed_tasks.copy()
    
    def get_failures(self) -> Dict[str, Exception]:
        """Get all failed tasks.
        
        Returns:
            Dictionary of failures
        """
        return self._failed_tasks.copy()
    
    def clear_queue(self) -> None:
        """Clear the task queue."""
        self._task_queue.clear()
        self._completed_tasks.clear()
        self._failed_tasks.clear()
        
    def _run_sequential(self) -> Dict[str, Any]:
        """Run strategies sequentially."""
        results = {}
        
        for task in self._task_queue:
            try:
                # Check dependencies
                if not self._check_dependencies(task):
                    logger.warning(f"Skipping task {task.task_id} due to failed dependencies")
                    continue
                    
                result = self._execute_strategy_task(task)
                results[task.task_id] = result
                self._completed_tasks[task.task_id] = result
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                self._failed_tasks[task.task_id] = e
                if not self.config.retry_failed:
                    results[task.task_id] = {"error": str(e)}
                    
        return results
    
    def _run_parallel_thread(self) -> Dict[str, Any]:
        """Run strategies in parallel using threads."""
        max_workers = self.config.max_workers or 4
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_task = {}
            for task in self._task_queue:
                if self._check_dependencies(task):
                    future = executor.submit(self._execute_strategy_task, task)
                    future_to_task[future] = task
                    
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results[task.task_id] = result
                    self._completed_tasks[task.task_id] = result
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    self._failed_tasks[task.task_id] = e
                    results[task.task_id] = {"error": str(e)}
                    
        return results
    
    def _run_parallel_process(self) -> Dict[str, Any]:
        """Run strategies in parallel using processes."""
        max_workers = self.config.max_workers or 4
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_task = {}
            for task in self._task_queue:
                if self._check_dependencies(task):
                    # Serialize task for process execution
                    future = executor.submit(
                        _execute_strategy_in_process,
                        task.strategy_class,
                        task.strategy_params,
                        task.data
                    )
                    future_to_task[future] = task
                    
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout)
                    results[task.task_id] = result
                    self._completed_tasks[task.task_id] = result
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    self._failed_tasks[task.task_id] = e
                    results[task.task_id] = {"error": str(e)}
                    
        return results
    
    def _run_distributed_ray(self) -> Dict[str, Any]:
        """Run strategies using Ray distributed computing."""
        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")
            
        results = {}
        
        # Create Ray actors for stateful execution
        if not self._ray_actors:
            n_actors = self.config.max_workers or 4
            self._ray_actors = [
                StrategyExecutorActor.remote() for _ in range(n_actors)
            ]
            
        # Distribute tasks among actors
        futures = []
        for i, task in enumerate(self._task_queue):
            if self._check_dependencies(task):
                actor = self._ray_actors[i % len(self._ray_actors)]
                future = actor.execute_strategy.remote(
                    task.strategy_class,
                    task.strategy_params,
                    task.data
                )
                futures.append((future, task))
                
        # Collect results
        for future, task in futures:
            try:
                result = ray.get(future, timeout=self.config.timeout)
                results[task.task_id] = result
                self._completed_tasks[task.task_id] = result
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                self._failed_tasks[task.task_id] = e
                results[task.task_id] = {"error": str(e)}
                
        return results
    
    def _run_distributed_dask(self) -> Dict[str, Any]:
        """Run strategies using Dask distributed computing."""
        from dask.distributed import get_client
        
        try:
            client = get_client()
        except ValueError:
            raise RuntimeError("Dask client not initialized")
            
        results = {}
        
        # Submit tasks
        futures = []
        for task in self._task_queue:
            if self._check_dependencies(task):
                future = client.submit(
                    _execute_strategy_in_process,
                    task.strategy_class,
                    task.strategy_params,
                    task.data,
                    key=task.task_id
                )
                futures.append((future, task))
                self._dask_futures.append(future)
                
        # Collect results
        for future, task in futures:
            try:
                result = future.result(timeout=self.config.timeout)
                results[task.task_id] = result
                self._completed_tasks[task.task_id] = result
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                self._failed_tasks[task.task_id] = e
                results[task.task_id] = {"error": str(e)}
                
        return results
    
    def _execute_strategy_task(self, task: StrategyTask) -> Dict[str, Any]:
        """Execute a single strategy task."""
        # Check cache
        if self.config.cache_results and task.task_id in self._results_cache:
            logger.info(f"Using cached result for task {task.task_id}")
            return self._results_cache[task.task_id]
            
        # Execute strategy
        strategy = task.strategy_class(**task.strategy_params)
        
        # Run backtest (placeholder - would call actual backtesting engine)
        result = self._run_strategy_backtest(strategy, task.data)
        
        # Cache result
        if self.config.cache_results:
            self._results_cache[task.task_id] = result
            
        # Progress callback
        if self.config.progress_callback:
            self.config.progress_callback(task.task_id, result)
            
        return result
    
    def _run_strategy_backtest(self, strategy: Any, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Run strategy backtest (placeholder)."""
        # This would integrate with the actual backtesting engine
        # For now, return mock results
        if isinstance(data, dict):
            data_length = sum(len(df) for df in data.values())
        else:
            data_length = len(data)
            
        returns = np.random.randn(data_length) * 0.01
        
        return {
            "total_return": np.prod(1 + returns) - 1,
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": np.min(np.minimum.accumulate(returns)) if len(returns) > 0 else 0,
            "n_trades": np.random.randint(10, 100),
            "win_rate": np.random.uniform(0.4, 0.6),
        }
    
    def _check_dependencies(self, task: StrategyTask) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self._completed_tasks:
                return False
        return True


# Ray actor for stateful strategy execution
@ray.remote
class StrategyExecutorActor:
    """Ray actor for executing strategies."""
    
    def __init__(self):
        """Initialize actor."""
        self.cache = {}
        self.execution_count = 0
        
    def execute_strategy(self, 
                        strategy_class: type,
                        strategy_params: Dict[str, Any],
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Execute a strategy.
        
        Args:
            strategy_class: Strategy class
            strategy_params: Strategy parameters
            data: Market data
            
        Returns:
            Execution results
        """
        # Generate cache key
        cache_key = f"{strategy_class.__name__}_{json.dumps(strategy_params, sort_keys=True)}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Execute strategy
        strategy = strategy_class(**strategy_params)
        
        # Run backtest (placeholder)
        if isinstance(data, dict):
            data_length = sum(len(df) for df in data.values())
        else:
            data_length = len(data)
            
        returns = np.random.randn(data_length) * 0.01
        
        result = {
            "total_return": np.prod(1 + returns) - 1,
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": np.min(np.minimum.accumulate(returns)) if len(returns) > 0 else 0,
            "n_trades": np.random.randint(10, 100),
            "win_rate": np.random.uniform(0.4, 0.6),
            "executor_id": ray.get_runtime_context().actor_id.hex(),
            "execution_count": self.execution_count,
        }
        
        # Update cache and counter
        self.cache[cache_key] = result
        self.execution_count += 1
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            "execution_count": self.execution_count,
            "cache_size": len(self.cache),
        }


def _execute_strategy_in_process(strategy_class: type,
                               strategy_params: Dict[str, Any],
                               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """Execute strategy in a separate process.
    
    Args:
        strategy_class: Strategy class
        strategy_params: Strategy parameters
        data: Market data
        
    Returns:
        Execution results
    """
    # This function must be at module level for pickling
    strategy = strategy_class(**strategy_params)
    
    # Run backtest (placeholder)
    if isinstance(data, dict):
        data_length = sum(len(df) for df in data.values())
    else:
        data_length = len(data)
        
    returns = np.random.randn(data_length) * 0.01
    
    return {
        "total_return": np.prod(1 + returns) - 1,
        "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
        "max_drawdown": np.min(np.minimum.accumulate(returns)) if len(returns) > 0 else 0,
        "n_trades": np.random.randint(10, 100),
        "win_rate": np.random.uniform(0.4, 0.6),
    }


class StrategyBatchProcessor:
    """Process strategies in batches for efficiency."""
    
    def __init__(self, batch_size: int = 10):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of strategies per batch
        """
        self.batch_size = batch_size
        self._batches: List[List[StrategyTask]] = []
        
    def add_strategies(self, tasks: List[StrategyTask]) -> None:
        """Add strategies and organize into batches.
        
        Args:
            tasks: List of strategy tasks
        """
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Create batches
        self._batches = [
            sorted_tasks[i:i + self.batch_size]
            for i in range(0, len(sorted_tasks), self.batch_size)
        ]
        
    def process_batches(self, executor: ParallelStrategyRunner) -> Dict[str, Any]:
        """Process all batches.
        
        Args:
            executor: Strategy runner
            
        Returns:
            Combined results
        """
        all_results = {}
        
        for i, batch in enumerate(self._batches):
            logger.info(f"Processing batch {i+1}/{len(self._batches)}")
            
            # Clear and add batch
            executor.clear_queue()
            for task in batch:
                executor.add_strategy(
                    task.strategy_class,
                    task.strategy_params,
                    task.data,
                    task.task_id,
                    task.priority,
                    task.dependencies
                )
                
            # Run batch
            results = executor.run_all()
            all_results.update(results)
            
        return all_results