"""Distributed backtesting engine using Ray and Dask."""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import ray
from dask import delayed
from dask.distributed import Client, as_completed
import time
from concurrent.futures import Future

from ..distributed.ray_cluster_manager import RayClusterManager, distributed_backtest_task
from ..distributed.dask_cluster_manager import DaskClusterManager, backtest_chunk

logger = logging.getLogger(__name__)


class DistributionFramework(Enum):
    """Distributed computing framework selection."""
    RAY = "ray"
    DASK = "dask"
    AUTO = "auto"


class ParallelizationStrategy(Enum):
    """Strategy for parallelizing backtests."""
    TIME_BASED = "time_based"  # Split by time periods
    SYMBOL_BASED = "symbol_based"  # Split by symbols/assets
    PARAMETER_BASED = "parameter_based"  # Split by parameter combinations
    MONTE_CARLO = "monte_carlo"  # Split Monte Carlo simulations


@dataclass
class DistributedBacktestConfig:
    """Configuration for distributed backtesting."""
    framework: DistributionFramework = DistributionFramework.AUTO
    parallelization_strategy: ParallelizationStrategy = ParallelizationStrategy.TIME_BASED
    chunk_size: Optional[int] = None  # Auto-calculate if None
    n_workers: Optional[int] = None  # Use all available if None
    memory_per_worker: str = "4GB"
    enable_progress_bar: bool = True
    fault_tolerance: bool = True
    max_retries: int = 3
    checkpoint_interval: Optional[int] = None
    result_cache: bool = True
    compression: bool = True


class DistributedBacktester:
    """Distributed backtesting engine."""
    
    def __init__(self, config: Optional[DistributedBacktestConfig] = None):
        """Initialize distributed backtester.
        
        Args:
            config: Distributed backtesting configuration
        """
        self.config = config or DistributedBacktestConfig()
        self.ray_manager: Optional[RayClusterManager] = None
        self.dask_manager: Optional[DaskClusterManager] = None
        self._results_cache: Dict[str, Any] = {}
        self._active_framework: Optional[DistributionFramework] = None
        
    def initialize(self, cluster_address: Optional[str] = None) -> bool:
        """Initialize distributed computing framework.
        
        Args:
            cluster_address: Cluster address (None for local)
            
        Returns:
            Success status
        """
        try:
            if self.config.framework == DistributionFramework.RAY:
                return self._initialize_ray(cluster_address)
            elif self.config.framework == DistributionFramework.DASK:
                return self._initialize_dask(cluster_address)
            else:  # AUTO
                # Try Ray first, then Dask
                if self._initialize_ray(cluster_address):
                    return True
                return self._initialize_dask(cluster_address)
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed framework: {e}")
            return False
    
    def run_backtest(self,
                    strategy: Any,
                    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    initial_capital: float = 100000,
                    **kwargs) -> Dict[str, Any]:
        """Run distributed backtest.
        
        Args:
            strategy: Trading strategy object
            data: Market data (DataFrame or dict of DataFrames)
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            **kwargs: Additional parameters
            
        Returns:
            Backtesting results
        """
        if not self._active_framework:
            raise RuntimeError("Distributed framework not initialized")
            
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data = {"default": data}
            
        # Filter by date range
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
            
        # Run distributed backtest based on strategy
        if self.config.parallelization_strategy == ParallelizationStrategy.TIME_BASED:
            return self._run_time_based_backtest(strategy, data, initial_capital, **kwargs)
        elif self.config.parallelization_strategy == ParallelizationStrategy.SYMBOL_BASED:
            return self._run_symbol_based_backtest(strategy, data, initial_capital, **kwargs)
        elif self.config.parallelization_strategy == ParallelizationStrategy.PARAMETER_BASED:
            return self._run_parameter_based_backtest(strategy, data, initial_capital, **kwargs)
        else:  # MONTE_CARLO
            return self._run_monte_carlo_backtest(strategy, data, initial_capital, **kwargs)
    
    def run_parameter_optimization(self,
                                 strategy_class: type,
                                 data: pd.DataFrame,
                                 param_grid: Dict[str, List[Any]],
                                 objective: str = "sharpe_ratio",
                                 n_trials: Optional[int] = None) -> Dict[str, Any]:
        """Run distributed parameter optimization.
        
        Args:
            strategy_class: Strategy class
            data: Market data
            param_grid: Parameter search space
            objective: Optimization objective
            n_trials: Number of trials (None for grid search)
            
        Returns:
            Optimization results
        """
        if self._active_framework == DistributionFramework.RAY and self.ray_manager:
            return self._run_ray_optimization(strategy_class, data, param_grid, objective, n_trials)
        elif self._active_framework == DistributionFramework.DASK and self.dask_manager:
            return self._run_dask_optimization(strategy_class, data, param_grid, objective, n_trials)
        else:
            raise RuntimeError("No active distributed framework")
    
    def run_walk_forward_analysis(self,
                                strategy: Any,
                                data: pd.DataFrame,
                                window_size: int,
                                step_size: int,
                                n_windows: Optional[int] = None) -> Dict[str, Any]:
        """Run distributed walk-forward analysis.
        
        Args:
            strategy: Trading strategy
            data: Market data
            window_size: Training window size (days)
            step_size: Step size between windows (days)
            n_windows: Number of windows (None for all)
            
        Returns:
            Walk-forward analysis results
        """
        # Calculate windows
        windows = self._calculate_walk_forward_windows(data, window_size, step_size, n_windows)
        
        # Run distributed analysis
        if self._active_framework == DistributionFramework.RAY:
            return self._run_ray_walk_forward(strategy, data, windows)
        else:
            return self._run_dask_walk_forward(strategy, data, windows)
    
    def shutdown(self) -> bool:
        """Shutdown distributed framework.
        
        Returns:
            Success status
        """
        try:
            if self.ray_manager:
                self.ray_manager.shutdown_cluster()
                self.ray_manager = None
                
            if self.dask_manager:
                self.dask_manager.shutdown_cluster()
                self.dask_manager = None
                
            self._active_framework = None
            self._results_cache.clear()
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown distributed framework: {e}")
            return False
    
    def _initialize_ray(self, cluster_address: Optional[str]) -> bool:
        """Initialize Ray framework."""
        try:
            from ..distributed.ray_cluster_manager import RayClusterManager, RayClusterConfig
            
            config = RayClusterConfig(
                min_workers=0,
                max_workers=self.config.n_workers or 10,
                enable_autoscaling=True
            )
            
            self.ray_manager = RayClusterManager(config)
            if self.ray_manager.initialize_cluster(cluster_address):
                self._active_framework = DistributionFramework.RAY
                logger.info("Ray framework initialized successfully")
                return True
            return False
            
        except ImportError:
            logger.warning("Ray not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return False
    
    def _initialize_dask(self, cluster_address: Optional[str]) -> bool:
        """Initialize Dask framework."""
        try:
            from ..distributed.dask_cluster_manager import DaskClusterManager, DaskClusterConfig
            
            config = DaskClusterConfig(
                n_workers=self.config.n_workers or 4,
                memory_limit=self.config.memory_per_worker
            )
            
            self.dask_manager = DaskClusterManager(config)
            if self.dask_manager.initialize_cluster(cluster_address):
                self._active_framework = DistributionFramework.DASK
                logger.info("Dask framework initialized successfully")
                return True
            return False
            
        except ImportError:
            logger.warning("Dask not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Dask: {e}")
            return False
    
    def _filter_data_by_date(self, 
                           data: Dict[str, pd.DataFrame],
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Filter data by date range."""
        filtered_data = {}
        
        for symbol, df in data.items():
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            filtered_data[symbol] = df
            
        return filtered_data
    
    def _calculate_chunks(self, data_length: int) -> List[Tuple[int, int]]:
        """Calculate optimal chunk boundaries."""
        if self.config.chunk_size:
            chunk_size = self.config.chunk_size
        else:
            # Auto-calculate based on data size and workers
            n_workers = self.config.n_workers or 4
            chunk_size = max(100, data_length // (n_workers * 4))
            
        chunks = []
        for i in range(0, data_length, chunk_size):
            chunks.append((i, min(i + chunk_size, data_length)))
            
        return chunks
    
    def _run_time_based_backtest(self,
                               strategy: Any,
                               data: Dict[str, pd.DataFrame],
                               initial_capital: float,
                               **kwargs) -> Dict[str, Any]:
        """Run time-based parallel backtest."""
        # Combine all data
        combined_data = pd.concat(data.values(), axis=1, keys=data.keys())
        
        # Calculate time chunks
        chunks = self._calculate_chunks(len(combined_data))
        
        if self._active_framework == DistributionFramework.RAY:
            return self._run_ray_time_based(strategy, combined_data, chunks, initial_capital, **kwargs)
        else:
            return self._run_dask_time_based(strategy, combined_data, chunks, initial_capital, **kwargs)
    
    def _run_ray_time_based(self,
                          strategy: Any,
                          data: pd.DataFrame,
                          chunks: List[Tuple[int, int]],
                          initial_capital: float,
                          **kwargs) -> Dict[str, Any]:
        """Run time-based backtest using Ray."""
        # Scatter data to all workers
        data_ref = ray.put(data.values)
        strategy_config = strategy.get_config()
        
        # Submit tasks
        tasks = []
        for start_idx, end_idx in chunks:
            task = distributed_backtest_task.remote(
                strategy_config, data_ref, start_idx, end_idx
            )
            tasks.append(task)
            
        # Wait for results
        results = ray.get(tasks)
        
        # Aggregate results
        return self._aggregate_time_based_results(results, initial_capital)
    
    def _run_dask_time_based(self,
                           strategy: Any,
                           data: pd.DataFrame,
                           chunks: List[Tuple[int, int]],
                           initial_capital: float,
                           **kwargs) -> Dict[str, Any]:
        """Run time-based backtest using Dask."""
        strategy_params = strategy.get_config()
        
        # Create delayed tasks
        tasks = []
        for start_idx, end_idx in chunks:
            chunk_data = data.iloc[start_idx:end_idx]
            task = backtest_chunk(chunk_data, strategy_params)
            tasks.append(task)
            
        # Compute results
        results = self.dask_manager.compute_delayed(*tasks)
        
        # Aggregate results
        return self._aggregate_time_based_results(results[0], initial_capital)
    
    def _run_symbol_based_backtest(self,
                                 strategy: Any,
                                 data: Dict[str, pd.DataFrame],
                                 initial_capital: float,
                                 **kwargs) -> Dict[str, Any]:
        """Run symbol-based parallel backtest."""
        if self._active_framework == DistributionFramework.RAY:
            return self._run_ray_symbol_based(strategy, data, initial_capital, **kwargs)
        else:
            return self._run_dask_symbol_based(strategy, data, initial_capital, **kwargs)
    
    def _run_ray_symbol_based(self,
                            strategy: Any,
                            data: Dict[str, pd.DataFrame],
                            initial_capital: float,
                            **kwargs) -> Dict[str, Any]:
        """Run symbol-based backtest using Ray."""
        strategy_config = strategy.get_config()
        
        # Submit tasks per symbol
        tasks = {}
        for symbol, symbol_data in data.items():
            data_ref = ray.put(symbol_data.values)
            task = distributed_backtest_task.remote(
                strategy_config, data_ref, 0, len(symbol_data)
            )
            tasks[symbol] = task
            
        # Wait for results
        results = {symbol: ray.get(task) for symbol, task in tasks.items()}
        
        # Aggregate results
        return self._aggregate_symbol_based_results(results, initial_capital)
    
    def _run_dask_symbol_based(self,
                             strategy: Any,
                             data: Dict[str, pd.DataFrame],
                             initial_capital: float,
                             **kwargs) -> Dict[str, Any]:
        """Run symbol-based backtest using Dask."""
        strategy_params = strategy.get_config()
        
        # Create delayed tasks per symbol
        tasks = {}
        for symbol, symbol_data in data.items():
            task = backtest_chunk(symbol_data, strategy_params)
            tasks[symbol] = task
            
        # Compute results
        results_list = self.dask_manager.compute_delayed(*tasks.values())
        results = dict(zip(tasks.keys(), results_list[0]))
        
        # Aggregate results
        return self._aggregate_symbol_based_results(results, initial_capital)
    
    def _aggregate_time_based_results(self, 
                                    chunk_results: List[Dict[str, Any]],
                                    initial_capital: float) -> Dict[str, Any]:
        """Aggregate results from time-based chunks."""
        # Combine returns
        all_returns = []
        for result in chunk_results:
            all_returns.extend(result["returns"])
            
        all_returns = np.array(all_returns)
        
        # Calculate portfolio metrics
        total_return = np.prod(1 + all_returns) - 1
        sharpe_ratio = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252)
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + all_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_capital": initial_capital * (1 + total_return),
            "n_chunks": len(chunk_results),
            "chunk_results": chunk_results,
        }
    
    def _aggregate_symbol_based_results(self,
                                      symbol_results: Dict[str, Dict[str, Any]],
                                      initial_capital: float) -> Dict[str, Any]:
        """Aggregate results from symbol-based backtests."""
        # Calculate portfolio-level metrics
        portfolio_returns = []
        symbol_weights = 1.0 / len(symbol_results)  # Equal weight
        
        for symbol, result in symbol_results.items():
            portfolio_returns.append(result["total_return"] * symbol_weights)
            
        total_return = sum(portfolio_returns)
        
        # Aggregate other metrics
        sharpe_ratios = [r["sharpe_ratio"] for r in symbol_results.values()]
        max_drawdowns = [r["max_drawdown"] for r in symbol_results.values()]
        
        return {
            "total_return": total_return,
            "average_sharpe_ratio": np.mean(sharpe_ratios),
            "worst_drawdown": min(max_drawdowns),
            "final_capital": initial_capital * (1 + total_return),
            "symbol_results": symbol_results,
        }
    
    def _calculate_walk_forward_windows(self,
                                      data: pd.DataFrame,
                                      window_size: int,
                                      step_size: int,
                                      n_windows: Optional[int]) -> List[Tuple[int, int, int]]:
        """Calculate walk-forward analysis windows."""
        windows = []
        data_length = len(data)
        
        i = 0
        while i + window_size < data_length:
            train_start = i
            train_end = i + window_size
            test_start = train_end
            test_end = min(test_start + step_size, data_length)
            
            windows.append((train_start, train_end, test_end))
            
            i += step_size
            
            if n_windows and len(windows) >= n_windows:
                break
                
        return windows
    
    def _run_ray_optimization(self,
                            strategy_class: type,
                            data: pd.DataFrame,
                            param_grid: Dict[str, List[Any]],
                            objective: str,
                            n_trials: Optional[int]) -> Dict[str, Any]:
        """Run hyperparameter optimization using Ray Tune."""
        from ray import tune
        
        # Define training function
        def train_fn(config):
            strategy = strategy_class(**config)
            results = self._run_single_backtest(strategy, data)
            return {objective: results[objective]}
        
        # Convert param_grid to Ray Tune search space
        search_space = {}
        for param, values in param_grid.items():
            if len(values) == 1:
                search_space[param] = values[0]
            else:
                search_space[param] = tune.grid_search(values)
                
        # Run optimization
        analysis = self.ray_manager.run_hyperparameter_optimization(
            train_fn,
            search_space,
            num_samples=n_trials or 1,
            metric=objective,
            mode="max" if objective in ["sharpe_ratio", "total_return"] else "min"
        )
        
        # Get best result
        best_config = analysis.get_best_config(metric=objective, mode="max")
        best_result = analysis.get_best_trial().last_result
        
        return {
            "best_params": best_config,
            "best_score": best_result[objective],
            "all_results": analysis.results_df.to_dict(),
        }
    
    def _run_dask_optimization(self,
                             strategy_class: type,
                             data: pd.DataFrame,
                             param_grid: Dict[str, List[Any]],
                             objective: str,
                             n_trials: Optional[int]) -> Dict[str, Any]:
        """Run hyperparameter optimization using Dask."""
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        # Sample if n_trials specified
        if n_trials and n_trials < len(param_combinations):
            import random
            param_combinations = random.sample(param_combinations, n_trials)
            
        # Create tasks
        tasks = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            @delayed
            def evaluate_params(p):
                strategy = strategy_class(**p)
                results = self._run_single_backtest(strategy, data)
                return p, results[objective]
                
            tasks.append(evaluate_params(param_dict))
            
        # Compute results
        results = self.dask_manager.compute_delayed(*tasks)[0]
        
        # Find best
        best_params, best_score = max(results, key=lambda x: x[1])
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
        }
    
    def _run_single_backtest(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Run a single backtest (placeholder)."""
        # This would call the actual backtesting engine
        # For now, return mock results
        returns = np.random.randn(len(data)) * 0.01
        
        return {
            "total_return": np.prod(1 + returns) - 1,
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": -0.1,  # Mock value
        }