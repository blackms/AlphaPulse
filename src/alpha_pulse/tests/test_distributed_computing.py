"""Comprehensive tests for distributed computing system."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch, MagicMock
import ray
import dask
from dask.distributed import Client, LocalCluster

from ..distributed.ray_cluster_manager import (
    RayClusterManager, RayClusterConfig, ClusterStatus
)
from ..distributed.dask_cluster_manager import (
    DaskClusterManager, DaskClusterConfig, DaskClusterStatus
)
from ..backtesting.distributed_backtester import (
    DistributedBacktester, DistributedBacktestConfig,
    DistributionFramework, ParallelizationStrategy
)
from ..backtesting.parallel_strategy_runner import (
    ParallelStrategyRunner, StrategyExecutionConfig,
    ExecutionMode, StrategyTask
)
from ..backtesting.result_aggregator import (
    ResultAggregator, AggregationConfig, MergeStrategy,
    AggregatedResult
)
from ..config.cluster_config import (
    ClusterConfig, ClusterType, validate_cluster_config,
    get_local_development_config
)
from ..utils.distributed_utils import (
    ResourceMonitor, DataPartitioner, CacheManager,
    retry_on_failure, optimize_partition_size
)
from ..services.distributed_computing_service import (
    DistributedComputingService, ServiceStatus, get_distributed_service
)


# Mock strategy for testing
class MockStrategy:
    """Mock trading strategy for tests."""
    
    def __init__(self, param1=1.0, param2=2.0):
        self.param1 = param1
        self.param2 = param2
        
    def get_config(self):
        return {"param1": self.param1, "param2": self.param2}


# Test fixtures
@pytest.fixture
def sample_data():
    """Generate sample market data."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = pd.DataFrame({
        "open": np.random.randn(len(dates)).cumsum() + 100,
        "high": np.random.randn(len(dates)).cumsum() + 101,
        "low": np.random.randn(len(dates)).cumsum() + 99,
        "close": np.random.randn(len(dates)).cumsum() + 100,
        "volume": np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)
    return data


@pytest.fixture
def multi_symbol_data():
    """Generate multi-symbol market data."""
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    data = {}
    for symbol in symbols:
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data[symbol] = pd.DataFrame({
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 101,
            "low": np.random.randn(len(dates)).cumsum() + 99,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
    return data


@pytest.fixture
def cluster_config():
    """Get test cluster configuration."""
    return get_local_development_config()


# Ray Cluster Manager Tests
class TestRayClusterManager:
    """Test Ray cluster manager functionality."""
    
    def test_initialization(self):
        """Test Ray cluster manager initialization."""
        config = RayClusterConfig(
            head_node_cpu=2,
            head_node_memory=4,
            min_workers=0,
            max_workers=4
        )
        manager = RayClusterManager(config)
        
        assert manager.status == ClusterStatus.UNINITIALIZED
        assert manager.config.head_node_cpu == 2
        assert manager.config.head_node_memory == 4
        
    @patch("ray.init")
    @patch("ray.is_initialized", return_value=False)
    def test_cluster_initialization_success(self, mock_is_init, mock_init):
        """Test successful cluster initialization."""
        manager = RayClusterManager()
        
        # Initialize cluster
        success = manager.initialize_cluster()
        
        assert success
        assert manager.status == ClusterStatus.READY
        mock_init.assert_called_once()
        
    @patch("ray.shutdown")
    @patch("ray.is_initialized", return_value=True)
    def test_cluster_shutdown(self, mock_is_init, mock_shutdown):
        """Test cluster shutdown."""
        manager = RayClusterManager()
        manager.status = ClusterStatus.READY
        
        # Shutdown cluster
        success = manager.shutdown_cluster()
        
        assert success
        assert manager.status == ClusterStatus.UNINITIALIZED
        mock_shutdown.assert_called_once()
        
    @patch("ray.is_initialized", return_value=True)
    @patch("ray.cluster_resources", return_value={"CPU": 8, "memory": 16000000000})
    @patch("ray.available_resources", return_value={"CPU": 4, "memory": 8000000000})
    def test_get_cluster_resources(self, mock_available, mock_total, mock_is_init):
        """Test getting cluster resources."""
        manager = RayClusterManager()
        manager.status = ClusterStatus.READY
        
        resources = manager.get_cluster_resources()
        
        assert "total" in resources
        assert "available" in resources
        assert "used" in resources
        assert resources["total"]["CPU"] == 8
        assert resources["available"]["CPU"] == 4
        assert resources["used"]["CPU"] == 4


# Dask Cluster Manager Tests
class TestDaskClusterManager:
    """Test Dask cluster manager functionality."""
    
    def test_initialization(self):
        """Test Dask cluster manager initialization."""
        config = DaskClusterConfig(
            n_workers=4,
            threads_per_worker=2,
            memory_limit="4GB"
        )
        manager = DaskClusterManager(config)
        
        assert manager.status == DaskClusterStatus.UNINITIALIZED
        assert manager.config.n_workers == 4
        assert manager.config.memory_limit == "4GB"
        
    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_cluster_initialization_success(self, mock_client, mock_cluster):
        """Test successful cluster initialization."""
        # Setup mocks
        mock_cluster_instance = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.dashboard_link = "http://localhost:8787"
        
        mock_cluster.return_value = mock_cluster_instance
        mock_client.return_value = mock_client_instance
        
        manager = DaskClusterManager()
        
        # Initialize cluster
        success = manager.initialize_cluster()
        
        assert success
        assert manager.status == DaskClusterStatus.READY
        mock_cluster.assert_called_once()
        mock_client.assert_called_once()
        
    def test_create_dask_dataframe(self):
        """Test creating Dask DataFrame from pandas."""
        manager = DaskClusterManager()
        
        # Create pandas DataFrame
        df = pd.DataFrame({
            "A": range(1000),
            "B": range(1000, 2000)
        })
        
        # Convert to Dask DataFrame
        ddf = manager.create_dask_dataframe(df, npartitions=4)
        
        assert ddf.npartitions == 4
        assert list(ddf.columns) == ["A", "B"]


# Distributed Backtester Tests
class TestDistributedBacktester:
    """Test distributed backtesting functionality."""
    
    def test_initialization(self):
        """Test distributed backtester initialization."""
        config = DistributedBacktestConfig(
            framework=DistributionFramework.RAY,
            parallelization_strategy=ParallelizationStrategy.TIME_BASED,
            chunk_size=100
        )
        backtester = DistributedBacktester(config)
        
        assert backtester.config.framework == DistributionFramework.RAY
        assert backtester.config.chunk_size == 100
        
    @patch.object(DistributedBacktester, "_initialize_ray", return_value=True)
    def test_framework_initialization(self, mock_init_ray):
        """Test framework initialization."""
        backtester = DistributedBacktester()
        
        # Initialize
        success = backtester.initialize()
        
        assert success
        assert backtester._active_framework == DistributionFramework.RAY
        mock_init_ray.assert_called_once()
        
    def test_calculate_chunks(self, sample_data):
        """Test chunk calculation."""
        backtester = DistributedBacktester()
        backtester.config.chunk_size = 50
        
        chunks = backtester._calculate_chunks(len(sample_data))
        
        assert len(chunks) > 1
        assert chunks[0] == (0, 50)
        assert chunks[-1][1] == len(sample_data)
        
    def test_filter_data_by_date(self, sample_data):
        """Test data filtering by date."""
        backtester = DistributedBacktester()
        
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 8, 31)
        
        filtered = backtester._filter_data_by_date(
            {"default": sample_data},
            start_date,
            end_date
        )
        
        assert len(filtered["default"]) < len(sample_data)
        assert filtered["default"].index[0] >= start_date
        assert filtered["default"].index[-1] <= end_date


# Parallel Strategy Runner Tests
class TestParallelStrategyRunner:
    """Test parallel strategy execution."""
    
    def test_initialization(self):
        """Test strategy runner initialization."""
        config = StrategyExecutionConfig(
            execution_mode=ExecutionMode.PARALLEL_THREAD,
            max_workers=4,
            cache_results=True
        )
        runner = ParallelStrategyRunner(config)
        
        assert runner.config.execution_mode == ExecutionMode.PARALLEL_THREAD
        assert runner.config.max_workers == 4
        
    def test_add_strategy(self):
        """Test adding strategy to queue."""
        runner = ParallelStrategyRunner()
        
        task_id = runner.add_strategy(
            MockStrategy,
            {"param1": 1.5, "param2": 2.5},
            pd.DataFrame(),
            priority=5
        )
        
        assert len(runner._task_queue) == 1
        assert runner._task_queue[0].priority == 5
        assert task_id is not None
        
    def test_run_sequential(self, sample_data):
        """Test sequential strategy execution."""
        runner = ParallelStrategyRunner()
        runner.config.execution_mode = ExecutionMode.SEQUENTIAL
        
        # Add strategies
        for i in range(3):
            runner.add_strategy(
                MockStrategy,
                {"param1": i, "param2": i * 2},
                sample_data
            )
            
        # Run all
        results = runner.run_all()
        
        assert len(results) == 3
        assert all("error" not in r for r in results.values())


# Result Aggregator Tests
class TestResultAggregator:
    """Test result aggregation functionality."""
    
    def test_initialization(self):
        """Test result aggregator initialization."""
        config = AggregationConfig(
            merge_strategy=MergeStrategy.PORTFOLIO,
            calculate_statistics=True
        )
        aggregator = ResultAggregator(config)
        
        assert aggregator.config.merge_strategy == MergeStrategy.PORTFOLIO
        assert aggregator.config.calculate_statistics
        
    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        aggregator = ResultAggregator()
        
        metrics_list = [
            {"sharpe_ratio": 1.5, "total_return": 0.15},
            {"sharpe_ratio": 1.8, "total_return": 0.20},
            {"sharpe_ratio": 1.2, "total_return": 0.10},
        ]
        
        aggregated = aggregator.aggregate_metrics(metrics_list)
        
        assert "sharpe_ratio" in aggregated
        assert "total_return" in aggregated
        assert aggregated["sharpe_ratio"] == pytest.approx(1.5, rel=0.01)
        
    def test_calculate_portfolio_metrics(self):
        """Test portfolio metric calculation."""
        aggregator = ResultAggregator()
        
        # Generate component returns
        component_returns = [
            np.random.randn(100) * 0.01,
            np.random.randn(100) * 0.015,
            np.random.randn(100) * 0.008,
        ]
        weights = [0.4, 0.3, 0.3]
        
        metrics = aggregator.calculate_portfolio_metrics(component_returns, weights)
        
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
        
    def test_aggregate_results_portfolio(self):
        """Test portfolio aggregation strategy."""
        aggregator = ResultAggregator()
        aggregator.config.merge_strategy = MergeStrategy.PORTFOLIO
        
        results = [
            {
                "returns": np.random.randn(100) * 0.01,
                "sharpe_ratio": 1.5,
                "total_return": 0.15
            },
            {
                "returns": np.random.randn(100) * 0.01,
                "sharpe_ratio": 1.8,
                "total_return": 0.20
            },
        ]
        
        aggregated = aggregator.aggregate_results(results)
        
        assert isinstance(aggregated, AggregatedResult)
        assert "total_return" in aggregated.aggregate_metrics
        assert len(aggregated.component_results) == 2


# Distributed Utils Tests
class TestDistributedUtils:
    """Test distributed utility functions."""
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        monitor = ResourceMonitor(check_interval=1)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Get current resources
        resources = monitor.get_current_resources()
        
        assert "cpu_percent" in resources
        assert "memory_percent" in resources
        assert "disk_percent" in resources
        
        # Check availability
        available, warnings = monitor.check_resource_availability()
        
        assert isinstance(available, bool)
        assert isinstance(warnings, list)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
    def test_data_partitioner_by_time(self, sample_data):
        """Test time-based data partitioning."""
        partitions = DataPartitioner.partition_by_time(
            sample_data, n_partitions=4, overlap=10
        )
        
        assert len(partitions) == 4
        assert all(isinstance(p, pd.DataFrame) for p in partitions)
        
        # Check overlap
        for i in range(len(partitions) - 1):
            overlap_indices = partitions[i].index[-10:].intersection(
                partitions[i + 1].index[:10]
            )
            assert len(overlap_indices) > 0
            
    def test_data_partitioner_by_symbol(self, multi_symbol_data):
        """Test symbol-based data partitioning."""
        partitions = DataPartitioner.partition_by_symbol(
            multi_symbol_data, n_partitions=2
        )
        
        assert len(partitions) == 2
        assert all(isinstance(p, dict) for p in partitions)
        assert sum(len(p) for p in partitions) == len(multi_symbol_data)
        
    def test_cache_manager(self):
        """Test cache management."""
        cache = CacheManager(cache_dir="./test_cache", max_size_gb=0.001)
        
        # Put value
        success = cache.put("test_key", {"data": [1, 2, 3]})
        assert success
        
        # Get value
        value = cache.get("test_key")
        assert value == {"data": [1, 2, 3]}
        
        # Clear cache
        cache.clear()
        assert cache.get("test_key") is None
        
    def test_retry_decorator(self):
        """Test retry on failure decorator."""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
            
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 3
        
    def test_optimize_partition_size(self):
        """Test partition size optimization."""
        chunk_size = optimize_partition_size(
            total_size=100000,
            n_workers=4,
            min_chunk_size=1000,
            max_chunk_size=10000
        )
        
        assert chunk_size >= 1000
        assert chunk_size <= 10000


# Cluster Config Tests
class TestClusterConfig:
    """Test cluster configuration."""
    
    def test_local_development_config(self):
        """Test local development configuration."""
        config = get_local_development_config()
        
        assert config.cluster_type == ClusterType.LOCAL
        assert config.autoscaling.enabled == False
        assert config.head_node.cpu_cores == 2
        
    def test_config_validation(self):
        """Test configuration validation."""
        config = ClusterConfig()
        config.head_node.cpu_cores = 1  # Too low
        config.autoscaling.min_nodes = 10
        config.autoscaling.max_nodes = 5  # Invalid
        
        errors = validate_cluster_config(config)
        
        assert len(errors) > 0
        assert any("CPU cores" in e for e in errors)
        assert any("minimum nodes" in e for e in errors)
        
    def test_config_serialization(self, tmp_path):
        """Test configuration serialization."""
        config = get_local_development_config()
        
        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        config.save(str(yaml_path))
        
        # Load from YAML
        loaded_config = ClusterConfig.load(str(yaml_path))
        
        assert loaded_config.cluster_name == config.cluster_name
        assert loaded_config.cluster_type == config.cluster_type


# Distributed Computing Service Tests
class TestDistributedComputingService:
    """Test distributed computing service."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, cluster_config):
        """Test service initialization."""
        service = DistributedComputingService()
        
        assert service.status == ServiceStatus.UNINITIALIZED
        assert service.cluster_config is not None
        
    @pytest.mark.asyncio
    @patch.object(DistributedComputingService, "_initialize_ray", return_value=True)
    async def test_service_startup(self, mock_init_ray):
        """Test service startup."""
        service = DistributedComputingService()
        
        # Initialize service
        success = await service.initialize(framework="ray")
        
        assert success
        assert service.status == ServiceStatus.READY
        mock_init_ray.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_run_distributed_backtest(self, sample_data):
        """Test running distributed backtest."""
        service = DistributedComputingService()
        service.status = ServiceStatus.READY
        
        # Mock backtester
        mock_backtester = Mock()
        mock_backtester.run_backtest.return_value = {
            "total_return": 0.15,
            "sharpe_ratio": 1.5
        }
        service.distributed_backtester = mock_backtester
        
        # Run backtest
        job_result = await service.run_distributed_backtest(
            MockStrategy(),
            sample_data
        )
        
        assert job_result.job_id is not None
        assert job_result.status == "running"
        assert job_result.metadata["type"] == "backtest"
        
    def test_get_cluster_status(self):
        """Test getting cluster status."""
        service = DistributedComputingService()
        service.status = ServiceStatus.READY
        
        status = service.get_cluster_status()
        
        assert "service_status" in status
        assert "active_jobs" in status
        assert "cluster_info" in status
        assert "resource_usage" in status
        
    def test_singleton_service(self):
        """Test singleton service pattern."""
        service1 = get_distributed_service()
        service2 = get_distributed_service()
        
        assert service1 is service2


# Integration Tests
@pytest.mark.integration
class TestDistributedIntegration:
    """Integration tests for distributed computing."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_distributed_backtest(self, sample_data):
        """Test end-to-end distributed backtesting."""
        # Initialize service
        service = DistributedComputingService()
        
        # Skip if no distributed framework available
        try:
            success = await service.initialize(framework="auto")
            if not success:
                pytest.skip("No distributed framework available")
        except Exception:
            pytest.skip("Failed to initialize distributed framework")
            
        try:
            # Run distributed backtest
            job_result = await service.run_distributed_backtest(
                MockStrategy(param1=1.5, param2=2.5),
                sample_data,
                initial_capital=100000
            )
            
            # Wait for completion (with timeout)
            max_wait = 30  # seconds
            start_time = datetime.now()
            
            while job_result.status == "running":
                if (datetime.now() - start_time).total_seconds() > max_wait:
                    break
                await asyncio.sleep(1)
                
            # Check results
            assert job_result.status in ["completed", "failed"]
            if job_result.status == "completed":
                assert job_result.result_data is not None
                assert "total_return" in job_result.result_data
                
        finally:
            # Cleanup
            await service.shutdown()


# Performance Tests
@pytest.mark.performance
class TestDistributedPerformance:
    """Performance tests for distributed computing."""
    
    def test_partition_performance(self):
        """Test data partitioning performance."""
        # Generate large dataset
        large_data = pd.DataFrame(
            np.random.randn(1000000, 5),
            columns=["A", "B", "C", "D", "E"]
        )
        
        # Time partitioning
        import time
        start = time.time()
        
        partitions = DataPartitioner.partition_by_size(
            large_data, max_size_mb=32
        )
        
        duration = time.time() - start
        
        assert duration < 1.0  # Should partition in under 1 second
        assert len(partitions) > 1
        
    @pytest.mark.asyncio
    async def test_parallel_execution_speedup(self, multi_symbol_data):
        """Test parallel execution speedup."""
        runner = ParallelStrategyRunner()
        
        # Add multiple strategies
        for symbol in multi_symbol_data:
            for i in range(3):
                runner.add_strategy(
                    MockStrategy,
                    {"param1": i, "param2": i * 2},
                    multi_symbol_data[symbol]
                )
                
        # Sequential execution
        runner.config.execution_mode = ExecutionMode.SEQUENTIAL
        start = time.time()
        sequential_results = runner.run_all()
        sequential_time = time.time() - start
        
        # Clear queue
        runner.clear_queue()
        
        # Re-add strategies
        for symbol in multi_symbol_data:
            for i in range(3):
                runner.add_strategy(
                    MockStrategy,
                    {"param1": i, "param2": i * 2},
                    multi_symbol_data[symbol]
                )
                
        # Parallel execution
        runner.config.execution_mode = ExecutionMode.PARALLEL_THREAD
        runner.config.max_workers = 4
        start = time.time()
        parallel_results = runner.run_all()
        parallel_time = time.time() - start
        
        # Check speedup (should be faster, but not necessarily 4x due to overhead)
        assert parallel_time < sequential_time
        assert len(parallel_results) == len(sequential_results)