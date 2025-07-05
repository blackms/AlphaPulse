# Distributed Computing for AlphaPulse

## Overview

AlphaPulse includes a comprehensive distributed computing system that enables parallel backtesting, hyperparameter optimization, and Monte Carlo simulations at scale. The system supports both Ray and Dask frameworks, allowing you to choose the best tool for your specific use case.

## Architecture

The distributed computing system consists of several key components:

1. **Cluster Managers**: Handle cluster initialization, scaling, and resource management
2. **Distributed Backtester**: Orchestrates parallel backtesting across multiple workers
3. **Parallel Strategy Runner**: Executes multiple strategies concurrently
4. **Result Aggregator**: Combines and analyzes distributed results
5. **Distributed Computing Service**: Unified interface for all distributed operations

## Quick Start

### Local Development

```python
from alpha_pulse.services.distributed_computing_service import get_distributed_service

# Get the service instance
service = get_distributed_service()

# Initialize with local cluster
await service.initialize(framework="auto")

# Run distributed backtest
strategy = MyTradingStrategy(param1=1.5, param2=2.0)
data = load_market_data()

job_result = await service.run_distributed_backtest(
    strategy=strategy,
    data=data,
    initial_capital=100000
)

# Check job status
status = service.get_job_status(job_result.job_id)
print(f"Job {job_result.job_id}: {status.status}")
```

### Production Deployment

```python
from alpha_pulse.config.cluster_config import get_aws_production_config
from alpha_pulse.services.distributed_computing_service import DistributedComputingService

# Load production configuration
config = get_aws_production_config()
service = DistributedComputingService(config_path="config/production_cluster.yaml")

# Initialize with existing cluster
await service.initialize(framework="ray")

# Scale cluster based on workload
service.scale_cluster(n_workers=20)
```

## Cluster Configuration

### Configuration Options

Create a cluster configuration file (`cluster_config.yaml`):

```yaml
cluster_name: alphapulse-cluster
cluster_type: aws  # local, aws, gcp, azure
region: us-east-1
availability_zones:
  - us-east-1a
  - us-east-1b

head_node:
  cpu_cores: 8
  memory_gb: 32
  disk_size_gb: 200
  instance_type: m5.2xlarge
  spot_instance: false

worker_node:
  cpu_cores: 16
  memory_gb: 64
  disk_size_gb: 500
  instance_type: m5.4xlarge
  spot_instance: true

autoscaling:
  enabled: true
  min_nodes: 2
  max_nodes: 50
  target_cpu_utilization: 70.0
  scaling_policy: cost_optimized

storage:
  shared_storage_type: s3
  s3_bucket: alphapulse-distributed-storage
  cache_size_gb: 100
  enable_data_compression: true

monitoring:
  enable_metrics: true
  enable_logging: true
  prometheus_endpoint: http://prometheus:9090
  grafana_endpoint: http://grafana:3000
```

### Pre-configured Templates

```python
from alpha_pulse.config.cluster_config import (
    get_local_development_config,
    get_aws_production_config,
    get_gpu_cluster_config
)

# Local development (4 workers)
local_config = get_local_development_config()

# AWS production (auto-scaling 2-50 nodes)
prod_config = get_aws_production_config()

# GPU cluster for ML training
gpu_config = get_gpu_cluster_config()
```

## Ray Integration

### Ray Cluster Manager

```python
from alpha_pulse.distributed.ray_cluster_manager import RayClusterManager, RayClusterConfig

# Configure Ray cluster
config = RayClusterConfig(
    head_node_cpu=4,
    head_node_memory=8,
    min_workers=0,
    max_workers=10,
    enable_autoscaling=True
)

# Initialize cluster
manager = RayClusterManager(config)
manager.initialize_cluster()

# Submit tasks
@ray.remote
def process_data(data):
    # Process data chunk
    return results

# Submit multiple tasks
futures = [process_data.remote(chunk) for chunk in data_chunks]
results = ray.get(futures)
```

### Ray Tune for Hyperparameter Optimization

```python
# Run distributed hyperparameter optimization
analysis = manager.run_hyperparameter_optimization(
    trainable=train_strategy,
    config={
        "param1": tune.uniform(0.1, 2.0),
        "param2": tune.choice([10, 20, 50, 100]),
        "param3": tune.loguniform(1e-4, 1e-1)
    },
    num_samples=100,
    metric="sharpe_ratio",
    mode="max",
    scheduler_type="asha"  # or "pbt" for population-based training
)

# Get best parameters
best_config = analysis.get_best_config(metric="sharpe_ratio", mode="max")
```

## Dask Integration

### Dask Cluster Manager

```python
from alpha_pulse.distributed.dask_cluster_manager import DaskClusterManager, DaskClusterConfig

# Configure Dask cluster
config = DaskClusterConfig(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="8GB",
    dashboard_address=":8787"
)

# Initialize cluster
manager = DaskClusterManager(config)
manager.initialize_cluster()

# Enable adaptive scaling
manager.adapt_cluster(minimum=2, maximum=20)

# Process large datasets
df = pd.read_csv("large_data.csv")
ddf = manager.create_dask_dataframe(df, npartitions=10)

# Parallel processing
result = ddf.groupby("symbol").close.mean().compute()
```

### Dask Distributed Arrays

```python
# Create distributed array
arr = np.random.randn(1000000, 100)
darr = manager.create_dask_array(arr, chunks=(10000, 100))

# Parallel computation
mean = darr.mean(axis=0).compute()
cov = da.cov(darr.T).compute()
```

## Distributed Backtesting

### Parallelization Strategies

1. **Time-Based**: Split historical data into time chunks
2. **Symbol-Based**: Process different assets in parallel
3. **Parameter-Based**: Test different parameter combinations
4. **Monte Carlo**: Run multiple simulation paths

```python
from alpha_pulse.backtesting.distributed_backtester import (
    DistributedBacktester,
    DistributedBacktestConfig,
    ParallelizationStrategy
)

# Configure distributed backtesting
config = DistributedBacktestConfig(
    framework=DistributionFramework.RAY,
    parallelization_strategy=ParallelizationStrategy.TIME_BASED,
    chunk_size=1000,  # Days per chunk
    n_workers=10,
    enable_progress_bar=True,
    fault_tolerance=True
)

# Initialize backtester
backtester = DistributedBacktester(config)
backtester.initialize()

# Run time-based parallel backtest
results = backtester.run_backtest(
    strategy=my_strategy,
    data=market_data,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=1000000
)
```

### Walk-Forward Analysis

```python
# Run distributed walk-forward analysis
wf_results = backtester.run_walk_forward_analysis(
    strategy=my_strategy,
    data=market_data,
    window_size=252,  # 1 year training
    step_size=63,     # 3 month steps
    n_windows=12      # 12 windows
)

# Analyze results
for window in wf_results['windows']:
    print(f"Window {window['index']}: "
          f"In-sample Sharpe: {window['in_sample_sharpe']:.2f}, "
          f"Out-sample Sharpe: {window['out_sample_sharpe']:.2f}")
```

## Parallel Strategy Execution

### Running Multiple Strategies

```python
from alpha_pulse.backtesting.parallel_strategy_runner import (
    ParallelStrategyRunner,
    StrategyExecutionConfig,
    ExecutionMode
)

# Configure parallel execution
config = StrategyExecutionConfig(
    execution_mode=ExecutionMode.DISTRIBUTED_RAY,
    max_workers=20,
    cache_results=True,
    retry_failed=True
)

# Create runner
runner = ParallelStrategyRunner(config)

# Add strategies to queue
strategies = [
    (TrendFollowing, {"lookback": 20, "threshold": 1.5}),
    (MeanReversion, {"window": 10, "zscore": 2.0}),
    (Momentum, {"period": 30, "top_n": 5}),
    (MachineLearning, {"model": "rf", "features": 50})
]

for strategy_class, params in strategies:
    runner.add_strategy(
        strategy_class=strategy_class,
        strategy_params=params,
        data=market_data,
        priority=1
    )

# Execute all strategies in parallel
results = runner.run_all()

# Get aggregated results
successful = runner.get_results()
failed = runner.get_failures()
```

### Batch Processing

```python
from alpha_pulse.backtesting.parallel_strategy_runner import StrategyBatchProcessor

# Process strategies in batches
processor = StrategyBatchProcessor(batch_size=10)
processor.add_strategies(strategy_tasks)

# Process all batches
all_results = processor.process_batches(runner)
```

## Result Aggregation

### Aggregation Strategies

```python
from alpha_pulse.backtesting.result_aggregator import (
    ResultAggregator,
    AggregationConfig,
    MergeStrategy
)

# Configure aggregation
config = AggregationConfig(
    merge_strategy=MergeStrategy.PORTFOLIO,
    aggregation_methods={
        "total_return": AggregationMethod.WEIGHTED_MEAN,
        "sharpe_ratio": AggregationMethod.MEAN,
        "max_drawdown": AggregationMethod.MIN
    },
    calculate_statistics=True,
    confidence_level=0.95
)

# Create aggregator
aggregator = ResultAggregator(config)

# Aggregate distributed results
aggregated = aggregator.aggregate_results(
    results=chunk_results,
    weights=[0.3, 0.3, 0.4]  # Weight by importance
)

# Access aggregated metrics
print(f"Portfolio Sharpe: {aggregated.aggregate_metrics['sharpe_ratio']:.2f}")
print(f"95% CI: {aggregated.confidence_intervals['sharpe_ratio']}")
print(f"Diversification Ratio: {aggregated.metadata['diversification_ratio']:.2f}")
```

### Custom Aggregation

```python
# Register custom aggregator
def custom_risk_metric(values, weights):
    # Custom risk calculation
    return np.sqrt(np.average(np.square(values), weights=weights))

aggregator.register_custom_aggregator("custom_risk", custom_risk_metric)
```

## Performance Optimization

### Resource Monitoring

```python
from alpha_pulse.utils.distributed_utils import ResourceMonitor

# Start monitoring
monitor = ResourceMonitor(
    cpu_threshold=80.0,
    memory_threshold=85.0,
    check_interval=5
)
monitor.start_monitoring()

# Check resource availability
available, warnings = monitor.check_resource_availability()
if not available:
    print(f"Resource warnings: {warnings}")
    
# Get resource history
history = monitor.get_resource_history()
```

### Data Partitioning

```python
from alpha_pulse.utils.distributed_utils import DataPartitioner

# Partition by time with overlap
time_partitions = DataPartitioner.partition_by_time(
    data=market_data,
    n_partitions=10,
    overlap=20  # 20 periods overlap for continuity
)

# Partition by symbol
symbol_partitions = DataPartitioner.partition_by_symbol(
    data=multi_symbol_data,
    n_partitions=4
)

# Partition by size (memory-aware)
size_partitions = DataPartitioner.partition_by_size(
    data=large_dataset,
    max_size_mb=128  # 128MB per partition
)
```

### Caching Results

```python
from alpha_pulse.utils.distributed_utils import CacheManager

# Initialize cache
cache = CacheManager(
    cache_dir="./distributed_cache",
    max_size_gb=50.0
)

# Cache expensive computations
result = cache.get("strategy_optimization_v1")
if result is None:
    result = run_expensive_optimization()
    cache.put("strategy_optimization_v1", result)
```

## Fault Tolerance

### Retry Mechanism

```python
from alpha_pulse.utils.distributed_utils import retry_on_failure

@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def unstable_computation(data):
    # Computation that might fail
    return process_data(data)
```

### Checkpointing

```python
# Enable checkpointing in distributed backtest
config = DistributedBacktestConfig(
    checkpoint_interval=100,  # Checkpoint every 100 iterations
    fault_tolerance=True,
    max_retries=3
)
```

## Monitoring and Debugging

### Cluster Status

```python
# Get cluster status
status = service.get_cluster_status()
print(f"Active Jobs: {status['active_jobs']}")
print(f"Cluster Resources: {status['cluster_info']}")
print(f"CPU Usage: {status['resource_usage']['cpu_percent']}%")

# Monitor specific job
job_status = service.get_job_status(job_id)
print(f"Job Progress: {job_status.metadata}")
print(f"Duration: {job_status.duration_seconds}s")
```

### Dashboard Access

- **Ray Dashboard**: http://localhost:8265
- **Dask Dashboard**: http://localhost:8787

## Best Practices

1. **Choose the Right Framework**:
   - Ray: Better for task-based parallelism, ML workloads
   - Dask: Better for DataFrame operations, large datasets

2. **Optimize Chunk Sizes**:
   ```python
   from alpha_pulse.utils.distributed_utils import optimize_partition_size
   
   chunk_size = optimize_partition_size(
       total_size=len(data),
       n_workers=20,
       min_chunk_size=1000,
       max_chunk_size=10000
   )
   ```

3. **Monitor Resource Usage**:
   - Set appropriate resource limits
   - Use adaptive scaling for variable workloads
   - Monitor memory usage to prevent OOM errors

4. **Handle Failures Gracefully**:
   - Enable fault tolerance
   - Use checkpointing for long-running jobs
   - Implement proper error handling

5. **Optimize Data Transfer**:
   - Use data locality when possible
   - Minimize data serialization
   - Cache frequently accessed data

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce chunk sizes
   - Increase worker memory limits
   - Enable data spilling to disk

2. **Slow Performance**:
   - Check network bandwidth
   - Optimize data partitioning
   - Use appropriate parallelization strategy

3. **Worker Failures**:
   - Enable fault tolerance
   - Check worker logs
   - Verify resource availability

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("alpha_pulse.distributed").setLevel(logging.DEBUG)

# Run with profiling
config.enable_profiling = True
```

## Examples

See the `examples/distributed_backtesting/` directory for complete examples:

- `basic_distributed_backtest.py`: Simple distributed backtesting
- `hyperparameter_optimization.py`: Distributed parameter search
- `monte_carlo_simulation.py`: Large-scale Monte Carlo
- `multi_strategy_portfolio.py`: Parallel portfolio optimization