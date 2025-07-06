# Database Optimization Guide

This guide covers the comprehensive database optimization features in AlphaPulse, designed to maximize performance, reliability, and scalability of database operations.

## Overview

The database optimization system provides:
- Advanced connection pooling
- Query optimization and analysis
- Intelligent index management
- Table partitioning strategies
- Read/write splitting with load balancing
- Automatic failover handling
- Performance monitoring and alerting

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│              Database Optimization Service                    │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ Connection  │    Query    │    Index    │   Read/Write     │
│   Pooling   │ Optimization│ Management  │    Routing       │
├─────────────┼─────────────┼─────────────┼──────────────────┤
│ Partitioning│   Failover  │ Monitoring  │  Load Balancing  │
│  Manager    │   Manager   │   System    │                  │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                     PostgreSQL/TimescaleDB                   │
│  ┌────────┐      ┌──────────┐      ┌──────────┐           │
│  │ Master │      │ Replica 1│      │ Replica 2│           │
│  └────────┘      └──────────┘      └──────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Basic Configuration

```python
from alpha_pulse.config.database_config import (
    DatabaseConfig, DatabaseNode, ConnectionPoolConfig
)

# Configure database
config = DatabaseConfig()

# Master node
config.master_node = DatabaseNode(
    host="db-master.example.com",
    port=5432,
    database="alphapulse",
    username="app_user",
    password="secure_password"
)

# Read replicas
config.read_replicas = [
    DatabaseNode(
        host="db-replica-1.example.com",
        port=5432,
        database="alphapulse",
        username="app_user",
        password="secure_password",
        is_master=False,
        weight=1  # Load balancing weight
    ),
    DatabaseNode(
        host="db-replica-2.example.com",
        port=5432,
        database="alphapulse",
        username="app_user",
        password="secure_password",
        is_master=False,
        weight=2  # Higher weight = more traffic
    )
]

# Connection pooling
config.connection_pool = ConnectionPoolConfig(
    min_size=10,
    max_size=50,
    pool_timeout=30.0,
    pool_pre_ping=True,  # Test connections before use
    pool_recycle=3600,   # Recycle connections after 1 hour
    load_balancing="weighted"  # or "round_robin", "least_connections"
)

# Enable features
config.enable_read_write_split = True
config.enable_performance_monitoring = True
```

## Usage

### Basic Usage

```python
from alpha_pulse.services.database_optimization_service import (
    DatabaseOptimizationService
)

# Initialize service
db_service = DatabaseOptimizationService(
    config=config,
    metrics_collector=metrics,
    alert_manager=alerts
)

# Initialize all components
await db_service.initialize()

# Get sessions for different operations
async with db_service.get_read_session() as session:
    # Read operations automatically routed to replicas
    result = await session.execute(
        text("SELECT * FROM market_data WHERE symbol = :symbol"),
        {"symbol": "BTC/USD"}
    )

async with db_service.get_write_session() as session:
    # Write operations always go to master
    await session.execute(
        text("INSERT INTO trades (symbol, price) VALUES (:symbol, :price)"),
        {"symbol": "BTC/USD", "price": 50000}
    )
    await session.commit()
```

### Query Analysis

```python
# Analyze query performance
analysis = await db_service.analyze_query(
    "SELECT * FROM trades WHERE created_at > '2024-01-01' ORDER BY volume DESC"
)

print(f"Original cost: {analysis['execution_plan']['cost']}")
print(f"Optimized query: {analysis['optimized_query']}")
print("Suggestions:")
for suggestion in analysis['suggestions']:
    print(f"- {suggestion['description']} (Priority: {suggestion['priority']})")
```

### Index Optimization

```python
# Analyze and optimize indexes
index_report = await db_service.optimize_indexes(
    tables=["trades", "market_data"],
    apply_recommendations=True  # Automatically create recommended indexes
)

print(f"Found {len(index_report['recommendations'])} recommendations")
print(f"Applied: {index_report['applied']}")
```

### Partitioning Setup

```python
# Set up partitioning for time-series data
await db_service.setup_partitioning(
    table_name="market_data",
    partition_column="timestamp",
    interval="daily",
    retention_days=90  # Automatically drop partitions older than 90 days
)
```

## Connection Pooling

### Features

- **Master/Replica Separation**: Automatic routing of reads to replicas
- **Connection Health Checks**: Pre-ping and validation before use
- **Timeout Handling**: Configurable acquisition timeouts
- **Connection Recycling**: Prevent long-lived connections
- **Overflow Management**: Handle traffic spikes

### Configuration Options

```python
ConnectionPoolConfig(
    # Pool size
    min_size=5,          # Minimum connections
    max_size=20,         # Maximum connections
    overflow=10,         # Additional connections during spikes
    
    # Timeouts
    pool_timeout=30.0,   # Max wait for connection (seconds)
    connect_timeout=10,  # Connection establishment timeout
    
    # Health checks
    pool_pre_ping=True,  # Test connections before use
    pool_recycle=3600,   # Max connection age (seconds)
    
    # Strategy
    pooling_strategy="transaction",  # or "session", "statement"
    load_balancing="weighted"        # or "round_robin", "least_connections"
)
```

## Query Optimization

### Automatic Analysis

The system automatically analyzes queries to:
- Detect inefficient execution plans
- Identify missing indexes
- Suggest query rewrites
- Estimate execution costs

### Slow Query Detection

```python
# Configure slow query detection
config.query_optimization = QueryOptimizationConfig(
    slow_query_threshold=1.0,  # Queries slower than 1 second
    log_slow_queries=True,
    analyze_query_plans=True,
    enable_query_hints=True
)

# Get slow query report
slow_queries = await db_service._get_slow_query_summary()
```

### Query Caching

```python
# Enable query result caching
config.query_optimization.enable_query_cache = True
config.query_optimization.query_cache_size = 1000
config.query_optimization.query_cache_ttl = 300  # 5 minutes
```

## Index Management

### Index Advisor

The index advisor automatically:
- Analyzes slow queries to find missing indexes
- Detects duplicate and redundant indexes
- Identifies unused indexes
- Monitors index bloat
- Suggests partial indexes for filtered queries

### Automatic Index Creation

```python
# Run index analysis
recommendations = await index_advisor.analyze_indexes(
    session,
    tables=["trades", "market_data"]
)

# Apply high-priority recommendations
await index_manager.apply_recommendations(
    session,
    recommendations,
    priority_threshold=4  # Only priority 4-5
)
```

### Index Maintenance

```python
# Check for bloated indexes
bloated = await index_manager.check_index_bloat(
    session,
    bloat_threshold=30.0  # 30% bloat
)

# Rebuild bloated indexes
for index in bloated:
    await index_manager.rebuild_index(session, index["index_name"])
```

## Table Partitioning

### Strategies

AlphaPulse supports several partitioning strategies:
- **Range Partitioning**: For time-series data (daily, monthly, yearly)
- **List Partitioning**: For categorical data
- **Hash Partitioning**: For even distribution

### Automatic Management

```python
# Configure partitioning
strategy = PartitionStrategy(
    table_name="market_data",
    partition_column="timestamp",
    partition_type=PartitionType.RANGE,
    interval=PartitionInterval.DAILY,
    retention_days=90,
    pre_create_days=7  # Create partitions 7 days in advance
)

# Start automatic maintenance
await partition_manager.start_maintenance(session)
```

## Read/Write Splitting

### Routing Logic

- **Writes**: Always routed to master
- **Reads**: Distributed across replicas based on:
  - Load balancing strategy
  - Replica health
  - Replication lag
  - Consistency requirements

### Consistency Options

```python
# Strong consistency (read from master)
async with db_service.get_read_session(consistency_required=True) as session:
    # Critical reads that need latest data
    pass

# Eventual consistency (read from replicas)
async with db_service.get_read_session() as session:
    # Normal reads that can tolerate slight lag
    pass
```

### Replica Lag Handling

```python
# Configure lag policy
router.max_replica_lag = 10.0  # seconds
router.lag_policy = ReplicaLagPolicy.FALLBACK  # Fall back to master if lag too high
```

## Load Balancing

### Strategies

1. **Round Robin**: Equal distribution across replicas
2. **Least Connections**: Route to replica with fewest active connections
3. **Weighted**: Distribute based on replica capacity
4. **Adaptive**: Automatically switch strategies based on load

### Circuit Breaker

```python
# Circuit breaker configuration
load_balancer.circuit_breaker_threshold = 5  # Open after 5 failures
load_balancer.circuit_breaker_timeout = 60   # Try again after 60 seconds
```

## Failover Management

### Automatic Failover

The system monitors master health and can:
- Detect master failures
- Select best replica for promotion
- Execute failover procedures
- Update routing configuration

### Configuration

```python
# Failover settings
failover_manager.health_check_interval = 10  # seconds
failover_manager.max_consecutive_failures = 3
failover_manager.promotion_strategy = PromotionStrategy.SEMI_AUTOMATIC
```

### Manual Failover

```python
# Manually trigger failover
await failover_manager.manual_failover(
    new_master_host="db-replica-1.example.com",
    new_master_port=5432
)
```

## Performance Monitoring

### Metrics Collection

The system collects:
- Connection pool statistics
- Query execution times
- Cache hit rates
- Replication lag
- Table and index statistics
- Lock contention

### Alerts

```python
# Configure alerts
monitor.connection_threshold = 0.8     # Alert at 80% connection usage
monitor.slow_query_threshold = 1.0     # Alert for queries > 1 second
monitor.replication_lag_threshold = 10 # Alert for lag > 10 seconds
```

### Dashboard Integration

```python
# Get monitoring data for dashboard
health_report = await db_service.analyze_database_health()

# Includes:
# - Connection pool status
# - Routing statistics
# - Slow query analysis
# - Index recommendations
# - Table metrics
# - Replication status
```

## Best Practices

### Connection Management

1. **Use appropriate pool sizes**: Start with min=5, max=20 and adjust based on load
2. **Enable connection validation**: Use `pool_pre_ping=True` for reliability
3. **Set reasonable timeouts**: Balance between responsiveness and stability
4. **Monitor connection usage**: Alert before hitting limits

### Query Optimization

1. **Analyze slow queries regularly**: Review slow query log weekly
2. **Create indexes strategically**: Balance query performance with write overhead
3. **Use prepared statements**: For frequently executed queries
4. **Avoid SELECT ***: Specify only needed columns

### Partitioning

1. **Choose appropriate intervals**: Daily for high-volume, monthly for moderate
2. **Set retention policies**: Automatically clean old data
3. **Pre-create partitions**: Avoid runtime partition creation
4. **Monitor partition sizes**: Ensure even distribution

### High Availability

1. **Use multiple replicas**: At least 2 for redundancy
2. **Monitor replication lag**: Alert on high lag
3. **Test failover procedures**: Regular drills
4. **Document recovery steps**: Clear runbooks

## Troubleshooting

### Common Issues

1. **High connection pool usage**
   - Check for connection leaks
   - Increase pool size if needed
   - Review query execution times

2. **Slow queries**
   - Run EXPLAIN ANALYZE
   - Check for missing indexes
   - Review table statistics

3. **Replication lag**
   - Check network latency
   - Review replica resources
   - Check for long-running transactions

4. **Failed failover**
   - Verify replica health
   - Check permissions
   - Review failover logs

### Diagnostic Commands

```python
# Get comprehensive health report
health = await db_service.analyze_database_health()

# Check specific components
pool_stats = await db_service.connection_pool.get_pool_stats()
routing_stats = db_service.router.get_routing_stats()
slow_queries = await db_service.slow_query_detector.get_statistics()
```

## Performance Tuning

### PostgreSQL Settings

```sql
-- Recommended settings for alphapulse.conf
max_connections = 200
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 10MB
min_wal_size = 1GB
max_wal_size = 4GB
```

### Application-Level Tuning

1. **Batch operations**: Use bulk inserts/updates
2. **Connection pooling**: Reuse connections
3. **Async operations**: Use asyncio for concurrent queries
4. **Caching**: Cache frequently accessed data
5. **Monitoring**: Track and optimize based on metrics