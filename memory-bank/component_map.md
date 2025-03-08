# Component Map for AI Hedge Fund System

## Core Components Map

### Data Pipeline
```yaml
base_path: src/alpha_pulse/data_pipeline/
components:
  market_data:
    path: providers/market_data.py
    interfaces: [DataProvider, MarketDataFetcher]
    dependencies: [ccxt, pandas]
    key_functions: [fetch_ohlcv, fetch_ticker]
  
  fundamental_data:
    path: providers/fundamental_data.py
    interfaces: [DataProvider, FundamentalDataFetcher]
    dependencies: [alpha_vantage, finnhub]
    
  sentiment_data:
    path: providers/sentiment_data.py
    interfaces: [DataProvider, SentimentAnalyzer]
    dependencies: [nltk, textblob]
```

### Agent System
```yaml
base_path: src/alpha_pulse/agents/
components:
  technical_agent:
    path: technical_agent.py
    interfaces: [BaseAgent, SignalGenerator]
    dependencies: [ta-lib, pandas]
    key_functions: [generate_signals, analyze_trends]
  
  fundamental_agent:
    path: fundamental_agent.py
    interfaces: [BaseAgent, SignalGenerator]
    
  sentiment_agent:
    path: sentiment_agent.py
    interfaces: [BaseAgent, SignalGenerator]
  
  value_agent:
    path: value_agent.py
    interfaces: [BaseAgent, SignalGenerator]
  
  activist_agent:
    path: activist_agent.py
    interfaces: [BaseAgent, SignalGenerator]
  
  agent_manager:
    path: manager.py
    interfaces: [AgentManager]
    key_functions: [aggregate_signals, manage_agents]
```

### Risk Management
```yaml
base_path: src/alpha_pulse/risk_management/
components:
  risk_manager:
    path: manager.py
    interfaces: [RiskManager]
    key_functions: [calculate_position_size, check_risk_limits]
  
  position_sizing:
    path: position_sizing.py
    interfaces: [PositionSizer]
    key_functions: [kelly_criterion, optimal_f]
  
  risk_metrics:
    path: metrics.py
    interfaces: [RiskMetrics]
    key_functions: [calculate_var, calculate_sharpe]
```

### Portfolio Management
```yaml
base_path: src/alpha_pulse/portfolio/
components:
  portfolio_manager:
    path: portfolio_manager.py
    interfaces: [PortfolioManager]
    key_functions: [rebalance_portfolio, update_positions]
  
  optimizer:
    path: optimizer.py
    interfaces: [PortfolioOptimizer]
    key_functions: [optimize_weights, calculate_efficient_frontier]
  
  allocation:
    path: allocation.py
    interfaces: [AssetAllocator]
    key_functions: [calculate_allocations, rebalance_targets]
```

### Execution
```yaml
base_path: src/alpha_pulse/execution/
components:
  broker:
    path: broker_interface.py
    interfaces: [ExecutionBroker]
    key_functions: [place_order, cancel_order]
  
  order_manager:
    path: order_manager.py
    interfaces: [OrderManager]
    key_functions: [manage_orders, track_fills]
```

### Monitoring
```yaml
base_path: src/alpha_pulse/monitoring/
components:
  metrics_collector:
    path: collector.py
    interfaces: [MetricsCollector]
    key_functions: [collect_metrics, store_metrics]
  
  alerting:
    path: alerting/
    subcomponents:
      manager:
        path: manager.py
        interfaces: [AlertManager]
      rules:
        path: rules.py
        interfaces: [RuleEvaluator]
      notifications:
        path: notifications/
        interfaces: [NotificationChannel]
```

### Database
```yaml
base_path: src/alpha_pulse/data_pipeline/database/
components:
  connection:
    path: connection.py
    interfaces: [DatabaseConnection]
    key_functions: [connect, execute_query]
  
  models:
    path: models.py
    interfaces: [BaseModel]
    key_functions: [create, update, delete]
  
  migrations:
    path: migrations/
    interfaces: [Migration]
```

### API
```yaml
base_path: src/alpha_pulse/api/
components:
  main:
    path: main.py
    interfaces: [FastAPI]
    key_functions: [startup, shutdown]
  
  routers:
    path: routers/
    subcomponents:
      metrics:
        path: metrics.py
        endpoints: [GET /metrics, GET /metrics/{metric_type}]
      alerts:
        path: alerts.py
        endpoints: [GET /alerts, POST /alerts/{alert_id}/acknowledge]
      portfolio:
        path: portfolio.py
        endpoints: [GET /portfolio, GET /portfolio/positions]
```

### Dashboard Frontend
```yaml
base_path: dashboard/src/
components:
  pages:
    path: pages/
    subcomponents:
      dashboard:
        path: dashboard/DashboardPage.tsx
      portfolio:
        path: portfolio/PortfolioPage.tsx
      alerts:
        path: alerts/AlertsPage.tsx
  
  components:
    path: components/
    subcomponents:
      charts:
        path: charts/
        interfaces: [LineChart, BarChart, PieChart]
      tables:
        path: tables/
        interfaces: [DataTable, SortableTable]
  
  services:
    path: services/
    subcomponents:
      api:
        path: api/
        interfaces: [APIClient]
      websocket:
        path: websocket/
        interfaces: [WebSocketClient]
```

## Integration Points

### Data Flow
```yaml
market_data -> technical_agent:
  interface: DataProvider
  data_type: OHLCV

agents -> agent_manager:
  interface: SignalGenerator
  data_type: TradingSignal

agent_manager -> risk_manager:
  interface: SignalAggregator
  data_type: AggregatedSignal

risk_manager -> portfolio_manager:
  interface: RiskAssessment
  data_type: PositionSize

portfolio_manager -> broker:
  interface: OrderGenerator
  data_type: TradeOrder
```

### Real-time Updates
```yaml
metrics_collector -> alerting:
  interface: MetricsProvider
  channel: metrics_update

alerting -> api:
  interface: AlertNotifier
  channel: alert_notification

api -> dashboard:
  interface: WebSocket
  channels: [metrics, alerts, portfolio, trades]
```

## Configuration Paths
```yaml
config_files:
  ai_hedge_fund: config/ai_hedge_fund_config.yaml
  alerting: config/alerting_config.yaml
  api: config/api_config.yaml
  data_pipeline: config/data_pipeline_config.yaml
  database: config/database_config.yaml
  monitoring: config/monitoring_config.yaml
  portfolio: config/portfolio_config.yaml
```

## Test Locations
```yaml
base_path: tests/
components:
  agents: agents/test_*.py
  risk: risk_management/test_*.py
  portfolio: portfolio/test_*.py
  execution: execution/test_*.py
  api: api/test_*.py
  integration: integration/test_*.py
```

## System State Indicators
```yaml
health_checks:
  database:
    path: src/alpha_pulse/data_pipeline/database/connection.py
    method: check_connection()
    status_endpoint: /api/v1/system/health/database
  
  exchange_connections:
    path: src/alpha_pulse/execution/broker_interface.py
    method: check_exchange_status()
    status_endpoint: /api/v1/system/health/exchanges
  
  data_pipeline:
    path: src/alpha_pulse/data_pipeline/collector.py
    method: check_pipeline_status()
    status_endpoint: /api/v1/system/health/pipeline

component_status:
  agents:
    status_check: src/alpha_pulse/agents/manager.py:check_agents_status
    required_state: all_running
  
  risk_management:
    status_check: src/alpha_pulse/risk_management/manager.py:check_status
    required_state: active
  
  portfolio:
    status_check: src/alpha_pulse/portfolio/portfolio_manager.py:check_status
    required_state: ready
```

## Initialization Order
```yaml
startup_sequence:
  1:
    component: database_connection
    path: src/alpha_pulse/data_pipeline/database/connection.py
    method: initialize_connection
  
  2:
    component: metrics_collector
    path: src/alpha_pulse/monitoring/collector.py
    method: start_collection
  
  3:
    component: alert_manager
    path: src/alpha_pulse/monitoring/alerting/manager.py
    method: initialize
  
  4:
    component: data_pipeline
    path: src/alpha_pulse/data_pipeline/collector.py
    method: start_pipeline
  
  5:
    component: agent_manager
    path: src/alpha_pulse/agents/manager.py
    method: initialize_agents
  
  6:
    component: risk_manager
    path: src/alpha_pulse/risk_management/manager.py
    method: initialize
  
  7:
    component: portfolio_manager
    path: src/alpha_pulse/portfolio/portfolio_manager.py
    method: initialize
  
  8:
    component: execution_broker
    path: src/alpha_pulse/execution/broker_interface.py
    method: connect
```

## Error Handling Patterns
```yaml
error_handlers:
  database:
    path: src/alpha_pulse/data_pipeline/database/error_handlers.py
    retry_strategy: exponential_backoff
    circuit_breaker: enabled
  
  exchange:
    path: src/alpha_pulse/execution/error_handlers.py
    retry_strategy: exponential_backoff
    circuit_breaker: enabled
    
  api:
    path: src/alpha_pulse/api/error_handlers.py
    retry_strategy: none
    circuit_breaker: disabled

graceful_degradation:
  database:
    fallback: in_memory_storage
    recovery: automatic
  
  exchange:
    fallback: paper_trading
    recovery: manual
  
  api:
    fallback: cached_responses
    recovery: automatic
```

## Component Dependencies
```yaml
critical_dependencies:
  database:
    required_by: [metrics_collector, alert_manager, portfolio_manager]
    fallback: none
  
  data_pipeline:
    required_by: [technical_agent, fundamental_agent, sentiment_agent]
    fallback: cached_data
  
  agent_manager:
    required_by: [risk_manager, portfolio_manager]
    fallback: last_known_signals

optional_dependencies:
  alerting:
    used_by: [risk_manager, portfolio_manager, execution_broker]
    fallback: logging_only
  
  metrics:
    used_by: [dashboard, monitoring]
    fallback: basic_metrics
```

## Configuration Requirements
```yaml
required_configs:
  database:
    file: config/database_config.yaml
    required_fields: [host, port, database, username, password]
    
  exchange:
    file: config/ai_hedge_fund_config.yaml
    required_fields: [api_key, api_secret, exchange_type]
    
  risk:
    file: config/portfolio_config.yaml
    required_fields: [max_position_size, risk_limits, stop_loss_settings]

environment_variables:
  required:
    - DB_CONNECTION_STRING
    - EXCHANGE_API_KEY
    - EXCHANGE_API_SECRET
    - JWT_SECRET
    
  optional:
    - LOG_LEVEL
    - METRICS_RETENTION_DAYS
    - CACHE_TTL
```

## Common Patterns
```yaml
design_patterns:
  factory:
    used_in: [exchange_creation, agent_creation]
    implementation: src/alpha_pulse/common/factories.py
    
  observer:
    used_in: [metrics_collection, alert_notification]
    implementation: src/alpha_pulse/common/observers.py
    
  strategy:
    used_in: [portfolio_optimization, risk_management]
    implementation: src/alpha_pulse/common/strategies.py

coding_conventions:
  error_handling:
    pattern: try_except_with_logging
    example_path: src/alpha_pulse/common/error_handling.py
    
  async_operations:
    pattern: async_with_timeout
    example_path: src/alpha_pulse/common/async_utils.py
    
  configuration:
    pattern: env_with_config_file
    example_path: src/alpha_pulse/common/config.py
```

## Component Status Tracking
```yaml
status_tracking:
  database:
    check_method: check_connection
    metrics: [connection_pool_size, query_latency]
    health_threshold: connection_pool_size > 0 && query_latency < 100ms
    
  agents:
    check_method: check_agents_status
    metrics: [active_agents, signal_generation_rate]
    health_threshold: active_agents == configured_agents && signal_generation_rate > 0
    
  portfolio:
    check_method: check_portfolio_status
    metrics: [position_count, total_value, cash_balance]
    health_threshold: total_value > 0 && cash_balance >= 0
```

## Memory Requirements
```yaml
memory_management:
  data_pipeline:
    cache_size: 1GB
    cleanup_interval: 1h
    
  portfolio:
    position_history: 7d
    trade_history: 30d
    
  metrics:
    retention_period: 90d
    aggregation_rules: [1m:1h, 1h:7d, 1d:90d]
```

## Performance Characteristics
```yaml
performance_metrics:
  api_endpoints:
    latency_threshold: 100ms
    timeout: 30s
    rate_limit: 100/minute
    
  websocket:
    max_connections: 1000
    message_rate: 10/second
    
  database:
    max_connections: 100
    query_timeout: 5s
    max_pool_size: 20
```