# Hedging Module

The hedging module provides a flexible system for managing and automating hedging strategies using futures contracts. The implementation follows SOLID principles to ensure maintainability, extensibility, and testability.

## Grid Hedging Strategy

The module now includes a grid-based hedging strategy that can operate in multiple trading modes:

### Trading Modes

1. **REAL**: Execute trades on actual exchanges (Binance/Bybit)
2. **PAPER**: Simulate trades without real execution
3. **RECOMMENDATION**: Log trade signals without execution

### Grid Strategy Components

- `GridHedgeConfig`: Configures grid parameters
  ```python
  config = GridHedgeConfig.create_symmetric_grid(
      symbol="BTCUSDT",
      center_price=40000.0,
      grid_spacing=100.0,    # Price difference between levels
      num_levels=5,          # Levels above/below center
      position_step_size=0.001,  # Position size per level
      max_position_size=0.01,    # Maximum total position
      grid_direction=GridDirection.BOTH
  )
  ```

- `GridHedgeBot`: Implements the grid strategy
  ```python
  from alpha_pulse.execution.broker_factory import create_broker, TradingMode
  
  # Create broker for desired mode
  broker = create_broker(
      trading_mode=TradingMode.PAPER,  # or REAL, RECOMMENDATION
      exchange_name="binance",  # Required for REAL mode
      api_key="...",           # Required for REAL mode
      api_secret="..."         # Required for REAL mode
  )
  
  # Create and run bot
  bot = GridHedgeBot(broker, config)
  bot.execute(current_price=40000.0)
  ```

### Grid Strategy Features

- Symmetric or asymmetric grid creation
- Dynamic grid rebalancing
- Position size limits
- Multiple trading directions (LONG, SHORT, BOTH)
- Configurable rebalance intervals
- Stop loss and take profit options

## Architecture

The module is built on several key interfaces that enable flexible component composition:

### Core Interfaces

- `IHedgeAnalyzer`: Analyzes positions and generates recommendations
- `IPositionFetcher`: Retrieves position data from data sources
- `IOrderExecutor`: Executes trading orders
- `IExecutionStrategy`: Implements hedge execution logic
- `IBroker`: Provides unified interface for different trading modes

### Components

1. **Position Management**
   - `SpotPosition`: Represents spot market positions
   - `FuturesPosition`: Represents futures market positions
   - `ExchangePositionFetcher`: Fetches positions from exchanges

2. **Analysis**
   - `BasicFuturesHedgeAnalyzer`: Implements basic hedging analysis
   - `HedgeRecommendation`: Contains analysis results
   - `HedgeAdjustment`: Represents recommended changes

3. **Execution**
   - `BasicExecutionStrategy`: Implements basic execution logic
   - `ExchangeOrderExecutor`: Executes orders through exchanges
   - `RealBroker`: Handles real exchange trading
   - `PaperBroker`: Simulates trading execution
   - `RecommendationOnlyBroker`: Logs trade signals

4. **Orchestration**
   - `HedgeManager`: Coordinates the hedging process
   - `HedgeConfig`: Configures hedging parameters
   - `GridHedgeBot`: Manages grid-based hedging

## SOLID Principles Implementation

### Single Responsibility Principle
Each class has a single, well-defined purpose:
- `HedgeManager`: Orchestrates the hedging process
- `ExchangePositionFetcher`: Handles position data retrieval
- `BasicExecutionStrategy`: Manages execution logic
- `GridHedgeBot`: Manages grid-based trading

### Open/Closed Principle
New functionality can be added without modifying existing code:
- New analyzers can implement `IHedgeAnalyzer`
- New execution strategies can implement `IExecutionStrategy`
- New position sources can implement `IPositionFetcher`
- New brokers can implement `IBroker`

### Liskov Substitution Principle
Components are interchangeable through their interfaces:
```python
# Any IHedgeAnalyzer implementation can be used
analyzer: IHedgeAnalyzer = BasicFuturesHedgeAnalyzer(config)

# Any IExecutionStrategy implementation can be used
strategy: IExecutionStrategy = BasicExecutionStrategy()

# Any IBroker implementation can be used
broker: IBroker = create_broker(trading_mode)
```

### Interface Segregation Principle
Interfaces are focused and minimal:
- `IPositionFetcher`: Only position retrieval methods
- `IOrderExecutor`: Only order execution methods
- `IExecutionStrategy`: Only strategy execution methods
- `IBroker`: Core trading operations

### Dependency Inversion Principle
High-level components depend on abstractions:
```python
class HedgeManager:
    def __init__(
        self,
        hedge_analyzer: IHedgeAnalyzer,
        position_fetcher: IPositionFetcher,
        execution_strategy: IExecutionStrategy,
        order_executor: IOrderExecutor
    ):
        ...

class GridHedgeBot:
    def __init__(self, broker: IBroker, config: GridHedgeConfig):
        ...
```

## Usage Example

```python
from alpha_pulse.exchanges.bybit import BybitExchange
from alpha_pulse.hedging import (
    HedgeConfig,
    BasicFuturesHedgeAnalyzer,
    ExchangePositionFetcher,
    BasicExecutionStrategy,
    ExchangeOrderExecutor,
    HedgeManager
)

# Create exchange connector
exchange = BybitExchange(api_key="...", api_secret="...")

# Create components
config = HedgeConfig.default_config()
position_fetcher = ExchangePositionFetcher(exchange)
order_executor = ExchangeOrderExecutor(exchange)
analyzer = BasicFuturesHedgeAnalyzer(config)
execution_strategy = BasicExecutionStrategy()

# Create manager
manager = HedgeManager(
    hedge_analyzer=analyzer,
    position_fetcher=position_fetcher,
    execution_strategy=execution_strategy,
    order_executor=order_executor,
    execute_hedge=False  # Set to True for live trading
)

# Run hedge management
await manager.manage_hedge()
```

## Configuration

### Hedge Parameters
```python
config = HedgeConfig(
    hedge_ratio_target=Decimal('0.0'),  # Fully hedged
    max_leverage=Decimal('3.0'),
    max_margin_usage=Decimal('0.8'),
    min_position_size={'BTC': Decimal('0.001')},
    max_position_size={'BTC': Decimal('1.0')},
    grid_bot_enabled=True  # Enable grid hedging
)
```

### Risk Management
- `max_drawdown`: Maximum allowed drawdown
- `stop_loss_threshold`: Stop loss trigger level
- `hedge_ratio_threshold`: Allowed hedge ratio deviation
- `grid_rebalance_interval`: Time between grid checks

### Execution Parameters
- `execution_delay`: Delay between trades
- `max_slippage`: Maximum allowed slippage
- `grid_spacing`: Price difference between grid levels

## Important Notes

1. **Risk Management**
   - Always test strategies in dry-run mode first
   - Monitor margin usage carefully
   - Consider funding rates for futures positions
   - Test grid parameters in paper trading mode

2. **Exchange Support**
   - Currently supports Binance and Bybit
   - Ensure API keys have proper permissions
   - Use testnet for initial testing

3. **Performance Monitoring**
   - Track hedge effectiveness over time
   - Monitor correlation between spot and futures
   - Adjust grid parameters based on market conditions

## Example Scripts

See the following examples for complete demonstrations:
- `examples/demo_hedging.py`: Basic hedging setup
- `examples/demo_grid_hedge_integration.py`: Grid hedging with multiple modes

## Contributing

When adding new features:
1. Follow SOLID principles
2. Implement appropriate interfaces
3. Add comprehensive tests
4. Update documentation
5. Consider backward compatibility

## License

This module is part of the AlphaPulse project and is subject to its licensing terms.