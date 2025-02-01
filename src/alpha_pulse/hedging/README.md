# Hedging Module

The hedging module provides a flexible system for managing and automating hedging strategies using futures contracts. The implementation follows SOLID principles to ensure maintainability, extensibility, and testability.

## Architecture

The module is built on several key interfaces that enable flexible component composition:

### Core Interfaces

- `IHedgeAnalyzer`: Analyzes positions and generates recommendations
- `IPositionFetcher`: Retrieves position data from data sources
- `IOrderExecutor`: Executes trading orders
- `IExecutionStrategy`: Implements hedge execution logic

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

4. **Orchestration**
   - `HedgeManager`: Coordinates the hedging process
   - `HedgeConfig`: Configures hedging parameters

## SOLID Principles Implementation

### Single Responsibility Principle
Each class has a single, well-defined purpose:
- `HedgeManager`: Orchestrates the hedging process
- `ExchangePositionFetcher`: Handles position data retrieval
- `BasicExecutionStrategy`: Manages execution logic

### Open/Closed Principle
New functionality can be added without modifying existing code:
- New analyzers can implement `IHedgeAnalyzer`
- New execution strategies can implement `IExecutionStrategy`
- New position sources can implement `IPositionFetcher`

### Liskov Substitution Principle
Components are interchangeable through their interfaces:
```python
# Any IHedgeAnalyzer implementation can be used
analyzer: IHedgeAnalyzer = BasicFuturesHedgeAnalyzer(config)

# Any IExecutionStrategy implementation can be used
strategy: IExecutionStrategy = BasicExecutionStrategy()
```

### Interface Segregation Principle
Interfaces are focused and minimal:
- `IPositionFetcher`: Only position retrieval methods
- `IOrderExecutor`: Only order execution methods
- `IExecutionStrategy`: Only strategy execution methods

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
    grid_bot_enabled=False
)
```

### Risk Management
- `max_drawdown`: Maximum allowed drawdown
- `stop_loss_threshold`: Stop loss trigger level
- `hedge_ratio_threshold`: Allowed hedge ratio deviation

### Execution Parameters
- `execution_delay`: Delay between trades
- `max_slippage`: Maximum allowed slippage

## Extending the System

### Adding a New Analyzer
```python
class AdvancedHedgeAnalyzer(IHedgeAnalyzer):
    def analyze(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> HedgeRecommendation:
        # Implement advanced analysis logic
        ...
```

### Adding a New Execution Strategy
```python
class SmartExecutionStrategy(IExecutionStrategy):
    async def execute_hedge_adjustments(
        self,
        recommendation: HedgeRecommendation,
        executor: IOrderExecutor
    ) -> bool:
        # Implement smart execution logic
        ...
```

## Important Notes

1. **Risk Management**
   - Always test strategies in dry-run mode first
   - Monitor margin usage carefully
   - Consider funding rates for futures positions

2. **Exchange Support**
   - Currently supports Bybit
   - Ensure API keys have proper permissions
   - Use testnet for initial testing

3. **Performance Monitoring**
   - Track hedge effectiveness over time
   - Monitor correlation between spot and futures
   - Adjust parameters based on market conditions

## Example Scripts

See `examples/demo_hedging.py` for a complete demonstration of:
- Component setup and configuration
- Position analysis
- Hedge management
- Emergency position closure

## Contributing

When adding new features:
1. Follow SOLID principles
2. Implement appropriate interfaces
3. Add comprehensive tests
4. Update documentation
5. Consider backward compatibility

## License

This module is part of the AlphaPulse project and is subject to its licensing terms.