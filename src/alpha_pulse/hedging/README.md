# Grid Hedging Module

A modular implementation of grid-based hedging strategies with advanced risk management.

## Architecture

The module follows SOLID principles with clear separation of concerns:

### Core Components

1. **GridHedgeBot** (`grid_hedge_bot.py`)
   - Main coordinator class
   - Manages component lifecycle
   - Handles high-level strategy execution

2. **GridCalculator** (`grid_calculator.py`)
   - Calculates grid levels
   - Adjusts for market conditions
   - Handles dynamic spacing

3. **RiskManager** (`risk_manager.py`)
   - Position sizing
   - Risk limit monitoring
   - Dynamic stop loss calculation

4. **OrderManager** (`order_manager.py`)
   - Order placement and cancellation
   - Risk order management
   - Order state tracking

5. **StateManager** (`state_manager.py`)
   - Position state tracking
   - Performance metrics
   - Status reporting

### Models and Interfaces

1. **Data Models** (`models.py`)
   - Immutable state classes
   - Grid level representation
   - Market state models

2. **Interfaces** (`interfaces.py`)
   - Component protocols
   - Abstract base classes
   - Clean dependency injection

## Features

- Dynamic grid spacing based on volatility
- Advanced risk management
  * Position sizing based on risk metrics
  * Dynamic stop loss levels
  * Value at Risk (VaR) monitoring
- Support/resistance level integration
- Funding rate optimization
- Comprehensive logging with loguru
- Multiple trading modes (Real/Paper/Recommendation)

## Usage

### Basic Example

```python
from alpha_pulse.execution.broker_factory import create_broker
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot

# Configuration
config = {
    "grid_spacing_pct": "0.01",    # 1% grid spacing
    "num_levels": 5,               # 5 levels each side
    "max_position_size": "1.0",    # 1 BTC max position
    "max_drawdown": "0.1",         # 10% max drawdown
    "stop_loss_pct": "0.04",       # 4% stop loss
    "var_limit": "10000",          # $10k VaR limit
}

# Create and start bot
broker = create_broker(trading_mode="PAPER")
bot = await GridHedgeBot.create(
    broker=broker,
    symbol="BTCUSDT",
    config=config
)

# Execute strategy
current_price = await bot.data_provider.get_current_price("BTCUSDT")
await bot.execute(current_price)
```

### Custom Components

You can inject custom implementations of any component:

```python
from alpha_pulse.hedging.interfaces import GridCalculator
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot

class CustomGridCalculator(GridCalculator):
    """Custom grid calculation logic."""
    def calculate_grid_levels(self, market, position):
        # Custom implementation
        pass

# Use custom calculator
bot = GridHedgeBot(
    broker=broker,
    data_provider=data_provider,
    symbol="BTCUSDT",
    grid_calculator=CustomGridCalculator(),
    config=config
)
```

## Risk Management

The strategy includes multiple layers of risk management:

1. **Position Level**
   - Maximum position size limits
   - Dynamic position sizing based on volatility
   - Correlation-aware exposure calculation

2. **Order Level**
   - Dynamic grid spacing
   - Support/resistance integration
   - Volume-based adjustments

3. **Portfolio Level**
   - Value at Risk (VaR) monitoring
   - Maximum drawdown limits
   - Portfolio-wide exposure tracking

4. **Market Level**
   - Volatility-based adjustments
   - Funding rate optimization
   - Market impact consideration

## Configuration

### Grid Parameters

```yaml
grid_spacing_pct: "0.01"    # Grid spacing as percentage
num_levels: 5               # Number of levels each side
min_price_distance: "1.0"   # Minimum distance between levels
```

### Risk Parameters

```yaml
max_position_size: "1.0"    # Maximum position size
max_drawdown: "0.1"         # Maximum allowed drawdown
stop_loss_pct: "0.04"       # Base stop loss percentage
var_limit: "10000"          # VaR limit in quote currency
```

### Execution Parameters

```yaml
max_active_orders: 50       # Maximum number of active orders
rebalance_interval: 60      # Seconds between rebalances
```

## Logging

The module uses loguru for structured logging:

```python
logger.add(
    "logs/grid_hedge.log",
    rotation="1 day",
    level="INFO"
)
```

Example log output:
```
2025-02-03 11:35:43 | INFO     | Initialized GridHedgeBot for BTCUSDT with 5 levels
2025-02-03 11:35:44 | INFO     | Grid Status - Price: 95236.15, Position: 1.00000000
2025-02-03 11:35:45 | WARNING  | VaR (12000.00) exceeds limit (10000.00)
```

## Testing

Run the test suite:
```bash
python -m pytest src/alpha_pulse/tests/hedging/
```

Run with specific mode:
```bash
TRADING_MODE=paper python -m alpha_pulse.examples.demo_grid_hedge_integration