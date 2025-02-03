# Hedging Module

This module provides hedging strategies for managing risk in cryptocurrency portfolios.

## Grid Hedging Strategy

The grid hedging strategy automatically manages hedge positions using a grid of orders placed at predefined price levels. It supports multiple trading modes and includes risk management features.

### Features

- Dynamic grid spacing based on market volatility
- Risk-based position sizing
- Stop loss and take profit management
- Multiple trading modes (Real, Paper, Recommendation)
- Real-time market data integration

### Components

- `GridHedgeBot`: Main strategy implementation
- `GridHedgeConfig`: Configuration and parameter calculation
- `IBroker`: Interface for order execution
- `ExchangeDataProvider`: Real-time market data

### Usage

```python
from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider
from alpha_pulse.exchanges import ExchangeType
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot

# Initialize components
data_provider = ExchangeDataProvider(
    exchange_type=ExchangeType.BINANCE,
    testnet=True
)
await data_provider.initialize()

# Create broker (Paper, Real, or Recommendation)
broker = create_broker(trading_mode=TradingMode.PAPER)

# Create and run grid bot
bot = await GridHedgeBot.create_for_spot_hedge(
    broker=broker,
    symbol="BTCUSDT",
    data_provider=data_provider,
    volatility=0.02,  # 2% daily volatility
    spot_quantity=1.0  # Size to hedge
)

# Execute strategy
current_price = await data_provider.get_current_price("BTCUSDT")
bot.execute(current_price)
```

### Configuration

The grid strategy can be configured with:

- Grid spacing: Calculated from market volatility
- Position sizing: Based on risk metrics and spot position
- Stop loss: Default 4% (2 standard deviations)
- Take profit: Default 6% (1.5x stop loss)
- Number of levels: Default 5

Example configuration in `grid_hedge_example.yaml`:

```yaml
symbol: BTCUSDT
trading_mode: PAPER  # REAL, PAPER, or RECOMMENDATION
grid:
  volatility: 0.02  # 2% daily volatility
  num_levels: 5
  direction: SHORT  # For hedging spot
risk:
  stop_loss_pct: 0.04  # 4% stop loss
  take_profit_pct: 0.06  # 6% take profit
```

### Trading Modes

1. **REAL Trading**
   - Connects to exchange APIs
   - Places actual orders
   - Requires API credentials
   ```bash
   export EXCHANGE_API_KEY="your-key"
   export EXCHANGE_API_SECRET="your-secret"
   ```

2. **PAPER Trading**
   - Simulates trading with real market data
   - Useful for testing strategies
   - No real orders placed

3. **RECOMMENDATION Mode**
   - Logs potential trades without execution
   - Useful for monitoring strategy signals

### Risk Management

The strategy includes several risk management features:

1. **Grid Parameters**
   - Spacing based on volatility
   - Position sizing based on risk metrics
   - Maximum position limits

2. **Stop Loss**
   - Default: 4% from entry (2σ)
   - Automatically placed and monitored
   - Cancels all grid orders when triggered

3. **Take Profit**
   - Default: 6% from entry (1.5x stop loss)
   - Automatically placed and monitored
   - Cancels all grid orders when triggered

4. **Position Monitoring**
   - Tracks spot and futures positions
   - Ensures hedge ratio stays within limits
   - Rebalances grid as needed

### Example

Running the grid hedge demo:

```bash
# Paper trading with real market data
python -m alpha_pulse.examples.demo_grid_hedge_integration

# Real trading (requires API keys)
python -m alpha_pulse.examples.demo_grid_hedge_integration --trading-mode REAL

# Recommendation mode
python -m alpha_pulse.examples.demo_grid_hedge_integration --trading-mode RECOMMENDATION
```

### Logging

The strategy uses loguru for comprehensive logging:

```
2025-02-03 11:11:25.255 | INFO | Creating hedge grid for BTCUSDT - 
  Spot: 1.00000000, 
  Price: 95203.05, 
  Grid Spacing: 952.03, 
  Position Step: 0.20000000, 
  Stop Loss: 4.00%, 
  Take Profit: 6.00%
```

Logs are stored in the `logs/` directory with rotation enabled.