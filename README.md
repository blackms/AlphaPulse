# AlphaPulse 🚀

A comprehensive cryptocurrency trading and portfolio management system.

## Features ✨

### Portfolio Management 💼
- 💹 Black-Litterman portfolio optimization
- 📊 Hierarchical Risk Parity (HRP) strategy
- 🏛 Modern Portfolio Theory (MPT) implementation
- 🤖 LLM-assisted portfolio analysis

### Risk Management 🔒
- 📉 Multi-asset risk analysis
- ⚖️ Position sizing optimization
- 🚦 Portfolio-level risk controls
- ⏱️ Real-time monitoring

### Hedging Strategies 🛡
- 🧮 Grid-based hedging with risk management
- 💱 Basic futures hedging
- 🔄 Position tracking and rebalancing
- 🎭 Multiple trading modes (Real/Paper/Recommendation)

### Data Pipeline 📡
- ⏱️ Real-time market data integration
- 🗄️ Historical data management
- 🛠️ Feature engineering
- 💾 Database integration

### Execution ⚡
- 🌐 Multi-exchange support (Binance, Bybit)
- 📝 Paper trading simulation
- 📈 Real-time order management
- 🛡️ Risk-aware execution

### Machine Learning 🤖
- 🔧 Feature generation
- 🎓 Model training pipeline
- 🧠 Reinforcement learning integration
- 💡 LLM-powered analysis

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AlphaPulse.git
cd AlphaPulse

# Install dependencies
pip install -e .
```

## Quick Start

### Portfolio Management

```python
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.portfolio.strategies import BlackLittermanStrategy

# Initialize portfolio manager
manager = PortfolioManager(strategy=BlackLittermanStrategy())

# Run optimization
optimal_weights = manager.optimize_portfolio(assets, returns)
```

### Grid Hedging

```python
from alpha_pulse.execution.broker_factory import create_broker, TradingMode
from alpha_pulse.hedging.grid_hedge_bot import GridHedgeBot

# Create paper trading broker
broker = create_broker(trading_mode=TradingMode.PAPER)

# Initialize grid hedging bot
bot = await GridHedgeBot.create_for_spot_hedge(
    broker=broker,
    symbol="BTCUSDT",
    volatility=0.02,  # 2% daily volatility
    spot_quantity=1.0  # Amount to hedge
)

# Run strategy
bot.execute(current_price)
```

### Data Pipeline

```python
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.features.feature_engineering import FeatureEngineer

# Fetch market data
fetcher = DataFetcher()
data = await fetcher.fetch_ohlcv("BTCUSDT", "1h")

# Generate features
engineer = FeatureEngineer()
features = engineer.calculate_features(data)
```

## Configuration

### Exchange Credentials

Create a credentials file at `src/alpha_pulse/exchanges/credentials/config.yaml`:

```yaml
binance:
  api_key: "your-api-key"
  api_secret: "your-api-secret"
  testnet: true  # Use testnet for testing

bybit:
  api_key: "your-api-key"
  api_secret: "your-api-secret"
  testnet: true
```

### Grid Hedging

Configure hedging parameters in `src/alpha_pulse/hedging/config/grid_hedge.yaml`:

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

## Examples

The `src/alpha_pulse/examples/` directory contains example scripts:

- `demo_portfolio_rebalancing.py`: Portfolio optimization
- `demo_grid_hedge_integration.py`: Grid hedging strategy
- `demo_feature_engineering.py`: Feature calculation
- `demo_model_training.py`: ML model training
- `demo_rl_trading.py`: Reinforcement learning
- `demo_llm_portfolio_analysis.py`: LLM integration

## Documentation

Detailed documentation for each module:

- [Portfolio Management](src/alpha_pulse/portfolio/README.md)
- [Risk Management](src/alpha_pulse/risk_management/README.md)
- [Hedging Strategies](src/alpha_pulse/hedging/README.md)
- [Data Pipeline](src/alpha_pulse/data_pipeline/README.md)
- [Feature Engineering](src/alpha_pulse/features/README.md)
- [Reinforcement Learning](src/alpha_pulse/rl/README.md)

## Testing

```bash
# Run all tests
python -m pytest src/alpha_pulse/tests/

# Run specific test file
python -m pytest src/alpha_pulse/tests/test_hedging.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.