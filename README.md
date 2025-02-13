# AlphaPulse 🚀

A comprehensive cryptocurrency and stock trading system with AI-powered portfolio management.

## Features ✨

### AI Hedge Fund System 🧠
- 🤖 Multi-agent trading system with specialized strategies:
  - Activist investing (Bill Ackman strategy)
  - Value investing (Warren Buffett strategy)
  - Fundamental analysis
  - Sentiment analysis
  - Technical analysis
  - Valuation analysis
- 📊 Intelligent signal aggregation
- ⚖️ Risk-aware position sizing
- 📈 Performance tracking and adaptation

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

### AI Hedge Fund

```python
from alpha_pulse.agents.manager import AgentManager
from alpha_pulse.data_pipeline.managers.mock_data import MockDataManager

# Initialize agent manager
manager = AgentManager()
await manager.initialize()

# Load market data
data_manager = MockDataManager()
market_data = await data_manager.get_market_data(symbols=["AAPL", "MSFT", "GOOGL"])

# Generate trading signals
signals = await manager.generate_signals(market_data)

# Get agent performance
performance = manager.get_agent_performance()
```

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

### AI Hedge Fund

Create a configuration file at `config/ai_hedge_fund_config.yaml`:

```yaml
agents:
  agent_weights:
    activist: 0.15
    value: 0.20
    fundamental: 0.20
    sentiment: 0.15
    technical: 0.15
    valuation: 0.15

risk:
  max_position_size: 0.20
  max_portfolio_leverage: 1.5
  max_drawdown: 0.25
  stop_loss: 0.10

execution:
  mode: paper
  initial_balance: 1000000
  slippage: 0.001
  fee_rate: 0.001
```

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

- `demo_ai_hedge_fund.py`: Multi-agent trading system
- `demo_portfolio_rebalancing.py`: Portfolio optimization
- `demo_grid_hedge_integration.py`: Grid hedging strategy
- `demo_feature_engineering.py`: Feature calculation
- `demo_model_training.py`: ML model training
- `demo_rl_trading.py`: Reinforcement learning
- `demo_llm_portfolio_analysis.py`: LLM integration

## Documentation

### Technical Documentation
- [AI Hedge Fund Technical Documentation](AI_HEDGE_FUND_DOCUMENTATION.md) - Comprehensive system architecture and implementation details
- [API Documentation](API_DOCUMENTATION.md) - REST API endpoints and usage

### Module Documentation
- [AI Hedge Fund Agents](src/alpha_pulse/agents/README.md) - Trading agents and signal generation
- [Portfolio Management](src/alpha_pulse/portfolio/README.md) - Portfolio optimization strategies
- [Risk Management](src/alpha_pulse/risk_management/README.md) - Risk controls and position sizing
- [Hedging Strategies](src/alpha_pulse/hedging/README.md) - Grid hedging and futures hedging
- [Data Pipeline](src/alpha_pulse/data_pipeline/README.md) - Data ingestion and processing
- [Feature Engineering](src/alpha_pulse/features/README.md) - Technical indicators and ML features
- [Reinforcement Learning](src/alpha_pulse/rl/README.md) - RL models and training

### Additional Resources
- [Deployment Guide](DEPLOYMENT.md) - Production deployment instructions
- [Release Notes](RELEASE.md) - Version history and changes

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