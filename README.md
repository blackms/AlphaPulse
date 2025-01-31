# AlphaPulse 📈 

A powerful and efficient trading data pipeline system for collecting, processing, and analyzing financial market data.

## 🌟 Features

- 🔄 Real-time data fetching from multiple exchanges
- 💾 Efficient database management and storage
- 🔍 Comprehensive testing suite
- ⚙️ Flexible configuration system
- 🚀 High-performance data processing
- 📊 Advanced feature engineering and ML pipeline
- 🤖 Machine learning model training and evaluation
- 📈 Backtesting and strategy evaluation framework
- 💹 Live and paper trading execution system

## 🏗️ Project Structure

```
AlphaPulse/
├── src/
│   ├── alpha_pulse/
│   │   ├── backtesting/    # Strategy backtesting framework
│   │   ├── config/         # Configuration management
│   │   ├── data_pipeline/  # Core data processing modules
│   │   ├── examples/       # Usage examples and demos
│   │   ├── execution/      # Live/paper trading system
│   │   ├── features/       # Feature engineering components
│   │   ├── models/         # ML model training and evaluation
│   │   ├── rl/            # Reinforcement Learning module
│   │   └── tests/         # Test suite
│   └── scripts/           # Utility scripts
├── feature_cache/         # Cache for computed features
├── logs/                  # Application logs
└── trained_models/       # Saved model artifacts
```

### 📦 Core Modules

- **data_fetcher.py**: Handles real-time market data collection
- **database.py**: Manages data storage and retrieval operations
- **exchange.py**: Implements exchange connectivity and interactions
- **models.py**: Defines data models and structures

### 🧮 Feature Engineering & ML Pipeline

The feature engineering and machine learning pipeline provides powerful tools for analyzing and predicting market movements:

#### 📊 Feature Engineering (`src/features/`)
- Technical indicators (EMA, SMA, RSI, MACD, Bollinger Bands)
- Rolling window statistics
- Feature caching and management
- Extensible FeatureStore system

```python
from alpha_pulse.features import FeatureStore

# Initialize feature store
feature_store = FeatureStore(cache_dir='feature_cache')

# Compute technical indicators
features = feature_store.compute_technical_indicators(
    price_data,
    windows=[12, 26, 50, 200]
)
```

#### 🤖 Model Training (`src/models/`)
- Support for multiple ML models (RandomForest, XGBoost)
- Model training and evaluation
- Cross-validation capabilities
- Model persistence and loading

```python
from alpha_pulse.models import ModelTrainer

# Initialize and train model
trainer = ModelTrainer(
    model_type='random_forest',
    task='regression'
)
metrics = trainer.train(features, target)
```

### 📊 Backtesting & Strategy Evaluation

The backtesting framework allows you to evaluate trading strategies using historical data and model predictions:

#### 🔧 Core Components (`src/backtesting/`)
- Flexible backtesting engine
- Customizable trading strategies
- Performance metrics calculation
- Trade visualization tools

```python
from alpha_pulse.backtesting import Backtester, DefaultStrategy

# Initialize backtester
backtester = Backtester(
    commission=0.001,  # 0.1% commission
    initial_capital=100000,
    position_size=0.1  # Risk 10% of capital per trade
)

# Run backtest
results = backtester.backtest(
    prices=historical_prices,
    signals=model_predictions,
    strategy=DefaultStrategy(threshold=0.5)
)

print(results)  # Display performance metrics
```

#### 📈 Available Strategies
- **DefaultStrategy**: Basic long-only strategy based on signal threshold
- **TrendFollowingStrategy**: Momentum-based approach with separate entry/exit thresholds
- **MeanReversionStrategy**: Mean reversion trading with overbought/oversold levels

#### 📊 Performance Metrics
- Total return and Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Trade-by-trade analysis
- Equity curve visualization

Check out `src/examples/demo_backtesting.py` for a complete demonstration of:
- Loading historical data
- Generating trading signals
- Running backtests with different strategies
- Analyzing and visualizing results

### 🤖 Reinforcement Learning Module

The RL module provides a framework for training and evaluating RL agents for automated trading:

#### 🔧 Core Components (`src/rl/`)
- OpenAI Gym-compatible trading environment
- Integration with Stable-Baselines3 for state-of-the-art RL algorithms
- Configurable reward functions and state representations
- Training and evaluation utilities

```python
from alpha_pulse.rl import TradingEnv, RLTrainer
from alpha_pulse.rl.env import TradingEnvConfig
from alpha_pulse.rl.rl_trainer import TrainingConfig

# Configure environment
env_config = TradingEnvConfig(
    initial_capital=100000.0,
    commission=0.001,      # 0.1% commission
    position_size=0.2,     # Risk 20% of capital per trade
    window_size=10        # Use 10 time steps of history
)

# Configure training
training_config = TrainingConfig(
    total_timesteps=100000,
    learning_rate=0.0003,
    batch_size=64
)

# Initialize trainer and train agent
trainer = RLTrainer(env_config, training_config)
model = trainer.train(
    prices=historical_prices,
    features=feature_data,
    algorithm='ppo'  # Supports PPO, A2C, DQN
)

# Evaluate the trained agent
metrics = trainer.evaluate(
    model=model,
    prices=test_prices,
    features=test_features
)
print(f"Agent Performance: {metrics}")
```

### 💹 Live/Paper Trading Module

The execution module enables real-time trading with both paper and live accounts:

#### 🔧 Core Components (`src/execution/`)
- Abstract broker interface for multiple broker implementations
- Paper trading simulation with realistic order execution
- Risk management and position tracking
- Real-time PnL calculation and monitoring

```python
from alpha_pulse.execution import PaperBroker, Order, OrderSide, OrderType, RiskLimits

# Configure risk limits
risk_limits = RiskLimits(
    max_position_size=50000.0,    # Maximum position size
    max_portfolio_size=150000.0,  # Maximum portfolio value
    max_drawdown_pct=0.25,        # 25% max drawdown
    stop_loss_pct=0.10            # 10% stop loss per position
)

# Initialize paper trading broker
broker = PaperBroker(
    initial_balance=100000.0,
    risk_limits=risk_limits
)

# Place a market order
order = Order(
    symbol="BTC/USD",
    side=OrderSide.BUY,
    quantity=1.0,
    order_type=OrderType.MARKET
)
executed_order = broker.place_order(order)

# Monitor positions and performance
position = broker.get_position("BTC/USD")
portfolio_value = broker.get_portfolio_value()
print(f"Current Position: {position}")
print(f"Portfolio Value: ${portfolio_value:,.2f}")
```

#### 📈 Features
- Market and limit order support
- Position tracking and PnL calculation
- Risk management controls
- Stop-loss implementation
- Real-time market data integration

Check out `src/examples/demo_paper_trading.py` for a complete demonstration of:
- Setting up the paper trading system
- Integrating with real-time data
- Executing trades based on model signals
- Monitoring performance and positions

#### 🔒 Risk Management
- Position size limits
- Portfolio value constraints
- Maximum drawdown controls
- Per-position stop losses
- Customizable risk parameters

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AlphaPulse.git
cd AlphaPulse
```

2. Install the package:
```bash
pip install -e .
```

## ⚙️ Configuration

Configure your settings in `src/config/settings.py`. This includes:
- Exchange API credentials
- Database connection parameters
- Data fetching intervals
- Other system configurations

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
python -m pytest src/tests/
```

The test suite includes:
- Connection debugging
- Data fetcher validation
- Database operations testing
- Exchange integration testing
- Feature engineering validation
- ML model training verification
- Backtesting framework validation
- Paper trading system validation

## 📝 Usage

```python
from alpha_pulse.data_pipeline import Exchange
from alpha_pulse.features import FeatureStore
from alpha_pulse.models import ModelTrainer
from alpha_pulse.backtesting import Backtester, DefaultStrategy
from alpha_pulse.execution import PaperBroker

# Initialize components
exchange = Exchange()
feature_store = FeatureStore()
trainer = ModelTrainer()
backtester = Backtester()
paper_broker = PaperBroker()

# Fetch historical data
historical_data = exchange.fetch_historical_data(
    symbol="BTC/USD",
    timeframe="1d"
)

# Compute features and train model
features = feature_store.compute_technical_indicators(historical_data)
model = trainer.train(features, target)

# Run backtest
predictions = model.predict(features)
results = backtester.backtest(
    prices=historical_data,
    signals=predictions,
    strategy=DefaultStrategy(threshold=0.5)
)
print(f"Strategy Performance: {results}")

# Paper trade the strategy
paper_broker.update_market_data("BTC/USD", historical_data['close'].iloc[-1])
if predictions[-1] > 0.7:  # Strong buy signal
    paper_broker.place_order(Order(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=1.0,
        order_type=OrderType.MARKET
    ))
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Acknowledgments

- Thanks to all contributors who have helped shape AlphaPulse
- Special thanks to the open-source community

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

---
⭐ Don't forget to star this repository if you find it useful!