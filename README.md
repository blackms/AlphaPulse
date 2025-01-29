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

## 🏗️ Project Structure

```
AlphaPulse/
├── src/
│   ├── config/          # Configuration management
│   ├── data_pipeline/   # Core data processing modules
│   ├── features/        # Feature engineering components
│   ├── models/          # ML model training and evaluation
│   ├── backtesting/     # Strategy backtesting framework
│   ├── examples/        # Usage examples and demos
│   └── tests/           # Test suite
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

#### �️ Available Algorithms
- **PPO**: Proximal Policy Optimization (recommended)
- **A2C**: Advantage Actor-Critic
- **DQN**: Deep Q-Network

#### 📈 Environment Features
- Realistic market simulation with transaction costs
- Customizable state space and reward function
- Support for both price and feature data
- Integration with existing backtesting metrics

Check out `src/examples/demo_rl_trading.py` for a complete demonstration of:
- Setting up the trading environment
- Training an RL agent
- Evaluating performance on test data
- Visualizing trading decisions

#### 📦 Dependencies
To use the RL module, install additional dependencies:
```bash
pip install stable-baselines3 gymnasium
```

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

## 📝 Usage

```python
from alpha_pulse.data_pipeline import Exchange
from alpha_pulse.features import FeatureStore
from alpha_pulse.models import ModelTrainer
from alpha_pulse.backtesting import Backtester, DefaultStrategy

# Initialize components
exchange = Exchange()
feature_store = FeatureStore()
trainer = ModelTrainer()
backtester = Backtester()

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