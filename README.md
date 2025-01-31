# AlphaPulse Trading System

A modular cryptocurrency trading system with support for backtesting, paper trading, and live execution.

## Features

- Data Pipeline: Fetch and store market data from multiple exchanges
- Feature Engineering: Calculate technical indicators and generate training features
- Model Training: Train and evaluate machine learning models
- Backtesting: Test strategies on historical data
- Paper Trading: Simulate trading with real-time market data
- Live Trading: Execute trades on supported exchanges (coming soon)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaPulse.git
cd AlphaPulse

# Install dependencies
pip install -e .
```

## Usage

### 1. Train a Model

Train a prediction model using historical data:

```bash
python src/alpha_pulse/examples/train_test_model.py
```

This will:
- Generate sample training data
- Calculate technical indicators
- Train a RandomForest model
- Save the model to trained_models/crypto_prediction_model.joblib

### 2. Paper Trading

Run paper trading simulation with real-time market data:

```bash
# Using default settings (Binance, BTC/USDT & ETH/USDT)
python src/alpha_pulse/examples/demo_paper_trading.py

# Using custom exchange and symbols
python src/alpha_pulse/examples/demo_paper_trading.py \
    --exchange binance \
    --symbols BTC/USDT ETH/USDT \
    --balance 100000 \
    --interval 60

# Using exchange with API credentials
python src/alpha_pulse/examples/demo_paper_trading.py \
    --exchange binance \
    --api-key YOUR_API_KEY \
    --api-secret YOUR_API_SECRET \
    --symbols BTC/USDT ETH/USDT
```

Command line arguments:
- `--exchange`: Exchange ID (default: binance, supports any CCXT exchange)
- `--api-key`: Exchange API key for data access
- `--api-secret`: Exchange API secret
- `--symbols`: Trading symbols (space-separated)
- `--balance`: Initial paper trading balance (default: 100000)
- `--interval`: Update interval in seconds (default: 60)

### 3. Backtesting

Run backtesting on historical data:

```bash
python src/alpha_pulse/examples/demo_backtesting.py
```

## Project Structure

```
AlphaPulse/
├── src/
│   └── alpha_pulse/
│       ├── data_pipeline/    # Data fetching and storage
│       ├── features/         # Feature engineering
│       ├── models/          # ML model implementations
│       ├── backtesting/     # Backtesting engine
│       ├── execution/       # Order execution and risk management
│       │   ├── __init__.py
│       │   ├── broker_interface.py  # Abstract broker interface
│       │   └── paper_broker.py      # Paper trading implementation
│       ├── config/          # Configuration settings
│       ├── examples/        # Example scripts
│       └── tests/           # Unit tests
├── trained_models/          # Saved model files
├── feature_cache/          # Cached feature data
└── logs/                   # Application logs
```

## Execution Module

The execution module provides a robust framework for paper trading and live execution:

### Components

1. Broker Interface (`broker_interface.py`):
   - Abstract base class defining broker operations
   - Methods for order placement, position tracking, and risk management
   - Support for market and limit orders

2. Paper Broker (`paper_broker.py`):
   - Implementation of broker interface for paper trading
   - Simulated order execution and position tracking
   - Real-time PnL calculation
   - Risk management features:
     * Position size limits
     * Portfolio value limits
     * Stop-loss implementation
     * Maximum drawdown control

### Risk Management

The system includes comprehensive risk management features:

- Position Sizing: Maximum 20% of account per position
- Portfolio Limits: Maximum 150% portfolio value (allowing leverage)
- Stop Loss: 10% per position
- Max Drawdown: 25% portfolio value

### Trading Logic

The paper trading system:

1. Fetches real-time market data
2. Calculates technical indicators
3. Generates trading signals using ML model
4. Executes trades based on signal strength:
   - Buy when signal > 0.75
   - Sell when signal < 0.25
5. Tracks performance metrics:
   - Portfolio value
   - Position PnL
   - Trade history

## Supported Exchanges

The system uses CCXT for exchange integration and supports all exchanges available in CCXT. Some popular options:

- Binance
- Coinbase Pro
- Kraken
- FTX
- Bitfinex
- And many more...

Check [CCXT documentation](https://docs.ccxt.com/#/README) for the full list of supported exchanges.

## Configuration

Exchange credentials and other settings can be configured in multiple ways:

1. Command line arguments (recommended for testing)
2. Environment variables
3. Configuration file (settings.py)

## Development

### Running Tests

```bash
pytest src/alpha_pulse/tests/
```

### Adding New Features

1. Create feature branch
2. Add tests
3. Implement feature
4. Run tests
5. Submit pull request

## License

MIT License - see LICENSE file for details