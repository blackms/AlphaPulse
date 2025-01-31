# AlphaPulse Trading System

A modular cryptocurrency trading system with support for backtesting, paper trading, and live execution.

## Features

- Data Pipeline: Fetch and store market data from multiple exchanges
- Feature Engineering: Calculate technical indicators and generate training features
- Model Training: Train and evaluate machine learning models
- Backtesting: Test strategies on historical data
- Paper Trading: Simulate trading with real-time market data
- Risk Management: Advanced position sizing and portfolio optimization
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

### 2. Multi-Asset Trading with Risk Management

Run the multi-asset trading system with advanced risk management:

```bash
# Using default settings (3 assets)
python src/alpha_pulse/examples/demo_multi_asset_risk.py

# Custom configuration
python src/alpha_pulse/examples/demo_multi_asset_risk.py \
    --symbols BTC/USDT ETH/USDT BNB/USDT SOL/USDT \
    --exchange binance \
    --interval 60
```

Command line arguments:
- `--symbols`: Trading symbols (space-separated)
- `--exchange`: Exchange ID (default: binance)
- `--api-key`: Exchange API key
- `--api-secret`: Exchange API secret
- `--interval`: Update interval in seconds (default: 60)

### 3. Paper Trading

Run paper trading simulation with real-time market data:

```bash
python src/alpha_pulse/examples/demo_paper_trading.py
```

### 4. Backtesting

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
│       ├── execution/       # Order execution
│       ├── risk_management/ # Risk and portfolio management
│       │   ├── interfaces.py     # Abstract interfaces
│       │   ├── position_sizing.py # Position sizing strategies
│       │   ├── analysis.py       # Risk metrics calculation
│       │   ├── portfolio.py      # Portfolio optimization
│       │   └── manager.py        # Risk management system
│       ├── config/          # Configuration settings
│       ├── examples/        # Example scripts
│       └── tests/           # Unit tests
├── trained_models/          # Saved model files
├── feature_cache/          # Cached feature data
└── logs/                   # Application logs
```

## Risk Management & Portfolio Optimization

The system includes comprehensive risk management and portfolio optimization features:

### Position Sizing Strategies

1. Kelly Criterion:
   - Optimal position sizing based on win rate and profit ratio
   - Adjusts for volatility and signal strength
   - Configurable fraction for conservative sizing

2. Volatility-Based:
   - Position sizing based on asset volatility
   - Targets specific portfolio risk level
   - Adapts to changing market conditions

3. Adaptive Strategy:
   - Combines multiple sizing approaches
   - Weights strategies based on market conditions
   - Self-adjusting based on performance

### Risk Analysis

1. Risk Metrics:
   - Value at Risk (VaR)
   - Expected Shortfall (CVaR)
   - Maximum Drawdown
   - Sharpe & Sortino Ratios
   - Volatility Analysis

2. Real-time Monitoring:
   - Rolling risk metrics
   - Drawdown tracking
   - Position exposure analysis

### Portfolio Optimization

1. Mean-Variance Optimization:
   - Modern Portfolio Theory implementation
   - Efficient frontier calculation
   - Risk-adjusted return optimization

2. Risk Parity:
   - Equal risk contribution approach
   - Volatility targeting
   - Dynamic rebalancing

3. Adaptive Portfolio Management:
   - Regime-based strategy selection
   - Dynamic asset allocation
   - Risk-based rebalancing

### Configuration

Risk management parameters can be configured through `RiskConfig`:

```python
from alpha_pulse.risk_management import RiskConfig

config = RiskConfig(
    max_position_size=0.2,      # Maximum 20% per position
    max_portfolio_leverage=1.5,  # Maximum 150% portfolio leverage
    max_drawdown=0.25,          # Maximum 25% drawdown
    stop_loss=0.1,              # 10% stop-loss per position
    target_volatility=0.15,     # Target 15% annual volatility
    var_confidence=0.95,        # 95% VaR confidence level
)
```

### Usage Example

```python
from alpha_pulse.risk_management import RiskManager, RiskConfig

# Initialize risk management system
risk_manager = RiskManager(
    config=RiskConfig(
        max_position_size=0.2,
        max_drawdown=0.25,
    )
)

# Calculate position size
size_result = risk_manager.calculate_position_size(
    symbol="BTC/USDT",
    current_price=50000.0,
    signal_strength=0.8,
)

# Evaluate trade
if risk_manager.evaluate_trade(
    symbol="BTC/USDT",
    side="buy",
    quantity=size_result.size,
    current_price=50000.0,
    portfolio_value=100000.0,
    current_positions=positions
):
    # Execute trade
    execute_trade(...)
```

## Development

### Running Tests

```bash
# Run all tests
pytest src/alpha_pulse/tests/

# Run specific test module
pytest src/alpha_pulse/tests/test_risk_management.py
```

### Adding New Features

1. Create feature branch
2. Add tests
3. Implement feature
4. Run tests
5. Submit pull request

## License

MIT License - see LICENSE file for details