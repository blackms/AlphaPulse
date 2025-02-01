# 🚀 AlphaPulse Trading System

A modular cryptocurrency trading system with support for backtesting, paper trading, and live execution.

## ✨ Features

- 📊 Data Pipeline: Fetch and store market data from multiple exchanges
- 🧮 Feature Engineering: Calculate technical indicators and generate training features
- 🤖 Model Training: Train and evaluate machine learning models
- 📈 Backtesting: Test strategies on historical data
- 🎮 Paper Trading: Simulate trading with real-time market data
- 🛡️ Risk Management: Advanced position sizing and portfolio optimization
- 💹 Live Trading: Execute trades on supported exchanges (coming soon)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaPulse.git
cd AlphaPulse

# Install dependencies
pip install -e .
```

## 🔐 Environment Setup

Create a `.env` file in the root directory with your API keys:

```env
# OpenAI API key for LLM-based portfolio analysis
OPENAI_API_KEY=your_openai_api_key

# Bybit exchange credentials
ALPHA_PULSE_BYBIT_API_KEY=your_bybit_api_key
ALPHA_PULSE_BYBIT_API_SECRET=your_bybit_api_secret
ALPHA_PULSE_BYBIT_TESTNET=false  # Set to true for testnet
```

> ⚠️ **Important**: Never commit your `.env` file to version control. It's already added to `.gitignore` to prevent accidental commits.

## 🎯 Usage

### 1. 🧠 Train a Model

Train a prediction model using historical data:

```bash
python src/alpha_pulse/examples/train_test_model.py
```

This will:
- Generate sample training data
- Calculate technical indicators
- Train a RandomForest model
- Save the model to trained_models/crypto_prediction_model.joblib

### 2. 📊 Multi-Asset Trading with Risk Management

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

### 3. 🎯 Paper Trading

Run paper trading simulation with real-time market data:

```bash
python src/alpha_pulse/examples/demo_paper_trading.py
```

### 4. 💼 Portfolio Rebalancing

Run portfolio analysis and rebalancing:

```bash
python src/alpha_pulse/examples/demo_portfolio_rebalancing.py \
    --exchange binance \  # or bybit
    --api-key your_api_key \
    --api-secret your_api_secret \
    --testnet  # Optional: use testnet for testing
```

This will:
- Connect to the exchange
- Analyze current portfolio allocation
- Calculate optimal allocation using MPT and HRP
- Generate rebalancing recommendations
- Plot current vs target allocations

### 5. 📊 Backtesting

Run backtesting on historical data:

```bash
python src/alpha_pulse/examples/demo_backtesting.py
```

## 📁 Project Structure

```
AlphaPulse/
├── src/
│   └── alpha_pulse/
│       ├── data_pipeline/    # Data fetching and storage
│       ├── features/         # Feature engineering
│       ├── models/          # ML model implementations
│       ├── backtesting/     # Backtesting engine
│       ├── execution/       # Order execution
│       ├── exchanges/       # Exchange integrations
│       │   ├── base.py          # Base exchange interface
│       │   ├── binance.py       # Binance implementation
│       │   ├── bybit.py         # Bybit implementation
│       │   └── credentials/     # Secure credential management
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

## 🛡️ Risk Management & Portfolio Optimization

The system includes comprehensive risk management and portfolio optimization features:

### 📊 Position Sizing Strategies

1. 🎯 Kelly Criterion:
   - Optimal position sizing based on win rate and profit ratio
   - Adjusts for volatility and signal strength
   - Configurable fraction for conservative sizing

2. 📈 Volatility-Based:
   - Position sizing based on asset volatility
   - Targets specific portfolio risk level
   - Adapts to changing market conditions

3. 🔄 Adaptive Strategy:
   - Combines multiple sizing approaches
   - Weights strategies based on market conditions
   - Self-adjusting based on performance

### 📊 Risk Analysis

1. 📉 Risk Metrics:
   - Value at Risk (VaR)
   - Expected Shortfall (CVaR)
   - Maximum Drawdown
   - Sharpe & Sortino Ratios
   - Volatility Analysis

2. 🔍 Real-time Monitoring:
   - Rolling risk metrics
   - Drawdown tracking
   - Position exposure analysis

### 💼 Portfolio Optimization & Rebalancing

1. 📈 Modern Portfolio Theory (MPT):
   - Mean-variance optimization
   - Efficient frontier calculation
   - Risk-adjusted return optimization
   - Sharpe ratio maximization
   - Support for custom constraints

2. 🌳 Hierarchical Risk Parity (HRP):
   - Clustering-based portfolio optimization
   - More robust to estimation errors
   - Better numerical stability
   - Handles high-dimensional portfolios
   - No expected returns estimation needed

3. 📊 Portfolio Analysis:
   - Current allocation analysis
   - Rebalancing score calculation
   - Trade size optimization
   - Transaction cost consideration
   - Visual allocation comparison

4. 🤖 LLM-Based Portfolio Analysis:
    - AI-powered portfolio insights
    - Risk assessment and recommendations
    - Custom rebalancing suggestions
    - Confidence scoring
    - Detailed reasoning for decisions

5. 🔄 Exchange Integration:
    - Real-time balance tracking
    - Multi-exchange support (Binance, Bybit)
    - Automated rebalancing suggestions
    - Support for spot and futures
    - Secure API key management

Example usage with LLM Portfolio Analysis:

```python
from alpha_pulse.portfolio.llm_analysis import OpenAILLMAnalyzer
from alpha_pulse.portfolio.portfolio_manager import PortfolioManager

import os
from dotenv import load_dotenv

async def analyze_portfolio():
    # Load environment variables from .env file
    load_dotenv()
    # Initialize LLM analyzer with API key from environment
    analyzer = OpenAILLMAnalyzer(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4",  # or other available models
    )
    
    # Initialize portfolio manager
    manager = PortfolioManager("path/to/portfolio_config.yaml")
    
    # Get AI-powered analysis
    analysis = await manager.analyze_portfolio_with_llm(
        analyzer=analyzer,
        exchange=exchange
    )
    
    print("Recommendations:", analysis.recommendations)
    print("Risk Assessment:", analysis.risk_assessment)
    if analysis.rebalancing_suggestions:
        print("Suggested Rebalancing:")
        for suggestion in analysis.rebalancing_suggestions:
            print(f"- {suggestion['asset']}: {suggestion['target_allocation']:.2%}")
    print(f"Confidence Score: {analysis.confidence_score:.2%}")
    print("Reasoning:", analysis.reasoning)
```

Example usage with the Portfolio Analyzer:

```python
from alpha_pulse.exchanges import ExchangeType, ExchangeFactory
from alpha_pulse.portfolio.analyzer import PortfolioAnalyzer
from alpha_pulse.portfolio.hrp_strategy import HRPStrategy

async def optimize_portfolio():
    # Initialize exchange
    exchange = await ExchangeFactory.create_exchange(
        exchange_type=ExchangeType.BYBIT,
        testnet=False
    )
    
    # Create portfolio analyzer with HRP strategy
    analyzer = PortfolioAnalyzer(
        exchange=exchange,
        strategy=HRPStrategy(),
        lookback_days=30
    )
    
    # Get optimal allocation
    result = await analyzer.analyze_portfolio(
        constraints={
            'min_weight': 0.05,  # Minimum 5% per asset
            'max_weight': 0.40   # Maximum 40% per asset
        }
    )
    
    # Get rebalancing trades
    trades = analyzer.get_rebalancing_trades(
        current_weights=await analyzer.get_current_allocation(),
        target_weights=result.weights,
        total_value=100000,
        min_trade_value=10
    )
    
    return result, trades
```

### ⚙️ Configuration

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

### 🔍 Usage Example

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

## 🛠️ Development

### 🧪 Running Tests

```bash
# Run all tests
pytest src/alpha_pulse/tests/

# Run specific test module
pytest src/alpha_pulse/tests/test_risk_management.py
```

### ✨ Adding New Features

1. Create feature branch
2. Add tests
3. Implement feature
4. Run tests
5. Submit pull request

## 📄 License

GNU Affero General Public License v3.0 (AGPL-3.0) - see LICENSE file for details

This means:
- 🔓 Source code must be made available when distributing the software
- 🔄 Modifications must be released under the same license
- 💼 Cannot be used for commercial purposes without explicit permission
- 🌐 Network use is considered distribution (must share modifications)
- 📝 Changes must be documented and dated