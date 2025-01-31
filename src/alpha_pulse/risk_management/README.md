# Risk Management Module

The risk management module provides comprehensive tools for position sizing, portfolio optimization, and risk analysis in algorithmic trading systems.

## Overview

The module consists of several key components:

1. Position Sizing (`position_sizing.py`)
2. Risk Analysis (`analysis.py`)
3. Portfolio Optimization (`portfolio.py`)
4. Risk Management System (`manager.py`)

Each component is designed to be modular and can be used independently or as part of the complete risk management system.

## Components

### 1. Position Sizing

Three position sizing strategies are available:

#### Kelly Criterion Sizer
```python
from alpha_pulse.risk_management import KellyCriterionSizer

sizer = KellyCriterionSizer(
    lookback_periods=100,  # Historical periods for win rate calculation
    min_trades=20,         # Minimum trades required
    max_size_pct=0.2      # Maximum position size (20% of portfolio)
)
```

Features:
- Calculates optimal position size based on win rate and profit ratio
- Adapts to historical performance
- Adjusts for volatility and signal strength

#### Volatility-Based Sizer
```python
from alpha_pulse.risk_management import VolatilityBasedSizer

sizer = VolatilityBasedSizer(
    target_volatility=0.01,  # Target daily volatility
    max_size_pct=0.2,        # Maximum position size
    volatility_lookback=20    # Periods for volatility calculation
)
```

Features:
- Sizes positions based on asset volatility
- Targets specific portfolio risk level
- Includes minimum and maximum size constraints

#### Adaptive Position Sizer
```python
from alpha_pulse.risk_management import AdaptivePositionSizer

sizer = AdaptivePositionSizer(
    kelly_sizer=KellyCriterionSizer(),
    vol_sizer=VolatilityBasedSizer(),
    max_size_pct=0.2
)
```

Features:
- Combines multiple sizing strategies
- Weights strategies based on confidence levels
- Adapts to market conditions

### 2. Risk Analysis

The risk analysis component calculates various risk metrics:

```python
from alpha_pulse.risk_management import RiskAnalyzer

analyzer = RiskAnalyzer(
    rolling_window=252,     # One year of daily data
    var_confidence=0.95,    # 95% VaR confidence level
    monte_carlo_sims=10000  # Simulations for Monte Carlo VaR
)

# Calculate risk metrics
metrics = analyzer.calculate_metrics(returns)
print(f"Volatility: {metrics.volatility:.2%}")
print(f"VaR (95%): {metrics.var_95:.2%}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

Features:
- Value at Risk (VaR) calculation methods:
  * Historical VaR
  * Parametric VaR
  * Monte Carlo VaR
- Drawdown analysis
- Rolling metrics calculation
- Performance ratios:
  * Sharpe Ratio
  * Sortino Ratio
  * Calmar Ratio

### 3. Portfolio Optimization

Multiple portfolio optimization strategies are available:

#### Mean-Variance Optimizer
```python
from alpha_pulse.risk_management import MeanVarianceOptimizer

optimizer = MeanVarianceOptimizer(
    target_return=0.10,    # Target annual return
    risk_aversion=1.0      # Risk aversion parameter
)

weights = optimizer.optimize(
    returns,
    risk_free_rate=0.02,
    constraints=PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.3,
        max_total_weight=1.0
    )
)
```

Features:
- Modern Portfolio Theory implementation
- Efficient frontier optimization
- Multiple objective functions:
  * Minimum risk
  * Maximum Sharpe ratio
  * Maximum utility

#### Risk Parity Optimizer
```python
from alpha_pulse.risk_management import RiskParityOptimizer

optimizer = RiskParityOptimizer(
    target_risk=0.15  # Target portfolio volatility
)

weights = optimizer.optimize(returns)
```

Features:
- Equal risk contribution approach
- Volatility targeting
- Risk-based asset allocation

#### Adaptive Portfolio Optimizer
```python
from alpha_pulse.risk_management import AdaptivePortfolioOptimizer

optimizer = AdaptivePortfolioOptimizer(
    volatility_threshold=0.2  # Threshold for strategy switching
)
```

Features:
- Regime-based strategy selection
- Automatic adaptation to market conditions
- Combines multiple optimization approaches

### 4. Risk Management System

The complete risk management system integrates all components:

```python
from alpha_pulse.risk_management import RiskManager, RiskConfig

manager = RiskManager(
    config=RiskConfig(
        max_position_size=0.2,        # 20% max per position
        max_portfolio_leverage=1.5,    # 150% max leverage
        max_drawdown=0.25,            # 25% max drawdown
        stop_loss=0.1,                # 10% stop-loss
        target_volatility=0.15,       # 15% target volatility
        rebalance_threshold=0.1       # 10% weight deviation for rebalance
    )
)

# Evaluate trade
if manager.evaluate_trade(
    symbol="BTC/USD",
    side="buy",
    quantity=1.0,
    current_price=50000.0,
    portfolio_value=100000.0,
    current_positions=positions
):
    # Execute trade
    execute_trade(...)

# Update risk metrics
manager.update_risk_metrics(portfolio_returns, asset_returns)

# Get risk report
report = manager.get_risk_report()
```

Features:
- Comprehensive risk management
- Position sizing and portfolio optimization
- Real-time risk monitoring
- Stop-loss management
- Portfolio rebalancing

## Usage in Multi-Asset Trading

See `examples/demo_multi_asset_risk.py` for a complete example of using the risk management system in a multi-asset trading context:

```bash
# Run demo with default settings
python src/alpha_pulse/examples/demo_multi_asset_risk.py

# Run with custom symbols and parameters
python src/alpha_pulse/examples/demo_multi_asset_risk.py \
    --symbols BTC/USDT ETH/USDT BNB/USDT \
    --exchange binance \
    --interval 60
```

## Testing

The module includes comprehensive unit tests:

```bash
# Run risk management tests
pytest src/alpha_pulse/tests/test_risk_management.py
```

Key test areas:
- Risk metrics calculation
- Position sizing logic
- Portfolio optimization
- Trade evaluation
- Stop-loss calculation

## Dependencies

- numpy: Numerical computations
- pandas: Data manipulation and analysis
- scipy: Optimization algorithms
- loguru: Logging

## Best Practices

1. Position Sizing:
   - Start with conservative position sizes
   - Use the adaptive sizer for most cases
   - Monitor and adjust parameters based on performance

2. Risk Analysis:
   - Calculate metrics on different timeframes
   - Use rolling windows for more stable estimates
   - Consider multiple VaR methods

3. Portfolio Optimization:
   - Start with risk parity in high volatility
   - Use mean-variance in stable markets
   - Regularly rebalance portfolio weights

4. Risk Management:
   - Set conservative risk limits initially
   - Monitor drawdown and leverage closely
   - Use stop-losses consistently