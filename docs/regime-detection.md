# Market Regime Detection with Hidden Markov Models

## Overview

The AlphaPulse Market Regime Detection system uses Hidden Markov Models (HMMs) to identify and classify market conditions into distinct regimes. This sophisticated approach enables the trading system to adapt strategies based on current market states, improving risk management and performance.

## Key Features

### 1. Multi-Factor Regime Detection
- **Volatility Analysis**: Multiple timeframe volatility measures
- **Return Characteristics**: Momentum, skewness, and kurtosis
- **Market Microstructure**: Volume patterns and liquidity indicators
- **Technical Indicators**: RSI, MACD, Bollinger Bands integration
- **Sentiment Integration**: VIX levels and put/call ratios

### 2. Advanced HMM Variants
- **Gaussian HMM**: Standard implementation with full/diagonal covariance
- **Regime-Switching GARCH**: Volatility regime modeling
- **Hierarchical HMM**: Multi-scale regime detection
- **Ensemble Methods**: Multiple model combination

### 3. Real-Time Classification
- **Online Inference**: Continuous regime monitoring
- **Confidence Estimation**: Probability-based regime assignment
- **Transition Detection**: Early warning for regime changes
- **Performance Tracking**: Accuracy and stability metrics

## Market Regime Types

### 1. Bull Market
- **Characteristics**: Low volatility, positive returns, trending
- **Optimal Strategies**: Trend following, momentum, growth investing
- **Risk Parameters**: Higher leverage allowed (1.5x)

### 2. Bear Market
- **Characteristics**: High volatility, negative returns, defensive
- **Optimal Strategies**: Short selling, volatility trading, hedging
- **Risk Parameters**: Reduced leverage (0.5x)

### 3. Sideways Market
- **Characteristics**: Moderate volatility, mean-reverting
- **Optimal Strategies**: Range trading, mean reversion, arbitrage
- **Risk Parameters**: Standard leverage (1.0x)

### 4. Crisis Market
- **Characteristics**: Extreme volatility, large drawdowns
- **Optimal Strategies**: Cash positions, safe haven assets
- **Risk Parameters**: Minimal leverage (0.2x)

### 5. Recovery Market
- **Characteristics**: Declining volatility, improving returns
- **Optimal Strategies**: Value investing, gradual accumulation
- **Risk Parameters**: Moderate leverage (1.2x)

## Architecture

### Feature Engineering Pipeline
```python
# Feature extraction example
from alpha_pulse.ml.regime.regime_features import RegimeFeatureEngineer

engineer = RegimeFeatureEngineer()
features = engineer.extract_features(market_data, additional_data={
    'vix': vix_data,
    'sentiment': sentiment_data
})
```

### HMM Training
```python
# Train regime detection model
from alpha_pulse.models.market_regime_hmm import MarketRegimeHMM

regime_model = MarketRegimeHMM()
regime_model.fit(historical_data)
```

### Real-Time Detection
```python
# Detect current regime
regime_info = regime_model.predict_regime(current_data)
print(f"Current regime: {regime_info.regime_type.value}")
print(f"Confidence: {regime_info.confidence:.2%}")
```

## Integration with Trading System

### 1. Dynamic Risk Management
- Position sizing adjusted based on regime
- Stop-loss levels adapted to regime volatility
- Leverage constraints per regime type

### 2. Strategy Selection
- Automatic strategy switching based on regime
- Regime-specific parameter optimization
- Ensemble weight adjustment

### 3. Portfolio Optimization
- Regime-conditioned asset allocation
- Dynamic hedging based on transition probabilities
- Risk budgeting per regime state

## Model Optimization

### Hyperparameter Tuning
```python
from alpha_pulse.utils.hmm_optimization import HMMOptimizer

optimizer = HMMOptimizer()
best_params = optimizer.optimize_hyperparameters(
    features,
    n_trials=100,
    use_optuna=True
)
```

### Model Selection
- Automatic selection of optimal number of states
- Comparison of different HMM variants
- Cross-validation for robust performance

## Monitoring and Alerts

### Prometheus Metrics
- `regime_classifications_total`: Total classifications by regime
- `regime_transitions_total`: Transition counts between regimes
- `current_market_regime`: Current regime identifier
- `regime_confidence`: Classification confidence score

### Alert Conditions
- Regime transitions
- Low confidence classifications
- High transition probabilities
- Model performance degradation

## Performance Analysis

### Regime Statistics
- Duration analysis per regime
- Transition probability matrices
- Historical regime distribution
- Economic value of regime timing

### Backtesting Integration
```python
# Backtest with regime awareness
results = backtest_with_regimes(
    strategy,
    historical_data,
    regime_model
)
```

## Configuration

### Basic Configuration
```yaml
regime_detection:
  n_states: 5
  update_interval: 60  # minutes
  min_confidence: 0.6
  features:
    volatility_windows: [5, 10, 20, 60]
    return_windows: [1, 5, 20, 60]
    use_vix: true
    use_sentiment: true
```

### Advanced Settings
```yaml
hmm_config:
  covariance_type: "full"
  init_method: "kmeans"
  n_iter: 100
  transition_penalty: 0.01
  
optimization:
  use_optuna: true
  n_trials: 100
  cv_splits: 5
```

## Best Practices

1. **Data Requirements**
   - Minimum 2 years of historical data for training
   - Include multiple market cycles
   - Ensure data quality and consistency

2. **Feature Selection**
   - Use domain knowledge for feature engineering
   - Validate feature importance regularly
   - Consider regime-specific features

3. **Model Validation**
   - Out-of-sample testing crucial
   - Monitor regime stability
   - Track false transition rates

4. **Production Deployment**
   - Regular model retraining (monthly)
   - A/B testing for strategy changes
   - Gradual position adjustments on transitions

## Troubleshooting

### Common Issues

1. **Too Many Regime Changes**
   - Increase transition penalty
   - Use longer confirmation windows
   - Check feature normalization

2. **Low Confidence Classifications**
   - Review feature quality
   - Consider fewer regime states
   - Check for data anomalies

3. **Poor Out-of-Sample Performance**
   - Avoid overfitting with regularization
   - Use proper cross-validation
   - Consider simpler models

## Future Enhancements

1. **Deep Learning Integration**
   - LSTM for sequence modeling
   - Attention mechanisms for feature selection
   - Neural HMMs for complex patterns

2. **Multi-Asset Regimes**
   - Cross-asset regime correlation
   - Sector-specific regime detection
   - Global macro regime integration

3. **Adaptive Learning**
   - Online HMM parameter updates
   - Regime definition evolution
   - Transfer learning across markets