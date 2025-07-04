# Ensemble Methods for Trading Signal Combination

## Overview

The AlphaPulse ensemble framework provides sophisticated methods for combining signals from multiple trading agents to create more robust and accurate trading decisions. This system implements state-of-the-art machine learning ensemble techniques adapted specifically for financial markets.

## Architecture

### Core Components

1. **Ensemble Manager** (`ensemble_manager.py`)
   - Manages multiple ensemble methods
   - Handles agent lifecycle (registration, activation, retirement)
   - Coordinates signal collection and combination
   - Tracks performance metrics

2. **Voting Methods** (`voting_classifiers.py`)
   - Hard Voting: Majority vote with tie-breaking strategies
   - Soft Voting: Probability-weighted averaging with Bayesian methods
   - Weighted Majority: Consensus-based decisions with veto mechanisms

3. **Stacking Methods** (`stacking_methods.py`)
   - Basic Stacking: Meta-learning with various algorithms
   - Hierarchical Stacking: Multi-level ensemble learning
   - Support for XGBoost, LightGBM, Neural Networks as meta-learners

4. **Boosting Algorithms** (`boosting_algorithms.py`)
   - Adaptive Boosting: Sequential learning with error correction
   - Gradient Boosting: Refined predictions through residual learning
   - XGBoost/LightGBM: High-performance gradient boosting
   - Online Boosting: Real-time model updates

5. **Signal Aggregation** (`signal_aggregation.py`)
   - Multiple aggregation methods (mean, median, trimmed mean, etc.)
   - Temporal aggregation with decay
   - Consensus mechanisms with quorum requirements

## Key Features

### Adaptive Weighting
- Performance-based weight updates
- Diversity-based optimization
- Market regime-dependent adjustments
- Real-time adaptation to changing conditions

### Meta-Learning
- Automated ensemble architecture search
- Cross-validation for robust training
- Feature engineering for enhanced predictions
- Transfer learning across markets

### Signal Quality Control
- Outlier detection and handling
- Confidence-weighted combinations
- Diversity maximization
- Signal stability monitoring

### Performance Monitoring
- Real-time performance tracking
- Individual agent contribution analysis
- Ensemble diversity metrics
- Alert system for performance degradation

## Usage Examples

### Creating an Ensemble

```python
from alpha_pulse.services.ensemble_service import EnsembleService

# Initialize service
service = EnsembleService(db_session)

# Create voting ensemble
ensemble_id = service.create_ensemble(
    name="multi_agent_voter",
    ensemble_type="voting",
    config={
        "method": "soft_voting",
        "temperature": 1.0,
        "min_confidence": 0.6
    }
)

# Create stacking ensemble
stacking_id = service.create_ensemble(
    name="meta_learner",
    ensemble_type="stacking",
    config={
        "meta_model": "xgboost",
        "use_cv": True,
        "cv_folds": 5
    }
)
```

### Registering and Activating Agents

```python
# Register trading agents
technical_agent = service.register_agent(
    name="technical_analyzer",
    agent_type="technical",
    config={"indicators": ["SMA", "RSI", "MACD"]}
)

fundamental_agent = service.register_agent(
    name="fundamental_analyzer",
    agent_type="fundamental",
    config={"data_sources": ["earnings", "sentiment"]}
)

# Activate agents for ensemble
service.activate_agent(technical_agent, ensemble_id)
service.activate_agent(fundamental_agent, ensemble_id)
```

### Generating Ensemble Predictions

```python
from alpha_pulse.models.ensemble_model import AgentSignalCreate

# Collect agent signals
signals = [
    AgentSignalCreate(
        agent_id=technical_agent,
        signal=0.7,  # Strong buy signal
        confidence=0.85,
        metadata={"reason": "bullish_pattern"}
    ),
    AgentSignalCreate(
        agent_id=fundamental_agent,
        signal=0.4,  # Moderate buy signal
        confidence=0.7,
        metadata={"pe_ratio": 15.2}
    )
]

# Generate ensemble prediction
prediction = await service.generate_ensemble_prediction(ensemble_id, signals)

print(f"Ensemble Signal: {prediction.signal}")
print(f"Confidence: {prediction.confidence}")
print(f"Contributing Agents: {prediction.contributing_agents}")
```

### Optimizing Ensemble Weights

```python
# Optimize based on recent performance
optimization_result = service.optimize_ensemble_weights(
    ensemble_id=ensemble_id,
    lookback_days=30
)

print(f"Old Weights: {optimization_result.old_weights}")
print(f"New Weights: {optimization_result.new_weights}")
print(f"Expected Improvement: {optimization_result.expected_improvement}")
```

## Configuration Options

### Voting Ensembles

```yaml
voting_config:
  # Hard Voting
  vote_threshold: 0.1        # Signal threshold for discretization
  tie_breaker: "confidence"  # confidence, random, neutral
  min_votes: 3              # Minimum votes required
  
  # Soft Voting
  temperature: 1.0          # Temperature for probability scaling
  use_bayesian: true        # Enable Bayesian averaging
  entropy_weighting: true   # Weight by signal entropy
  
  # Weighted Majority
  consensus_threshold: 0.6  # Threshold for consensus
  super_majority: 0.8       # Super-majority threshold
  veto_threshold: 0.9       # Confidence for veto power
```

### Stacking Configuration

```yaml
stacking_config:
  meta_model: "xgboost"     # Meta-learner type
  use_cv: true              # Use cross-validation
  cv_folds: 5               # Number of CV folds
  blend_meta_models: true   # Blend multiple meta-models
  feature_engineering: true  # Enable feature engineering
  
  # Model-specific parameters
  xgboost_params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

### Boosting Configuration

```yaml
boosting_config:
  # Adaptive Boosting
  n_estimators: 50
  learning_rate: 1.0
  base_estimator: "decision_tree"
  
  # Online Boosting
  window_size: 1000        # Samples to keep in memory
  update_frequency: 100    # Update model every N samples
  min_samples: 50          # Minimum samples for training
```

## Performance Metrics

The ensemble framework tracks various performance metrics:

- **Accuracy Metrics**: Directional accuracy, hit rate, correlation
- **Risk Metrics**: Sharpe ratio, max drawdown, Value at Risk
- **Signal Quality**: Stability, conviction, confidence consistency
- **Ensemble Metrics**: Diversity, consensus strength, agent contributions

## Best Practices

1. **Agent Diversity**: Include agents with different strategies and timeframes
2. **Regular Retraining**: Update ensemble models periodically
3. **Performance Monitoring**: Set up alerts for performance degradation
4. **Weight Management**: Balance between stability and adaptability
5. **Backtesting**: Always validate ensemble performance on historical data

## Troubleshooting

### Low Ensemble Performance
- Check individual agent performance
- Verify sufficient signal diversity
- Ensure proper data preprocessing
- Consider adjusting ensemble parameters

### High Latency
- Reduce number of active agents
- Use lighter meta-models
- Enable parallel signal collection
- Implement caching for predictions

### Weight Instability
- Increase weight update smoothing
- Use longer performance windows
- Implement weight constraints
- Check for agent signal quality

## API Reference

See the full API documentation in the [API Reference](./api-reference.md) section.