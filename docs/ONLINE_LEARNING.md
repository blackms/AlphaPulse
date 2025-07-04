# Online Learning System Documentation

## Overview

The AlphaPulse online learning system provides real-time model adaptation capabilities for trading agents. It enables continuous learning from streaming market data with automatic drift detection and adaptation strategies.

## Architecture

### Core Components

1. **Online Learners** (`online_learner.py`)
   - Base framework for incremental learning
   - Prequential evaluation (test-then-train)
   - Automatic drift detection integration
   - Memory-efficient buffering

2. **Incremental Models** (`incremental_models.py`)
   - **IncrementalSGD**: Stochastic Gradient Descent for regression/classification
   - **IncrementalNaiveBayes**: Gaussian Naive Bayes for probabilistic classification
   - **IncrementalPassiveAggressive**: Online learning with aggressive updates
   - **HoeffdingTree**: Very Fast Decision Tree (VFDT) for streaming data
   - **AdaptiveRandomForest**: Ensemble with per-tree drift detection
   - **OnlineGradientBoosting**: Sequential boosting for streaming data

3. **Adaptive Algorithms** (`adaptive_algorithms.py`)
   - **AdaptiveLearningRateScheduler**: Dynamic learning rate adjustment
     - Exponential decay
     - Polynomial decay
     - Cosine annealing
     - Performance-based adaptation
   - **AdaptiveOptimizer**: Advanced optimization strategies
     - Adam
     - RMSprop
     - AdaGrad
     - Momentum
   - **MultiArmedBandit**: Strategy selection
     - UCB (Upper Confidence Bound)
     - Thompson Sampling
     - ε-greedy
     - Gradient bandits
   - **AdaptiveMetaLearner**: Meta-learning for strategy optimization

4. **Concept Drift Detection** (`concept_drift_detector.py`)
   - **ADWIN**: Adaptive Windowing for detecting distribution changes
   - **DDM**: Drift Detection Method based on error rates
   - **Page-Hinkley**: Sequential change detection test
   - **KSWIN**: Kolmogorov-Smirnov test with sliding windows
   - Ensemble drift detection with voting mechanisms

5. **Memory Management** (`memory_manager.py`)
   - Efficient memory usage with configurable limits
   - Multiple eviction policies:
     - LRU (Least Recently Used)
     - LFU (Least Frequently Used)
     - Importance-based
     - Time-based
     - Adaptive (composite scoring)
   - **SlidingWindowBuffer**: Fixed-size circular buffer
   - **ReservoirSampler**: Uniform sampling from streams

6. **Streaming Validation** (`streaming_validation.py`)
   - **PrequentialEvaluator**: Test-then-train evaluation
   - **StreamingValidator**: Real-time performance monitoring
   - **StabilityTracker**: Detect performance stability
   - **AnomalyDetector**: Identify outlier predictions
   - **StreamingCrossValidator**: Time series cross-validation

## Usage Examples

### Basic Online Learning

```python
from alpha_pulse.ml.online import IncrementalSGD, OnlineDataPoint

# Initialize learner with drift detection
config = {
    'learning_rate': 0.01,
    'drift_detection': {
        'method': 'adwin',
        'delta': 0.002
    }
}

learner = IncrementalSGD(config, task_type='regression')

# Process streaming data
for timestamp, features, label in data_stream:
    data_point = OnlineDataPoint(
        timestamp=timestamp,
        features=features,
        label=label
    )
    
    # Learn and predict
    prediction = learner.learn_one(data_point)
    
    # Check for drift
    if learner.drift_detector.detected_change():
        print(f"Drift detected at {timestamp}")
```

### Ensemble Learning

```python
from alpha_pulse.ml.online import OnlineLearnerEnsemble

# Create ensemble
ensemble_config = {
    'max_models': 5,
    'combination_method': 'weighted_average'
}

ensemble = OnlineLearnerEnsemble(ensemble_config)

# Add diverse learners
ensemble.add_learner(IncrementalSGD({'learning_rate': 0.01}))
ensemble.add_learner(HoeffdingTree({'grace_period': 200}))
ensemble.add_learner(AdaptiveRandomForest({'n_estimators': 10}))

# Process data
prediction = ensemble.learn_one(data_point)

# Update weights based on performance
ensemble.update_weights()
```

### Adaptive Learning Rate

```python
from alpha_pulse.ml.online import AdaptiveLearningRateScheduler

# Configure adaptive scheduler
scheduler = AdaptiveLearningRateScheduler({
    'schedule_type': 'adaptive',
    'initial_rate': 0.1,
    'adapt_to_performance': True,
    'adapt_to_volatility': True
})

# Update based on performance
for step in range(training_steps):
    current_rate = scheduler.step(performance_metric)
    # Apply learning rate to optimizer
```

### Service Integration

```python
from alpha_pulse.ml.online import OnlineLearningService

# Initialize service
service = OnlineLearningService(db_session, config)

# Start learning session
request = LearningSessionRequest(
    agent_id='tech_agent_001',
    strategy='adaptive',
    config={'model_type': 'adaptive_forest'}
)

response = await service.start_session(request)
session_id = response.session_id

# Process streaming batches
batch = StreamingBatch(
    batch_id='batch_001',
    data_points=data_points,
    source='market_feed'
)

result = await service.process_batch(session_id, batch)

# Make predictions
pred_request = PredictionRequest(
    session_id=session_id,
    features=new_features
)

predictions = await service.predict(pred_request)
```

## Key Features

### 1. Concept Drift Handling

The system automatically detects and adapts to concept drift:

- **Detection**: Multiple algorithms monitor for distribution changes
- **Adaptation**: Strategies include:
  - Model reset
  - Learning rate increase
  - Ensemble member addition
  - Adaptive hyperparameter tuning

### 2. Memory Efficiency

- Configurable memory limits
- Intelligent eviction policies
- Streaming algorithms that don't store all data
- Efficient data structures (circular buffers, reservoir sampling)

### 3. Real-time Performance

- Incremental updates without full retraining
- Parallel processing for ensembles
- Optimized algorithms for streaming data
- Low-latency predictions

### 4. Robustness

- Multiple drift detectors for reliability
- Ensemble methods for stability
- Anomaly detection for outliers
- Continuous validation

## Configuration

### Online Learning Config

```yaml
online_learning:
  # Core settings
  learning_rate: 0.01
  batch_size: 32
  update_frequency: 1
  performance_window_size: 100
  
  # Drift detection
  drift_detection:
    method: "ensemble"  # adwin, ddm, page_hinkley, kswin, ensemble
    ensemble_mode: true
    check_frequency: 100
    
  # Memory management
  memory_management:
    max_memory_mb: 1024
    eviction_policy: "adaptive"  # lru, lfu, importance, time, adaptive
    gc_threshold: 0.9
    
  # Adaptive optimization
  adaptive_optimization:
    optimizer_type: "adam"
    adapt_betas: true
    gradient_clipping: 1.0
    
  # Ensemble configuration
  ensemble_config:
    max_models: 5
    combination_method: "weighted_average"
    diversity_threshold: 0.1
```

## Best Practices

1. **Choose the Right Model**
   - SGD for simple linear relationships
   - Trees for non-linear patterns
   - Ensembles for robustness

2. **Configure Drift Detection**
   - ADWIN for gradual drift
   - DDM for sudden changes
   - Ensemble for reliability

3. **Monitor Performance**
   - Use streaming validation
   - Track stability metrics
   - Set up alerts for anomalies

4. **Optimize Memory Usage**
   - Set appropriate memory limits
   - Choose efficient data structures
   - Use adaptive eviction policies

5. **Handle Market Regimes**
   - Use adaptive learning rates
   - Implement regime-specific models
   - Monitor for concept drift

## Performance Metrics

The system tracks various metrics:

- **Accuracy Metrics**: MSE, MAE, RMSE, R², MAPE
- **Classification Metrics**: Accuracy, Precision, Recall, F1
- **Streaming Metrics**: Processing rate, memory usage, latency
- **Drift Metrics**: Drift frequency, adaptation success rate

## Integration with Trading Agents

Online learning enhances agent capabilities:

1. **Real-time Adaptation**: Agents continuously improve predictions
2. **Market Regime Detection**: Automatic adjustment to market changes
3. **Risk Management**: Better prediction confidence estimates
4. **Performance Optimization**: Dynamic strategy selection

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce window sizes
   - Increase eviction frequency
   - Use more aggressive eviction policies

2. **Slow Processing**
   - Reduce ensemble size
   - Increase batch size
   - Use simpler models

3. **Poor Drift Detection**
   - Adjust sensitivity parameters
   - Try different detection methods
   - Use ensemble detection

4. **Unstable Predictions**
   - Check for data quality issues
   - Increase stability thresholds
   - Use ensemble methods

## Future Enhancements

- Neural network incremental learning
- Federated learning support
- AutoML for online settings
- Advanced meta-learning strategies
- GPU acceleration for streaming