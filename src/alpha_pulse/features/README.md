# Feature Engineering Package

This package provides a comprehensive suite of tools for financial data feature engineering, model training, and visualization. The package follows enterprise-grade best practices including SOLID principles, proper error handling, and comprehensive documentation.

## Package Structure

```
features/
├── __init__.py
├── data_generation.py    # Synthetic data generation utilities
├── feature_engineering.py # Technical indicator calculations
├── model_training.py     # ML model training and evaluation
├── visualization.py      # Plotting and visualization tools
└── README.md            # This file
```

## Modules

### Data Generation (`data_generation.py`)
Provides utilities for generating synthetic financial data for testing and demonstration purposes.

```python
from alpha_pulse.features.data_generation import create_sample_data, create_target_variable

# Generate sample OHLCV data
df = create_sample_data(days=365)

# Create target variable for prediction
target = create_target_variable(df, forward_returns_days=1)
```

### Feature Engineering (`feature_engineering.py`)
Implements technical indicators and feature calculations using TA-Lib.

```python
from alpha_pulse.features.feature_engineering import calculate_technical_indicators, FeatureStore

# Calculate technical indicators
features = calculate_technical_indicators(df)

# Cache features for later use
store = FeatureStore()
store.add_features('my_features', features)
```

### Model Training (`model_training.py`)
Handles model training, evaluation, and persistence.

```python
from alpha_pulse.features.model_training import ModelTrainer, ModelFactory

# Create and train a model
model = ModelFactory.create_random_forest(n_estimators=100)
trainer = ModelTrainer(model=model)

# Train and evaluate
X_train, X_test, y_train, y_test = trainer.prepare_data(features, target)
trainer.train(X_train, y_train)
metrics = trainer.evaluate(X_test, y_test)

# Save trained model
trainer.save_model('my_model')
```

### Visualization (`visualization.py`)
Provides plotting utilities for feature analysis and model evaluation.

```python
from alpha_pulse.features.visualization import FeatureVisualizer

visualizer = FeatureVisualizer()

# Plot feature importance
visualizer.plot_feature_importance(importance_scores)

# Plot model predictions
visualizer.plot_predictions_vs_actual(y_test, predictions)
```

## Best Practices

The package implements several best practices:

1. **SOLID Principles**
   - Single Responsibility: Each module has a specific focus
   - Open/Closed: Easy to extend with new features/models
   - Interface Segregation: Clean, focused interfaces
   - Dependency Inversion: Uses dependency injection

2. **Error Handling**
   - Comprehensive error checking
   - Informative error messages
   - Proper exception handling

3. **Type Hints**
   - All functions include type annotations
   - Improves code readability and IDE support

4. **Documentation**
   - Detailed docstrings
   - Usage examples
   - Clear API documentation

5. **Testing**
   - Comprehensive unit tests
   - Test coverage for critical functionality
   - Easy to run test suite

## Usage Example

See `examples/demo_feature_engineering.py` for a complete demonstration of the package's capabilities:

```python
from alpha_pulse.features import (
    create_sample_data,
    calculate_technical_indicators,
    ModelTrainer,
    FeatureVisualizer
)

# Generate data
df = create_sample_data(days=365)

# Calculate features
features = calculate_technical_indicators(df)

# Train model
trainer = ModelTrainer()
trainer.train(features, target)

# Visualize results
visualizer = FeatureVisualizer()
visualizer.plot_feature_importance(trainer.get_feature_importance())
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- TA-Lib
- loguru

## Installation

Ensure you have TA-Lib installed:

```bash
# On Ubuntu/Debian
sudo apt-get install ta-lib

# On macOS
brew install ta-lib

# Then install Python dependencies
pip install -r requirements.txt
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Use type hints
5. Handle errors appropriately