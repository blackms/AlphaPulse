"""
Tests for model training module.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from pathlib import Path
from sklearn.linear_model import LinearRegression

from ..features.model_training import ModelTrainer, ModelFactory


@pytest.fixture
def sample_data():
    """Fixture for sample training data."""
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 30, tzinfo=UTC),  # Increased to 30 days
        freq='D'
    )
    np.random.seed(42)  # For reproducibility
    
    # Create synthetic features with stronger signal
    n_samples = len(dates)
    features = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    }, index=dates)
    
    # Create synthetic target with clearer relationship
    target = pd.Series(
        features['feature1'] * 2.0 +  # Stronger coefficients
        features['feature2'] * -1.5 +
        features['feature3'] * 1.0 +
        np.random.randn(n_samples) * 0.1,  # Less noise
        index=dates
    )
    
    return features, target


@pytest.fixture
def trainer(tmp_path):
    """Fixture for ModelTrainer instance."""
    return ModelTrainer(model_dir=str(tmp_path))  # Use default model initialization


def test_initialization(tmp_path):
    """Test model trainer initialization."""
    # Test default initialization
    trainer = ModelTrainer(model_dir=str(tmp_path))
    assert trainer.model is not None
    assert trainer.random_state == 42
    assert trainer.model_dir == tmp_path
    
    # Test custom model initialization
    custom_model = LinearRegression()
    trainer = ModelTrainer(model=custom_model, random_state=123)
    assert trainer.model is custom_model
    assert trainer.random_state == 123


def test_data_preparation(trainer, sample_data):
    """Test data preparation and splitting."""
    features, target = sample_data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, target, test_size=0.2  # Explicit test size
    )
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(features)
    assert len(y_train) + len(y_test) == len(target)
    
    # Check feature names are stored
    assert trainer._feature_names == features.columns.tolist()
    
    # Test handling of NaN values
    features_with_nan = features.copy()
    features_with_nan.iloc[0, 0] = np.nan
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features_with_nan, target, test_size=0.2
    )
    assert len(X_train) + len(X_test) == len(features) - 1


def test_training_workflow(trainer, sample_data):
    """Test complete training and evaluation workflow."""
    features, target = sample_data
    
    # Prepare data with smaller test size
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, target, test_size=0.2
    )
    
    # Train model
    model = trainer.train(X_train, y_train)
    assert model is not None
    
    # Make predictions and evaluate
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    for metric in ['mse', 'rmse', 'mae', 'r2']:
        assert metric in metrics
    
    # Check metric values are reasonable
    assert metrics['mse'] >= 0.0  # MSE should be non-negative
    assert metrics['rmse'] >= 0.0  # RMSE should be non-negative
    assert metrics['mae'] >= 0.0   # MAE should be non-negative
    assert -1.0 <= metrics['r2'] <= 1.0  # R² should be between -1 and 1


def test_feature_importance(trainer, sample_data):
    """Test feature importance calculation."""
    features, target = sample_data
    
    # Train model first
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, target, test_size=0.2
    )
    trainer.train(X_train, y_train)
    
    # Get feature importance
    importance = trainer.get_feature_importance()
    assert isinstance(importance, pd.Series)
    assert len(importance) == len(features.columns)
    assert importance.index.tolist() == features.columns.tolist()
    
    # Test with model that doesn't support feature importance
    trainer = ModelTrainer(model=LinearRegression())
    trainer.train(X_train, y_train)
    with pytest.raises(AttributeError):
        trainer.get_feature_importance()


def test_model_persistence(trainer, sample_data, tmp_path):
    """Test model saving and loading."""
    features, target = sample_data
    
    # Train model
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, target, test_size=0.2
    )
    original_model = trainer.train(X_train, y_train)
    
    # Save model
    save_path = trainer.save_model("test_model")
    assert save_path.exists()
    
    # Create new trainer and load model
    new_trainer = ModelTrainer(model_dir=str(tmp_path))
    new_trainer.load_model(str(save_path))
    
    # Verify loaded model makes same predictions
    original_preds = original_model.predict(X_test)
    loaded_preds = new_trainer.model.predict(X_test)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)
    
    # Verify feature names are preserved
    assert new_trainer._feature_names == trainer._feature_names


def test_model_factory():
    """Test ModelFactory functionality."""
    # Test random forest creation with specific parameters
    rf_model = ModelFactory.create_random_forest(
        n_estimators=50,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    # Verify model parameters
    params = rf_model.get_params()
    assert params['n_estimators'] == 50
    assert params['max_depth'] == 3
    assert params['random_state'] == 42
    assert params['min_samples_split'] == 2
    assert params['min_samples_leaf'] == 1