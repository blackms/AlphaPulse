"""
Unit tests for feature engineering modules.

Tests the functionality of data generation, feature engineering,
visualization, and model training modules.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from alpha_pulse.features.data_generation import (
    create_sample_data,
    create_target_variable
)
from alpha_pulse.features.feature_engineering import (
    calculate_technical_indicators,
    calculate_rolling_stats,
    FeatureStore
)
from alpha_pulse.features.visualization import FeatureVisualizer
from alpha_pulse.features.model_training import ModelTrainer, ModelFactory


class TestDataGeneration(unittest.TestCase):
    """Test data generation functionality."""

    def setUp(self):
        self.days = 100
        self.df = create_sample_data(days=self.days)

    def test_create_sample_data(self):
        """Test sample data creation."""
        self.assertEqual(len(self.df), self.days)
        self.assertTrue(all(col in self.df.columns 
                          for col in ['open', 'high', 'low', 'close', 'volume']))
        
        # Test data integrity
        self.assertTrue(all(self.df['high'] >= self.df['low']))
        self.assertTrue(all(self.df['high'] >= self.df['close']))
        self.assertTrue(all(self.df['high'] >= self.df['open']))
        self.assertTrue(all(self.df['low'] <= self.df['close']))
        self.assertTrue(all(self.df['low'] <= self.df['open']))

    def test_create_target_variable(self):
        """Test target variable creation."""
        forward_days = 5
        target = create_target_variable(self.df, forward_returns_days=forward_days)
        
        self.assertEqual(len(target), len(self.df))
        self.assertTrue(target.isna().sum() == forward_days)  # Last n days should be NaN


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality."""

    def setUp(self):
        self.df = create_sample_data(days=100)
        self.features = calculate_technical_indicators(self.df)

    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        self.assertGreater(len(self.features.columns), 0)
        self.assertEqual(len(self.features), len(self.df))

    def test_calculate_rolling_stats(self):
        """Test rolling statistics calculation."""
        window = 20
        stats = calculate_rolling_stats(self.df['close'], window)
        
        self.assertIsInstance(stats, dict)
        self.assertTrue(all(len(v) == len(self.df) for v in stats.values()))
        self.assertTrue(all(k in stats for k in 
                          ['mean', 'std', 'min', 'max', 'median']))

    def test_feature_store(self):
        """Test feature store functionality."""
        store = FeatureStore(cache_dir='test_cache')
        try:
            # Test saving features
            store.add_features('test_features', self.features)
            
            # Test loading features
            loaded_features = store.get_features('test_features')
            self.assertTrue(loaded_features.equals(self.features))
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree('test_cache', ignore_errors=True)


class TestVisualization(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        self.df = create_sample_data(days=100)
        self.features = calculate_technical_indicators(self.df)
        self.visualizer = FeatureVisualizer(output_dir='test_plots')

    def test_plot_creation(self):
        """Test plot creation and saving."""
        try:
            # Test feature importance plot
            importance = pd.Series(np.random.random(len(self.features.columns)),
                                 index=self.features.columns)
            plot_path = self.visualizer.plot_feature_importance(importance)
            self.assertTrue(plot_path.exists())
            
            # Test predictions vs actual plot
            y_true = pd.Series(np.random.random(100))
            y_pred = np.random.random(100)
            plot_path = self.visualizer.plot_predictions_vs_actual(y_true, y_pred)
            self.assertTrue(plot_path.exists())
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree('test_plots', ignore_errors=True)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""

    def setUp(self):
        self.df = create_sample_data(days=100)
        self.features = calculate_technical_indicators(self.df)
        self.target = create_target_variable(self.df)
        self.trainer = ModelTrainer(model_dir='test_models')

    def test_model_training(self):
        """Test model training and evaluation."""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.trainer.prepare_data(
                self.features.fillna(0),  # Fill NaN for testing
                self.target.fillna(0)
            )
            
            # Train model
            self.trainer.train(X_train, y_train)
            
            # Evaluate model
            metrics = self.trainer.evaluate(X_test, y_test)
            self.assertTrue(all(k in metrics for k in ['mse', 'rmse', 'mae', 'r2']))
            
            # Test feature importance
            importance = self.trainer.get_feature_importance()
            self.assertEqual(len(importance), len(self.features.columns))
            
            # Test model saving and loading
            save_path = self.trainer.save_model('test_model')
            self.assertTrue(save_path.exists())
            
            new_trainer = ModelTrainer()
            new_trainer.load_model(save_path)
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree('test_models', ignore_errors=True)

    def test_model_factory(self):
        """Test model factory functionality."""
        model = ModelFactory.create_random_forest(
            n_estimators=50,
            max_depth=3
        )
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 3)


if __name__ == '__main__':
    unittest.main()