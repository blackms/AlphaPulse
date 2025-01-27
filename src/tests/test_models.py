"""
Unit tests for the model training module.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from sklearn.exceptions import NotFittedError

from models.basic_models import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test suite for ModelTrainer class."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample regression data
        np.random.seed(42)
        self.X_reg = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y_reg = pd.Series(
            self.X_reg['feature1'] * 2 + self.X_reg['feature2'] + np.random.randn(100) * 0.1
        )
        
        # Create sample classification data
        self.X_clf = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y_clf = pd.Series(
            (self.X_clf['feature1'] + self.X_clf['feature2'] > 0).astype(int)
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test valid initialization
        trainer = ModelTrainer(model_type='random_forest', task='regression')
        self.assertIsNotNone(trainer.model)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            ModelTrainer(model_type='invalid_model')
        
        # Test invalid task
        with self.assertRaises(ValueError):
            ModelTrainer(task='invalid_task')
        
        # Test custom parameters
        params = {'n_estimators': 50, 'max_depth': 5}
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_params=params
        )
        self.assertEqual(trainer.model.n_estimators, 50)
        self.assertEqual(trainer.model.max_depth, 5)

    def test_regression_training(self):
        """Test model training for regression task."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        # Train model
        metrics = trainer.train(self.X_reg, self.y_reg)
        
        # Check metrics
        self.assertIn('rmse', metrics)
        self.assertIn('mse', metrics)
        self.assertTrue(all(v >= 0 for v in metrics.values()))  # Metrics should be non-negative
        
        # Test predictions
        predictions = trainer.predict(self.X_reg)
        self.assertEqual(len(predictions), len(self.y_reg))
        self.assertTrue(np.all(np.isfinite(predictions)))  # No NaN or infinite values

    def test_classification_training(self):
        """Test model training for classification task."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='classification',
            model_dir=self.temp_dir
        )
        
        # Train model
        metrics = trainer.train(self.X_clf, self.y_clf)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertTrue(all(0 <= v <= 1 for v in metrics.values()))  # Metrics should be between 0 and 1
        
        # Test predictions
        predictions = trainer.predict(self.X_clf)
        self.assertEqual(len(predictions), len(self.y_clf))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))  # Binary predictions

    def test_model_persistence(self):
        """Test model saving and loading."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        # Train and save model
        trainer.train(self.X_reg, self.y_reg)
        save_path = trainer.save_model('test_model.joblib')
        self.assertTrue(save_path.exists())
        
        # Create new trainer and load model
        new_trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        new_trainer.load_model(save_path)
        
        # Compare predictions
        pred1 = trainer.predict(self.X_reg)
        pred2 = new_trainer.predict(self.X_reg)
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression'
        )
        
        # Perform cross-validation
        cv_metrics = trainer.cross_validate(self.X_reg, self.y_reg, n_splits=3)
        
        # Check metrics
        self.assertTrue(all(isinstance(v, list) for v in cv_metrics.values()))
        self.assertTrue(all(len(v) == 3 for v in cv_metrics.values()))  # 3 splits
        self.assertTrue(all(np.isfinite(v) for values in cv_metrics.values() for v in values))

    def test_feature_importance(self):
        """Test feature importance calculation."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression'
        )
        
        # Train model and get feature importance
        trainer.train(self.X_reg, self.y_reg)
        importance = trainer.get_feature_importance()
        
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), len(self.X_reg.columns))
        self.assertTrue(all(importance >= 0))  # Importance scores should be non-negative

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression'
        )
        
        # Test prediction without training
        with self.assertRaises(ValueError):
            trainer.predict(self.X_reg)
        
        # Test saving without training
        with self.assertRaises(ValueError):
            trainer.save_model()
        
        # Test loading non-existent model
        with self.assertRaises(FileNotFoundError):
            trainer.load_model('non_existent_model.joblib')
        
        # Test training with mismatched X, y lengths
        with self.assertRaises(ValueError):
            trainer.train(self.X_reg, self.y_reg[:50])

    def test_xgboost_models(self):
        """Test XGBoost model functionality."""
        trainer = ModelTrainer(
            model_type='xgboost',
            task='regression',
            model_params={'n_estimators': 50}
        )
        
        # Train and evaluate
        metrics = trainer.train(self.X_reg, self.y_reg)
        self.assertIn('rmse', metrics)
        
        # Test classification
        trainer = ModelTrainer(
            model_type='xgboost',
            task='classification',
            model_params={'n_estimators': 50}
        )
        metrics = trainer.train(self.X_clf, self.y_clf)
        self.assertIn('accuracy', metrics)


if __name__ == '__main__':
    unittest.main()