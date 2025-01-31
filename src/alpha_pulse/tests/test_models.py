
"""
Tests for machine learning models.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from ..models.basic_models import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test suite for ModelTrainer class."""

    def setUp(self):
        """Set up test data and model trainer."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        self.X = np.random.randn(n_samples, n_features)
        self.y_reg = np.sum(self.X * np.random.randn(n_features), axis=1)
        self.y_cls = (self.y_reg > 0).astype(int)
        
        # Convert to pandas for column names
        self.X_df = pd.DataFrame(
            self.X,
            columns=[f'feature_{i}' for i in range(n_features)]
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test default initialization
        trainer = ModelTrainer(model_dir=self.temp_dir)
        self.assertEqual(trainer.model_type, 'random_forest')
        self.assertEqual(trainer.task, 'regression')
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            ModelTrainer(model_type='invalid_model')
        
        # Test invalid task
        with self.assertRaises(ValueError):
            ModelTrainer(task='invalid_task')

    def test_regression_training(self):
        """Test regression model training."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        metrics = trainer.train(self.X_df, self.y_reg)
        
        # Check metrics
        self.assertIn('train_r2', metrics)
        self.assertIn('test_r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertTrue(0 <= metrics['train_r2'] <= 1)
        self.assertTrue(0 <= metrics['test_r2'] <= 1)
        self.assertTrue(metrics['rmse'] >= 0)

    def test_classification_training(self):
        """Test classification model training."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='classification',
            model_dir=self.temp_dir
        )
        
        metrics = trainer.train(self.X_df, self.y_cls)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        cv_metrics = trainer.cross_validate(
            self.X_df,
            self.y_reg,
            n_splits=5,
            metrics=['r2']
        )
        
        self.assertIn('r2_scores', cv_metrics)
        self.assertEqual(len(cv_metrics['r2_scores']), 5)
        self.assertTrue(all(0 <= score <= 1 for score in cv_metrics['r2_scores']))

    def test_feature_importance(self):
        """Test feature importance calculation."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        trainer.train(self.X_df, self.y_reg)
        importance = trainer.get_feature_importance()
        
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), self.X_df.shape[1])
        self.assertTrue(all(imp >= 0 for imp in importance))

    def test_model_persistence(self):
        """Test model saving and loading."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        # Train and save model
        trainer.train(self.X_df, self.y_reg)
        save_path = trainer.save_model('test_model.joblib')
        self.assertTrue(save_path.exists())
        
        # Load model and make predictions
        new_trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        new_trainer.load_model('test_model.joblib')
        
        predictions = new_trainer.predict(self.X_df)
        self.assertEqual(len(predictions), len(self.y_reg))

    def test_error_handling(self):
        """Test error handling."""
        trainer = ModelTrainer(
            model_type='random_forest',
            task='regression',
            model_dir=self.temp_dir
        )
        
        # Test prediction without training
        with self.assertRaises(RuntimeError):
            trainer.predict(self.X_df)
        
        # Test saving without training
        with self.assertRaises(RuntimeError):
            trainer.save_model('test_model.joblib')
        
        # Test loading non-existent model
        with self.assertRaises(FileNotFoundError):
            trainer.load_model('non_existent.joblib')


if __name__ == '__main__':
    unittest.main()