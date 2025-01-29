"""
Model training module for financial prediction models.

This module provides utilities for training, evaluating, and managing
machine learning models for financial prediction tasks.
"""
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger
import joblib


class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        model_dir: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize model trainer.

        Args:
            model: Scikit-learn compatible model (defaults to RandomForestRegressor)
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model = model or RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
        self.model_dir = Path(model_dir) if model_dir else Path('trained_models')
        self.model_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self._feature_names = None
        logger.info(f"Initialized ModelTrainer with {type(self.model).__name__}")

    def prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into train/test sets.

        Args:
            features: Feature DataFrame
            target: Target Series
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Store feature names
        self._feature_names = features.columns.tolist()
        
        # Remove any rows with NaN values
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features.loc[valid_mask]
        target = target.loc[valid_mask]
        
        # Split data
        return train_test_split(
            features, target,
            test_size=test_size,
            random_state=self.random_state
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> BaseEstimator:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments passed to model.fit()

        Returns:
            Trained model
        """
        if self._feature_names is None:
            self._feature_names = X_train.columns.tolist()
            
        logger.info(f"Training {type(self.model).__name__} on {len(X_train)} samples")
        self.model.fit(X_train, y_train, **kwargs)
        return self.model

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        logger.info("Model Performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.6f}")
        
        return metrics

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores if model supports it.

        Returns:
            Series of feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError(
                f"{type(self.model).__name__} does not support feature importance"
            )
        
        if self._feature_names is None:
            raise ValueError("Feature names not available. Train the model first.")
            
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names
        )

    def save_model(self, name: Optional[str] = None) -> Path:
        """
        Save trained model to disk.

        Args:
            name: Name to save model under (defaults to timestamp)

        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"model_{timestamp}"
        save_path = self.model_dir / f"{name}.joblib"
        
        model_info = {
            'model': self.model,
            'feature_names': self._feature_names,
            'metadata': {
                'model_type': type(self.model).__name__,
                'timestamp': timestamp,
                'parameters': self.model.get_params()
            }
        }
        
        joblib.dump(model_info, save_path)
        logger.info(f"Saved model to {save_path}")
        return save_path

    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.

        Args:
            path: Path to saved model
        """
        model_info = joblib.load(path)
        self.model = model_info['model']
        self._feature_names = model_info['feature_names']
        logger.info(f"Loaded {type(self.model).__name__} from {path}")


class ModelFactory:
    """Factory class for creating different types of models."""
    
    @staticmethod
    def create_random_forest(
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
        **kwargs: Any
    ) -> RandomForestRegressor:
        """Create a RandomForestRegressor with given parameters."""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    # Add more factory methods for other model types as needed