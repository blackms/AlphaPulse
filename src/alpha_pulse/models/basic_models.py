"""
Basic machine learning models for AlphaPulse.
"""
from typing import Dict, Optional, Union, List
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import multiprocessing


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model_type: str = 'random_forest',
        task: str = 'regression',
        model_params: Optional[Dict] = None,
        model_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model to use
            task: Type of task ('regression' or 'classification')
            model_params: Model hyperparameters
            model_dir: Directory to save trained models
        """
        if model_type not in ['random_forest']:
            raise ValueError(f"Unsupported model type: {model_type}")
        if task not in ['regression', 'classification']:
            raise ValueError(f"Unsupported task: {task}")

        # Get number of CPU cores
        n_cores = multiprocessing.cpu_count()
        logger.info(f"Using {n_cores} CPU cores for training")

        self.model_type = model_type
        self.task = task
        base_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': n_cores  # Use all CPU cores
        }
        if model_params:
            base_params.update(model_params)
        self.model_params = base_params
        
        self.model_dir = Path(model_dir) if model_dir else Path('trained_models')
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize model
        if task == 'regression':
            self._model = RandomForestRegressor(**self.model_params)
        else:
            self._model = RandomForestClassifier(**self.model_params)
        
        self._is_fitted = False
        logger.info(f"Initialized {model_type} model for {task} with params: {self.model_params}")

    def train(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model and return performance metrics.

        Args:
            features: Training features
            target: Target values
            test_size: Fraction of data to use for testing

        Returns:
            Dict of performance metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        
        y_pred = self._model.predict(X_test)
        
        metrics = {}
        if self.task == 'regression':
            metrics['train_r2'] = self._model.score(X_train, y_train)
            metrics['test_r2'] = self._model.score(X_test, y_test)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            
            logger.info(
                f"Model trained. Train R²: {metrics['train_r2']:.4f}, "
                f"Test R²: {metrics['test_r2']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}"
            )
        else:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(
                f"Model trained. Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}"
            )
        
        return metrics

    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            features: Features to make predictions on

        Returns:
            Array of predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self._model.predict(features)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores if available.

        Returns:
            Series of feature importance scores or None
        """
        if not self._is_fitted:
            return None
        
        if hasattr(self._model, 'feature_importances_'):
            return pd.Series(self._model.feature_importances_)
        
        return None

    def cross_validate(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation.

        Args:
            features: Training features
            target: Target values
            n_splits: Number of CV folds
            metrics: List of metrics to compute

        Returns:
            Dict of CV scores
        """
        metrics = metrics or ['r2' if self.task == 'regression' else 'accuracy']
        scores = {}
        
        # Get number of CPU cores
        n_cores = multiprocessing.cpu_count()
        
        for metric in metrics:
            cv_scores = cross_val_score(
                self._model, features, target,
                cv=n_splits, scoring=metric,
                n_jobs=n_cores  # Use all CPU cores for cross-validation
            )
            scores[f'{metric}_scores'] = cv_scores
            logger.info(
                f"Cross-validation {metric}: "
                f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )
        
        return scores

    def save_model(self, filename: str) -> Path:
        """
        Save the trained model to disk.

        Args:
            filename: Name of the file to save the model to

        Returns:
            Path to the saved model file
        """
        if not self._is_fitted:
            raise RuntimeError("No model to save")
        
        save_path = self.model_dir / filename
        joblib.dump(self._model, save_path)
        logger.info(f"Model saved to {save_path}")
        
        return save_path

    def load_model(self, filename: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filename: Name of the file to load the model from
        """
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        self._model = joblib.load(load_path)
        self._is_fitted = True
        logger.info(f"Model loaded from {load_path}")