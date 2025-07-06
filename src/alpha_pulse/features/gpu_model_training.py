"""
GPU-accelerated model training module.

This module extends the base model training functionality with GPU acceleration,
automatically leveraging GPU resources when available for faster training and inference.
"""
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger
import joblib
import asyncio

from .model_training import ModelTrainer, ModelFactory
from ..ml.gpu.gpu_service import GPUService
from ..ml.gpu.gpu_config import get_training_config, get_inference_config


class GPUModelTrainer(ModelTrainer):
    """GPU-accelerated model trainer that extends base ModelTrainer functionality."""
    
    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        model_dir: Optional[str] = None,
        random_state: int = 42,
        use_gpu: bool = True,
        gpu_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GPU-accelerated model trainer.

        Args:
            model: Scikit-learn compatible model or None for GPU models
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration
            gpu_config: GPU configuration override
        """
        super().__init__(model, model_dir, random_state)
        
        self.use_gpu = use_gpu
        self.gpu_service = None
        self.gpu_model_name = None
        self._is_gpu_model = False
        
        if self.use_gpu:
            try:
                # Initialize GPU service with training configuration
                config = get_training_config() if gpu_config is None else gpu_config
                self.gpu_service = GPUService(config)
                asyncio.create_task(self.gpu_service.start())
                logger.info("GPU acceleration enabled for model training")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.use_gpu = False

    def create_gpu_model(
        self,
        model_type: str = 'neural_network',
        **model_params
    ) -> None:
        """
        Create a GPU-optimized model.

        Args:
            model_type: Type of GPU model (linear, neural_network, lstm, transformer)
            **model_params: Model-specific parameters
        """
        if not self.use_gpu or not self.gpu_service:
            raise RuntimeError("GPU service not available")
        
        # Generate unique model name
        self.gpu_model_name = f"gpu_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create GPU model
        self.model = self.gpu_service.create_model(
            model_type=model_type,
            model_name=self.gpu_model_name,
            **model_params
        )
        self._is_gpu_model = True
        logger.info(f"Created GPU {model_type} model: {self.gpu_model_name}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> BaseEstimator:
        """
        Train the model with GPU acceleration when available.

        Args:
            X_train: Training features
            y_train: Training targets
            validation_data: Optional (X_val, y_val) tuple
            **kwargs: Additional arguments passed to model.fit()

        Returns:
            Trained model
        """
        if self._feature_names is None:
            self._feature_names = X_train.columns.tolist()
        
        # Convert to numpy arrays
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        # Handle validation data
        val_data_np = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
            val_data_np = (X_val_np, y_val_np)
        
        if self._is_gpu_model and self.gpu_service:
            # GPU training
            logger.info(f"Training {self.gpu_model_name} on GPU with {len(X_train)} samples")
            
            # Run async training
            loop = asyncio.get_event_loop()
            training_results = loop.run_until_complete(
                self.gpu_service.train_model(
                    model_name=self.gpu_model_name,
                    X=X_train_np,
                    y=y_train_np,
                    validation_data=val_data_np,
                    **kwargs
                )
            )
            
            logger.info(f"GPU training completed in {training_results['training_time']:.2f}s")
            return self.model
        else:
            # Fallback to CPU training
            return super().train(X_train, y_train, **kwargs)

    def predict(
        self,
        X: pd.DataFrame,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions with GPU acceleration when available.

        Args:
            X: Features for prediction
            batch_size: Batch size for GPU processing

        Returns:
            Predictions
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        if self._is_gpu_model and self.gpu_service:
            # GPU prediction
            loop = asyncio.get_event_loop()
            predictions = loop.run_until_complete(
                self.gpu_service.predict(
                    model_name=self.gpu_model_name,
                    X=X_np,
                    batch_size=batch_size
                )
            )
            return predictions
        else:
            # CPU prediction
            return self.model.predict(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance with GPU acceleration.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        # Add GPU-specific metrics if available
        if self._is_gpu_model and self.gpu_service:
            gpu_metrics = self.gpu_service.get_metrics()
            if 'devices' in gpu_metrics:
                for device_id, device_info in gpu_metrics['devices'].items():
                    metrics[f'gpu_{device_id}_utilization'] = device_info['utilization']
                    metrics[f'gpu_{device_id}_memory_usage'] = device_info['memory_usage']
        
        logger.info("Model Performance:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric.upper()}: {value:.6f}")
        
        return metrics

    def calculate_technical_indicators(
        self,
        prices: np.ndarray,
        indicators: list
    ) -> Dict[str, np.ndarray]:
        """
        Calculate technical indicators using GPU acceleration.

        Args:
            prices: Price data
            indicators: List of indicators to calculate

        Returns:
            Dictionary of indicator values
        """
        if self.gpu_service:
            return self.gpu_service.calculate_technical_indicators(prices, indicators)
        else:
            raise NotImplementedError("Technical indicators require GPU service")

    def run_monte_carlo_simulation(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        time_horizon: int,
        n_simulations: int = 10000
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation on GPU for risk analysis.

        Args:
            initial_price: Initial asset price
            drift: Expected return
            volatility: Volatility
            time_horizon: Number of time steps
            n_simulations: Number of simulations

        Returns:
            Simulated price paths
        """
        if self.gpu_service:
            return self.gpu_service.run_monte_carlo(
                initial_price=initial_price,
                drift=drift,
                volatility=volatility,
                time_horizon=time_horizon,
                n_simulations=n_simulations
            )
        else:
            raise NotImplementedError("Monte Carlo simulation requires GPU service")

    def optimize_portfolio_weights(
        self,
        returns: np.ndarray,
        method: str = 'mean_variance',
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio weights using GPU acceleration.

        Args:
            returns: Asset returns
            method: Optimization method
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        if self.gpu_service:
            return self.gpu_service.optimize_portfolio(
                returns=returns,
                method=method,
                constraints=constraints
            )
        else:
            raise NotImplementedError("Portfolio optimization requires GPU service")

    def get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU utilization and performance metrics."""
        if self.gpu_service:
            return self.gpu_service.get_metrics()
        return None

    def save_model(self, name: Optional[str] = None) -> Path:
        """
        Save trained model to disk (GPU or CPU).

        Args:
            name: Name to save model under (defaults to timestamp)

        Returns:
            Path to saved model
        """
        if self._is_gpu_model and self.gpu_service:
            # Save GPU model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = name or f"gpu_model_{timestamp}"
            save_path = self.model_dir / f"{name}.pt"
            
            self.gpu_service.save_model(self.gpu_model_name, str(save_path))
            
            # Save metadata
            metadata_path = self.model_dir / f"{name}_metadata.json"
            import json
            metadata = {
                'model_type': 'gpu_model',
                'gpu_model_name': self.gpu_model_name,
                'feature_names': self._feature_names,
                'timestamp': timestamp,
                'gpu_metrics': self.get_gpu_metrics()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return save_path
        else:
            # Use parent class method for CPU models
            return super().save_model(name)

    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.

        Args:
            path: Path to saved model
        """
        path = Path(path)
        
        if path.suffix in ['.pt', '.pth']:
            # Load GPU model
            if not self.gpu_service:
                raise RuntimeError("GPU service required to load GPU model")
            
            # Load metadata
            metadata_path = path.parent / f"{path.stem}_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.gpu_model_name = metadata.get('gpu_model_name', path.stem)
                    self._feature_names = metadata.get('feature_names')
            
            self.gpu_service.load_model(str(path), self.gpu_model_name)
            self._is_gpu_model = True
            logger.info(f"Loaded GPU model from {path}")
        else:
            # Use parent class method for CPU models
            super().load_model(str(path))

    def __del__(self):
        """Cleanup GPU resources."""
        if self.gpu_service:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.gpu_service.stop())
            else:
                loop.run_until_complete(self.gpu_service.stop())


class GPUModelFactory(ModelFactory):
    """Factory class for creating GPU-accelerated models."""
    
    @staticmethod
    def create_gpu_linear_regression(**kwargs) -> GPUModelTrainer:
        """Create a GPU-accelerated linear regression model."""
        trainer = GPUModelTrainer(model=None, use_gpu=True)
        trainer.create_gpu_model('linear', **kwargs)
        return trainer
    
    @staticmethod
    def create_gpu_neural_network(
        input_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        **kwargs
    ) -> GPUModelTrainer:
        """Create a GPU-accelerated neural network model."""
        trainer = GPUModelTrainer(model=None, use_gpu=True)
        trainer.create_gpu_model(
            'neural_network',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs
        )
        return trainer
    
    @staticmethod
    def create_gpu_lstm(
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        **kwargs
    ) -> GPUModelTrainer:
        """Create a GPU-accelerated LSTM model."""
        trainer = GPUModelTrainer(model=None, use_gpu=True)
        trainer.create_gpu_model(
            'lstm',
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            **kwargs
        )
        return trainer
    
    @staticmethod
    def create_gpu_transformer(
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        output_dim: int = 1,
        **kwargs
    ) -> GPUModelTrainer:
        """Create a GPU-accelerated transformer model."""
        trainer = GPUModelTrainer(model=None, use_gpu=True)
        trainer.create_gpu_model(
            'transformer',
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=output_dim,
            **kwargs
        )
        return trainer