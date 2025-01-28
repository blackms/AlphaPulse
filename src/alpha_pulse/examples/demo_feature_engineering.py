"""
Demo Feature Engineering and Model Training

This script demonstrates the complete ML pipeline:
1. Data loading from the data pipeline
2. Feature engineering using technical indicators
3. Model training and evaluation
4. Results visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from data_pipeline.exchange import CCXTExchangeFactory
from data_pipeline.storage import SQLAlchemyStorage
from data_pipeline.data_fetcher import DataFetcher
from features.feature_engineering import FeatureStore
from models.basic_models import ModelTrainer


def create_target_variable(df: pd.DataFrame, forward_returns_days: int = 1) -> pd.Series:
    """Create target variable based on forward returns."""
    close_prices = df['close']
    forward_returns = close_prices.shift(-forward_returns_days) / close_prices - 1
    return forward_returns


def plot_feature_importance(importance: pd.Series, title: str) -> None:
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 6))
    importance.sort_values().plot(kind='barh')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')


def plot_predictions_vs_actual(y_test: pd.Series, predictions: np.ndarray, title: str) -> None:
    """Plot predicted vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title(title)
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'predictions_vs_actual_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')


def main():
    # 1. Load historical data
    exchange_factory = CCXTExchangeFactory()
    storage = SQLAlchemyStorage()
    data_fetcher = DataFetcher(exchange_factory, storage)
    
    # Fetch one year of hourly data
    data_fetcher.update_historical_data(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        days_back=365
    )
    
    # Get the data from storage
    df = storage.get_historical_data("binance", "BTC/USDT", "1h")
    print(f"Loaded {len(df)} data points")
    
    # 2. Feature Engineering
    feature_store = FeatureStore(cache_dir='feature_cache')
    
    # Compute technical indicators
    features = feature_store.compute_technical_indicators(
        df['close'],
        windows=[12, 26, 50, 100, 200]  # Common technical analysis periods
    )
    
    # Add some price-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    features['volatility'] = features['returns'].rolling(window=20).std()
    
    print(f"Generated {len(features.columns)} features")
    
    # 3. Create target variable (next day returns)
    target = create_target_variable(df, forward_returns_days=1)
    
    # Align features and target indices
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    
    # Remove any rows with NaN values
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]
    
    print(f"Final dataset size: {len(features)} samples")
    
    # 4. Train and evaluate model
    model_dir = Path('trained_models')
    model_dir.mkdir(exist_ok=True)
    
    # Initialize model trainer
    trainer = ModelTrainer(
        model_type='xgboost',
        task='regression',
        model_params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        },
        model_dir=model_dir
    )
    
    # Train model
    metrics = trainer.train(features, target, test_size=0.2)
    print("\nTraining Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Cross-validation
    cv_metrics = trainer.cross_validate(features, target, n_splits=5)
    print("\nCross-validation Metrics:")
    for metric, values in cv_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # 5. Feature Importance Analysis
    importance = trainer.get_feature_importance()
    if importance is not None:
        importance.index = features.columns
        print("\nTop 10 Most Important Features:")
        print(importance.nlargest(10))
        
        # Plot feature importance
        plot_feature_importance(importance, "Feature Importance Scores")
    
    # 6. Plot predictions vs actual
    predictions = trainer.predict(features)
    plot_predictions_vs_actual(target, predictions, "Predicted vs Actual Returns")
    
    # 7. Save the model
    model_path = trainer.save_model("btc_returns_predictor.joblib")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()