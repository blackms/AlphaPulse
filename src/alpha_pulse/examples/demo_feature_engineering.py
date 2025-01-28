"""
Demo Feature Engineering and Model Training

This script demonstrates the ML pipeline with mock data:
1. Generate sample price data
2. Feature engineering using technical indicators
3. Model training and evaluation
4. Results visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


def create_sample_data(days: int = 365) -> pd.DataFrame:
    """Create sample price data for demonstration."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days-1),
        end=datetime.now(),
        freq='D'
    )
    
    # Generate random walk prices
    rw = np.random.normal(0, 0.02, len(dates)).cumsum()
    prices = 100 * np.exp(rw)
    
    return pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
        'high': prices * (1 + abs(np.random.normal(0, 0.02, len(dates)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.02, len(dates)))),
        'volume': np.random.lognormal(10, 1, len(dates))
    }, index=dates)


def create_target_variable(df: pd.DataFrame, forward_returns_days: int = 1) -> pd.Series:
    """Create target variable based on forward returns."""
    close_prices = df['close']
    forward_returns = close_prices.shift(-forward_returns_days) / close_prices - 1
    return forward_returns


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and other features."""
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    features['volatility'] = features['returns'].rolling(window=20).std()
    
    # Moving averages
    for window in [12, 26, 50, 100, 200]:
        features[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
    
    # Price momentum
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = df['close'].pct_change(window)
    
    # Volume features
    features['volume_ma'] = df['volume'].rolling(window=20).mean()
    features['volume_std'] = df['volume'].rolling(window=20).std()
    
    return features


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
    plot_path = plots_dir / f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path)
    logger.info(f"Saved feature importance plot to {plot_path}")


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
    plot_path = plots_dir / f'predictions_vs_actual_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_path)
    logger.info(f"Saved predictions plot to {plot_path}")


def main():
    """Run feature engineering and model training demonstration."""
    logger.info("Starting feature engineering demonstration...")

    # 1. Generate sample data
    df = create_sample_data(days=365)
    logger.info(f"Generated {len(df)} days of sample data")
    
    # 2. Feature Engineering
    features = compute_features(df)
    logger.info(f"Generated {len(features.columns)} features")
    
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
    
    logger.info(f"Final dataset size: {len(features)} samples")
    
    # 4. Train a simple model (using sklearn for demonstration)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logger.info("\nModel Performance:")
    logger.info(f"Mean Squared Error: {mse:.6f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    # 5. Feature Importance Analysis
    importance = pd.Series(
        model.feature_importances_,
        index=features.columns
    )
    logger.info("\nTop 10 Most Important Features:")
    logger.info(importance.nlargest(10))
    
    # Plot results
    plot_feature_importance(importance, "Feature Importance Scores")
    plot_predictions_vs_actual(y_test, predictions, "Predicted vs Actual Returns")


if __name__ == "__main__":
    main()