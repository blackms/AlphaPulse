"""
Quick script to train and save a test model for demo_paper_trading.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime, timedelta

from alpha_pulse.features.feature_engineering import calculate_technical_indicators

# Set up MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("crypto_prediction")


# Create sample historical data with clear trends and patterns
def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data with clear patterns."""
    current_time = datetime.now()
    dates = [current_time - timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate price series with strong trends and cycles
    t = np.linspace(0, 10, n_samples)
    
    # Create stronger trends and patterns
    trend = 0.005 * t  # Stronger upward trend
    cycles = (
        0.1 * np.sin(t) +      # Main cycle (stronger)
        0.05 * np.sin(3 * t) +  # Faster cycle (stronger)
        0.02 * np.sin(0.5 * t)  # Slower cycle (stronger)
    )
    
    # Add volatility clusters
    volatility = 0.02 * (1 + 0.5 * np.abs(np.sin(t/2)))
    noise = np.random.normal(0, volatility, n_samples)
    
    # Combine components
    returns = trend + cycles + noise
    prices = 50000 * np.exp(np.cumsum(returns))  # Base price for BTC/USD
    
    # Generate volume with correlation to absolute returns
    volume_base = 100 + 50 * np.abs(returns)
    volume = volume_base * np.exp(np.random.normal(0, 0.5, n_samples))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 - 0.001),
        'high': prices * (1 + 0.002),
        'low': prices * (1 - 0.002),
        'close': prices,
        'volume': volume
    })
    df.set_index('timestamp', inplace=True)
    
    return df

# Generate sample data
print("Generating sample data...")
df = generate_sample_data(1000)

# Calculate technical indicators
print("Calculating features...")
features_df = calculate_technical_indicators(df)

# Create target (1 if significant price increase in next period)
future_returns = df['close'].pct_change(-1)  # Forward returns
features_df['target'] = (future_returns > 0.002).astype(int)  # 0.2% threshold

# Drop rows with NaN values
features_df.dropna(inplace=True)

# Prepare features and target
X = features_df.drop('target', axis=1)
feature_names = list(X.columns)  # Save feature names
y = features_df['target']

print(f"\nFeature distribution:")
print(y.value_counts(normalize=True))

# Train model with more trees and class weights
print("\nTraining model...")

# Define model parameters
params = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_leaf': 20,
    'class_weight': {0: 1, 1: 2},
    'random_state': 42
}

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)
    
    # Create and train model
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    
    # Calculate and log cross-validation metrics
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    mlflow.log_metric("cv_f1_mean", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())
    
    # Log training set metrics
    train_predictions = model.predict(X)
    train_proba = model.predict_proba(X)[:, 1]
    mlflow.log_metric("train_accuracy", (train_predictions == y).mean())
    mlflow.log_metric("train_f1", cv_scores.mean())
    
    # Log feature importances
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)
    
    # Log feature importance plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    mlflow.log_artifact('feature_importances.png')
    plt.close()
    
    # Save model with MLflow
    mlflow.sklearn.log_model(model, "model")
    
    # Also save locally for compatibility
    model_dir = Path('trained_models')
    model_dir.mkdir(exist_ok=True)
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    local_path = model_dir / 'crypto_prediction_model.joblib'
    joblib.dump(model_data, local_path)
    mlflow.log_artifact(local_path)

print(f"\nModel trained on {len(features_df)} samples with features:")
for feature in feature_names:
    print(f"- {feature}")
print("\nTest model trained and saved successfully!")

# Print some predictions on the training data to verify signal distribution
predictions = model.predict_proba(X)[:, 1]
print("\nSignal distribution in training data:")
print(pd.Series(predictions).describe([0.1, 0.25, 0.5, 0.75, 0.9]))