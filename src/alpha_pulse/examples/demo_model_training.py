"""
Example script demonstrating how to train a model using AlphaPulse.
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger

from alpha_pulse.models import ModelTrainer
from alpha_pulse.data_pipeline import CCXTExchangeFactory
from alpha_pulse.features.feature_engineering import (
    calculate_technical_indicators,
    add_target_column
)

def main():
    # 1. Fetch historical data
    logger.info("Fetching historical data...")
    factory = CCXTExchangeFactory()
    exchange = factory.create_exchange("binance")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    historical_data = exchange.fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start_time=start_date,
        end_time=end_date
    )
    
    # Convert to DataFrame and ensure column names
    df = pd.DataFrame(historical_data)
    df.columns = df.columns.astype(str)  # Convert column names to strings
    logger.info(f"Initial DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame head:\n{df.head()}")
    logger.info(f"NaN values before feature calculation:\n{df.isna().sum()}")

    # 2. Prepare features
    logger.info("Preparing features...")
    # Make a copy of the close price before feature calculation
    close_price = df['close'].copy()
    
    # Add technical indicators
    df = calculate_technical_indicators(df)
    logger.info(f"DataFrame shape after technical indicators: {df.shape}")
    logger.info(f"NaN values after technical indicators:\n{df.isna().sum()}")
    
    # Ensure we still have the close price
    df['close'] = close_price
    
    # Add target (price change in next 24 hours)
    df = add_target_column(df, target_column='close', periods=24)
    logger.info(f"DataFrame shape after adding target: {df.shape}")
    logger.info(f"NaN values after adding target:\n{df.isna().sum()}")
    
    # Handle NaN values
    # Forward fill NaN values in features
    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'sma_20', 'sma_50', 'ema_20', 'bollinger_upper',
        'bollinger_lower', 'atr'
    ]
    
    df[feature_columns] = df[feature_columns].fillna(method='ffill')
    
    # Drop remaining rows with NaN values (should only be at the start and end of the dataset)
    df = df.dropna()
    logger.info(f"Final DataFrame shape after handling NaN values: {df.shape}")
    
    # Verify we have enough data
    if len(df) < 100:  # Minimum required samples
        raise ValueError(f"Not enough data points after preprocessing: {len(df)}")
    
    # Verify all feature columns exist
    logger.info(f"Available columns: {df.columns.tolist()}")
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    X = df[feature_columns]
    y = df['target']

    # 3. Initialize and train model
    logger.info("Training model...")
    trainer = ModelTrainer(
        model_type='random_forest',
        task='regression',
        model_params={
            'n_estimators': 200,
            'max_depth': 10,
            'random_state': 42
        }
    )

    # Train and get metrics
    metrics = trainer.train(X, y, test_size=0.2)
    logger.info(f"Training metrics: {metrics}")

    # 4. Cross-validation
    logger.info("Performing cross-validation...")
    cv_scores = trainer.cross_validate(X, y, n_splits=5, metrics=['r2', 'neg_mean_squared_error'])
    
    # 5. Feature importance analysis
    logger.info("Analyzing feature importance...")
    importance = trainer.get_feature_importance()
    if importance is not None:
        importance = pd.Series(importance, index=feature_columns)
        plt.figure(figsize=(10, 6))
        importance.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(plots_dir / f'feature_importance_{timestamp}.png')

    # 6. Make predictions
    logger.info("Making predictions...")
    predictions = trainer.predict(X.iloc[-10:])  # Predict last 10 samples
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    actual = y.iloc[-10:]
    plt.plot(actual.index, actual.values, label='Actual', marker='o')
    plt.plot(actual.index, predictions, label='Predicted', marker='x')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / f'predictions_vs_actual_{timestamp}.png')

    # 7. Save the model
    logger.info("Saving model...")
    model_path = trainer.save_model(f'price_predictor_{timestamp}.joblib')
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()