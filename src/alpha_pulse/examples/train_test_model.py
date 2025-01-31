"""
Quick script to train and save a test model for demo_paper_trading.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from datetime import datetime, timedelta

from alpha_pulse.features.feature_engineering import calculate_technical_indicators


# Create sample historical data
def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    current_time = datetime.now()
    dates = [current_time - timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate random walk prices
    base_price = 50000  # Starting price
    returns = np.random.normal(0, 0.001, n_samples)  # 0.1% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 - 0.0005),  # Slight spread
        'high': prices * (1 + 0.001),
        'low': prices * (1 - 0.001),
        'close': prices,
        'volume': np.random.normal(100, 10, n_samples)
    })
    df.set_index('timestamp', inplace=True)
    
    return df

# Generate sample data
df = generate_sample_data(1000)

# Calculate technical indicators
features_df = calculate_technical_indicators(df)

# Create target (1 if price goes up in next period, 0 otherwise)
features_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop rows with NaN values
features_df.dropna(inplace=True)

# Prepare features and target
X = features_df.drop('target', axis=1)
y = features_df['target']

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X, y)

# Save the model
model_dir = Path('trained_models')
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / 'crypto_prediction_model.joblib')

print(f"Model trained on {len(features_df)} samples with features: {list(X.columns)}")
print("Test model trained and saved successfully!")