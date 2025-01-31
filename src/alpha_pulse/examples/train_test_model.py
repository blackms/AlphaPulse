"""
Quick script to train and save a test model for demo_paper_trading.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Create sample data
np.random.seed(42)
n_samples = 1000

# Generate random features
data = {
    'rsi': np.random.normal(50, 15, n_samples).clip(0, 100),
    'sma_ratio': np.random.normal(1, 0.1, n_samples),
    'volatility': np.abs(np.random.normal(0, 0.02, n_samples)),
    'volume': np.abs(np.random.normal(1000, 300, n_samples))
}

# Generate target (up/down) based on features
df = pd.DataFrame(data)
df['target'] = (df['rsi'] > 70) | (df['sma_ratio'] > 1.05) | (df['volatility'] < 0.01)
df['target'] = df['target'].astype(int)

# Train a simple model
features = ['rsi', 'sma_ratio', 'volatility', 'volume']
X = df[features]
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
model_dir = Path('trained_models')
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / 'crypto_prediction_model.joblib')

print("Test model trained and saved successfully!")