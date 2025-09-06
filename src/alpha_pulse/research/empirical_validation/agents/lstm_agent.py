"""
LSTM-based Time Series Forecasting Agent for empirical validation.

Implements a simplified version of the LSTM forecasting agent described in the paper.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    sequence_length: int = 24  # 24 hours lookback
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    forecast_horizon: int = 1  # 1-hour ahead prediction


class PriceDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, data: np.ndarray, sequence_length: int, forecast_horizon: int = 1):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMPriceForecaster(nn.Module):
    """
    LSTM model for price and volatility forecasting.
    
    Architecture:
    - Input: [batch, sequence, features]
    - LSTM layers with dropout
    - Output: [batch, forecast_horizon, targets]
    """
    
    def __init__(self, input_size: int, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.price_head = nn.Linear(config.hidden_size, config.forecast_horizon)
        self.volatility_head = nn.Linear(config.hidden_size, config.forecast_horizon)
        self.confidence_head = nn.Linear(config.hidden_size, config.forecast_horizon)
        
        # Activation for confidence (sigmoid to get [0,1])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for predictions
        last_output = lstm_out[:, -1, :]
        
        # Generate predictions
        price_pred = self.price_head(last_output)
        volatility_pred = torch.relu(self.volatility_head(last_output))  # Volatility must be positive
        confidence = self.sigmoid(self.confidence_head(last_output))
        
        return {
            'price': price_pred,
            'volatility': volatility_pred, 
            'confidence': confidence
        }


class LSTMForecastingAgent:
    """
    LSTM-based Time-Series Forecasting Agent (TSFA) from the paper.
    
    Features:
    - Price and volatility forecasting
    - Confidence estimation
    - Online learning capability
    - Feature engineering from OHLCV data
    """
    
    def __init__(self, config: LSTMConfig = None):
        """
        Initialize LSTM agent.
        
        Args:
            config: LSTM configuration parameters
        """
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        
        logger.info(f"Initialized LSTM agent with config: {self.config}")
        
    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Engineer features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Engineered feature matrix
        """
        features = []
        
        # Price features
        features.extend([
            data['close'].values,
            data['high'].values,
            data['low'].values,
            data['volume'].values,
        ])
        
        # Technical indicators
        # Returns
        returns = data['close'].pct_change().fillna(0)
        log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        features.extend([returns.values, log_returns.values])
        
        # Moving averages
        ma_5 = data['close'].rolling(5, min_periods=1).mean()
        ma_20 = data['close'].rolling(20, min_periods=1).mean()
        features.extend([ma_5.values, ma_20.values])
        
        # Volatility features
        volatility = returns.rolling(24, min_periods=1).std().fillna(0)
        features.append(volatility.values)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50).values)
        
        # Price position within range
        high_max = data['high'].rolling(24, min_periods=1).max()
        low_min = data['low'].rolling(24, min_periods=1).min()
        price_position = (data['close'] - low_min) / (high_max - low_min + 1e-8)
        features.append(price_position.fillna(0.5).values)
        
        return np.column_stack(features)
    
    def _prepare_targets(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare target variables (price and volatility).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Target matrix [price, volatility]
        """
        # Price targets (log returns for better stability)
        price_targets = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        
        # Volatility targets (realized volatility)
        returns = data['close'].pct_change().fillna(0)
        vol_targets = returns.rolling(self.config.forecast_horizon, min_periods=1).std().fillna(0)
        
        return np.column_stack([price_targets.values, vol_targets.values])
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train the LSTM model on historical data.
        
        Args:
            data: Historical market data
            validation_split: Fraction of data for validation
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Training LSTM model on {len(data)} data points...")
        
        # Engineer features and targets
        X = self._engineer_features(data)
        y = self._prepare_targets(data)
        
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Create datasets
        train_dataset = PriceDataset(
            np.concatenate([X_train, y_train], axis=1),
            self.config.sequence_length,
            self.config.forecast_horizon
        )
        val_dataset = PriceDataset(
            np.concatenate([X_val, y_val], axis=1),
            self.config.sequence_length,
            self.config.forecast_horizon
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_scaled.shape[1]
        self.model = LSTMPriceForecaster(input_size, self.config).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                # Split input and target
                batch_X_features = batch_X[:, :, :input_size].to(self.device)
                batch_y_targets = batch_X[:, -1, input_size:].to(self.device)  # Only price and vol targets
                
                optimizer.zero_grad()
                outputs = self.model(batch_X_features)
                
                # Combine price and volatility predictions
                predictions = torch.cat([outputs['price'], outputs['volatility']], dim=1)
                loss = criterion(predictions, batch_y_targets)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X_features = batch_X[:, :, :input_size].to(self.device)
                    batch_y_targets = batch_X[:, -1, input_size:].to(self.device)
                    
                    outputs = self.model(batch_X_features)
                    predictions = torch.cat([outputs['price'], outputs['volatility']], dim=1)
                    loss = criterion(predictions, batch_y_targets)
                    val_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")
        
        self.is_trained = True
        logger.info("Training completed!")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions for the next time step.
        
        Args:
            data: Recent market data (must have at least sequence_length points)
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(data) < self.config.sequence_length:
            raise ValueError(f"Need at least {self.config.sequence_length} data points for prediction")
        
        # Use the most recent data
        recent_data = data.tail(self.config.sequence_length)
        
        # Engineer features
        X = self._engineer_features(recent_data)
        X_scaled = self.scaler_X.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        # Generate prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        # Convert predictions back to original scale (simplified)
        price_pred = outputs['price'].cpu().numpy().flatten()
        volatility_pred = outputs['volatility'].cpu().numpy().flatten()
        confidence = outputs['confidence'].cpu().numpy().flatten()
        
        return {
            'price_forecast': price_pred,
            'volatility_forecast': volatility_pred,
            'confidence': confidence,
            'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else len(data)
        }
    
    def get_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals from LSTM predictions.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary with trading signals
        """
        if not self.is_trained:
            return {'price_signal': 0.0, 'volatility_signal': 0.0, 'confidence': 0.0}
        
        predictions = self.predict(data)
        
        # Convert predictions to signals
        # Price signal: positive if expecting price increase
        price_forecast = predictions['price_forecast'][0]
        price_signal = np.tanh(price_forecast * 10)  # Scale to [-1, 1]
        
        # Volatility signal: higher volatility suggests more uncertainty
        vol_forecast = predictions['volatility_forecast'][0]
        current_vol = data['close'].pct_change().tail(24).std()
        vol_ratio = vol_forecast / (current_vol + 1e-8)
        vol_signal = np.clip(vol_ratio, 0.1, 3.0)  # Scale volatility signal
        
        confidence = predictions['confidence'][0]
        
        return {
            'price_signal': float(price_signal),
            'volatility_signal': float(vol_signal), 
            'confidence': float(confidence),
            'raw_price_forecast': float(price_forecast),
            'raw_vol_forecast': float(vol_forecast)
        }


if __name__ == "__main__":
    # Example usage
    from ..simulation.market_simulator import MarketDataSimulator
    
    # Generate synthetic data
    simulator = MarketDataSimulator(random_seed=42)
    data = simulator.generate_dataset(n_days=10)
    
    # Initialize and train LSTM agent
    config = LSTMConfig(epochs=20, batch_size=16)
    agent = LSTMForecastingAgent(config)
    
    # Train on first 80% of data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    history = agent.train(train_data)
    
    # Generate predictions
    signals = agent.get_signals(test_data)
    print("LSTM signals:", signals)