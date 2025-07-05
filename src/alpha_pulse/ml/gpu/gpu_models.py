"""
GPU-Optimized ML Models for Trading.

This module provides GPU-accelerated implementations of machine learning
models commonly used in algorithmic trading.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # Placeholder

logger = logging.getLogger(__name__)


class GPUModel(ABC):
    """Base class for GPU-accelerated models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def to_gpu(self, device_id: int = 0):
        """Move model to GPU."""
        pass


class GPULinearRegression(GPUModel):
    """GPU-accelerated linear regression using PyTorch."""
    
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPULinearRegression")
        
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.model = nn.Linear(n_features, 1)
        self.device = torch.device('cpu')
        self.optimizer = None
        self.criterion = nn.MSELoss()
    
    def to_gpu(self, device_id: int = 0):
        """Move model to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
            logger.info(f"Model moved to GPU {device_id}")
        else:
            logger.warning("CUDA not available, using CPU")
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Fit the model on GPU."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_tensor = X_tensor[perm]
            y_tensor = y_tensor[perm]
            
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                
                # Forward pass
                outputs = self.model(X_tensor[start:end])
                loss = self.criterion(outputs, y_tensor[start:end])
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {epoch_loss/n_batches:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on GPU."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()


class GPUNeuralNetwork(nn.Module, GPUModel):
    """GPU-accelerated neural network for trading predictions."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int = 1,
                 dropout: float = 0.2,
                 activation: str = 'relu'):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPUNeuralNetwork")
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.device = torch.device('cpu')
        self.optimizer = None
        self.scaler = GradScaler()  # For mixed precision training
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def to_gpu(self, device_id: int = 0):
        """Move model to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
            self.to(self.device)
            logger.info(f"Neural network moved to GPU {device_id}")
        else:
            logger.warning("CUDA not available, using CPU")
        return self
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            epochs: int = 100,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            use_amp: bool = True):
        """
        Train the neural network on GPU with mixed precision.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional (X_val, y_val) tuple
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_amp: Use automatic mixed precision
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, self.output_size).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Validation data
        if validation_data:
            X_val, y_val = validation_data
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, self.output_size).to(self.device)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                
                # Mixed precision training
                if use_amp and self.device.type == 'cuda':
                    with autocast():
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            if validation_data and epoch % 10 == 0:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                
                logger.debug(f"Epoch {epoch}, Train Loss: {epoch_loss/len(loader):.6f}, "
                           f"Val Loss: {val_loss.item():.6f}")
                self.train()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on GPU."""
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self(X_tensor)
            return predictions.cpu().numpy()


class GPULSTMModel(nn.Module, GPUModel):
    """GPU-accelerated LSTM for time series prediction."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPULSTMModel")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        self.device = torch.device('cpu')
        self.optimizer = None
    
    def forward(self, x, lengths=None):
        """Forward pass with optional sequence lengths."""
        # LSTM forward
        if lengths is not None:
            # Pack sequences for variable length handling
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)
        
        # Take last output for each sequence
        if self.bidirectional:
            # Combine forward and backward
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = lstm_out[:, -1, :]
        
        # Output projection
        output = self.fc(lstm_out)
        return output
    
    def to_gpu(self, device_id: int = 0):
        """Move model to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
            self.to(self.device)
            logger.info(f"LSTM model moved to GPU {device_id}")
        else:
            logger.warning("CUDA not available, using CPU")
        return self
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sequence_lengths: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001):
        """
        Train LSTM on GPU.
        
        Args:
            X: Input sequences (batch, seq_len, features)
            y: Target values
            sequence_lengths: Optional actual lengths of sequences
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, self.output_size).to(self.device)
        
        if sequence_lengths is not None:
            lengths_tensor = torch.LongTensor(sequence_lengths)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, lengths_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in loader:
                if sequence_lengths is not None:
                    batch_X, batch_y, batch_lengths = batch
                    outputs = self(batch_X, batch_lengths)
                else:
                    batch_X, batch_y = batch
                    outputs = self(batch_X)
                
                loss = criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {epoch_loss/len(loader):.6f}")
    
    def predict(self, X: np.ndarray, sequence_lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions on GPU."""
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            if sequence_lengths is not None:
                lengths_tensor = torch.LongTensor(sequence_lengths)
                predictions = self(X_tensor, lengths_tensor)
            else:
                predictions = self(X_tensor)
            
            return predictions.cpu().numpy()


class GPUTransformerModel(nn.Module, GPUModel):
    """GPU-accelerated Transformer for sequence modeling."""
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 output_size: int = 1,
                 max_seq_length: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPUTransformerModel")
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
        self.device = torch.device('cpu')
        self.optimizer = None
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x, mask=mask)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        output = self.output_projection(x)
        return output
    
    def to_gpu(self, device_id: int = 0):
        """Move model to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
            self.to(self.device)
            self.positional_encoding = self.positional_encoding.to(self.device)
            logger.info(f"Transformer model moved to GPU {device_id}")
        else:
            logger.warning("CUDA not available, using CPU")
        return self
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 16,
            learning_rate: float = 0.0001,
            warmup_steps: int = 1000):
        """Train Transformer on GPU with learning rate scheduling."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, self.output_size).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer with warmup
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        criterion = nn.MSELoss()
        
        # Training loop
        self.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {epoch_loss/len(loader):.6f}, "
                           f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on GPU."""
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self(X_tensor)
            return predictions.cpu().numpy()


class MultiGPUWrapper:
    """Wrapper for multi-GPU training using DataParallel or DistributedDataParallel."""
    
    def __init__(self, model: nn.Module, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU wrapper.
        
        Args:
            model: PyTorch model
            device_ids: List of GPU device IDs to use
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MultiGPUWrapper")
        
        self.base_model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        
        if len(self.device_ids) > 1:
            # Use DataParallel for multi-GPU
            self.model = DataParallel(model, device_ids=self.device_ids)
            logger.info(f"Using DataParallel with GPUs: {self.device_ids}")
        else:
            self.model = model
            if self.device_ids:
                model.to(f'cuda:{self.device_ids[0]}')
    
    def get_model(self) -> nn.Module:
        """Get the wrapped model."""
        return self.model
    
    def get_base_model(self) -> nn.Module:
        """Get the base model (unwrapped)."""
        if isinstance(self.model, DataParallel):
            return self.model.module
        return self.model