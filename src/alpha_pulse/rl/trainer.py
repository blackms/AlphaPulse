"""
Advanced RL trainer module for AlphaPulse.

This module provides sophisticated training capabilities for RL agents with
features like curriculum learning, experience replay, and distributed training.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from alpha_pulse.data_pipeline.models import MarketData
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from loguru import logger

from .trading_env import TradingEnv, TradingEnvConfig


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    hidden_sizes: List[int] = None
    activation_fn: str = "relu"
    use_lstm: bool = True
    lstm_units: int = 64
    attention_heads: int = 4
    dropout_rate: float = 0.1

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]
            
    @staticmethod
    def _get_activation(name: str) -> type:
        """Get activation function from string name."""
        activation_map = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU
        }
        return activation_map.get(name.lower(), nn.ReLU)


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    model_path: str = "trained_models/rl"
    log_path: str = "logs/rl"
    checkpoint_freq: int = 10_000  # Added for checkpointing
    n_envs: int = 4  # Added for parallel environments
    device: str = "auto"  # Added for device configuration
    cuda_deterministic: bool = True  # Added for CUDA settings
    precision: str = "float32"  # Added for precision settings


class CustomNetwork(nn.Module):
    """
    Advanced neural network architecture with LSTM and attention mechanisms.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: NetworkConfig
    ):
        """
        Initialize the network.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output action dimension
            config: Network configuration
        """
        super().__init__()
        
        # Feature extraction layers
        layers = []
        prev_size = input_dim
        for size in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                self._get_activation(config.activation_fn),
                nn.Dropout(config.dropout_rate)
            ])
            prev_size = size
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # LSTM layer
        self.use_lstm = config.use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.hidden_sizes[-1],
                hidden_size=config.lstm_units,
                batch_first=True
            )
            prev_size = config.lstm_units
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=prev_size,
            num_heads=config.attention_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Output layers
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            self._get_activation(config.activation_fn),
            nn.Linear(prev_size // 2, output_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            self._get_activation(config.activation_fn),
            nn.Linear(prev_size // 2, 1)
        )
        
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            hidden_states: LSTM hidden states
            
        Returns:
            tuple containing:
            - action_logits: Action probability logits
            - value: State value estimate
            - new_hidden_states: Updated LSTM hidden states
        """
        # Feature extraction
        features = self.feature_extractor(x)
        
        # LSTM processing
        if self.use_lstm:
            if hidden_states is None:
                batch_size = x.size(0)
                hidden_states = (
                    torch.zeros(1, batch_size, self.lstm.hidden_size, device=x.device),
                    torch.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
                )
            features = features.unsqueeze(1)  # Add sequence dimension
            features, new_hidden_states = self.lstm(features, hidden_states)
            features = features.squeeze(1)  # Remove sequence dimension
        else:
            new_hidden_states = None
            
        # Self-attention
        features = features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(features, features, features)
        features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Output heads
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return action_logits, value, new_hidden_states


class RLTrainer:
    """Advanced RL trainer with sophisticated training capabilities."""
    
    def __init__(
        self,
        env_config: TradingEnvConfig,
        network_config: NetworkConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            env_config: Trading environment configuration
            network_config: Neural network configuration
            training_config: Training configuration
            device: Optional torch device for training
        """
        self.env_config = env_config
        self.network_config = network_config
        self.training_config = training_config
        self.device = device or self._get_device(training_config)
        
        # Create directories
        Path(training_config.model_path).mkdir(parents=True, exist_ok=True)
        Path(training_config.log_path).mkdir(parents=True, exist_ok=True)
        
    def _get_device(self, config: TrainingConfig) -> torch.device:
        """Determine the appropriate device based on configuration."""
        if config.device == "cpu":
            return torch.device("cpu")
        elif config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device("cpu")
        
    def _make_env(self, market_data: pd.DataFrame, rank: int = 0) -> gym.Env:
        """Create a trading environment instance."""
        def _init():
            # Create MarketData object from DataFrame
            data = MarketData(
                prices=market_data,
                volumes=market_data[['volume']] if 'volume' in market_data.columns else None,
                timestamp=market_data.index[-1] if isinstance(market_data.index, pd.DatetimeIndex) else None
            )
            env = TradingEnv(data, self.env_config)
            return env
        return _init
        
    def _create_envs(
        self,
        market_data: pd.DataFrame,
        n_envs: int = 4
    ) -> gym.Env:
        """Create vectorized environments for parallel training."""
        env_fns = [self._make_env(market_data, i) for i in range(n_envs)]
        
        if n_envs == 1:
            return DummyVecEnv(env_fns)
        else:
            return SubprocVecEnv(env_fns)
            
    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        n_envs: Optional[int] = None,
        callbacks: Optional[List[BaseCallback]] = None
    ) -> PPO:
        """
        Train an RL agent.
        
        Args:
            train_data: Training market data
            eval_data: Evaluation market data
            n_envs: Number of parallel environments (overrides config)
            callbacks: Optional list of callbacks
            
        Returns:
            Trained PPO model
        """
        logger.info("Starting RL training...")
        
        # Use n_envs from config if not specified
        n_envs = n_envs or self.training_config.n_envs
        
        # Create environments
        train_env = self._create_envs(train_data, n_envs)
        eval_env = None if eval_data is None else self._create_envs(eval_data, 1)
        
        # Configure network architecture
        policy_kwargs = {
            "activation_fn": self.network_config._get_activation(self.network_config.activation_fn),
            "net_arch": dict(
                pi=self.network_config.hidden_sizes,
                vf=self.network_config.hidden_sizes
            )
        }
        
        # Initialize model
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=self.training_config.learning_rate,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            clip_range=self.training_config.clip_range,
            ent_coef=self.training_config.ent_coef,
            vf_coef=self.training_config.vf_coef,
            max_grad_norm=self.training_config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.training_config.log_path,
            device=self.device,
            verbose=1
        )
        
        # Set up evaluation callback if not provided in callbacks
        if eval_env is not None and not any(isinstance(cb, EvalCallback) for cb in (callbacks or [])):
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.training_config.model_path,
                log_path=self.training_config.log_path,
                eval_freq=self.training_config.eval_freq,
                n_eval_episodes=self.training_config.n_eval_episodes,
                deterministic=True
            )
            callbacks = callbacks or []
            callbacks.append(eval_callback)
            
        try:
            # Train model
            model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model if no evaluation
            if eval_env is None:
                model.save(f"{self.training_config.model_path}/final_model")
                
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Clean up CUDA memory if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    def evaluate(
        self,
        model: PPO,
        eval_data: pd.DataFrame,
        n_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained PPO model
            eval_data: Evaluation market data
            n_episodes: Number of evaluation episodes (overrides config)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use n_episodes from config if not specified
        n_episodes = n_episodes or self.training_config.n_eval_episodes
        
        # Create evaluation environment
        eval_env = self._create_envs(eval_data, 1)
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()[0]  # Updated for gymnasium API
            done = False
            episode_reward = 0
            episode_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action[0])  # Updated for gymnasium API
            episode_reward += reward
            episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        # Calculate metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths))
        }
        
        return metrics
