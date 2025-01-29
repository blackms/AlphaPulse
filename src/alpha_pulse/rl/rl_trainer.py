"""
Reinforcement learning trainer module.

This module provides functionality for training and evaluating RL agents
using stable-baselines3.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn

from .env import TradingEnv, TradingEnvConfig


@dataclass
class ComputeConfig:
    """Configuration for compute resources."""
    n_envs: int = 1
    device: str = "auto"
    n_threads: int = 1


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    hidden_sizes: List[int] = None
    activation_fn: str = "tanh"

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 64]


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    total_timesteps: int = 100_000
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    model_path: str = "trained_models/rl/trading_agent"
    log_path: str = "logs/rl"
    checkpoint_freq: int = 10_000


class RLTrainer:
    """Trainer class for RL agents."""
    
    def __init__(
        self,
        env_config: TradingEnvConfig,
        training_config: TrainingConfig,
        compute_config: Optional[ComputeConfig] = None,
        network_config: Optional[NetworkConfig] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            env_config: Configuration for the trading environment
            training_config: Configuration for training
            compute_config: Configuration for compute resources
            network_config: Configuration for neural network architecture
        """
        self.env_config = env_config
        self.training_config = training_config
        self.compute_config = compute_config or ComputeConfig()
        self.network_config = network_config or NetworkConfig()
        
        # Create directories
        Path(self.training_config.model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.training_config.log_path).mkdir(parents=True, exist_ok=True)
        
    def _get_activation_fn(self) -> type:
        """Get activation function from string name."""
        activation_map = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU
        }
        return activation_map.get(
            self.network_config.activation_fn.lower(),
            nn.Tanh
        )
        
    def _create_model(
        self,
        env: TradingEnv,
        algorithm: str = 'ppo'
    ) -> PPO:
        """
        Create an RL model.
        
        Args:
            env: Trading environment
            algorithm: RL algorithm to use (currently only PPO supported)
            
        Returns:
            Initialized model
        """
        if algorithm.lower() != 'ppo':
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Policy kwargs
        policy_kwargs = dict(
            activation_fn=self._get_activation_fn(),
            net_arch=self.network_config.hidden_sizes
        )
        
        # Initialize model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.training_config.learning_rate,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            gamma=self.training_config.gamma,
            gae_lambda=self.training_config.gae_lambda,
            ent_coef=self.training_config.ent_coef,
            vf_coef=self.training_config.vf_coef,
            max_grad_norm=self.training_config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.training_config.log_path,
            device=self.compute_config.device,
            n_epochs=10,
            verbose=1
        )
        
        return model
        
    def train(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        algorithm: str = 'ppo',
        eval_prices: Optional[pd.Series] = None,
        eval_features: Optional[pd.DataFrame] = None,
        model_path: Optional[str] = None
    ) -> PPO:
        """
        Train an RL agent.
        
        Args:
            prices: Price data for training
            features: Feature data for training
            algorithm: RL algorithm to use
            eval_prices: Price data for evaluation
            eval_features: Feature data for evaluation
            model_path: Path to save the trained model
            
        Returns:
            Trained model
        """
        logger.info("Starting PPO training...")
        
        # Create training environment
        train_env = TradingEnv(
            prices=prices,
            features=features,
            config=self.env_config
        )
        
        # Create evaluation environment if data provided
        eval_env = None
        eval_callback = None
        if eval_prices is not None and eval_features is not None:
            eval_env = TradingEnv(
                prices=eval_prices,
                features=eval_features,
                config=self.env_config
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=model_path or self.training_config.model_path,
                log_path=self.training_config.log_path,
                eval_freq=self.training_config.eval_freq,
                n_eval_episodes=self.training_config.n_eval_episodes,
                deterministic=True,
                render=False
            )
            
        try:
            # Create and train model
            model = self._create_model(train_env, algorithm)
            model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Save final model if no evaluation callback
            if eval_callback is None and model_path:
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate(
        self,
        model: PPO,
        prices: pd.Series,
        features: pd.DataFrame,
        n_episodes: int = 1,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model to evaluate
            prices: Price data for evaluation
            features: Feature data for evaluation
            n_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create evaluation environment
        eval_env = TradingEnv(
            prices=prices,
            features=features,
            config=self.env_config
        )
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                if render:
                    eval_env.render()
                    
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        # Calculate metrics
        metrics = {
            'mean_reward': sum(episode_rewards) / n_episodes,
            'mean_length': sum(episode_lengths) / n_episodes
        }
        
        return metrics