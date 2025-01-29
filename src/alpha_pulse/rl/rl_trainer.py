"""
Reinforcement learning trainer module for AlphaPulse.

This module provides functionality to train RL agents using the TradingEnv.
It uses Stable-Baselines3 for implementing common RL algorithms.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, Union
import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv, TradingEnvConfig


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_steps: int = 2048  # For PPO/A2C
    gamma: float = 0.99  # Discount factor
    train_freq: int = 1
    gradient_steps: int = 1
    save_freq: int = 10000
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    model_path: str = "trained_models/rl"
    log_path: str = "logs/rl"


class RLTrainer:
    """Handles training and evaluation of RL agents for trading."""
    
    SUPPORTED_ALGORITHMS = {
        'ppo': PPO,
        'a2c': A2C,
        'dqn': DQN
    }
    
    def __init__(
        self,
        env_config: Optional[TradingEnvConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the RL trainer.
        
        Args:
            env_config: Configuration for the trading environment
            training_config: Configuration for training
        """
        self.env_config = env_config or TradingEnvConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Create directories if they don't exist
        Path(self.training_config.model_path).mkdir(parents=True, exist_ok=True)
        Path(self.training_config.log_path).mkdir(parents=True, exist_ok=True)
        
    def _create_env(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        is_eval: bool = False
    ) -> DummyVecEnv:
        """Create a vectorized environment."""
        def make_env():
            env = TradingEnv(
                prices=prices,
                features=features,
                config=self.env_config
            )
            return env
            
        return DummyVecEnv([make_env])
        
    def _get_model_class(
        self,
        algorithm: str
    ) -> Type[BaseAlgorithm]:
        """Get the model class for the specified algorithm."""
        algo = algorithm.lower()
        if algo not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported algorithms: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )
        return self.SUPPORTED_ALGORITHMS[algo]
        
    def train(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        algorithm: str = 'ppo',
        eval_prices: Optional[pd.Series] = None,
        eval_features: Optional[pd.DataFrame] = None,
        model_path: Optional[str] = None
    ) -> BaseAlgorithm:
        """
        Train an RL agent.
        
        Args:
            prices: Training price data
            features: Training feature data
            algorithm: RL algorithm to use ('ppo', 'a2c', or 'dqn')
            eval_prices: Optional evaluation price data
            eval_features: Optional evaluation feature data
            model_path: Optional path to save the model
            
        Returns:
            Trained model
        """
        logger.info(f"Starting {algorithm.upper()} training...")
        
        # Create environments
        train_env = self._create_env(prices, features)
        
        # Create evaluation environment if data is provided
        eval_env = None
        eval_callback = None
        if eval_prices is not None and eval_features is not None:
            eval_env = self._create_env(
                eval_prices,
                eval_features,
                is_eval=True
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.training_config.model_path,
                log_path=self.training_config.log_path,
                eval_freq=self.training_config.eval_freq,
                n_eval_episodes=self.training_config.n_eval_episodes,
                deterministic=True
            )
        
        # Initialize model
        model_class = self._get_model_class(algorithm)
        model = model_class(
            "MlpPolicy",
            train_env,
            learning_rate=self.training_config.learning_rate,
            batch_size=self.training_config.batch_size,
            n_steps=self.training_config.n_steps,
            gamma=self.training_config.gamma,
            verbose=1,
            tensorboard_log=self.training_config.log_path
        )
        
        # Train the model
        try:
            model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Save the final model
            if model_path:
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
        return model
        
    def load_model(
        self,
        path: str,
        algorithm: str = 'ppo'
    ) -> BaseAlgorithm:
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            algorithm: Algorithm type of the saved model
            
        Returns:
            Loaded model
        """
        model_class = self._get_model_class(algorithm)
        return model_class.load(path)
        
    def evaluate(
        self,
        model: BaseAlgorithm,
        prices: pd.Series,
        features: pd.DataFrame,
        n_episodes: int = 1,
        render: bool = False
    ) -> dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained RL model
            prices: Price data for evaluation
            features: Feature data for evaluation
            n_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            dict: Evaluation metrics
        """
        env = self._create_env(prices, features, is_eval=True)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
                    
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }