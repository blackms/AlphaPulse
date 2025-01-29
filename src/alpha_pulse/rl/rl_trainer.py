"""
Reinforcement learning trainer module with configurable compute settings.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pandas as pd
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
import torch.nn as nn
import multiprocessing

from .env import TradingEnv, TradingEnvConfig


@dataclass
class ComputeConfig:
    """Configuration for compute resources."""
    n_envs: int = multiprocessing.cpu_count()  # Number of parallel environments
    device: str = "auto"  # 'auto', 'cuda', 'cuda:0', 'cpu'
    n_threads: int = multiprocessing.cpu_count()  # Number of threads for training


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    policy_network: List[int] = (256, 256)  # Policy network architecture
    value_network: List[int] = (256, 256)   # Value network architecture
    activation_fn: str = "relu"  # Activation function: 'relu', 'tanh', 'elu'
    shared_layers: bool = False  # Whether to use shared layers for policy and value


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    total_timesteps: int = 1_000_000
    learning_rate: float = 0.0003
    batch_size: int = 256
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
    checkpoint_freq: int = 50_000  # Save checkpoints every n steps


class RLTrainer:
    """Enhanced trainer class for RL agents with configurable compute settings."""
    
    def __init__(
        self,
        env_config: TradingEnvConfig,
        training_config: TrainingConfig,
        compute_config: Optional[ComputeConfig] = None,
        network_config: Optional[NetworkConfig] = None
    ):
        """Initialize the trainer with configurable settings."""
        self.env_config = env_config
        self.training_config = training_config
        self.compute_config = compute_config or ComputeConfig()
        self.network_config = network_config or NetworkConfig()
        
        # Set up compute device
        if self.compute_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.compute_config.device
        
        logger.info(f"Using device: {self.device}")
        torch.set_num_threads(self.compute_config.n_threads)
        
    def _make_env(self, prices: pd.Series, features: pd.DataFrame, rank: int) -> callable:
        """Create a vectorized environment factory."""
        def _init() -> TradingEnv:
            env = TradingEnv(
                prices=prices,
                features=features,
                config=self.env_config
            )
            return env
        return _init
        
    def _create_model(
        self,
        env: Union[SubprocVecEnv, DummyVecEnv],
        algorithm: str = 'ppo'
    ) -> PPO:
        """Create an RL model with configurable architecture."""
        if algorithm.lower() != 'ppo':
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        # Set activation function
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
        }
        activation_fn = activation_map.get(
            self.network_config.activation_fn.lower(),
            nn.ReLU
        )
        
        # Configure network architecture
        if self.network_config.shared_layers:
            net_arch = [
                *self.network_config.policy_network,
                dict(
                    pi=self.network_config.policy_network,
                    vf=self.network_config.value_network
                )
            ]
        else:
            net_arch = dict(
                pi=self.network_config.policy_network,
                vf=self.network_config.value_network
            )
        
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=net_arch
        )
        
        # Initialize model with enhanced settings
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
            device=self.device,
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
        """Train an RL agent with enhanced compute capabilities."""
        logger.info(f"Starting {algorithm.upper()} training with {self.compute_config.n_envs} environments...")
        
        # Create vectorized training environment
        env_fns = [self._make_env(prices, features, i) for i in range(self.compute_config.n_envs)]
        if self.compute_config.n_envs > 1:
            train_env = SubprocVecEnv(env_fns)
        else:
            train_env = DummyVecEnv(env_fns)
        
        # Set up evaluation
        eval_callback = None
        if eval_prices is not None and eval_features is not None:
            eval_env = DummyVecEnv([self._make_env(eval_prices, eval_features, 0)])
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=model_path or self.training_config.model_path,
                log_path=self.training_config.log_path,
                eval_freq=self.training_config.eval_freq,
                n_eval_episodes=self.training_config.n_eval_episodes,
                deterministic=True
            )
            
        try:
            model = self._create_model(train_env, algorithm)
            model.learn(
                total_timesteps=self.training_config.total_timesteps,
                callback=eval_callback,
                progress_bar=True
            )
            
            if eval_callback is None and model_path:
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            train_env.close()
            if eval_callback:
                eval_callback.eval_env.close()

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