"""
Tests for reinforcement learning module.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from pathlib import Path
import tempfile
import shutil
import gymnasium as gym
from stable_baselines3 import PPO

from ..rl.env import TradingEnv, TradingEnvConfig
from ..rl.rl_trainer import (
    RLTrainer,
    TrainingConfig,
    ComputeConfig,
    NetworkConfig
)


@pytest.fixture
def sample_data():
    """Fixture for sample trading data."""
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 30, tzinfo=UTC),
        freq='D'
    )
    np.random.seed(42)
    
    # Create price data with less volatility
    prices = pd.Series(
        100 + np.random.randn(len(dates)).cumsum() * 0.1,  # Reduced volatility
        index=dates
    )
    
    # Create feature data
    features = pd.DataFrame({
        'sma': prices.rolling(5).mean(),
        'rsi': np.random.randn(len(dates)),
        'volume': np.abs(np.random.randn(len(dates)) * 1000)
    }, index=dates)
    
    return prices, features


@pytest.fixture
def env_config():
    """Fixture for environment configuration."""
    return TradingEnvConfig(
        initial_capital=100000.0,
        commission=0.001,
        position_size=0.1,
        window_size=5
    )


@pytest.fixture
def training_config(tmp_path):
    """Fixture for training configuration."""
    return TrainingConfig(
        total_timesteps=1000,  # Small number for testing
        model_path=str(tmp_path / "test_model"),
        log_path=str(tmp_path / "test_logs")
    )


@pytest.fixture
def trading_env(sample_data, env_config):
    """Fixture for trading environment."""
    prices, features = sample_data
    return TradingEnv(prices, features, env_config)


@pytest.fixture
def rl_trainer(env_config, training_config):
    """Fixture for RL trainer."""
    compute_config = ComputeConfig(device="cpu")
    network_config = NetworkConfig(hidden_sizes=[32, 32])
    
    return RLTrainer(
        env_config=env_config,
        training_config=training_config,
        compute_config=compute_config,
        network_config=network_config
    )


def test_trading_env_initialization(trading_env, env_config):
    """Test trading environment initialization."""
    assert isinstance(trading_env, gym.Env)
    assert trading_env.action_space.n == 3
    assert trading_env.observation_space.shape == (
        env_config.window_size * (1 + trading_env.features.shape[1]),
    )
    assert trading_env.equity == env_config.initial_capital
    assert trading_env.current_position is None


def test_trading_env_reset(trading_env, env_config):
    """Test environment reset."""
    initial_state, info = trading_env.reset()
    
    assert isinstance(initial_state, np.ndarray)
    assert initial_state.shape == trading_env.observation_space.shape
    assert trading_env.current_step == env_config.window_size
    assert trading_env.equity == env_config.initial_capital
    assert trading_env.current_position is None
    assert isinstance(info, dict)


def test_trading_env_step(trading_env):
    """Test environment step function."""
    trading_env.reset()
    
    # Test buy action
    state, reward, done, truncated, info = trading_env.step(2)  # Buy
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert trading_env.current_position is not None
    assert trading_env.current_position.size > 0  # Long position
    
    # Test hold action
    state, reward, done, truncated, info = trading_env.step(1)  # Hold
    assert isinstance(state, np.ndarray)
    assert trading_env.current_position is not None  # Position maintained
    
    # Test invalid action
    with pytest.raises(ValueError):
        trading_env.step(3)


def test_trading_env_position_management(trading_env):
    """Test position management."""
    trading_env.reset()
    
    # Open long position
    trading_env.step(2)  # Buy
    assert trading_env.current_position is not None
    assert trading_env.current_position.size > 0
    initial_position = trading_env.current_position
    
    # Hold position
    trading_env.step(1)  # Hold
    assert trading_env.current_position is initial_position  # Same position maintained
    
    # Switch to short position
    trading_env.step(0)  # Sell
    assert len(trading_env.positions) == 1  # Previous position added to history
    assert trading_env.positions[0].exit_price is not None
    assert trading_env.current_position is not None
    assert trading_env.current_position.size < 0  # New short position
    assert trading_env.current_position is not initial_position  # Different position object


def test_trading_env_reward_calculation(trading_env):
    """Test reward calculation."""
    trading_env.reset()
    
    # Open position and check reward
    _, reward, _, _, _ = trading_env.step(2)  # Buy
    assert isinstance(reward, float)
    
    # Hold position and check reward
    _, reward, _, _, _ = trading_env.step(1)  # Hold
    assert isinstance(reward, float)


def test_rl_trainer_initialization(env_config, training_config, tmp_path):
    """Test RL trainer initialization."""
    trainer = RLTrainer(
        env_config=env_config,
        training_config=training_config
    )
    
    assert isinstance(trainer.env_config, TradingEnvConfig)
    assert isinstance(trainer.training_config, TrainingConfig)
    assert isinstance(trainer.compute_config, ComputeConfig)
    assert isinstance(trainer.network_config, NetworkConfig)
    
    # Check directory creation
    assert Path(trainer.training_config.model_path).parent.exists()
    assert Path(trainer.training_config.log_path).exists()


def test_rl_trainer_model_creation(rl_trainer, trading_env):
    """Test model creation."""
    model = rl_trainer._create_model(trading_env)
    assert isinstance(model, PPO)
    
    # Test unsupported algorithm
    with pytest.raises(ValueError):
        rl_trainer._create_model(trading_env, algorithm='unsupported')


def test_rl_trainer_training(rl_trainer, sample_data):
    """Test training workflow."""
    prices, features = sample_data
    
    # Basic training without evaluation
    model = rl_trainer.train(
        prices=prices,
        features=features,
        algorithm='ppo'
    )
    assert isinstance(model, PPO)
    
    # Training with evaluation data
    eval_prices = prices.iloc[-10:]
    eval_features = features.iloc[-10:]
    
    model = rl_trainer.train(
        prices=prices,
        features=features,
        algorithm='ppo',
        eval_prices=eval_prices,
        eval_features=eval_features
    )
    assert isinstance(model, PPO)


def test_rl_trainer_evaluation(rl_trainer, sample_data):
    """Test model evaluation."""
    prices, features = sample_data
    
    # Train a model first
    model = rl_trainer.train(prices=prices, features=features)
    
    # Evaluate the model
    metrics = rl_trainer.evaluate(
        model=model,
        prices=prices,
        features=features,
        n_episodes=2
    )
    
    assert isinstance(metrics, dict)
    assert 'mean_reward' in metrics
    assert 'mean_length' in metrics
    assert isinstance(metrics['mean_reward'], float)
    assert isinstance(metrics['mean_length'], float)