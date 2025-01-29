"""
Tests for the reinforcement learning components.
"""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from alpha_pulse.rl.env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.rl_trainer import RLTrainer, TrainingConfig


@pytest.fixture
def sample_data():
    """Create sample price and feature data for testing."""
    # Create 100 time steps of data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    
    # Generate sample price data with some trend and noise
    prices = pd.Series(
        np.linspace(100, 120, 100) + np.random.normal(0, 1, 100),
        index=dates
    )
    
    # Generate sample features (e.g., moving averages)
    features = pd.DataFrame({
        'sma': prices.rolling(window=5).mean(),
        'std': prices.rolling(window=5).std()
    }).fillna(0)
    
    return prices, features


def test_trading_env_initialization(sample_data):
    """Test that the trading environment initializes correctly."""
    prices, features = sample_data
    
    env = TradingEnv(
        prices=prices,
        features=features,
        config=TradingEnvConfig(window_size=5)
    )
    
    # Check spaces
    assert env.action_space.n == 3  # sell, hold, buy
    assert env.observation_space.shape == (15,)  # 5 time steps * (1 price + 2 features)
    
    # Check initial reset
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    assert info['equity'] == env.config.initial_capital
    assert info['position'] is None


def test_trading_env_step(sample_data):
    """Test the environment step function."""
    prices, features = sample_data
    
    env = TradingEnv(
        prices=prices,
        features=features,
        config=TradingEnvConfig(window_size=5)
    )
    
    # Reset and take a step
    obs, _ = env.reset()
    next_obs, reward, done, truncated, info = env.step(2)  # Buy action
    
    # Check observation
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (15,)
    
    # Check reward is float
    assert isinstance(reward, float)
    
    # Check info contains expected keys
    assert 'equity' in info
    assert 'realized_pnl' in info
    assert 'position' in info


def test_trading_env_position_management(sample_data):
    """Test that positions are managed correctly."""
    prices, features = sample_data
    
    env = TradingEnv(
        prices=prices,
        features=features,
        config=TradingEnvConfig(
            window_size=5,
            position_size=1.0,
            commission=0.001
        )
    )
    
    # Reset and open position
    env.reset()
    _, _, _, _, info = env.step(2)  # Buy
    assert info['position'] is not None
    
    # Close position
    _, _, _, _, info = env.step(0)  # Sell
    assert info['position'] is None
    assert isinstance(info['realized_pnl'], float)


def test_trading_env_done_condition(sample_data):
    """Test that environment properly signals episode completion."""
    prices, features = sample_data
    
    env = TradingEnv(
        prices=prices,
        features=features,
        config=TradingEnvConfig(window_size=5)
    )
    
    env.reset()
    done = False
    steps = 0
    
    while not done:
        _, _, done, _, _ = env.step(1)  # Hold action
        steps += 1
    
    assert steps == len(prices) - 6  # window_size + 1


def test_rl_trainer_initialization():
    """Test RLTrainer initialization."""
    trainer = RLTrainer()
    assert trainer.env_config is not None
    assert trainer.training_config is not None


def test_rl_trainer_basic_training(sample_data):
    """Test that basic training runs without errors."""
    prices, features = sample_data
    
    trainer = RLTrainer(
        training_config=TrainingConfig(
            total_timesteps=1000,  # Small number for testing
            eval_freq=500
        )
    )
    
    # Train for a small number of steps
    model = trainer.train(
        prices=prices,
        features=features,
        algorithm='ppo',
        eval_prices=prices,
        eval_features=features
    )
    
    assert isinstance(model, PPO)


def test_rl_trainer_evaluation(sample_data):
    """Test model evaluation."""
    prices, features = sample_data
    
    trainer = RLTrainer(
        training_config=TrainingConfig(total_timesteps=1000)
    )
    
    # Train and evaluate
    model = trainer.train(prices=prices, features=features, algorithm='ppo')
    metrics = trainer.evaluate(
        model=model,
        prices=prices,
        features=features,
        n_episodes=2
    )
    
    assert 'mean_reward' in metrics
    assert 'std_reward' in metrics
    assert 'mean_length' in metrics
    assert 'std_length' in metrics


def test_invalid_data():
    """Test that invalid data raises appropriate errors."""
    # Test mismatched indices
    prices = pd.Series([1, 2, 3])
    features = pd.DataFrame({'f1': [1, 2, 3, 4]})
    
    with pytest.raises(ValueError, match="must have matching timestamps"):
        TradingEnv(prices=prices, features=features)
    
    # Test negative prices
    prices = pd.Series([-1, 2, 3])
    features = pd.DataFrame({'f1': [1, 2, 3]}, index=prices.index)
    
    with pytest.raises(ValueError, match="must be positive"):
        TradingEnv(prices=prices, features=features)