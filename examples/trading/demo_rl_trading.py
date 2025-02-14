"""
Example script demonstrating the RL trading system.

This script shows how to:
1. Load and preprocess market data
2. Configure and create the RL environment
3. Train an RL agent
4. Evaluate its performance
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from alpha_pulse.data_pipeline.models import MarketData
from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.trainer import RLTrainer, NetworkConfig, TrainingConfig
from alpha_pulse.rl.features import FeatureEngineer
from alpha_pulse.rl.utils import RewardParams, calculate_trade_statistics


def generate_mock_data(days: int = 365) -> MarketData:
    """
    Generate synthetic market data for testing.
    
    Args:
        days: Number of days of data to generate
        
    Returns:
        MarketData object with synthetic data
    """
    # Generate timestamps
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Generate synthetic price data with more realistic trends
    np.random.seed(42)
    # Add a slight upward bias and more volatility
    returns = np.random.normal(0.0002, 0.025, days)
    # Add some trending behavior
    trend = np.sin(np.linspace(0, 4*np.pi, days)) * 0.001
    returns += trend
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data with more realistic patterns
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.normal(0, 0.003, days)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.005, days))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.005, days))),
        'close': price,
        'volume': np.random.lognormal(10, 1.2, days)  # More variable volume
    }).set_index('timestamp')
    
    # Calculate features
    feature_engineer = FeatureEngineer(window_size=100)
    features = feature_engineer.calculate_features(df)
    
    return MarketData(
        prices=features,
        volumes=df[['volume']],
        timestamp=end_date
    )


def prepare_data(data: MarketData) -> tuple:
    """
    Prepare data for RL training.
    
    Args:
        data: Market data
        
    Returns:
        tuple: (train_data, eval_data)
    """
    # Split into train/eval sets (80/20)
    split_idx = int(len(data.prices) * 0.8)
    train_data = data.prices[:split_idx]
    eval_data = data.prices[split_idx:]
    
    logger.info(f"Prepared {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    return train_data, eval_data


def main():
    # Set up logging
    logger.add("logs/rl_trading.log", rotation="1 day")
    
    try:
        # Generate synthetic data
        market_data = generate_mock_data(days=365)
        
        # Prepare data
        train_data, eval_data = prepare_data(market_data)
        
        # Configure environment with more aggressive trading parameters
        env_config = TradingEnvConfig(
            initial_capital=100000.0,
            commission=0.001,
            position_size=0.3,  # Increased position size
            window_size=10,
            reward_scaling=2.0,  # Increased reward scaling
            risk_aversion=0.05,  # Reduced risk aversion
            max_position=5.0,
            stop_loss_pct=0.03,  # Increased stop loss
            take_profit_pct=0.06  # Increased take profit
        )
        
        # Configure neural network with deeper architecture
        network_config = NetworkConfig(
            hidden_sizes=[256, 128, 64],  # Larger network
            activation_fn="relu",
            use_lstm=True,
            lstm_units=128,  # Increased LSTM capacity
            attention_heads=4,
            dropout_rate=0.1
        )
        
        # Configure training with emphasis on exploration
        training_config = TrainingConfig(
            total_timesteps=1_000_000,
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,  # Increased entropy coefficient for more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            eval_freq=10_000,
            n_eval_episodes=10,
            model_path="trained_models/rl",
            log_path="logs/rl"
        )
        
        # Initialize trainer
        trainer = RLTrainer(
            env_config=env_config,
            network_config=network_config,
            training_config=training_config
        )
        
        # Train model
        logger.info("Starting training...")
        model = trainer.train(
            train_data=train_data,
            eval_data=eval_data,
            n_envs=4
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(
            model=model,
            eval_data=eval_data,
            n_episodes=10
        )
        
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error in RL trading demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()