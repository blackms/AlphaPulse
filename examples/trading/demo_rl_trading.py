"""
Example script demonstrating the RL trading system.

This script shows how to:
1. Load and preprocess market data
2. Configure and create the RL environment
3. Train an RL agent
4. Evaluate its performance
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.trainer import RLTrainer, NetworkConfig, TrainingConfig
from alpha_pulse.rl.features import FeatureEngineer
from alpha_pulse.rl.utils import RewardParams, calculate_trade_statistics
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher


def prepare_data(data_fetcher: DataFetcher, symbol: str) -> tuple:
    """
    Prepare data for RL training.
    
    Args:
        data_fetcher: DataFetcher instance
        symbol: Trading symbol
        
    Returns:
        tuple: (train_data, eval_data)
    """
    # Fetch historical data
    historical_data = data_fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe="1h",
        start_time="2023-01-01",
        end_time="2024-01-01"
    )
    
    # Calculate features
    feature_engineer = FeatureEngineer(window_size=100)
    features = feature_engineer.calculate_features(historical_data)
    
    # Split into train/eval sets (80/20)
    split_idx = int(len(features) * 0.8)
    train_data = features[:split_idx]
    eval_data = features[split_idx:]
    
    logger.info(f"Prepared {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    return train_data, eval_data


def main():
    # Set up logging
    logger.add("logs/rl_trading.log", rotation="1 day")
    
    try:
        # Initialize data fetcher
        data_fetcher = DataFetcher()
        
        # Prepare data
        train_data, eval_data = prepare_data(data_fetcher, "BTC/USDT")
        
        # Configure environment
        env_config = TradingEnvConfig(
            initial_capital=100000.0,
            commission=0.001,
            position_size=0.2,
            window_size=10,
            reward_scaling=1.0,
            risk_aversion=0.1,
            max_position=5.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )
        
        # Configure neural network
        network_config = NetworkConfig(
            hidden_sizes=[128, 64, 32],
            activation_fn="relu",
            use_lstm=True,
            lstm_units=64,
            attention_heads=4,
            dropout_rate=0.1
        )
        
        # Configure training
        training_config = TrainingConfig(
            total_timesteps=1_000_000,
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
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