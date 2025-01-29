"""
Demo script for training and evaluating an RL trading agent.

This script demonstrates how to:
1. Load historical data and generate features
2. Create and configure a trading environment
3. Train an RL agent using Stable-Baselines3
4. Evaluate the trained agent's performance
"""
import pandas as pd
from loguru import logger

from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.features.feature_engineering import FeatureEngineer
from alpha_pulse.rl.env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.rl_trainer import RLTrainer, TrainingConfig


def main():
    """Run the RL trading demo."""
    logger.info("Starting RL trading demo...")
    
    # 1. Load historical data
    logger.info("Loading historical data...")
    fetcher = DataFetcher()
    data = fetcher.fetch_historical_data(
        symbol="BTC/USD",
        timeframe="1h",
        start_time="2023-01-01",
        end_time="2023-12-31"
    )
    
    # 2. Generate features
    logger.info("Generating features...")
    engineer = FeatureEngineer()
    features = engineer.calculate_features(data['close'])
    
    # Split data into train and test sets (80/20)
    split_idx = int(len(data) * 0.8)
    
    train_prices = data['close'][:split_idx]
    train_features = features[:split_idx]
    
    test_prices = data['close'][split_idx:]
    test_features = features[split_idx:]
    
    # 3. Configure and create the environment
    logger.info("Creating trading environment...")
    env_config = TradingEnvConfig(
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
        position_size=0.2,  # Risk 20% of capital per trade
        window_size=10,    # Use 10 time steps of history
        reward_scaling=1.0
    )
    
    training_config = TrainingConfig(
        total_timesteps=100000,
        learning_rate=0.0003,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        eval_freq=10000,
        n_eval_episodes=5,
        model_path="trained_models/rl/trading_agent",
        log_path="logs/rl"
    )
    
    # 4. Create trainer and train the agent
    logger.info("Training RL agent...")
    trainer = RLTrainer(env_config, training_config)
    
    try:
        model = trainer.train(
            prices=train_prices,
            features=train_features,
            algorithm='ppo',  # Using PPO algorithm
            eval_prices=test_prices,
            eval_features=test_features,
            model_path=training_config.model_path
        )
        
        # 5. Evaluate the trained agent
        logger.info("Evaluating trained agent...")
        eval_metrics = trainer.evaluate(
            model=model,
            prices=test_prices,
            features=test_features,
            n_episodes=5,
            render=True
        )
        
        logger.info("Evaluation metrics:")
        for metric, value in eval_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()