"""
Demo script for training and evaluating an RL trading agent.

This script demonstrates how to:
1. Load historical data and generate features
2. Create and configure a trading environment
3. Train an RL agent using Stable-Baselines3
4. Evaluate the trained agent's performance
"""
from datetime import datetime, UTC, timedelta
import pandas as pd
from loguru import logger

from ..data_pipeline.data_fetcher import DataFetcher
from ..data_pipeline.exchange import CCXTExchangeFactory
from ..data_pipeline.storage import SQLAlchemyStorage
from ..features.feature_engineering import calculate_technical_indicators
from ..rl.env import TradingEnv, TradingEnvConfig
from ..rl.rl_trainer import RLTrainer, TrainingConfig, ComputeConfig, NetworkConfig


def main():
    """Run the RL trading demo."""
    logger.info("Starting RL trading demo...")
    
    # 1. Load historical data
    logger.info("Loading historical data...")
    exchange_factory = CCXTExchangeFactory()
    storage = SQLAlchemyStorage()
    fetcher = DataFetcher(exchange_factory, storage)
    
    # Update historical data - use 30 days for faster training
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=30)
    days = (end_time - start_time).days
    
    fetcher.update_historical_data(
        exchange_id="binance",
        symbol="BTC/USDT",  # Using USDT pair for Binance
        timeframe="1h",
        days_back=days,
        end_time=end_time
    )
    
    # Retrieve the data from storage
    ohlcv_data = storage.get_historical_data(
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start_time=start_time,
        end_time=end_time
    )
    
    # Convert OHLCV objects to DataFrame
    df = pd.DataFrame([{
        'timestamp': candle.timestamp,
        'open': candle.open,
        'high': candle.high,
        'low': candle.low,
        'close': candle.close,
        'volume': candle.volume
    } for candle in ohlcv_data])
    
    if len(df) == 0:
        logger.error("No data retrieved from storage")
        return
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # 2. Generate features
    logger.info("Generating features...")
    features = calculate_technical_indicators(df)
    
    # Split data into train and test sets (80/20)
    split_idx = int(len(df) * 0.8)
    
    train_prices = df['close'][:split_idx]
    train_features = features[:split_idx]
    
    test_prices = df['close'][split_idx:]
    test_features = features[split_idx:]
    
    # 3. Configure environment and training settings
    logger.info("Creating trading environment...")
    
    # Environment configuration
    env_config = TradingEnvConfig(
        initial_capital=100000.0,
        commission=0.001,     # 0.1% commission
        position_size=0.2,    # Risk 20% of capital per trade
        window_size=10,       # Use 10 time steps of history
        reward_scaling=1.0
    )
    
    # Network configuration
    network_config = NetworkConfig(
        hidden_sizes=[64, 64],  # Simple network architecture
        activation_fn="tanh"
    )
    
    # Training configuration
    training_config = TrainingConfig(
        total_timesteps=10000,  # Reduced for faster training
        learning_rate=0.0003,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        eval_freq=1000,
        n_eval_episodes=2,
        model_path="trained_models/rl/trading_agent",
        log_path="logs/rl"
    )
    
    # 4. Create trainer and train the agent
    logger.info("Training RL agent...")
    trainer = RLTrainer(
        env_config=env_config,
        training_config=training_config,
        network_config=network_config
    )
    
    try:
        model = trainer.train(
            prices=train_prices,
            features=train_features,
            algorithm='ppo',
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
            n_episodes=5,  # More evaluation episodes
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