"""
Demo script for training and evaluating an RL trading agent.

This script demonstrates how to:
1. Load historical data and generate features
2. Create and configure a trading environment
3. Train an RL agent using Stable-Baselines3
4. Evaluate the trained agent's performance
"""
from datetime import datetime, UTC
import pandas as pd
from loguru import logger

from ..data_pipeline.data_fetcher import DataFetcher
from ..data_pipeline.exchange import Exchange
from ..data_pipeline.storage import SQLAlchemyStorage
from ..features.feature_engineering import calculate_technical_indicators
from ..rl.env import TradingEnv, TradingEnvConfig
from ..rl.rl_trainer import RLTrainer, TrainingConfig


class MockExchangeFactory:
    """Factory for creating mock exchange instances."""
    def create_exchange(self, exchange_id: str) -> Exchange:
        """Create a mock exchange instance."""
        return Exchange()


def main():
    """Run the RL trading demo."""
    logger.info("Starting RL trading demo...")
    
    # 1. Load historical data
    logger.info("Loading historical data...")
    exchange_factory = MockExchangeFactory()
    storage = SQLAlchemyStorage()
    fetcher = DataFetcher(exchange_factory, storage)
    
    # Update historical data
    start_time = datetime(2023, 1, 1, tzinfo=UTC)
    end_time = datetime(2023, 12, 31, tzinfo=UTC)
    days = (end_time - start_time).days
    
    fetcher.update_historical_data(
        exchange_id="mock",
        symbol="BTC/USD",
        timeframe="1h",
        days_back=days,
        end_time=end_time
    )
    
    # Retrieve the data from storage
    ohlcv_data = storage.get_historical_data(
        exchange_id="mock",
        symbol="BTC/USD",
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