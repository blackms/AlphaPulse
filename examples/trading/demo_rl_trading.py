"""
Advanced RL trading system demonstration.

This script implements a comprehensive RL-based trading system with:
1. Reproducible seeding and configuration management
2. Sophisticated environment creation and data handling
3. Advanced model training with checkpointing
4. Detailed performance analysis and logging
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import random
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta, timezone
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from loguru import logger

from alpha_pulse.data_pipeline.models import MarketData
from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.trainer import RLTrainer, NetworkConfig, TrainingConfig
from alpha_pulse.rl.features import FeatureEngineer
from alpha_pulse.rl.utils import RewardParams, calculate_trade_statistics
from alpha_pulse.data_pipeline.data_fetcher import DataFetcher
from alpha_pulse.data_pipeline.storage.sql import SQLAlchemyStorage
from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.types import ExchangeType


@dataclass
class ExperimentConfig:
    """Configuration container for the RL trading experiment."""
    environment: Dict[str, Any]
    network: Dict[str, Any]
    training: Dict[str, Any]
    data: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


class MetricsCallback(BaseCallback):
    """Callback for logging training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_history: List[Dict[str, float]] = []
        
    def _on_step(self) -> bool:
        """Log metrics at each step."""
        metrics = {
            'reward_mean': float(np.mean(self.locals['rewards'])),
            'reward_std': float(np.std(self.locals['rewards'])),
            'value_loss': float(self.locals.get('value_loss', 0)),
            'policy_loss': float(self.locals.get('policy_loss', 0))
        }
        self.metrics_history.append(metrics)
        return True


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


async def fetch_historical_data(symbol: str, days: int = 365) -> MarketData:
    """
    Fetch historical market data from Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        days: Number of days of data to fetch
        
    Returns:
        MarketData object with historical data
    """
    # Load Binance credentials
    with open("src/alpha_pulse/exchanges/credentials/binance_credentials.json", "r") as f:
        credentials = json.load(f)
    api_key = credentials["api_key"]
    api_secret = credentials["api_secret"]
    
    # Initialize data storage and exchange factory
    storage = SQLAlchemyStorage()
    exchange_factory = ExchangeFactory()
    
    # Create exchange instance
    exchange = exchange_factory.create_exchange(
        exchange_type=ExchangeType.BINANCE,
        api_key=api_key,
        api_secret=api_secret
    )
    await exchange.initialize()
    
    # Calculate start date
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    # Fetch historical data
    ohlcv_data = await exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe="1d",
        since=int(start_date.timestamp() * 1000),
        limit=days
    )
    
    # Convert OHLCV data to DataFrame with float values
    df = pd.DataFrame([
        {
            'timestamp': ohlcv.timestamp,
            'open': float(ohlcv.open),
            'high': float(ohlcv.high),
            'low': float(ohlcv.low),
            'close': float(ohlcv.close),
            'volume': float(ohlcv.volume)
        }
        for ohlcv in ohlcv_data
    ]).set_index('timestamp')
    
    # Calculate features
    feature_engineer = FeatureEngineer(window_size=100)
    features = feature_engineer.calculate_features(df)
    
    return MarketData(
        prices=features,
        volumes=df[['volume']],
        timestamp=end_date
    )


def prepare_data(data: MarketData, eval_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for RL training.
    
    Args:
        data: Market data
        eval_split: Fraction of data to use for evaluation
        
    Returns:
        Tuple of (train_data, eval_data)
        
    Raises:
        ValueError: If eval_split is not between 0 and 1
    """
    if not 0 < eval_split < 1:
        raise ValueError("eval_split must be between 0 and 1")
        
    split_idx = int(len(data.prices) * (1 - eval_split))
    train_data = data.prices[:split_idx]
    eval_data = data.prices[split_idx:]
    
    logger.info(f"Prepared {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    logger.debug(f"Train data shape: {train_data.shape}, Eval data shape: {eval_data.shape}")
    
    return train_data, eval_data


def create_trading_env(
    market_data: MarketData,
    config: Dict[str, Any]
) -> TradingEnv:
    """
    Create trading environment with specified configuration.
    
    Args:
        market_data: Market data for the environment
        config: Environment configuration dictionary
        
    Returns:
        Configured TradingEnv instance
    """
    env_config = TradingEnvConfig(**config)
    return TradingEnv(market_data, config=env_config)


def evaluate_and_save_trades(
    model: Any,
    eval_data: pd.DataFrame,
    env_config: TradingEnvConfig
) -> pd.DataFrame:
    """
    Evaluate model and save detailed trade information.
    
    Args:
        model: Trained RL model
        eval_data: Evaluation market data
        env_config: Environment configuration
        
    Returns:
        DataFrame containing trade information
    """
    env = create_trading_env(
        MarketData(prices=eval_data, volumes=None, timestamp=None),
        env_config.__dict__
    )
    
    trades = []
    obs = env.reset()[0]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        if info['trade_executed'] and env.positions:
            latest_trade = env.positions[-1]
            trades.append({
                'entry_time': latest_trade.timestamp,
                'exit_time': latest_trade.exit_time,
                'duration': latest_trade.exit_time - latest_trade.timestamp,
                'entry_price': latest_trade.avg_entry_price,
                'exit_price': latest_trade.exit_price,
                'position_size': latest_trade.quantity,
                'profit': latest_trade.pnl,
                'equity': info['equity']
            })
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Save trades and metrics
        Path("reports").mkdir(parents=True, exist_ok=True)
        trades_df.to_csv("reports/rl_trades.csv", index=False)
        
        # Calculate and save additional metrics
        metrics = calculate_trade_statistics(trades_df)
        with open("reports/final_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Saved {len(trades)} trades and metrics to reports/")
        return trades_df
    else:
        logger.warning("No trades were executed during evaluation")
        return pd.DataFrame()


async def main():
    """Main execution function."""
    try:
        # Set up logging
        log_path = Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            "logs/rl_trading.log",
            rotation="1 day",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # Load configuration
        config = ExperimentConfig.from_yaml("examples/trading/rl_config.yaml")
        logger.info("Loaded configuration")
        logger.debug(f"Config: {config}")
        
        # Set random seeds
        set_random_seeds(config.data.get('random_seed', 42))
        
        # Generate and prepare data
        market_data = await fetch_historical_data(
            symbol=config.data['symbol'],
            days=365  # Fetch last 365 days
        )
        train_data, eval_data = prepare_data(
            market_data,
            eval_split=config.data['eval_split']
        )
        
        # Initialize trainer components
        env_config = TradingEnvConfig(**config.environment)
        network_config = NetworkConfig(**config.network)
        training_config = TrainingConfig(**config.training)
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=training_config.model_path,
            name_prefix="rl_model"
        )
        
        # Create metrics callback
        metrics_callback = MetricsCallback()
        
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
            n_envs=4,
            callback=checkpoint_callback
        )
        
        # Save training metrics history
        pd.DataFrame(metrics_callback.metrics_history).to_csv(
            "reports/training_metrics.csv",
            index=False
        )
        
        # Evaluate model and save trades
        logger.info("Evaluating model and collecting trades...")
        trades_df = evaluate_and_save_trades(model, eval_data, env_config)
        
        # Get standard evaluation metrics
        metrics = trainer.evaluate(
            model=model,
            eval_data=eval_data,
            n_episodes=10
        )
        
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except ValueError as ve:
        logger.error(f"Value error in RL trading demo: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RL trading demo: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())