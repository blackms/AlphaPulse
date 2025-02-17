"""
PPO-based Trend Following Strategy.

This strategy implements a trend-following approach using PPO (Proximal Policy Optimization)
with the following key features:
1. Technical indicator-based state space
2. Position sizing based on trend strength
3. Risk management with stop-loss and take-profit
4. Multi-timeframe trend analysis
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Literal
import json
import random
import yaml
import numpy as np
import pandas as pd
import torch
import signal
import os
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


# Global flag for graceful shutdown
should_exit = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global should_exit
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    should_exit = True


def check_cuda_support(force_gpu: bool = False) -> torch.device:
    """
    Check CUDA support and return appropriate device.
    
    Args:
        force_gpu: If True, raise error when GPU is not available
        
    Returns:
        torch.device: Device to use for training
        
    Raises:
        RuntimeError: If force_gpu is True but CUDA is not available
    """
    # For PPO with MlpPolicy, CPU is recommended
    logger.info("Using CPU for PPO with MlpPolicy as recommended by stable-baselines3")
    if force_gpu:
        logger.warning(
            "GPU flag is set but using CPU anyway as PPO with MlpPolicy is optimized for CPU. "
            "See: https://github.com/DLR-RM/stable-baselines3/issues/1245"
        )
    return torch.device("cpu")


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
        return not should_exit  # Stop training if shutdown requested


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


def get_model_paths(
    algorithm: Literal["ppo", "dqn", "a2c"],
    asset_class: Literal["crypto", "stocks", "forex"],
    model_name: str
) -> Dict[str, Path]:
    """
    Generate paths for model artifacts based on algorithm and asset class.
    
    Args:
        algorithm: RL algorithm type
        asset_class: Asset class for trading
        model_name: Name of the model
        
    Returns:
        Dictionary containing paths for model, checkpoints, and logs
    """
    try:
        base_path = Path("trained_models/rl") / algorithm / asset_class
        model_path = base_path / f"{model_name}"
        checkpoint_path = base_path / "checkpoints" / model_name
        log_path = Path("logs/rl") / algorithm / asset_class / model_name
        
        # Create directories with detailed logging
        logger.info("Creating required directories...")
        
        for path, desc in [
            (base_path, "base model directory"),
            (model_path.parent, "model directory"),
            (checkpoint_path, "checkpoint directory"),
            (log_path, "log directory")
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created/verified {desc} at: {path}")
            except PermissionError as pe:
                logger.error(f"Permission denied when creating {desc}: {str(pe)}")
                logger.info("Please ensure you have write permissions for these directories")
                raise
            except Exception as e:
                logger.error(f"Failed to create {desc}: {str(e)}")
                raise
        
        paths = {
            "model": model_path,
            "checkpoint": checkpoint_path,
            "log": log_path
        }
        
        logger.info("Directory setup complete:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")
            
        return paths
        
    except Exception as e:
        logger.error(f"Error setting up model paths: {str(e)}")
        raise


async def fetch_historical_data(symbol: str, days: int = 365) -> MarketData:
    """
    Fetch historical market data from Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        days: Number of days of data to fetch
        
    Returns:
        MarketData object with historical data
    """
    exchange = None
    try:
        logger.info(f"Fetching historical data for {symbol} over {days} days")
        
        # Load Binance credentials
        creds_path = "src/alpha_pulse/exchanges/credentials/binance_credentials.json"
        try:
            with open(creds_path, "r") as f:
                credentials = json.load(f)
            api_key = credentials["api_key"]
            api_secret = credentials["api_secret"]
            logger.info("Loaded API credentials successfully")
        except FileNotFoundError:
            logger.error(f"Credentials file not found at: {creds_path}")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in credentials file")
            raise
        except KeyError as e:
            logger.error(f"Missing required credential: {e}")
            raise
        
        # Initialize data storage and exchange factory
        storage = SQLAlchemyStorage()
        exchange_factory = ExchangeFactory()
        logger.debug("Initialized storage and exchange factory")
        
        # Create exchange instance
        exchange = exchange_factory.create_exchange(
            exchange_type=ExchangeType.BINANCE,
            api_key=api_key,
            api_secret=api_secret
        )
        await exchange.initialize()
        logger.info("Exchange initialized successfully")
        
        # Calculate start date
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        logger.debug(f"Date range: {start_date} to {end_date}")
        
        # Fetch historical data
        logger.info("Fetching OHLCV data...")
        ohlcv_data = await exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe="1d",
            since=int(start_date.timestamp() * 1000),
            limit=days
        )
        
        if not ohlcv_data:
            raise ValueError(f"No data returned for {symbol}")
            
        logger.info(f"Received {len(ohlcv_data)} data points")
        
        # Convert OHLCV data to DataFrame with float values
        logger.debug("Converting data to DataFrame")
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
        
        # Log data statistics
        logger.debug("Data statistics:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            stats = df[col].describe()
            logger.debug(f"  {col}:")
            logger.debug(f"    Mean: {stats['mean']:.2f}")
            logger.debug(f"    Std: {stats['std']:.2f}")
            logger.debug(f"    Min: {stats['min']:.2f}")
            logger.debug(f"    Max: {stats['max']:.2f}")
        
        # Calculate features
        logger.info("Calculating technical features")
        feature_engineer = FeatureEngineer(window_size=100)
        features = feature_engineer.calculate_features(df)
        logger.info(f"Generated {len(features.columns)} features")
        
        # Create MarketData object
        market_data = MarketData(
            prices=features,
            volumes=df[['volume']],
            timestamp=end_date
        )
        logger.info("Historical data fetching completed successfully")
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise
        
    finally:
        if exchange:
            try:
                await exchange.close()
                logger.debug("Exchange connection closed")
            except Exception as e:
                logger.error(f"Error closing exchange connection: {str(e)}")


def prepare_data(data: MarketData, eval_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for RL training.
    
    Args:
        data: Market data
        eval_split: Fraction of data to use for evaluation
        
    Returns:
        Tuple of (train_data, eval_data)
        
    Raises:
        ValueError: If eval_split is not between 0 and 1 or if data validation fails
    """
    logger.info("Preparing data for training...")
    
    try:
        # Validate input data
        if not isinstance(data, MarketData):
            raise ValueError("Input must be a MarketData object")
            
        if not isinstance(data.prices, pd.DataFrame):
            raise ValueError("Prices must be a pandas DataFrame")
            
        if data.prices.empty:
            raise ValueError("Price data is empty")
            
        # Validate eval_split
        if not 0 < eval_split < 1:
            raise ValueError(f"eval_split must be between 0 and 1, got {eval_split}")
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.prices.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for NaN values
        nan_counts = data.prices.isna().sum()
        if nan_counts.any():
            logger.warning("Found NaN values in data:")
            for col, count in nan_counts.items():
                if count > 0:
                    logger.warning(f"  {col}: {count} NaN values")
            logger.info("Filling NaN values with forward fill method")
            data.prices = data.prices.ffill()
            
        # Calculate split index
        total_samples = len(data.prices)
        split_idx = int(total_samples * (1 - eval_split))
        
        if split_idx <= 0 or split_idx >= total_samples:
            raise ValueError(f"Invalid split index {split_idx} for {total_samples} samples")
            
        # Split data
        train_data = data.prices[:split_idx]
        eval_data = data.prices[split_idx:]
        
        # Log data statistics
        logger.info("Data preparation statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Training samples: {len(train_data)} ({(1-eval_split)*100:.1f}%)")
        logger.info(f"  Evaluation samples: {len(eval_data)} ({eval_split*100:.1f}%)")
        logger.info(f"  Features: {len(data.prices.columns)}")
        
        # Log data shapes
        logger.debug("Data shapes:")
        logger.debug(f"  Train data: {train_data.shape}")
        logger.debug(f"  Eval data: {eval_data.shape}")
        
        # Log basic statistics for both sets
        for name, df in [("Training", train_data), ("Evaluation", eval_data)]:
            logger.debug(f"\n{name} set statistics:")
            for col in df.columns:
                stats = df[col].describe()
                logger.debug(f"  {col}:")
                logger.debug(f"    Mean: {stats['mean']:.2f}")
                logger.debug(f"    Std: {stats['std']:.2f}")
                logger.debug(f"    Min: {stats['min']:.2f}")
                logger.debug(f"    Max: {stats['max']:.2f}")
        
        logger.info("Data preparation completed successfully")
        return train_data, eval_data
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


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
        
    Raises:
        ValueError: If configuration validation fails
    """
    logger.info("Creating trading environment...")
    
    try:
        # Validate market data
        if not isinstance(market_data, MarketData):
            raise ValueError("market_data must be a MarketData instance")
            
        if not isinstance(market_data.prices, pd.DataFrame):
            raise ValueError("market_data.prices must be a DataFrame")
            
        if market_data.prices.empty:
            raise ValueError("market_data.prices is empty")
            
        # Validate configuration
        required_config = {
            'initial_capital', 'commission', 'position_size', 'window_size',
            'reward_scaling', 'risk_aversion', 'max_position',
            'stop_loss_pct', 'take_profit_pct'
        }
        
        missing_config = required_config - set(config.keys())
        if missing_config:
            raise ValueError(f"Missing required configuration parameters: {missing_config}")
            
        # Log configuration
        logger.info("Environment configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
            
        # Create environment config
        logger.debug("Creating environment configuration...")
        env_config = TradingEnvConfig(**config)
        
        # Log market data statistics
        logger.info("Market data statistics:")
        logger.info(f"  Total timesteps: {len(market_data.prices)}")
        logger.info(f"  Features: {len(market_data.prices.columns)}")
        if isinstance(market_data.prices.index, pd.DatetimeIndex):
            logger.info(f"  Time range: {market_data.prices.index[0]} to {market_data.prices.index[-1]}")
        
        # Create environment
        logger.debug("Initializing trading environment...")
        env = TradingEnv(market_data, config=env_config)
        
        # Verify environment spaces
        logger.debug("Environment spaces:")
        logger.debug(f"  Action space: {env.action_space}")
        logger.debug(f"  Observation space: {env.observation_space}")
        
        logger.info("Trading environment created successfully")
        return env
        
    except Exception as e:
        logger.error(f"Error creating trading environment: {str(e)}")
        raise


def evaluate_and_save_trades(
    model: Any,
    eval_data: pd.DataFrame,
    env_config: TradingEnvConfig,
    save_path: Path
) -> pd.DataFrame:
    """
    Evaluate model and save detailed trade information.
    
    Args:
        model: Trained RL model
        eval_data: Evaluation market data
        env_config: Environment configuration
        save_path: Path to save evaluation results
        
    Returns:
        DataFrame containing trade information
    """
    env = create_trading_env(
        MarketData(prices=eval_data, volumes=None, timestamp=None),
        env_config.__dict__
    )
    
    # Create evaluation log file
    save_path.mkdir(parents=True, exist_ok=True)
    eval_log_path = save_path / "evaluation_steps.log"
    
    trades = []
    attempted_trades = []
    obs = env.reset()[0]
    done = False
    step = 0
    
    logger.info("Starting evaluation...")
    
    while not done and not should_exit:
        # Get model prediction and log it
        action, _states = model.predict(obs, deterministic=True)
        action_value = action.item() if isinstance(action, np.ndarray) else action
        
        # Execute step
        obs, reward, done, _, info = env.step(action_value)
        
        # Log step details
        step_info = {
            'step': step,
            'action': action_value,
            'reward': reward,
            'equity': info['equity'],
            'position': str(info['position']),
            'trade_executed': info['trade_executed'],
            'realized_pnl': info['realized_pnl']
        }
        
        # Save step info to log
        with open(eval_log_path, 'a') as f:
            f.write(f"Step {step}: {json.dumps(step_info)}\n")
            
        # Track attempted trades
        if info['trade_executed']:
            attempted_trades.append({
                'step': step,
                'action': action_value,
                'position': info['position'],
                'realized_pnl': info['realized_pnl'],
                'equity': info['equity']
            })
            
            # If trade was completed (has position history), add to trades list
            if env.positions:
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
        
        step += 1
        
    # Save all evaluation data
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(save_path / "completed_trades.csv", index=False)
        
        # Calculate and save metrics for completed trades
        metrics = calculate_trade_statistics(trades_df)
        with open(save_path / "trade_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Saved {len(trades)} completed trades and metrics to {save_path}")
    else:
        logger.warning("No completed trades during evaluation")
        
    # Save attempted trades
    if attempted_trades:
        attempted_df = pd.DataFrame(attempted_trades)
        attempted_df.to_csv(save_path / "attempted_trades.csv", index=False)
        logger.info(f"Saved {len(attempted_trades)} attempted trades to {save_path}")
    else:
        logger.warning("No trade attempts during evaluation")
        
    # Log evaluation summary
    logger.info(f"Evaluation completed:")
    logger.info(f"Total steps: {step}")
    logger.info(f"Attempted trades: {len(attempted_trades)}")
    logger.info(f"Completed trades: {len(trades)}")
    logger.info(f"Final equity: {info['equity']}")
    
    return pd.DataFrame(trades)


async def main(
    algorithm: Literal["ppo", "dqn", "a2c"] = "ppo",
    asset_class: Literal["crypto", "stocks", "forex"] = "crypto",
    model_name: str = "default",
    use_gpu: bool = False
):
    """
    Main execution function.
    
    Args:
        algorithm: RL algorithm to use
        asset_class: Asset class to trade
        model_name: Name for the model
        use_gpu: If True, force GPU usage and raise error if not available
    """
    # Remove signal handlers to avoid BrokenPipeError
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    
    exchange = None
    model = None
    env = None
    
    try:
        # Get model paths
        paths = get_model_paths(algorithm, asset_class, model_name)
        
        # Set up logging
        logger.add(
            str(paths["log"] / "training.log"),
            rotation="1 day",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

        # Load configuration
        # Load configuration
        config = ExperimentConfig.from_yaml("examples/trading/rl/ppo_strategies/trend_following/config.yaml")
        
        # Update paths in config
        config.training["model_path"] = str(paths["model"])
        config.training["checkpoint_path"] = str(paths["checkpoint"])
        config.training["log_path"] = str(paths["log"])
        
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
        training_config = TrainingConfig(
            total_timesteps=config.training['total_timesteps'],
            learning_rate=config.training['learning_rate'],
            batch_size=config.training['batch_size'],
            n_steps=config.training['n_steps'],
            gamma=config.training['gamma'],
            gae_lambda=config.training['gae_lambda'],
            clip_range=config.training['clip_range'],
            ent_coef=config.training['ent_coef'],
            vf_coef=config.training['vf_coef'],
            max_grad_norm=config.training['max_grad_norm'],
            eval_freq=config.training['eval_freq'],
            n_eval_episodes=config.training['n_eval_episodes'],
            checkpoint_freq=config.training['checkpoint_freq'],
            n_envs=config.training['n_envs'],
            device=config.training['device'],
            cuda_deterministic=config.training['cuda_deterministic'],
            precision=config.training['precision'],
            model_path=config.training['model_path'],
            log_path=config.training['log_path']
        )
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.checkpoint_freq,
            save_path=paths["checkpoint"],
            name_prefix=model_name
        )
        
        # Create metrics callback
        metrics_callback = MetricsCallback()
        
        # Initialize trainer with device configuration
        device = check_cuda_support(use_gpu)
        logger.info(f"Using device: {device}")
        
        # Log detailed configuration
        logger.info("\nTraining Configuration:")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Asset Class: {asset_class}")
        logger.info(f"Model Name: {model_name}")
        logger.info("Environment Config:")
        for k, v in env_config.__dict__.items():
            logger.info(f"  {k}: {v}")
        logger.info("\nNetwork Config:")
        for k, v in network_config.__dict__.items():
            logger.info(f"  {k}: {v}")
        logger.info("\nTraining Config:")
        for k, v in training_config.__dict__.items():
            logger.info(f"  {k}: {v}")
            
        trainer = RLTrainer(
            env_config=env_config,
            network_config=network_config,
            training_config=training_config,
            device=device
        )
        
        # Train model
        logger.info("Starting training...")
        model = trainer.train(
            train_data=train_data,
            eval_data=eval_data,
            callbacks=[checkpoint_callback, metrics_callback]
        )
        
        # Save training metrics history
        metrics_df = pd.DataFrame(metrics_callback.metrics_history)
        metrics_df.to_csv(
            paths["log"] / "training_metrics.csv",
            index=False
        )
        
        if not should_exit:
            # Evaluate model and save trades
            logger.info("Evaluating model and collecting trades...")
            trades_df = evaluate_and_save_trades(
                model,
                eval_data,
                env_config,
                paths["log"] / "evaluation"
            )
            
            # Get standard evaluation metrics
            metrics = trainer.evaluate(
                model=model,
                eval_data=eval_data,
                n_episodes=training_config.n_eval_episodes
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
    finally:
        # Clean up resources
        if model:
            try:
                model.env.close()
            except:
                pass
        if env:
            try:
                env.close()
            except:
                pass
        # Clean up CUDA memory if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        import asyncio
        import sys
        import argparse
        import torch
        from pathlib import Path
        from loguru import logger
        print("Successfully imported all required packages")
    except ImportError as e:
        print(f"Failed to import required packages: {str(e)}")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PPO Trend Following Strategy")
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "dqn", "a2c"],
        default="ppo",
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--asset-class",
        choices=["crypto", "stocks", "forex"],
        default="crypto",
        help="Asset class to trade"
    )
    parser.add_argument(
        "--model-name",
        default="default",
        help="Name for the model"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage. Will raise error if GPU is not available."
    )
    args = parser.parse_args()
    
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU requested: {args.gpu}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    print("===========================\n")
    logger.info("Initializing PPO Trend Following Strategy")
    
    try:
        if sys.platform == 'win32':
            print("Setting Windows event loop policy...")
            from asyncio import WindowsSelectorEventLoopPolicy
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
            
        print("Running main function...")
        asyncio.run(main(
            algorithm=args.algorithm,
            asset_class=args.asset_class,
            model_name=args.model_name,
            use_gpu=args.gpu
        ))
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down gracefully...")
        should_exit = True
    except RuntimeError as re:
        print(f"Runtime error: {str(re)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise