"""
Script to analyze trades made by a trained RL model.

This script demonstrates how to:
1. Load a trained model
2. Run it on market data
3. Analyze the resulting trades
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
    """Generate synthetic market data for testing."""
    # Generate timestamps
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Generate synthetic price data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, days)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.normal(0, 0.002, days)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.004, days))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.004, days))),
        'close': price,
        'volume': np.random.lognormal(10, 1, days)
    }).set_index('timestamp')
    
    # Calculate features
    feature_engineer = FeatureEngineer(window_size=100)
    features = feature_engineer.calculate_features(df)
    
    return MarketData(
        prices=features,
        volumes=df[['volume']],
        timestamp=end_date
    )


def analyze_trades(model, eval_data: pd.DataFrame, env_config: TradingEnvConfig) -> dict:
    """
    Analyze trades made by the model.
    
    Args:
        model: Trained RL model
        eval_data: Evaluation market data
        env_config: Environment configuration
        
    Returns:
        Dictionary containing trade analysis
    """
    # Create evaluation environment
    env = TradingEnv(MarketData(prices=eval_data, volumes=None, timestamp=None), env_config)
    
    # Run evaluation episode
    obs, info = env.reset()
    done = False
    trades = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        # Record trade if one was executed
        if info['trade_executed'] and info['position'] is not None:
            trades.append({
                'timestamp': eval_data.index[env.current_step],
                'action': action,
                'price': float(eval_data['close'].iloc[env.current_step]),
                'quantity': info['position'].quantity,
                'profit': info['realized_pnl'],
                'duration': info['position'].exit_time - info['position'].timestamp if info['position'].exit_time else 0
            })
    
    # Calculate trade statistics
    stats = calculate_trade_statistics(trades)
    
    # Add additional analysis
    if trades:
        df_trades = pd.DataFrame(trades)
        stats.update({
            'total_profit': df_trades['profit'].sum(),
            'avg_trade_duration': df_trades['duration'].mean(),
            'max_profit_trade': df_trades['profit'].max(),
            'max_loss_trade': df_trades['profit'].min(),
            'profit_factor': abs(df_trades[df_trades['profit'] > 0]['profit'].sum() / 
                               df_trades[df_trades['profit'] < 0]['profit'].sum())
                               if len(df_trades[df_trades['profit'] < 0]) > 0 else float('inf')
        })
        
        # Save trades to CSV for further analysis
        df_trades.to_csv('reports/rl_trades.csv')
        
    return stats


def main():
    # Set up logging
    logger.add("logs/trade_analysis.log", rotation="1 day")
    
    try:
        # Generate or load market data
        market_data = generate_mock_data(days=365)
        
        # Prepare data
        split_idx = int(len(market_data.prices) * 0.8)
        train_data = market_data.prices[:split_idx]
        eval_data = market_data.prices[split_idx:]
        
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
            total_timesteps=100_000,  # Reduced for demonstration
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
        
        # Analyze trades
        logger.info("Analyzing trades...")
        trade_analysis = analyze_trades(model, eval_data, env_config)
        
