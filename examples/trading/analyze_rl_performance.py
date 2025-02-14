"""
Script to analyze the performance of a trained RL trading model.
This script evaluates:
1. Trading statistics (win rate, profit factor, etc.)
2. Risk-adjusted metrics (Sharpe, Sortino, max drawdown)
3. Position analysis (sizes, durations, PnL)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from loguru import logger

from stable_baselines3 import PPO

from alpha_pulse.data_pipeline.models import MarketData
from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.features import FeatureEngineer
from alpha_pulse.rl.utils import calculate_trade_statistics

def plot_equity_curve(trades_df: pd.DataFrame, save_path: str):
    """Plot cumulative returns over time."""
    cumulative_returns = (1 + trades_df['profit']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, cumulative_returns)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_trade_distribution(trades_df: pd.DataFrame, save_path: str):
    """Plot distribution of trade profits."""
    plt.figure(figsize=(12, 6))
    plt.hist(trades_df['profit'], bins=50)
    plt.title('Trade Profit Distribution')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_drawdown(trades_df: pd.DataFrame, save_path: str):
    """Plot drawdown over time."""
    cumulative_returns = (1 + trades_df['profit']).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, drawdown)
    plt.title('Drawdown')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown %')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def analyze_model_performance(
    model_path: str,
    eval_data: pd.DataFrame,
    env_config: TradingEnvConfig,
    plots_dir: str = 'plots/rl'
):
    """
    Analyze trading performance of a trained model.
    
    Args:
        model_path: Path to saved model
        eval_data: Market data for evaluation
        env_config: Trading environment configuration
        plots_dir: Directory to save performance plots
    """
    # Create plots directory
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = TradingEnv(
        MarketData(prices=eval_data, volumes=None, timestamp=None),
        env_config
    )
    
    # Run evaluation episode
    obs, info = env.reset()
    done = False
    trades = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        if info['trade_executed'] and info['position'] is not None:
            trades.append({
                'timestamp': eval_data.index[env.current_step],
                'action': action,
                'price': float(eval_data['close'].iloc[env.current_step]),
                'quantity': info['position'].quantity,
                'profit': info['realized_pnl'],
                'duration': info['position'].exit_time - info['position'].timestamp if info['position'].exit_time else 0
            })
    
    if not trades:
        logger.warning("No trades were executed during evaluation")
        return
        
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate trade statistics
    stats = calculate_trade_statistics(trades)
    
    # Generate performance plots
    plot_equity_curve(trades_df, f'{plots_dir}/equity_curve.png')
    plot_trade_distribution(trades_df, f'{plots_dir}/trade_distribution.png')
    plot_drawdown(trades_df, f'{plots_dir}/drawdown.png')
    
    # Print performance metrics
    logger.info("\nTrading Performance Metrics:")
    logger.info(f"Total Trades: {stats['total_trades']}")
    logger.info(f"Win Rate: {stats['win_rate']:.2%}")
    logger.info(f"Average Profit: {stats['avg_profit']:.4f}")
    logger.info(f"Average Win: {stats['avg_win']:.4f}")
    logger.info(f"Average Loss: {stats['avg_loss']:.4f}")
    logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
    logger.info(f"Average Duration: {stats['avg_duration']:.2f}")
    logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {stats['sortino_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {stats['max_drawdown']:.2%}")
    
    # Save trades to CSV
    trades_df.to_csv(f'{plots_dir}/trades.csv')
    
    return stats

def main():
    # Set up logging
    logger.add("logs/performance_analysis.log", rotation="1 day")
    
    try:
        # Load market data
        feature_engineer = FeatureEngineer(window_size=100)
        data = feature_engineer.load_market_data(days=360)
        
        # Split data
        split_idx = int(len(data) * 0.8)
        eval_data = data[split_idx:]
        
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
        
        # Analyze model performance
        stats = analyze_model_performance(
            model_path="trained_models/rl/best_model",
            eval_data=eval_data,
            env_config=env_config
        )
        
        logger.info("Performance analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during performance analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()