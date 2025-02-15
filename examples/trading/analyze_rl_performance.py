"""
Script to analyze RL model performance and generate evaluation metrics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig

def calculate_trading_metrics(df):
    """Calculate key trading performance metrics."""
    metrics = {}
    
    # Calculate returns
    df['returns'] = df['pnl'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    
    # Basic metrics
    metrics['total_trades'] = len(df)
    metrics['profitable_trades'] = len(df[df['pnl'] > 0])
    metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
    
    # Risk metrics
    returns = df['returns'].dropna()
    metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
    metrics['max_drawdown'] = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
    
    # Trade characteristics
    metrics['avg_trade_duration'] = df['duration'].mean()
    metrics['avg_profit_per_trade'] = df['pnl'].mean()
    metrics['profit_factor'] = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum())
    
    return metrics

def plot_performance(df, save_dir):
    """Generate performance visualization plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative_returns'])
    plt.title('Cumulative Returns')
    plt.xlabel('Trade Number')
    plt.ylabel('Returns')
    plt.savefig(save_dir / 'cumulative_returns.png')
    plt.close()
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    drawdown = df['cumulative_returns'].cummax() - df['cumulative_returns']
    plt.plot(drawdown)
    plt.title('Drawdown')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown')
    plt.savefig(save_dir / 'drawdown.png')
    plt.close()
    
    # Plot trade PnL distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['pnl'], bins=50)
    plt.title('Trade PnL Distribution')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.savefig(save_dir / 'pnl_distribution.png')
    plt.close()
    
    # Plot position sizes
    plt.figure(figsize=(12, 6))
    plt.hist(df['position_size'], bins=50)
    plt.title('Position Size Distribution')
    plt.xlabel('Position Size')
    plt.ylabel('Frequency')
    plt.savefig(save_dir / 'position_sizes.png')
    plt.close()

def evaluate_model(model_path, env_config):
    """Evaluate model performance on test data."""
    model = PPO.load(model_path)
    env = TradingEnv(env_config)
    
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True
    )
    
    return mean_reward, std_reward

def main():
    """Main analysis function."""
    # Load trade history
    trades_df = pd.read_csv('reports/rl_trades.csv')
    
    # Calculate metrics
    metrics = calculate_trading_metrics(trades_df)
    
    # Print metrics
    print("\nTrading Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate plots
    plot_performance(trades_df, 'reports/plots')
    
    # Evaluate model
    env_config = TradingEnvConfig(
        initial_capital=100000.0,
        commission=0.001,
        position_size=0.3,
        window_size=10,
        reward_scaling=2.0,
        risk_aversion=0.05,
        max_position=5.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06
    )
    
    mean_reward, std_reward = evaluate_model('trained_models/rl/model', env_config)
    print(f"\nModel Evaluation:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()