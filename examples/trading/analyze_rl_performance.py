"""
Script to analyze RL model performance and generate evaluation metrics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig

# Set up plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

def calculate_trading_metrics(df):
    """Calculate comprehensive trading performance metrics."""
    metrics = {}
    
    # Calculate returns
    df['returns'] = df['pnl'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    
    # Basic metrics
    metrics['total_trades'] = len(df)
    metrics['profitable_trades'] = len(df[df['pnl'] > 0])
    metrics['losing_trades'] = len(df[df['pnl'] < 0])
    metrics['breakeven_trades'] = len(df[df['pnl'] == 0])
    metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
    
    # Profit metrics
    metrics['total_profit'] = df[df['pnl'] > 0]['pnl'].sum()
    metrics['total_loss'] = abs(df[df['pnl'] < 0]['pnl'].sum())
    metrics['net_profit'] = metrics['total_profit'] - metrics['total_loss']
    metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss'] if metrics['total_loss'] != 0 else float('inf')
    metrics['avg_profit_per_winning_trade'] = df[df['pnl'] > 0]['pnl'].mean()
    metrics['avg_loss_per_losing_trade'] = df[df['pnl'] < 0]['pnl'].mean()
    
    # Risk metrics
    returns = df['returns'].dropna()
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = annualized_return / annualized_vol if annualized_vol != 0 else 0
    metrics['sortino_ratio'] = annualized_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
    metrics['max_drawdown'] = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
    metrics['max_drawdown_duration'] = (df['cumulative_returns'].cummax() - df['cumulative_returns']).idxmax()
    
    # Trade characteristics
    metrics['avg_trade_duration'] = pd.Timedelta(df['duration'].mean()).total_seconds() / 3600  # in hours
    metrics['max_trade_duration'] = pd.Timedelta(df['duration'].max()).total_seconds() / 3600
    metrics['min_trade_duration'] = pd.Timedelta(df['duration'].min()).total_seconds() / 3600
    metrics['avg_position_size'] = df['position_size'].mean()
    metrics['max_position_size'] = df['position_size'].max()
    
    # Risk-adjusted returns
    metrics['calmar_ratio'] = annualized_return / metrics['max_drawdown'] if metrics['max_drawdown'] != 0 else 0
    metrics['risk_reward_ratio'] = abs(metrics['avg_profit_per_winning_trade'] / metrics['avg_loss_per_losing_trade']) if metrics['avg_loss_per_losing_trade'] != 0 else float('inf')
    
    # Format metrics
    for key in metrics:
        if isinstance(metrics[key], float):
            metrics[key] = round(metrics[key], 4)
    
    return metrics

def plot_performance(df, save_dir):
    """Generate performance visualization plots with enhanced styling."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Cumulative Returns
    ax1 = plt.subplot(2, 2, 1)
    df['cumulative_returns'].plot(ax=ax1, color='blue', linewidth=2)
    ax1.set_title('Cumulative Returns', fontsize=12, pad=10)
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Returns')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    ax2 = plt.subplot(2, 2, 2)
    drawdown = df['cumulative_returns'].cummax() - df['cumulative_returns']
    drawdown.plot(ax=ax2, color='red', linewidth=2)
    ax2.set_title('Drawdown', fontsize=12, pad=10)
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trade PnL Distribution
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='pnl', bins=50, ax=ax3, color='green')
    ax3.set_title('Trade PnL Distribution', fontsize=12, pad=10)
    ax3.set_xlabel('PnL')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Position Sizes Over Time
    ax4 = plt.subplot(2, 2, 4)
    df['position_size'].plot(ax=ax4, color='purple', linewidth=2)
    ax4.set_title('Position Sizes Over Time', fontsize=12, pad=10)
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Position Size')
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle('Trading Performance Analysis', fontsize=16, y=0.95)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_dir / f'performance_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Monthly Returns Heatmap
    if 'entry_time' in df.columns:
        df['month'] = pd.to_datetime(df['entry_time']).dt.month
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        monthly_returns = df.groupby(['year', 'month'])['returns'].sum().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns Heatmap', fontsize=12, pad=10)
        plt.savefig(save_dir / f'monthly_returns_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_model(model_path, env_config):
    """Evaluate model performance on test data."""
    model = PPO.load(model_path)
    
    # Create environment with market data
    from alpha_pulse.data_pipeline.models import MarketData
    env = TradingEnv(
        MarketData(prices=None, volumes=None, timestamp=None),
        config=env_config
    )
    
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True
    )
    
    env.close()
    return mean_reward, std_reward

def main():
    """Main analysis function."""
    # Create reports directory
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nStarting trading performance analysis...")
    
    try:
        # Load trade history
        trades_df = pd.read_csv('reports/rl_trades.csv')
        print(f"Loaded {len(trades_df)} trades for analysis")
        
        # Calculate metrics
        metrics = calculate_trading_metrics(trades_df)
        
        # Organize metrics by category
        metric_categories = {
            'Basic Statistics': ['total_trades', 'profitable_trades', 'losing_trades', 'breakeven_trades', 'win_rate'],
            'Profit Metrics': ['total_profit', 'total_loss', 'net_profit', 'profit_factor', 'avg_profit_per_winning_trade', 'avg_loss_per_losing_trade'],
            'Risk Metrics': ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'max_drawdown_duration', 'calmar_ratio', 'risk_reward_ratio'],
            'Trade Characteristics': ['avg_trade_duration', 'max_trade_duration', 'min_trade_duration', 'avg_position_size', 'max_position_size']
        }
        
        # Generate report
        report = [
            "\nTrading Performance Analysis",
            "=" * 50,
            f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {trades_df['entry_time'].min()} to {trades_df['entry_time'].max()}\n"
        ]
        
        for category, metric_names in metric_categories.items():
            report.extend([f"\n{category}:", "-" * len(category)])
            for metric in metric_names:
                if metric in metrics:
                    report.append(f"{metric:30}: {metrics[metric]}")
        
        # Save and print report
        report_path = reports_dir / f'performance_metrics_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print('\n'.join(report))
        
        # Generate visualizations
        print("\nGenerating performance visualizations...")
        plot_performance(trades_df, reports_dir / 'plots')
        
        # Model evaluation
        print("\nEvaluating model performance...")
        with open("examples/trading/rl_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        env_config = TradingEnvConfig(**config['environment'])
        
        mean_reward, std_reward = evaluate_model('trained_models/rl/model', env_config)
        
        eval_results = [
            "\nModel Evaluation Results",
            "-" * 25,
            f"Mean reward: {mean_reward:.2f}",
            f"Std deviation: {std_reward:.2f}",
            f"95% CI: [{mean_reward - 1.96*std_reward:.2f}, {mean_reward + 1.96*std_reward:.2f}]"
        ]
        print('\n'.join(eval_results))
        
        # Append evaluation results to report
        with open(report_path, 'a') as f:
            f.write('\n\n' + '\n'.join(eval_results))
        
        print(f"\nAnalysis complete. Reports saved to {reports_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {str(e)}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()