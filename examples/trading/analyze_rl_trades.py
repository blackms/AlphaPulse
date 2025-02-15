"""
Analyze and visualize RL trading performance.

This script reads the saved trades from rl_trades.csv and generates:
1. Trade statistics and performance metrics
2. Equity curve visualization
3. Trade distribution analysis
4. Risk metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import seaborn as sns
from loguru import logger


def calculate_performance_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        trades_df: DataFrame containing trade information
        
    Returns:
        Dictionary of performance metrics
    """
    if trades_df.empty:
        return {}
        
    # Basic trade statistics
    total_trades = len(trades_df)
    profitable_trades = len(trades_df[trades_df['profit'] > 0])
    win_rate = profitable_trades / total_trades
    
    # Profit metrics
    total_profit = trades_df['profit'].sum()
    avg_profit = trades_df['profit'].mean()
    profit_std = trades_df['profit'].std()
    
    # Separate wins and losses
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]
    
    avg_win = winning_trades['profit'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['profit'].mean() if not losing_trades.empty else 0
    
    # Calculate profit factor
    gross_profits = winning_trades['profit'].sum() if not winning_trades.empty else 0
    gross_losses = abs(losing_trades['profit'].sum()) if not losing_trades.empty else 0
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Risk metrics
    max_drawdown = calculate_max_drawdown(trades_df['equity'])
    sharpe_ratio = calculate_sharpe_ratio(trades_df['profit'])
    
    # Trade duration analysis
    avg_duration = trades_df['duration'].mean()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_profit': total_profit,
        'average_profit': avg_profit,
        'profit_std': profit_std,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'average_duration': avg_duration
    }


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve - rolling_max
    return abs(drawdowns.min())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if len(excess_returns) > 0:
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    return 0.0


def plot_equity_curve(trades_df: pd.DataFrame, save_path: str = "reports/equity_curve.png"):
    """Plot equity curve with drawdown overlay."""
    plt.figure(figsize=(12, 6))
    
    # Plot equity curve
    plt.plot(trades_df.index, trades_df['equity'], label='Equity', color='blue')
    
    # Calculate and plot drawdown
    rolling_max = trades_df['equity'].expanding().max()
    drawdown = trades_df['equity'] - rolling_max
    plt.fill_between(trades_df.index, 0, drawdown, alpha=0.3, color='red', label='Drawdown')
    
    plt.title('Equity Curve with Drawdown')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()


def plot_trade_distribution(trades_df: pd.DataFrame, save_path: str = "reports/trade_distribution.png"):
    """Plot trade profit distribution."""
    plt.figure(figsize=(12, 6))
    
    # Create profit distribution plot
    sns.histplot(data=trades_df['profit'], bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Trade Profit Distribution')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()


def analyze_trades():
    """Analyze RL trading performance from saved trades."""
    try:
        # Load trades
        trades_file = Path("reports/rl_trades.csv")
        if not trades_file.exists():
            logger.error("No trades file found at reports/rl_trades.csv")
            return
            
        trades_df = pd.read_csv(trades_file)
        if trades_df.empty:
            logger.error("No trades found in the trades file")
            return
            
        # Calculate performance metrics
        metrics = calculate_performance_metrics(trades_df)
        
        # Print performance summary
        logger.info("\nTrading Performance Summary:")
        logger.info("-" * 40)
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Total Profit: ${metrics['total_profit']:.2f}")
        logger.info(f"Average Profit: ${metrics['average_profit']:.2f}")
        logger.info(f"Average Win: ${metrics['average_win']:.2f}")
        logger.info(f"Average Loss: ${metrics['average_loss']:.2f}")
        logger.info(f"Max Drawdown: ${metrics['max_drawdown']:.2f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Average Trade Duration: {metrics['average_duration']:.1f} periods")
        
        # Create reports directory if it doesn't exist
        Path("reports").mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plot_equity_curve(trades_df)
        plot_trade_distribution(trades_df)
        
        logger.info("\nAnalysis complete. Plots saved in reports/")
        
    except Exception as e:
        logger.error(f"Error analyzing trades: {str(e)}")
        raise


if __name__ == "__main__":
    analyze_trades()
