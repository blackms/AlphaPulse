import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from alpha_pulse.rl.trading_env import TradingEnvironment
from alpha_pulse.rl.features import prepare_data
import matplotlib.pyplot as plt

def calculate_metrics(trades_df):
    """Calculate key trading performance metrics"""
    metrics = {}
    
    # PnL metrics
    metrics['total_pnl'] = trades_df['pnl'].sum()
    metrics['avg_trade_pnl'] = trades_df['pnl'].mean()
    metrics['win_rate'] = (trades_df['pnl'] > 0).mean()
    
    # Risk metrics
    returns = trades_df['pnl'].pct_change()
    metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
    
    cumulative_pnl = trades_df['pnl'].cumsum()
    rolling_max = cumulative_pnl.expanding().max()
    drawdowns = cumulative_pnl - rolling_max
    metrics['max_drawdown'] = drawdowns.min()
    
    # Trade characteristics
    metrics['avg_trade_duration'] = trades_df['duration'].mean()
    metrics['total_trades'] = len(trades_df)
    metrics['avg_position_size'] = trades_df['position_size'].abs().mean()
    
    return metrics

def plot_equity_curve(trades_df):
    """Plot equity curve and drawdowns"""
    cumulative_pnl = trades_df['pnl'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl.index, cumulative_pnl.values)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.savefig('plots/equity_curve.png')
    plt.close()

def main():
    # Load the trained model
    model = PPO.load("trained_models/rl/best_model")
    
    # Prepare evaluation data
    _, eval_data = prepare_data(days=360)
    
    # Create environment for evaluation
    env = TradingEnvironment(eval_data, mode='eval')
    
    # Run evaluation episode
    obs = env.reset()
    done = False
    trades = []
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        if info.get('trade_executed'):
            trades.append({
                'timestamp': info['timestamp'],
                'pnl': info['trade_pnl'],
                'position_size': info['position_size'],
                'duration': info['trade_duration']
            })
    
    # Create trades dataframe
    trades_df = pd.DataFrame(trades)
    
    # Calculate and print metrics
    metrics = calculate_metrics(trades_df)
    print("\nTrading Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    plot_equity_curve(trades_df)
    
    # Save detailed trade log
    trades_df.to_csv('reports/trade_log.csv', index=False)

if __name__ == "__main__":
    main()