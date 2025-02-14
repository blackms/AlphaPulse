"""
Script to evaluate RL model trading performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from stable_baselines3 import PPO

from alpha_pulse.rl.trading_env import TradingEnv, TradingEnvConfig
from alpha_pulse.rl.trainer import RLTrainer, NetworkConfig, TrainingConfig
from alpha_pulse.rl.utils import calculate_trade_statistics
from alpha_pulse.data_pipeline.models import MarketData

def analyze_model_performance(model_path: str, eval_data: MarketData):
    """
    Analyze model performance on evaluation data.
    
    Args:
        model_path: Path to saved model
        eval_data: Market data for evaluation
    """
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
    
    # Initialize trainer with minimal configs since we're just evaluating
    trainer = RLTrainer(
        env_config=env_config,
        network_config=NetworkConfig(),
        training_config=TrainingConfig()
    )
    
    # Create evaluation environment
    env = trainer._create_envs(eval_data, n_envs=1)
    
    # Load model using stable_baselines3
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    # Run evaluation episodes
    logger.info("Running evaluation episodes...")
    trades = []
    
    # Run multiple episodes
    n_episodes = 10
    for episode in range(n_episodes):
        obs = env.reset()  # VecEnv reset() returns just the observation
        done = False
        episode_trades = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # VecEnv wraps info in a list
            info = info[0] if isinstance(info, list) else info
            
            if info.get('trade_executed', False):
                position = info.get('position')
                if position:
                    trade = {
                        'profit': info.get('realized_pnl', 0.0),
                        'duration': position.exit_time - position.timestamp if hasattr(position, 'exit_time') else 0
                    }
                    episode_trades.append(trade)
        
        trades.extend(episode_trades)
    
    # Calculate trade statistics
    stats = calculate_trade_statistics(trades) if trades else {
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_profit': 0.0,
        'profit_factor': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown': 0.0,
        'avg_duration': 0.0
    }
    
    # Print performance metrics
    logger.info("\nPerformance Metrics:")
    logger.info(f"Total Trades: {stats['total_trades']}")
    logger.info(f"Win Rate: {stats['win_rate']:.2%}")
    logger.info(f"Average Profit per Trade: {stats['avg_profit']:.4f}")
    logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {stats['sortino_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {stats['max_drawdown']:.2%}")
    logger.info(f"Average Trade Duration: {stats['avg_duration']:.1f} periods")
    
    if trades:
        # Plot equity curve
        profits = [t['profit'] for t in trades]
        cumulative_returns = np.cumsum(profits)
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns, label='Equity Curve')
        plt.title('Trading Performance')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plot_path = Path("plots/rl_performance.png")
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"\nPerformance plot saved to {plot_path}")
    else:
        logger.warning("No trades were executed during evaluation")
    
    # Return statistics for further analysis if needed
    return stats, trades

def main():
    # Set up logging
    logger.add("logs/rl_analysis.log", rotation="1 day")
    
    try:
        # Load your evaluation data here
        # For demo purposes, we'll use the mock data generator from demo_rl_trading.py
        from demo_rl_trading import generate_mock_data, prepare_data
        market_data = generate_mock_data(days=365)
        _, eval_data = prepare_data(market_data)
        
        # Analyze model performance
        model_path = "trained_models/rl/best_model"  # Updated path to best model
        stats, trades = analyze_model_performance(model_path, eval_data)
        
    except Exception as e:
        logger.error(f"Error in RL performance analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    main()