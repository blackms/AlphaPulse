"""
Demo script showcasing portfolio rebalancing functionality.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.exchanges.mock import MockExchange


def create_sample_data(assets: list, days: int = 180) -> pd.DataFrame:
    """Create synthetic price data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    data = {}
    
    # Generate correlated random walks for crypto assets
    for asset in assets:
        if asset in ['USDT', 'USDC']:  # Stablecoins
            data[asset] = np.ones(days)
        else:  # Crypto assets
            # Random walk with drift and volatility
            returns = np.random.normal(0.0002, 0.02, days)  # Daily returns
            price = 100 * np.exp(np.cumsum(returns))  # Price series
            data[asset] = price
            
    return pd.DataFrame(data, index=dates)


def main():
    """Run portfolio rebalancing demonstration."""
    print("Starting Portfolio Rebalancing Demo")
    print("=" * 50)

    # Load configuration
    config_path = Path(__file__).parent.parent / "portfolio" / "portfolio_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Initialize portfolio manager
    manager = PortfolioManager(str(config_path))
    
    # Create mock exchange with sample data
    assets = ['BTC', 'ETH', 'BNB', 'SOL', 'USDT', 'USDC']
    historical_data = create_sample_data(assets)
    
    exchange = MockExchange(
        initial_balances={
            'BTC': 1.0,
            'ETH': 10.0,
            'BNB': 50.0,
            'SOL': 100.0,
            'USDT': 50000.0,
            'USDC': 50000.0
        },
        price_data=historical_data
    )

    # Get current portfolio state
    print("\nCurrent Portfolio Allocation:")
    current_allocation = manager.get_current_allocation(exchange)
    for asset, weight in current_allocation.items():
        print(f"{asset}: {weight:.2%}")

    # Check if rebalancing is needed
    print("\nChecking if rebalancing is needed...")
    if manager.needs_rebalancing(current_allocation):
        print("Portfolio requires rebalancing")
        
        # Execute rebalancing
        print("\nExecuting portfolio rebalancing...")
        result = manager.rebalance_portfolio(exchange, historical_data)
        
        if result['status'] == 'completed':
            print("\nRebalancing completed successfully!")
            print("\nTarget Allocation:")
            for asset, weight in result['target_allocation'].items():
                print(f"{asset}: {weight:.2%}")
                
            print("\nExecuted Trades:")
            for trade in result['trades']:
                print(f"{trade['type'].upper()} {trade['asset']}: "
                      f"${abs(trade['value']):,.2f} "
                      f"({trade['weight_change']:+.2%})")
        else:
            print(f"\nRebalancing {result['status']}: {result.get('reason', 'Unknown error')}")
    else:
        print("Portfolio is already well-balanced")

    # Show final portfolio state
    print("\nFinal Portfolio Allocation:")
    final_allocation = manager.get_current_allocation(exchange)
    for asset, weight in final_allocation.items():
        print(f"{asset}: {weight:.2%}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")