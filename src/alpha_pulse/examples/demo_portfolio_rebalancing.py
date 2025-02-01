"""
Demo script showcasing portfolio rebalancing functionality with Bybit exchange.
"""

import os
import asyncio
from pathlib import Path
from decimal import Decimal

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.exchanges.bybit import BybitExchange
from alpha_pulse.exchanges.credentials.manager import credentials_manager


async def main():
    """Run portfolio rebalancing demonstration."""
    print("Starting Portfolio Rebalancing Demo")
    print("=" * 50)

    # Save Bybit credentials
    credentials_manager.save_credentials(
        exchange_id='bybit',
        api_key='3m3tcGApIqThJLIlhx',
        api_secret='kIy1WOlHjWZ0yMmKIuatL5ryu2v8BBl31YFa',
        testnet=False
    )

    # Load configuration
    config_path = Path(__file__).parent.parent / "portfolio" / "portfolio_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Initialize portfolio manager
    manager = PortfolioManager(str(config_path))
    
    # Initialize Bybit exchange
    exchange = BybitExchange(testnet=False)
    await exchange.initialize()

    try:
        # Get current portfolio state
        print("\nCurrent Portfolio Allocation:")
        current_allocation = await manager.get_current_allocation(exchange)
        for asset, weight in current_allocation.items():
            print(f"{asset}: {weight:.2%}")

        # Check if rebalancing is needed
        print("\nChecking if rebalancing is needed...")
        if await manager.needs_rebalancing(exchange):
            print("Portfolio requires rebalancing")
            
            # Execute rebalancing
            print("\nExecuting portfolio rebalancing...")
            result = await manager.rebalance_portfolio(exchange)
            
            if result['status'] == 'completed':
                print("\nRebalancing completed successfully!")
                print("\nTarget Allocation:")
                for asset, weight in result['target_allocation'].items():
                    print(f"{asset}: {weight:.2%}")
                    
                print("\nExecuted Trades:")
                for trade in result['trades']:
                    print(f"{trade['type'].upper()} {trade['asset']}: "
                          f"${abs(float(trade['value'])):,.2f} "
                          f"({float(trade['weight_change']):+.2%})")
            else:
                print(f"\nRebalancing {result['status']}: {result.get('reason', 'Unknown error')}")
        else:
            print("Portfolio is already well-balanced")

        # Show final portfolio state
        print("\nFinal Portfolio Allocation:")
        final_allocation = await manager.get_current_allocation(exchange)
        for asset, weight in final_allocation.items():
            print(f"{asset}: {weight:.2%}")

    finally:
        # Always close exchange connection
        await exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {str(e)}")