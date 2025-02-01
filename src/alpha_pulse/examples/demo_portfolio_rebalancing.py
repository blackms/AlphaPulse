"""
Demo script showcasing portfolio rebalancing functionality with Bybit exchange.
"""

import os
import asyncio
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone

from loguru import logger

from alpha_pulse.portfolio.portfolio_manager import PortfolioManager
from alpha_pulse.exchanges.bybit import BybitExchange
from alpha_pulse.exchanges.credentials.manager import credentials_manager


async def main():
    """Run portfolio rebalancing demonstration."""
    logger.info("Starting Portfolio Rebalancing Demo")
    logger.info("=" * 50)

    try:
        # Load Bybit credentials from environment variables
        api_key = os.getenv('ALPHA_PULSE_BYBIT_API_KEY')
        api_secret = os.getenv('ALPHA_PULSE_BYBIT_API_SECRET')
        testnet = os.getenv('ALPHA_PULSE_BYBIT_TESTNET', 'false').lower() == 'true'

        if not api_key or not api_secret:
            raise ValueError("Bybit API credentials not found in environment variables")

        credentials_manager.save_credentials(
            exchange_id='bybit',
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )

        # Load configuration
        config_path = Path(__file__).parent.parent / "portfolio" / "portfolio_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        logger.debug(f"Loading portfolio configuration from {config_path}")
        
        # Initialize portfolio manager
        manager = PortfolioManager(str(config_path))
        
        # Initialize Bybit exchange
        exchange = BybitExchange(testnet=testnet)
        await exchange.initialize()

        # Get current portfolio state
        logger.info("\nCurrent Portfolio Allocation:")
        current_allocation = await manager.get_current_allocation(exchange)
        for asset, weight in current_allocation.items():
            logger.info(f"{asset}: {weight:.2%}")

        # Check if rebalancing is needed
        logger.info("\nChecking if rebalancing is needed...")
        logger.debug("Comparing current allocation with target weights...")
        logger.debug(f"Current allocation: {current_allocation}")
        
        needs_rebalancing = await manager.needs_rebalancing(exchange)
        if needs_rebalancing:
            logger.warning("Portfolio requires rebalancing - Deviation from target weights detected")
            
            # Execute rebalancing
            logger.info("\nExecuting portfolio rebalancing...")
            result = await manager.rebalance_portfolio(exchange)
            
            if result['status'] == 'completed':
                logger.success("\nRebalancing completed successfully!")
                logger.info("\nTarget Allocation:")
                for asset, weight in result['target_allocation'].items():
                    logger.info(f"{asset}: {weight:.2%}")
                    
                logger.info("\nExecuted Trades:")
                for trade in result['trades']:
                    logger.info(
                        f"{trade['type'].upper()} {trade['asset']}: "
                        f"${abs(float(trade['value'])):,.2f} "
                        f"({float(trade['weight_change']):+.2%})"
                    )
            else:
                logger.error(
                    f"\nRebalancing {result['status']}: {result.get('reason', 'Unknown error')}"
                )
        else:
            logger.info("Portfolio is already well-balanced")
            logger.debug("All asset weights are within tolerance thresholds")

        # Show final portfolio state
        logger.info("\nFinal Portfolio Allocation:")
        final_allocation = await manager.get_current_allocation(exchange)
        for asset, weight in final_allocation.items():
            logger.info(f"{asset}: {weight:.2%}")

    except Exception as e:
        logger.exception(f"An error occurred during portfolio rebalancing: {str(e)}")
        raise

    finally:
        # Always close exchange connection
        logger.debug("Closing exchange connection...")
        await exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Fatal error in main: {str(e)}")
        exit(1)
