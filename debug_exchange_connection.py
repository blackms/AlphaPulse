#!/usr/bin/env python3
"""
Debug script to test the exchange connection with the current environment variables.
"""
import asyncio
import os
from loguru import logger

from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.types import ExchangeType


async def test_exchange_connection():
    """Test exchange connection with environment credentials."""
    logger.info("=== EXCHANGE CONNECTION TEST ===")
    
    # Get credentials from environment
    exchange_type = os.environ.get('EXCHANGE_TYPE', 'bybit').lower()
    api_key = os.environ.get('BYBIT_API_KEY', os.environ.get('EXCHANGE_API_KEY', ''))
    api_secret = os.environ.get('BYBIT_API_SECRET', os.environ.get('EXCHANGE_API_SECRET', ''))
    testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
    
    # Log what we found (without revealing full secrets)
    logger.info(f"Exchange type: {exchange_type}")
    logger.info(f"API key found: {bool(api_key)} (length: {len(api_key) if api_key else 0})")
    logger.info(f"API secret found: {bool(api_secret)} (length: {len(api_secret) if api_secret else 0})")
    logger.info(f"Using testnet: {testnet}")
    
    try:
        # Create exchange based on type
        if exchange_type == 'binance':
            logger.info("Creating Binance exchange")
            exchange = ExchangeFactory.create_exchange(
                ExchangeType.BINANCE,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        elif exchange_type == 'bybit':
            logger.info("Creating Bybit exchange")
            exchange = ExchangeFactory.create_exchange(
                ExchangeType.BYBIT,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        else:
            logger.warning(f"Unsupported exchange type: {exchange_type}, falling back to Bybit")
            exchange = ExchangeFactory.create_exchange(
                ExchangeType.BYBIT,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        
        # Initialize the exchange
        logger.info("Initializing exchange...")
        await exchange.initialize()
        logger.info("Exchange initialized successfully")
        
        # Try to get balances
        logger.info("Attempting to get balances...")
        balances = await exchange.get_balances()
        logger.info(f"Got balances for {len(balances)} assets:")
        for asset, balance in balances.items():
            logger.info(f"  {asset}: {balance.total}")
        
        # Try to get portfolio value
        logger.info("Attempting to get portfolio value...")
        value = await exchange.get_portfolio_value()
        logger.info(f"Portfolio value: {value}")
        
        logger.info("Exchange connection test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during exchange test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'exchange' in locals():
            logger.info("Closing exchange connection...")
            await exchange.close()


if __name__ == "__main__":
    asyncio.run(test_exchange_connection())
