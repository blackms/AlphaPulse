#!/usr/bin/env python3
"""
Quick test script for verifying the fixed Bybit order history functionality.
"""

import os
import sys
import asyncio
from datetime import datetime
from loguru import logger

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Use the fixed adapter
from src.alpha_pulse.exchanges.adapters.ccxt_adapter_fixed import CCXTAdapter
from src.alpha_pulse.exchanges.interfaces import ExchangeConfiguration

async def test_bybit_order_history():
    """Test retrieving order history from Bybit with the fixed adapter."""
    logger.info("Testing order history retrieval with fixed adapter...")
    
    # Get API credentials
    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet = os.environ.get("BYBIT_TESTNET", "false").lower() == "true"
    
    # Check if credentials are available
    if not api_key or not api_secret:
        try:
            # Try to get from credentials_manager if available
            from src.alpha_pulse.exchanges.credentials.manager import credentials_manager
            creds = credentials_manager.get_credentials('bybit')
            if creds:
                api_key = creds.api_key
                api_secret = creds.api_secret
                logger.info(f"Using credentials from credentials_manager")
        except ImportError:
            logger.warning("Could not import credentials_manager")
    
    if not api_key or not api_secret:
        logger.error("API credentials are required. Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables.")
        return False
    
    # Create exchange config
    config = ExchangeConfiguration(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        options={
            '_exchange_id': 'bybit',
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
            'recvWindow': 60000,
            'createMarketBuyOrderRequiresPrice': True
        }
    )
    
    # Initialize adapter
    adapter = CCXTAdapter(config)
    
    try:
        await adapter.initialize()
        logger.info("Adapter initialized successfully")
        
        # Test with ETH/USDT instead of BTC/USDT
        symbol = "ETH/USDT"
        logger.info(f"Testing order history for {symbol}...")
        
        # Get order history
        orders = await adapter.get_order_history(symbol)
        
        if orders:
            logger.info(f"SUCCESS: Found {len(orders)} orders for {symbol}")
            logger.info(f"Sample order: {orders[0]}")
        else:
            logger.warning(f"No orders found for {symbol}")
        
        # Also try with market format
        market_symbol = "ETHUSDT"
        logger.info(f"Testing order history with market format: '{market_symbol}'")
        
        # Try with direct call to _get_bybit_order_history
        time_params = {
            'since': int((datetime.now().timestamp() - 90*24*60*60) * 1000),
            'limit': 100
        }
        direct_orders = await adapter._get_bybit_order_history(market_symbol, time_params)
        
        if direct_orders:
            logger.info(f"SUCCESS: Found {len(direct_orders)} orders with direct call for {market_symbol}")
        else:
            logger.warning(f"No orders found with direct call for {market_symbol}")
        
        return bool(orders or direct_orders)
    
    except Exception as e:
        logger.error(f"Error testing order history: {str(e)}")
        return False
    
    finally:
        if adapter:
            await adapter.close()
            logger.info("Adapter closed")

if __name__ == "__main__":
    success = asyncio.run(test_bybit_order_history())
    sys.exit(0 if success else 1)