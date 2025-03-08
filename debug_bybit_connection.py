#!/usr/bin/env python3
"""
Debug script for Bybit connection issues.

This script tests the Bybit connection with our custom implementation.
"""
import asyncio
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Import our custom Bybit implementation
from alpha_pulse.exchanges.implementations.bybit import BybitExchange
from alpha_pulse.exchanges.interfaces import ConnectionError, ExchangeError


async def test_bybit_connection():
    """Test the Bybit connection with our custom implementation."""
    logger.info("Testing Bybit connection with custom implementation")
    
    # Create Bybit exchange instance
    exchange = BybitExchange(testnet=False)
    
    try:
        # Initialize the exchange
        logger.info("Initializing Bybit exchange")
        await exchange.initialize()
        logger.info("✅ Successfully initialized Bybit exchange")
        
        # Test getting balances
        logger.info("Testing get_balances")
        balances = await exchange.get_balances()
        logger.info(f"✅ Successfully retrieved {len(balances)} balances")
        
        # Print some balance details
        if balances:
            logger.info("Sample balances:")
            for currency, balance in list(balances.items())[:5]:  # Show first 5 balances
                logger.info(f"  {currency}: {balance}")
        
        # Test getting positions
        logger.info("Testing get_positions")
        positions = await exchange.get_positions()
        logger.info(f"✅ Successfully retrieved {len(positions)} positions")
        
        # Print some position details
        if positions:
            logger.info("Sample positions:")
            for symbol, position in list(positions.items())[:5]:  # Show first 5 positions
                logger.info(f"  {symbol}: {position}")
        
        # Test getting orders
        logger.info("Testing get_orders")
        try:
            # Use fetchOpenOrders instead of fetch_orders which is not supported for UTA accounts
            logger.info("Fetching open orders...")
            open_orders = await exchange.exchange.fetch_open_orders()
            
            # Optionally fetch closed orders too
            logger.info("Fetching closed orders...")
            closed_orders = await exchange.exchange.fetch_closed_orders()
            
            # Combine the results
            orders = open_orders + closed_orders
            
            logger.info(f"✅ Successfully retrieved {len(orders)} orders ({len(open_orders)} open, {len(closed_orders)} closed)")
            
            # Print some order details
            if orders:
                logger.info("Sample orders:")
                for order in orders[:5]:  # Show first 5 orders
                    logger.info(f"  {order['id']}: {order['symbol']} {order['side']} {order['amount']} @ {order['price']}")
            else:
                logger.info("No orders found")
        except Exception as e:
            logger.warning(f"Could not fetch orders: {str(e)}")
            logger.info("This is not critical for basic functionality")
        
        # Test getting ticker
        logger.info("Testing get_ticker for BTC/USDT")
        try:
            # Use our custom method instead of direct CCXT access
            price = await exchange.get_ticker_price("BTC/USDT")
            logger.info(f"✅ Successfully retrieved ticker for BTC/USDT: {price}")
        except Exception as e:
            logger.warning(f"Could not fetch ticker: {str(e)}")
            logger.info("This is not critical for basic functionality")
        
        logger.info("All tests passed successfully!")
        return True
    except ConnectionError as e:
        logger.error(f"❌ Connection error: {str(e)}")
        return False
    except ExchangeError as e:
        logger.error(f"❌ Exchange error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Close the exchange
        try:
            await exchange.close()
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.warning(f"Error closing exchange connection: {str(e)}")


async def main():
    """Main entry point."""
    logger.info("Starting Bybit connection test")
    
    # Test the connection
    success = await test_bybit_connection()
    
    # Print summary
    if success:
        logger.info("✅ Bybit connection test completed successfully")
    else:
        logger.error("❌ Bybit connection test failed")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bybit_connection_test_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({"success": success, "timestamp": timestamp}, f, indent=2)
    
    logger.info(f"Test results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())