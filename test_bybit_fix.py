#!/usr/bin/env python3
"""
Test script to fix Bybit order history retrieval issues.

This script addresses the problem where 'category' parameter is incorrectly
passed to CCXT methods that don't support it.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('bybit_fix')

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.alpha_pulse.exchanges.credentials.manager import credentials_manager
    from src.alpha_pulse.exchanges.implementations.bybit import BybitExchange
    from src.alpha_pulse.exchanges.interfaces import ExchangeConfiguration
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory.")
    sys.exit(1)


async def test_fixed_order_history(symbol="BTC/USDT"):
    """Test the fixed order history retrieval for Bybit."""
    try:
        # Get credentials
        creds = credentials_manager.get_credentials('bybit')
        
        if not creds or not creds.api_key or not creds.api_secret:
            logger.error("No credentials found for Bybit. Please set up credentials first.")
            return
        
        # Initialize Bybit exchange
        exchange = BybitExchange(testnet=False)
        await exchange.initialize()
        
        # Set up time parameters (last 90 days)
        since = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
        
        # First, try with market format
        logger.info(f"Testing order history retrieval for {symbol}")
        
        # Get orders using our exchange implementation
        orders = await exchange.get_orders(symbol)
        
        if orders:
            logger.info(f"Success! Found {len(orders)} orders for {symbol}")
            # Log first order details
            if len(orders) > 0:
                logger.info(f"Sample order: {orders[0]}")
        else:
            logger.warning(f"No orders found for {symbol}")
            
            # Try different format for symbol
            if '/' in symbol:
                alt_symbol = symbol.replace('/', '')
                logger.info(f"Trying alternative symbol format: {alt_symbol}")
                orders = await exchange.get_orders(alt_symbol)
                
                if orders:
                    logger.info(f"Success with alternative format! Found {len(orders)} orders for {alt_symbol}")
                    if len(orders) > 0:
                        logger.info(f"Sample order: {orders[0]}")
                else:
                    logger.warning(f"No orders found for {alt_symbol} either")
        
        # Access the adapter to test the direct fix
        adapter = exchange._exchange
        logger.info("Testing the direct fix in the CCXT adapter...")
        
        # Create test parameters similar to those used in the adapter
        test_params = {
            'since': since,
            'limit': 50
        }
        
        # Try to get order history using the adapter method directly
        adapter_orders = await adapter._get_bybit_order_history(symbol, test_params)
        
        if adapter_orders:
            logger.info(f"Fixed adapter success! Found {len(adapter_orders)} orders via adapter")
            # Log first order details
            if len(adapter_orders) > 0:
                logger.info(f"Sample adapter order: {adapter_orders[0]}")
        else:
            logger.warning(f"No orders found via fixed adapter for {symbol}")
            
        # Close the exchange
        await exchange.close()
        
    except Exception as e:
        logger.error(f"Error during test: {e}")


def apply_fix():
    """
    Apply the fix to the ccxt_adapter.py file.
    
    This function makes a backup of the original file and then creates a fixed version.
    """
    try:
        adapter_path = "src/alpha_pulse/exchanges/adapters/ccxt_adapter.py"
        backup_path = "src/alpha_pulse/exchanges/adapters/ccxt_adapter.py.backup"
        
        # Make a backup if it doesn't exist
        if not os.path.exists(backup_path):
            with open(adapter_path, 'r') as src_file, open(backup_path, 'w') as backup_file:
                backup_file.write(src_file.read())
                logger.info(f"Created backup at {backup_path}")
        
        # Read the file content
        with open(adapter_path, 'r') as file:
            content = file.readlines()
        
        # Find and fix the problematic lines
        in_bybit_history = False
        for i, line in enumerate(content):
            # Detect the _get_bybit_order_history method
            if "async def _get_bybit_order_history" in line:
                in_bybit_history = True
                
            # Fix for open orders
            if in_bybit_history and "await self.exchange.fetch_open_orders(symbol, **category_params)" in line:
                logger.info(f"Fixing fetch_open_orders call at line {i+1}")
                content[i] = line.replace("**category_params", "**open_params")
                
            # Fix for closed orders
            if in_bybit_history and "await self.exchange.fetch_closed_orders(symbol, **category_params)" in line:
                logger.info(f"Fixing fetch_closed_orders call at line {i+1}")
                content[i] = line.replace("**category_params", "**closed_params")
                
            # End of method
            if in_bybit_history and line.strip() == "return []":
                in_bybit_history = False
        
        # Write the fixed content back
        with open(adapter_path, 'w') as file:
            file.writelines(content)
            
        logger.info(f"Fix applied to {adapter_path}")
        
    except Exception as e:
        logger.error(f"Error applying fix: {e}")


async def main():
    """Main entry point."""
    # Print banner
    print("\n" + "=" * 80)
    print(" BYBIT ORDER HISTORY FIX TEST ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Apply the fix
    print("Applying the fix to ccxt_adapter.py...")
    apply_fix()
    
    # Run the test
    print("\nTesting order history retrieval with the fix...\n")
    await test_fixed_order_history()
    
    print("\n" + "=" * 80)
    print(" TEST COMPLETED ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())