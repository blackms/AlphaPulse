"""
Test the order history retrieval functionality to identify and fix the bug causing 
'No order history found for BTC/USDT' warnings despite orders being present.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import modules
from alpha_pulse.exchanges.factories import ExchangeFactory
from alpha_pulse.exchanges.types import ExchangeType
from alpha_pulse.exchanges.adapters.ccxt_adapter import CCXTAdapter
from alpha_pulse.data_pipeline.database.connection import get_pg_connection
from alpha_pulse.data_pipeline.database.exchange_cache import ExchangeCacheRepository


async def test_order_history_retrieval():
    """Test the order history retrieval functionality."""
    logger.info("Testing order history retrieval...")
    
    # Get exchange type from environment
    exchange_type_str = os.environ.get('EXCHANGE_TYPE', 'bybit')
    exchange_id = exchange_type_str.lower()
    
    # Get API credentials from environment
    api_key = os.environ.get(f'{exchange_id.upper()}_API_KEY', 
                            os.environ.get('EXCHANGE_API_KEY', ''))
    api_secret = os.environ.get(f'{exchange_id.upper()}_API_SECRET', 
                               os.environ.get('EXCHANGE_API_SECRET', ''))
    
    # Determine if using testnet
    if exchange_id == 'bybit':
        if 'BYBIT_TESTNET' in os.environ:
            testnet = os.environ.get('BYBIT_TESTNET', '').lower() == 'true'
        else:
            testnet = False
    else:
        testnet = os.environ.get('EXCHANGE_TESTNET', 'true').lower() == 'true'
    
    logger.info(f"Creating {exchange_id} exchange with testnet={testnet}")
    
    # Create exchange
    if exchange_id == 'binance':
        exchange_type = ExchangeType.BINANCE
    else:
        exchange_type = ExchangeType.BYBIT
    
    # Create the exchange and initialize it
    exchange = ExchangeFactory.create_exchange(
        exchange_type,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )
    
    await exchange.initialize()
    
    # Get the specific symbols to test
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    # Test each symbol
    for symbol in symbols:
        logger.info(f"Testing order history retrieval for {symbol}")
        
        # Direct method test 
        if isinstance(exchange, CCXTAdapter):
            # Test the get_closed_orders method
            try:
                logger.info(f"Requesting closed orders for {symbol}...")
                closed_orders = await exchange.get_closed_orders(symbol)
                logger.info(f"Retrieved {len(closed_orders)} closed orders for {symbol}")
                
                # Debug output of a few orders if available
                if closed_orders:
                    logger.info(f"Sample order 1: {closed_orders[0]}")
                    if len(closed_orders) > 1:
                        logger.info(f"Sample order 2: {closed_orders[1]}")
                
                # Test the get_average_entry_price method
                logger.info(f"Requesting average entry price for {symbol}...")
                avg_price = await exchange.get_average_entry_price(symbol)
                logger.info(f"Average entry price for {symbol}: {avg_price}")
                
            except Exception as e:
                logger.error(f"Error testing order history for {symbol}: {str(e)}")
        else:
            logger.warning(f"Exchange is not a CCXTAdapter, skipping direct testing")
    
    # Check cached orders in database
    logger.info("Checking cached orders in database...")
    async with get_pg_connection() as conn:
        repo = ExchangeCacheRepository(conn)
        orders = await repo.get_orders(exchange_id)
        
        if orders:
            logger.info(f"Found {len(orders)} cached orders for {exchange_id}")
            # Check BTC/USDT orders specifically
            btc_orders = [o for o in orders if o.symbol == "BTC/USDT"]
            logger.info(f"Found {len(btc_orders)} BTC/USDT orders in cache")
            
            # Display sample orders
            if btc_orders:
                logger.info(f"Sample BTC/USDT order: {btc_orders[0]}")
        else:
            logger.warning(f"No cached orders found for {exchange_id}")
    
    logger.info("Order history retrieval test complete")


async def main():
    """Run the order history retrieval test."""
    try:
        await test_order_history_retrieval()
    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())