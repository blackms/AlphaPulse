from alpha_pulse.data_pipeline.exchange import ExchangeManager
import pytest
from loguru import logger

def test_exchange_connection():
    # Initialize exchange manager
    manager = ExchangeManager()
    
    # Test connection to Binance
    try:
        exchange = manager.get_exchange('binance')
        assert exchange is not None
        logger.info("Successfully connected to Binance")
        
        # Test ticker fetching
        ticker = manager.get_ticker('binance', 'BTC/USDT')
        assert ticker is not None
        assert 'last' in ticker
        logger.info(f"Successfully fetched BTC/USDT ticker: {ticker['last']}")
        
    except Exception as e:
        logger.error(f"Exchange test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_exchange_connection() 