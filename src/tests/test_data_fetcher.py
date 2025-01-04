from data_pipeline.data_fetcher import DataFetcher
from data_pipeline.database import get_db
from data_pipeline.models import OHLCV
from datetime import datetime, timedelta
from loguru import logger

def test_data_fetcher():
    fetcher = DataFetcher()
    
    # Test fetching and storing recent data
    try:
        # Fetch last 24 hours of data
        fetcher.update_historical_data(
            exchange_id='binance',
            symbol='BTC/USDT',
            timeframe='1h',
            days_back=1
        )
        
        logger.info("Successfully fetched and stored historical data")
        
        # Verify data in database
        with get_db() as db:
            count = db.query(OHLCV).filter(
                OHLCV.exchange == 'binance',
                OHLCV.symbol == 'BTC/USDT',
                OHLCV.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).count()
            
            assert count > 0
            logger.info(f"Found {count} records in database")
            
    except Exception as e:
        logger.error(f"Data fetcher test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_data_fetcher()