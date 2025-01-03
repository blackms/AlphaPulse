from data_pipeline.database import get_db
from data_pipeline.models import Base, OHLCV
from datetime import datetime
import pytest

def test_database_connection():
    # Try to create tables and insert sample data
    with get_db() as db:
        # Create all tables
        Base.metadata.create_all(bind=db.get_bind())
        
        # Create sample OHLCV data
        sample_data = OHLCV(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=datetime.utcnow(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0
        )
        
        # Add and commit
        db.add(sample_data)
        db.commit()
        
        # Query back
        result = db.query(OHLCV).first()
        assert result is not None
        assert result.symbol == "BTC/USDT"
        
        # Clean up
        db.query(OHLCV).delete()
        db.commit()

if __name__ == "__main__":
    test_database_connection() 