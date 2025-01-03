from data_pipeline.database import get_db
from data_pipeline.models import Base, OHLCV, Strategy, Trade
from datetime import datetime
import pytest

def test_database_connection():
    with get_db() as db:
        # Create all tables
        Base.metadata.create_all(bind=db.get_bind())
        
        # Create sample strategy
        strategy = Strategy(
            name="Simple Moving Average",
            type="ta",
            parameters={"short_window": 10, "long_window": 20},
            created_at=datetime.utcnow()
        )
        db.add(strategy)
        db.commit()
        
        # Create sample OHLCV data
        sample_ohlcv = OHLCV(
            exchange="binance",
            symbol="BTC/USDT",
            timestamp=datetime.utcnow(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0
        )
        db.add(sample_ohlcv)
        
        # Create sample trade
        sample_trade = Trade(
            strategy_id=strategy.id,
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            price=50000.0,
            timestamp=datetime.utcnow()
        )
        db.add(sample_trade)
        
        db.commit()
        
        # Verify data
        assert db.query(OHLCV).count() == 1
        assert db.query(Strategy).count() == 1
        assert db.query(Trade).count() == 1
        
        # Clean up
        db.query(Trade).delete()
        db.query(OHLCV).delete()
        db.query(Strategy).delete()
        db.commit()

if __name__ == "__main__":
    test_database_connection() 