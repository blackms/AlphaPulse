"""
Tests for the database module.
"""
from datetime import datetime, UTC
import pytest
from loguru import logger

from alpha_pulse.data_pipeline.database import get_db
from alpha_pulse.data_pipeline.models import Base, OHLCV, Strategy, Trade


@pytest.fixture(autouse=True)
def setup_database():
    """Set up test database."""
    with get_db() as db:
        # Drop all tables
        Base.metadata.drop_all(bind=db.get_bind())
        # Create all tables
        Base.metadata.create_all(bind=db.get_bind())
        yield db


def test_database_connection(setup_database):
    """Test database connection and basic operations."""
    with get_db() as db:
        try:
            # Create sample strategy
            strategy = Strategy(
                name="Simple Moving Average",
                type="ta",
                parameters={"short_window": 10, "long_window": 20},
                created_at=datetime.now(UTC)
            )
            db.add(strategy)
            db.flush()  # Flush to get the strategy ID
            
            # Create sample OHLCV data
            sample_ohlcv = OHLCV(
                exchange="binance",
                symbol="BTC/USDT",
                timestamp=datetime.now(UTC),
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
                timestamp=datetime.now(UTC)
            )
            db.add(sample_trade)
            
            db.commit()
            
            # Verify data
            assert db.query(OHLCV).count() == 1
            assert db.query(Strategy).count() == 1
            assert db.query(Trade).count() == 1
            
            logger.info("Database operations completed successfully")
            
        except Exception as e:
            logger.error(f"Database test failed: {str(e)}")
            raise


if __name__ == "__main__":
    pytest.main([__file__])