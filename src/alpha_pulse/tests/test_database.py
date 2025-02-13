"""
Tests for the database module.
"""
from datetime import datetime, UTC
from decimal import Decimal
import pytest
import pytest_asyncio
from loguru import logger
import aiosqlite
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..data_pipeline.models import Base, OHLCVRecord, OHLCV
from ..exchanges.factories import ExchangeType


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Set up test database."""
    # Use SQLite memory database for testing
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    # Create async session factory
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.mark.asyncio
async def test_ohlcv_conversion(setup_database):
    """Test OHLCV data conversion and storage."""
    db = setup_database
    
    try:
        # Create sample OHLCV data
        ohlcv = OHLCV(
            timestamp=datetime.now(UTC),
            open=Decimal('50000.0'),
            high=Decimal('51000.0'),
            low=Decimal('49000.0'),
            close=Decimal('50500.0'),
            volume=Decimal('100.0')
        )
        
        # Convert to database record
        record = OHLCVRecord.from_ohlcv(
            exchange_type=ExchangeType.BINANCE,
            symbol="BTC/USDT",
            timeframe="1h",
            ohlcv=ohlcv
        )
        
        # Save to database
        db.add(record)
        await db.commit()
        
        # Query and verify
        stmt = select(OHLCVRecord).where(OHLCVRecord.id == record.id)
        result = await db.execute(stmt)
        saved_record = result.scalar_one()
        assert saved_record is not None
        
        # Convert back to OHLCV
        converted = saved_record.to_ohlcv()
        assert isinstance(converted, OHLCV)
        assert converted.open == ohlcv.open
        assert converted.high == ohlcv.high
        assert converted.low == ohlcv.low
        assert converted.close == ohlcv.close
        assert converted.volume == ohlcv.volume
        assert converted.exchange == ExchangeType.BINANCE.value
        assert converted.symbol == "BTC/USDT"
        assert converted.timeframe == "1h"
        
        logger.info("OHLCV conversion test completed successfully")
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_bulk_ohlcv_operations(setup_database):
    """Test bulk OHLCV operations."""
    db = setup_database
    
    try:
        # Create multiple OHLCV records
        records = []
        for i in range(5):
            ohlcv = OHLCV(
                timestamp=datetime.now(UTC),
                open=Decimal(f'{50000.0 + i}'),
                high=Decimal(f'{51000.0 + i}'),
                low=Decimal(f'{49000.0 + i}'),
                close=Decimal(f'{50500.0 + i}'),
                volume=Decimal(f'{100.0 + i}')
            )
            record = OHLCVRecord.from_ohlcv(
                exchange_type=ExchangeType.BINANCE,
                symbol="BTC/USDT",
                timeframe="1h",
                ohlcv=ohlcv
            )
            records.append(record)
        
        # Bulk insert
        db.add_all(records)
        await db.commit()
        
        # Query and verify
        stmt = select(func.count()).select_from(OHLCVRecord)
        result = await db.execute(stmt)
        count = result.scalar()
        assert count == 5
        
        logger.info("Bulk OHLCV operations completed successfully")
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise