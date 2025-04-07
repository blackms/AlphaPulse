"""
Database writing functions for the data pipeline.
"""
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from loguru import logger
from typing import List, Dict, Any

# Assuming models are defined here or imported
from ..models import OHLCVRecord # Adjust import based on actual location

async def write_ohlcv_bulk(
    session: AsyncSession,
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str
):
    """
    Bulk inserts or updates OHLCV data into the database.

    Uses PostgreSQL's INSERT ... ON CONFLICT DO UPDATE to handle duplicates
    based on the composite primary key (timestamp, symbol, timeframe).

    Args:
        session: The SQLAlchemy AsyncSession.
        df: Pandas DataFrame containing OHLCV data (Open, High, Low, Close, Volume)
            with a DatetimeIndex.
        exchange: The name of the exchange (e.g., 'yfinance', 'binance').
        symbol: The trading symbol (e.g., '^GSPC', 'BTC/USDT').
        timeframe: The data timeframe (e.g., '1d', '1h').
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame received for {exchange}:{symbol}:{timeframe}. Skipping write.")
        return

    records_to_insert: List[Dict[str, Any]] = []
    for timestamp, row in df.iterrows():
        # Ensure timestamp is timezone-aware (should be handled by provider, but double-check)
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')

        record = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": ts,
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": float(row['Volume']),
        }
        records_to_insert.append(record)

    if not records_to_insert:
        logger.warning(f"No valid records generated from DataFrame for {exchange}:{symbol}:{timeframe}.")
        return

    try:
        # Use INSERT ... ON CONFLICT DO UPDATE (Upsert)
        # Assumes a composite primary key or unique constraint on (timestamp, symbol, timeframe)
        # Adjust index_elements if your primary key / unique constraint is different
        stmt = pg_insert(OHLCVRecord).values(records_to_insert)

        # Define the columns to update on conflict
        update_dict = {
            col.name: col
            for col in stmt.excluded
            if col.name not in ["timestamp", "symbol", "timeframe"] # Don't update PK columns
        }

        # Define the ON CONFLICT clause
        # Requires PostgreSQL >= 9.5
        # Ensure the constraint name matches your actual unique constraint or primary key
        # If using default PK name from SQLAlchemy convention: pk_ohlcv_data
        # If defining a specific UNIQUE constraint: uq_ohlcv_data_ts_sym_tf (example)
        # For composite PK, list the columns:
        on_conflict_stmt = stmt.on_conflict_do_update(
            index_elements=['timestamp', 'symbol', 'timeframe'], # Columns forming the unique constraint/PK
            set_=update_dict
        )

        await session.execute(on_conflict_stmt)
        await session.commit()
        logger.info(f"Successfully upserted {len(records_to_insert)} records for {exchange}:{symbol}:{timeframe}.")

    except Exception as e:
        await session.rollback() # Rollback on error
        logger.error(f"Error writing bulk OHLCV data for {exchange}:{symbol}:{timeframe}: {e}")
        # Optionally re-raise or handle specific exceptions

# Example usage (within an async function)
async def example_usage(session: AsyncSession):
    # Assume df is a DataFrame fetched by a provider
    # df = YFinanceProvider().fetch_ohlcv(...)
    df = pd.DataFrame({
        'Open': [100, 101], 'High': [102, 102], 'Low': [99, 100],
        'Close': [101, 101.5], 'Volume': [1000, 1200]
    }, index=pd.to_datetime(['2023-01-01 10:00:00+00:00', '2023-01-01 11:00:00+00:00']))

    await write_ohlcv_bulk(session, df, 'example_exchange', 'EXAMPLE/USD', '1h')