"""
Data Loader for the AlphaPulse Backtesting Framework.

Loads historical market data from the database.
"""
import pandas as pd
import datetime as dt # Use alias for datetime module
from typing import Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

# Assuming models and session management are available from data_pipeline
# Adjust imports based on actual project structure
try:
    from alpha_pulse.data_pipeline.models import OHLCVRecord
    from alpha_pulse.data_pipeline.session import get_session_context # Use context manager for session handling
except ImportError as e:
    logger.error(f"Error importing data_pipeline components: {e}")
    # Define dummy classes or raise error to indicate dependency issue
    class OHLCVRecord: pass
    async def get_session_context(): raise NotImplementedError("Database session module not found")


async def load_ohlcv_data(
    symbols: List[str],
    timeframe: str,
    start_dt: dt.datetime, # Use dt.datetime for type hint
    end_dt: dt.datetime,   # Use dt.datetime for type hint
    exchange: str = 'yfinance' # Default to yfinance for now
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Loads OHLCV data for specified symbols and timeframe from the database.

    Args:
        symbols: A list of ticker symbols to load.
        timeframe: The timeframe string (e.g., '1d', '1h').
        start_dt: The start datetime (inclusive).
        end_dt: The end datetime (exclusive).
        exchange: The exchange name to filter by (defaults to 'yfinance').

    Returns:
        A dictionary where keys are symbols and values are pandas DataFrames
        containing OHLCV data, indexed by timestamp. Returns None if loading fails.
    """
    logger.info(
        f"Loading data from DB for symbols: {symbols}, timeframe: {timeframe}, "
        f"range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}, "
        f"exchange: {exchange}"
    )
    all_data: Dict[str, pd.DataFrame] = {}

    try:
        async with get_session_context() as session: # Use async context manager
            for symbol in symbols:
                logger.debug(f"Querying data for {symbol}...")
                stmt = (
                    select(OHLCVRecord)
                    .where(
                        OHLCVRecord.symbol == symbol,
                        OHLCVRecord.timeframe == timeframe,
                        OHLCVRecord.exchange == exchange,
                        OHLCVRecord.timestamp >= start_dt,
                        OHLCVRecord.timestamp < end_dt
                    )
                    .order_by(OHLCVRecord.timestamp)
                )
                result = await session.execute(stmt)
                records = result.scalars().all()

                if not records:
                    logger.warning(f"No data found in DB for {symbol} {timeframe} in the specified range.")
                    all_data[symbol] = pd.DataFrame() # Return empty DataFrame for this symbol
                    continue

                # Convert records to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': r.timestamp,
                        'Open': r.open,
                        'High': r.high,
                        'Low': r.low,
                        'Close': r.close,
                        'Volume': r.volume
                    } for r in records
                ])
                df.set_index('timestamp', inplace=True)
                # Ensure index is UTC (should be already from DB)
                if df.index.tz is None:
                     df.index = df.index.tz_localize('UTC')
                elif df.index.tz != dt.timezone.utc: # Use alias dt here
                     df.index = df.index.tz_convert('UTC')

                all_data[symbol] = df
                logger.info(f"Loaded {len(df)} records for {symbol} {timeframe} from DB.")

        return all_data

    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (requires database to be populated)
    import asyncio
    import os

    async def run_example():
        # Setup environment variables for DB connection
        os.environ['DB_HOST'] = 'localhost'
        os.environ['DB_PORT'] = '5432'
        os.environ['DB_NAME'] = 'backtesting'
        os.environ['DB_USER'] = 'devuser'
        os.environ['DB_PASS'] = 'devpassword'

        symbols_to_load = ["^GSPC", "^VIX"]
        timeframe_to_load = "1d"
        start_date = dt.datetime(2022, 1, 1)
        end_date = dt.datetime(2023, 1, 1)

        loaded_data = await load_ohlcv_data(
            symbols=symbols_to_load,
            timeframe=timeframe_to_load,
            start_dt=start_date,
            end_dt=end_date
        )

        if loaded_data:
            for symbol, df in loaded_data.items():
                print(f"\n--- Data for {symbol} ({timeframe_to_load}) ---")
                if not df.empty:
                    print(df.head())
                    print("...")
                    print(df.tail())
                    print(f"Shape: {df.shape}")
                else:
                    print("No data loaded.")
        else:
            print("Failed to load data.")

    asyncio.run(run_example())