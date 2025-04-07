#!/usr/bin/env python3
"""
Main script to run the AlphaPulse data download pipeline.

Fetches market data from configured providers (e.g., yfinance) and stores it
in the database (e.g., TimescaleDB).
"""

import asyncio
import os
from datetime import datetime, timedelta
from loguru import logger
import sys
from pathlib import Path

# Ensure src directory is in path if running script directly from root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Import necessary components from alpha_pulse
# (Adjust imports if structure changes)
try:
    from alpha_pulse.data_pipeline.providers.yfinance_provider import YFinanceProvider
    from alpha_pulse.data_pipeline.storage.db_writer import write_ohlcv_bulk
    # Import from the package __init__ which now exposes session stuff
    from alpha_pulse.data_pipeline import async_session
except ImportError as e:
    logger.error(f"Error importing alpha_pulse components: {e}")
    logger.error("Ensure the project is installed correctly (pip install -e .) and PYTHONPATH is set if needed.")
    sys.exit(1)

# --- Configuration ---
# TODO: Load this from a config file (e.g., config/downloader_config.yaml)
SYMBOLS_TO_DOWNLOAD = ["^GSPC", "^VIX"] # SP500 and VIX
TIMEFRAMES_TO_DOWNLOAD = ["1d", "1h"]
DEFAULT_START_DATE = datetime.now() - timedelta(days=5*365) # Default to 5 years back
DEFAULT_END_DATE = datetime.now()

# Configure logger
log_file_path = project_root / "logs" / "downloader_{time}.log"
logger.add(log_file_path, rotation="10 MB", retention="7 days", level="DEBUG")
# --- End Configuration ---


async def download_data_for_symbol(provider, symbol, timeframes, start_date, end_date):
    """Fetches and writes data for a single symbol across specified timeframes."""
    logger.info(f"Processing symbol: {symbol}")
    async with async_session() as session: # Get a new session for this symbol/task
        for timeframe in timeframes:
            try:
                df = provider.fetch_ohlcv(symbol, timeframe, start_date, end_date)
                if df is not None and not df.empty:
                    # Assuming yfinance provider maps to 'yfinance' exchange name
                    await write_ohlcv_bulk(session, df, 'yfinance', symbol, timeframe)
                else:
                    logger.warning(f"No data fetched for {symbol} - {timeframe}, skipping write.")
                # Add a small delay to avoid hitting API limits too quickly
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Failed processing {symbol} - {timeframe}: {e}")
                # Continue to next timeframe/symbol even if one fails

async def main():
    """Main asynchronous function to run the download process."""
    logger.info("Starting data download process...")

    # Get date range from environment variables or use defaults
    start_str = os.environ.get("DOWNLOAD_START_DATE")
    end_str = os.environ.get("DOWNLOAD_END_DATE")

    try:
        start_date = datetime.fromisoformat(start_str) if start_str else DEFAULT_START_DATE
        end_date = datetime.fromisoformat(end_str) if end_str else DEFAULT_END_DATE
        # Ensure dates are timezone-naive or consistently timezone-aware (UTC) for yfinance
        start_date = start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)
        logger.info(f"Data download range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    except ValueError:
        logger.error("Invalid date format in DOWNLOAD_START_DATE or DOWNLOAD_END_DATE. Use YYYY-MM-DD.")
        sys.exit(1)

    provider = YFinanceProvider()

    tasks = []
    for symbol in SYMBOLS_TO_DOWNLOAD:
        # Create a separate task for each symbol
        task = asyncio.create_task(
            download_data_for_symbol(provider, symbol, TIMEFRAMES_TO_DOWNLOAD, start_date, end_date)
        )
        tasks.append(task)

    # Wait for all symbol processing tasks to complete
    await asyncio.gather(*tasks)

    logger.info("Data download process finished.")

if __name__ == "__main__":
    # Setup environment variables for DB connection (replace with your actual values or use .env file)
    # These should match the user/db created and configured in alembic.ini
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '5432'
    os.environ['DB_NAME'] = 'backtesting'
    os.environ['DB_USER'] = 'devuser'
    os.environ['DB_PASS'] = 'devpassword' # Be cautious with hardcoding passwords

    # Optional: Set download date range via environment variables
    # os.environ['DOWNLOAD_START_DATE'] = '2020-01-01'
    # os.environ['DOWNLOAD_END_DATE'] = '2024-01-01'

    asyncio.run(main())